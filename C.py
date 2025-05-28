import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
# Load utility functions from cloned repository

from src.loadData import GraphDataset
from src.utils import set_seed
from src.modelsASY import GNN
from sklearn.metrics import f1_score

import argparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

from itertools import product
import copy
import shutil
import gc

import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn as nn



# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def mixup_graphs(emb1, emb2, y1, y2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Optional: favor equal mix
    mixed_emb = lam * emb1 + (1 - lam) * emb2
    mixed_targets = lam * F.one_hot(y1, num_classes=emb1.size(1)) + (1 - lam) * F.one_hot(y2, num_classes=emb1.size(1))
    return mixed_emb, mixed_targets


def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            logits, _ = model(data)  # We only need logits for evaluation
            pred = logits.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

            # Only collect true labels if they exist
            if hasattr(data, 'y') and data.y is not None:
                true_labels.extend(data.y.cpu().numpy())

                if calculate_accuracy:
                    correct += (pred == data.y).sum().item()
                    total += data.y.size(0)
                    total_loss += criterion(logits, data.y).item()


    if calculate_accuracy and true_labels:
        accuracy = correct / total
        return total_loss / len(data_loader), accuracy, predictions, true_labels

    return None, None, predictions, true_labels 

class SoftCoTeaching:
    def __init__(self, model1, model2, optimizer1, optimizer2, criterion, 
                 forget_rate=0.2, num_gradual=10, device='cuda', 
                 temperature=1.0, weight_smoothing='sigmoid', use_mixup=False, mixup_start_epoch=10):
        """
        Soft Co-Teaching implementation for noise-robust training
        
        Args:
            model1, model2: Two identical GNN models
            optimizer1, optimizer2: Optimizers for each model
            criterion: Loss function
            forget_rate: Rate controlling the strength of downweighting
            num_gradual: Number of epochs to gradually increase forget rate
            device: Training device
            temperature: Temperature parameter for soft weighting
            weight_smoothing: Method for computing weights ('sigmoid', 'exp', 'linear')
            """
        self.model1 = model1
        self.model2 = model2
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.criterion = criterion
        self.forget_rate = forget_rate
        self.num_gradual = num_gradual
        self.device = device
        self.temperature = temperature
        self.weight_smoothing = weight_smoothing
        self.use_mixup = use_mixup
        self.mixup_start_epoch = mixup_start_epoch
        
    def get_forget_rate(self, epoch):
        """Gradually increase forget rate over epochs"""
        if epoch < self.num_gradual:
            return self.forget_rate * epoch / self.num_gradual
        return self.forget_rate
    
    def compute_soft_weights(self, losses, forget_rate):
        """
        Compute soft weights for samples based on their losses
        
        Args:
            losses: Tensor of individual sample losses
            forget_rate: Current forget rate
            Returns:
            weights: Soft weights for each sample
        """
        if self.weight_smoothing == 'sigmoid':
            # Sigmoid-based soft weighting
            # Normalize losses to [0, 1] range
            normalized_losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
            
            # Apply sigmoid with temperature and shift based on forget_rate
            # Higher forget_rate makes the function steeper, downweighting noisy samples more
            shift = torch.quantile(normalized_losses, 1 - forget_rate)
            weights = torch.sigmoid(-(normalized_losses - shift) / self.temperature)
            
        elif self.weight_smoothing == 'exp':
            # Exponential decay based on loss rank
            sorted_losses, sorted_indices = torch.sort(losses)
            ranks = torch.argsort(sorted_indices).float()
            normalized_ranks = ranks / (len(losses) - 1)

    # Exponential decay: higher loss -> lower weight
            decay_factor = forget_rate * 5  # Scale factor for exponential decay
            weights = torch.exp(-decay_factor * normalized_ranks)
            
        elif self.weight_smoothing == 'linear':
            # Linear decay based on loss percentile
            sorted_losses, _ = torch.sort(losses)
            percentiles = torch.searchsorted(sorted_losses, losses).float() / len(losses)
            
            # Linear decay from 1 to (1-forget_rate)
            weights = 1.0 - forget_rate * percentiles
            weights = torch.clamp(weights, min=0.1)  # Minimum weight to avoid complete exclusion
            
        else:
            raise ValueError(f"Unknown weight_smoothing method: {self.weight_smoothing}")
        
        return weights

    def weighted_loss(self, outputs, targets, weights):
        """Compute weighted cross-entropy loss"""
        losses = F.cross_entropy(outputs, targets, reduction='none')
        weighted_losses = losses * weights
        return weighted_losses.mean()
         
    def train_epoch(self, data_loader, epoch):
        """Train both models for one epoch using soft co-teaching or NCOD+ with optional MixUp"""
        self.model1.train()
        self.model2.train()

        total_loss1 = 0
        total_loss2 = 0
        correct1 = 0
        correct2 = 0
        total = 0
        total_weight_sum1 = 0
        total_weight_sum2 = 0
        current_forget_rate = self.get_forget_rate(epoch)
        use_mixup_now = self.use_mixup and epoch >= getattr(self, "mixup_start_epoch", 7)

        if self.use_mixup and epoch == getattr(self, "mixup_start_epoch", 7):
            print(f"🔀 MixUp activated starting from epoch {epoch}")

        for data in data_loader:
            data = data.to(self.device)

            logits1, emb1 = self.model1(data)
            logits2, emb2 = self.model2(data)

            if use_mixup_now and not hasattr(self.criterion, 'update_centroids'):
                lam = np.random.beta(1.0, 1.0)
                batch_size = data.y.size(0)
                index = torch.randperm(batch_size).to(self.device)

                y1 = F.one_hot(data.y, num_classes=logits1.size(1)).float()
                y2 = F.one_hot(data.y[index], num_classes=logits1.size(1)).float()

                mixed_emb1 = lam * emb1 + (1 - lam) * emb1[index]
                mixed_emb2 = lam * emb2 + (1 - lam) * emb2[index]

                mixed_y1 = lam * y1 + (1 - lam) * y2
                mixed_y2 = lam * y2 + (1 - lam) * y1

                logits1 = self.model1.graph_pred_linear(mixed_emb1)
                logits2 = self.model2.graph_pred_linear(mixed_emb2)

                loss1 = F.kl_div(F.log_softmax(logits1, dim=1), mixed_y1, reduction='batchmean')
                loss2 = F.kl_div(F.log_softmax(logits2, dim=1), mixed_y2, reduction='batchmean')

                total_weight_sum1 += 1
                total_weight_sum2 += 1

            else:
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()

                if hasattr(self.criterion, 'update_centroids'):
                    if use_mixup_now:
                        lam = np.random.beta(1.0, 1.0)
                        index = torch.randperm(data.y.size(0)).to(self.device)

                        y1 = data.y
                        y2 = data.y[index]

                        mixed_emb1 = lam * emb1 + (1 - lam) * emb1[index]
                        mixed_emb2 = lam * emb2 + (1 - lam) * emb2[index]

                        logits1 = self.model1.graph_pred_linear(mixed_emb1)
                        logits2 = self.model2.graph_pred_linear(mixed_emb2)

                        self.criterion.update_centroids(emb1.detach(), y1)

                        loss1 = self.criterion(mixed_emb1, logits1, y1, alt_logits=logits2.detach())
                        loss2 = self.criterion(mixed_emb2, logits2, y2, alt_logits=logits1.detach())

                    else:
                        self.criterion.update_centroids(emb1.detach(), data.y)
                        loss1 = self.criterion(emb1, logits1, data.y, alt_logits=logits2.detach())
                        loss2 = self.criterion(emb2, logits2, data.y, alt_logits=logits1.detach())

                    total_weight_sum1 += 1
                    total_weight_sum2 += 1

                else:
                    loss1_vec = F.cross_entropy(logits1, data.y, reduction='none')
                    loss2_vec = F.cross_entropy(logits2, data.y, reduction='none')

                    weights_for_model2 = self.compute_soft_weights(loss1_vec.detach(), current_forget_rate)
                    weights_for_model1 = self.compute_soft_weights(loss2_vec.detach(), current_forget_rate)

                    loss1 = self.weighted_loss(logits1, data.y, weights_for_model1)
                    loss2 = self.weighted_loss(logits2, data.y, weights_for_model2)

                    total_weight_sum1 += weights_for_model1.sum().item()
                    total_weight_sum2 += weights_for_model2.sum().item()

            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()

            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()

            pred1 = logits1.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)
            correct1 += (pred1 == data.y).sum().item()
            correct2 += (pred2 == data.y).sum().item()
            total += data.y.size(0)
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

        avg_loss1 = total_loss1 / len(data_loader)
        avg_loss2 = total_loss2 / len(data_loader)
        acc1 = correct1 / total
        acc2 = correct2 / total
        avg_weight1 = total_weight_sum1 / len(data_loader)
        avg_weight2 = total_weight_sum2 / len(data_loader)

        return avg_loss1, avg_loss2, acc1, acc2, avg_weight1, avg_weight2







class AdaptiveSoftCoTeaching(SoftCoTeaching):
    """
    Adaptive version that adjusts temperature based on training progress
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_temperature = self.temperature
        self.loss_history = []
        
    def adapt_temperature(self, epoch, current_loss):
        """Adapt temperature based on training progress"""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) > 5:  # Wait for some history
            recent_improvement = self.loss_history[-5] - current_loss
            
            if recent_improvement < 0.01:  # Slow improvement
                # Increase temperature to be more selective
                self.temperature = min(2.0, self.temperature * 1.1)
            elif recent_improvement > 0.05:  # Fast improvement
                # Decrease temperature to be less selective
                self.temperature = max(0.1, self.temperature * 0.9)
    
    def train_epoch(self, data_loader, epoch):
        """Train with adaptive temperature"""
        results = super().train_epoch(data_loader, epoch)
        avg_loss = (results[0] + results[1]) / 2
        self.adapt_temperature(epoch, avg_loss)
        return results
    
def train_with_soft_co_teaching(args, device, use_adaptive=False, checkpoint_path=None):
    """Main training function using soft co-teaching with early stopping"""
    model1, model2 = create_co_teaching_models(args, device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=args.epochs, eta_min=1e-5)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.epochs, eta_min=1e-5)

    if args.baseline_mode == 2:
        criterion = NoisyCrossEntropyLoss(args.noise_prob)
    elif args.baseline_mode == 3:
        criterion = GeneralizedCrossEntropyLoss(q=0.7)
    elif args.baseline_mode == 4:
        criterion = SymmetricCrossEntropyLoss(alpha=1.0, beta=1.0)
    elif args.baseline_mode == 5:
        criterion = NCODPlusLoss(num_classes=6, emb_dim=args.emb_dim)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    CoTeachingClass = AdaptiveSoftCoTeaching if use_adaptive else SoftCoTeaching
    co_teacher = CoTeachingClass(
        model1=model1,
        model2=model2, 
        optimizer1=optimizer1,
        optimizer2=optimizer2,
        criterion=criterion,
        forget_rate=args.noise_prob,
        num_gradual=args.epochs // 3,
        device=device,
        temperature=1.0,
        weight_smoothing='sigmoid',
        use_mixup = True,
        mixup_start_epoch = 7
    )
    

    full_dataset = GraphDataset(args.train_path, transform=add_zeros)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(12))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_f1 = 0.0
    best_val_accuracy = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(args.epochs):
        loss1, loss2, acc1, acc2, avg_weight1, avg_weight2 = co_teacher.train_epoch(train_loader, epoch)

        val_loss1, val_acc1, val_preds1, val_labels1 = evaluate(val_loader, model1, device, calculate_accuracy=True)
        val_loss2, val_acc2, val_preds2, val_labels2 = evaluate(val_loader, model2, device, calculate_accuracy=True)

        val_f1_1 = f1_score(val_labels1, val_preds1, average='macro')
        val_f1_2 = f1_score(val_labels2, val_preds2, average='macro')

        val_f1 = max(val_f1_1, val_f1_2)
        best_model = model1 if val_f1_1 >= val_f1_2 else model2

        scheduler1.step()
        scheduler2.step()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Learning Rates - Model1: {scheduler1.get_last_lr()[0]:.6f}, Model2: {scheduler2.get_last_lr()[0]:.6f}")
        print(f"Model1 - Train Loss: {loss1:.4f}, Train Acc: {acc1:.4f}, Val Acc: {val_acc1:.4f}, Val F1: {val_f1_1:.4f}")
        print(f"Model2 - Train Loss: {loss2:.4f}, Train Acc: {acc2:.4f}, Val Acc: {val_acc2:.4f}, Val F1: {val_f1_2:.4f}")
        print(f"Average Weights - Model1: {avg_weight1:.4f}, Model2: {avg_weight2:.4f}")
        if use_adaptive:
            print(f"Current Temperature: {co_teacher.temperature:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            checkpoint_path = f"checkpoints/model_{os.path.basename(os.path.dirname(args.test_path))}_best.pth"
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
                'temperature': co_teacher.temperature if use_adaptive else 1.0
            }, checkpoint_path)

            print(f"✅ Best model updated and saved at {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
            break

    return (model1 if val_f1_1 >= val_f1_2 else model2), val_loader


def create_co_teaching_models(args, device):
    """Create two identical models for co-teaching"""
    if args.gnn == 'gin-virtual':
        model1 = GNN(gnn_type='gin', num_class=6, num_layer=args.num_layer,
                     emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
        model2 = GNN(gnn_type='gin', num_class=6, num_layer=args.num_layer,
                     emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    elif args.gnn == 'gcn-virtual':
        model1 = GNN(gnn_type='gcn', num_class=6, num_layer=args.num_layer,
                     emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
        model2 = GNN(gnn_type='gcn', num_class=6, num_layer=args.num_layer,
                     emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    else:
        raise ValueError("Unsupported GNN type, expected 'gin-virtual' or 'gcn-virtual'")
    
    return model1, model2


def save_predictions(predictions, test_path):
    script_dir = os.getcwd() 
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()


class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()
    
class GeneralizedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = torch.nn.functional.softmax(logits, dim=1)
        probs_correct = probs[torch.arange(len(targets)), targets]
        loss = (1 - probs_correct ** self.q) / self.q
        return loss.mean()


class SymmetricCrossEntropyLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        pred = torch.nn.functional.softmax(logits, dim=1)
        target_onehot = torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float()
        rce = (-pred * torch.log(target_onehot + 1e-6)).sum(dim=1)
        return self.alpha * self.ce(logits, targets) + self.beta * rce.mean()



class NCODLoss(nn.Module):
    def __init__(self, num_classes, emb_dim, use_outlier_discounting=True):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.use_outlier_discounting = use_outlier_discounting
        self.register_buffer("centroids", torch.zeros(num_classes, emb_dim))
        self.class_counts = torch.zeros(num_classes, dtype=torch.long)
        self.u = nn.Parameter(torch.ones(1))

    def update_centroids(self, embeddings, targets):
        for i in range(self.num_classes):
            mask = (targets == i)
            if mask.any():
                self.centroids[i] = embeddings[mask].mean(dim=0).detach()
                self.class_counts[i] = mask.sum().item()

    def forward(self, embeddings, logits, targets):
        # Classification loss (CE)
        ce_loss = F.cross_entropy(logits, targets)

        # Compute similarities to centroids
        sim = F.cosine_similarity(embeddings.unsqueeze(1), self.centroids.unsqueeze(0), dim=2)  # (B, C)
        soft_labels = F.softmax(sim, dim=1)  # pseudo labels from centroid similarity

        # Outlier discounting term
        if self.use_outlier_discounting:
            hard_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
            similarity_to_target = (soft_labels * hard_one_hot).sum(dim=1)
            discount = 1.0 - similarity_to_target
            reg_term = discount.mean() * self.u
        else:
            reg_term = 0.0

        # Optional: distance between true one-hot and soft label
        dist_reg = F.mse_loss(hard_one_hot, soft_labels.detach())

        return ce_loss + dist_reg + reg_term

class NCODPlusLoss(nn.Module):
    def __init__(self, num_classes, emb_dim, lambda_c=1.0, lambda_b=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.lambda_c = lambda_c
        self.lambda_b = lambda_b
        self.register_buffer("centroids", torch.zeros(num_classes, emb_dim))
        self.class_counts = torch.zeros(num_classes, dtype=torch.long)
        self.u = nn.Parameter(torch.ones(1))

    def update_centroids(self, embeddings, targets):
        for i in range(self.num_classes):
            mask = (targets == i)
            if mask.any():
                self.centroids[i] = embeddings[mask].mean(dim=0).detach()
                self.class_counts[i] = mask.sum().item()

    def forward(self, embeddings, logits, targets, alt_logits=None):
        device = logits.device  # Make sure all new tensors go here

        ce_loss = F.cross_entropy(logits, targets)

        sim = F.cosine_similarity(embeddings.unsqueeze(1), self.centroids.to(device).unsqueeze(0), dim=2)
        soft_labels = F.softmax(sim, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).float().to(device)

        similarity_to_target = (soft_labels * one_hot_targets).sum(dim=1)
        discount = 1.0 - similarity_to_target
        reg_term = discount.mean() * self.u.to(device)

        mse_term = F.mse_loss(soft_labels, one_hot_targets.detach())

        loss = ce_loss + mse_term + reg_term

        if alt_logits is not None:
            p1 = F.softmax(logits, dim=1)
            p2 = F.softmax(alt_logits, dim=1)

            kl_c = F.kl_div(p1.log(), p2, reduction='batchmean') + F.kl_div(p2.log(), p1, reduction='batchmean')
            avg_pred = (p1 + p2) / 2
            uniform_dist = torch.full_like(avg_pred, 1.0 / self.num_classes).to(device)
            kl_b = F.kl_div(avg_pred.log(), uniform_dist, reduction='batchmean')

            loss += self.lambda_c * kl_c + self.lambda_b * kl_b

        return loss



def run_evaluation(data_loader, model, device, label_map=None, save_csv_path=None, calculate_accuracy=True):
    result = evaluate(data_loader, model, device, calculate_accuracy=calculate_accuracy)

    if calculate_accuracy:
        loss, acc, preds, labels = result
        if labels:
            print("\nClassification Report:")
            print(classification_report(labels, preds, digits=5))
            macro_f1 = f1_score(labels, preds, average='macro')
            micro_f1 = f1_score(labels, preds, average='micro')
            print(f"Macro F1-score: {macro_f1:.5f}")
            print(f"Micro F1-score: {micro_f1:.5f}")
            if label_map:
                cm = confusion_matrix(labels, preds)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=label_map, yticklabels=label_map)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")
                plt.tight_layout()
                plt.show()
    else:
        _, _, preds, _ = result

    if save_csv_path and preds is not None:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        df = pd.DataFrame({
            "id": list(range(len(preds))),
            "pred": preds
        })
        df.to_csv(save_csv_path, index=False)
        print(f"Predictions saved to {save_csv_path}")




def main(args):
    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well


    # Define checkpoint path relative to the script's directory
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Create model before loading weights
    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', num_class=6, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', num_class=6, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', num_class=6, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', num_class=6, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    else:
        raise ValueError("Invalid GNN type")

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {checkpoint_path}")

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if hasattr(args, 'use_co_teaching') and args.use_co_teaching and args.train_path:
        model, val_loader = train_with_soft_co_teaching(args, device, use_adaptive=True, checkpoint_path=checkpoint_path)

    


    # Load the best model Generate predictions for the test set using the best model
    if 'model' not in locals():
        raise RuntimeError("Model was not initialized. This usually means training did not occur.")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])


    # Evaluation on validation set (if available)
    if 'val_loader' in locals():
        run_evaluation(val_loader, model, device,
                    label_map=["class0", "class1", "class2", "class3", "class4", "class5"],
                    calculate_accuracy=True)

    # Evaluation on test set (no labels, just predictions)
    run_evaluation(test_loader, model, device,
                save_csv_path=os.path.join("submission", f"testset_{os.path.basename(os.path.dirname(args.test_path))}.csv"),
               calculate_accuracy=False)
    
    if 'val_loader' in locals():
        del val_loader
        torch.cuda.empty_cache()
        gc.collect()


def get_arguments():
    args = {}

    # Default argument values
    args['train_path'] = "datasets/A/0.2_train.json"
    args['test_path'] = "datasets/A/0.2_test.json"
    args['num_checkpoints'] = 3
    args['device'] = 0
    args['gnn'] = 'gcn-virtual'
    args['drop_ratio'] = 0.0
    args['num_layer'] = 5
    args['emb_dim'] = 256
    args['batch_size'] = 32
    args['epochs'] = 200
    args['baseline_mode'] = 5
    args['noise_prob'] = 0.2
    args['use_co_teaching'] = True

    return argparse.Namespace(**args)



if __name__ == "__main__":

    args = get_arguments()
    main(args)
    #hyperparameter_search(args)