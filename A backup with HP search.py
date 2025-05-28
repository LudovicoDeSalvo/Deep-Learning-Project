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
from src.models import GNN

import argparse
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

from itertools import product
import copy
import shutil
import gc
from sklearn.metrics import f1_score

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR



# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        #loss = criterion(logits, targets, data.idx) #this is for ELR
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader),  correct / total

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
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())

            # Only collect true labels if they exist
            if hasattr(data, 'y') and data.y is not None:
                true_labels.extend(data.y.cpu().numpy())

                if calculate_accuracy:
                    correct += (pred == data.y).sum().item()
                    total += data.y.size(0)
                    total_loss += criterion(output, data.y).item()

    if calculate_accuracy and true_labels:
        accuracy = correct / total
        return total_loss / len(data_loader), accuracy, predictions, true_labels

    return predictions, None, None, None

class SoftCoTeaching:
    def __init__(self, model1, model2, optimizer1, optimizer2, criterion, 
                 forget_rate=0.2, num_gradual=10, device='cuda', 
                 temperature=1.0, weight_smoothing='sigmoid'):
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
        """Train both models for one epoch using soft co-teaching"""
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
        
        for batch_idx, data in enumerate(data_loader):
            data = data.to(self.device)
            batch_size = data.y.size(0)
            
            # Forward pass for both models
            output1 = self.model1(data)
            output2 = self.model2(data)
            # Calculate losses for each sample
            loss1_vec = F.cross_entropy(output1, data.y, reduction='none')
            loss2_vec = F.cross_entropy(output2, data.y, reduction='none')
            
            # Compute soft weights
            # Model 1 provides weights for Model 2's training
            weights_for_model2 = self.compute_soft_weights(loss1_vec.detach(), current_forget_rate)
            
            # Model 2 provides weights for Model 1's training
            weights_for_model1 = self.compute_soft_weights(loss2_vec.detach(), current_forget_rate)
            
            # Update Model 1 with soft weights from Model 2
            self.optimizer1.zero_grad()
            loss1_weighted = self.weighted_loss(output1, data.y, weights_for_model1)
            loss1_weighted.backward()
            self.optimizer1.step()

            # Update Model 2 with soft weights from Model 1
            self.optimizer2.zero_grad()
            loss2_weighted = self.weighted_loss(output2, data.y, weights_for_model2)
            loss2_weighted.backward()
            self.optimizer2.step()
            
            # Calculate accuracy and statistics
            pred1 = output1.argmax(dim=1)
            pred2 = output2.argmax(dim=1)
            correct1 += (pred1 == data.y).sum().item()
            correct2 += (pred2 == data.y).sum().item()
            total += data.y.size(0)
            
            total_loss1 += loss1_weighted.item()
            total_loss2 += loss2_weighted.item()
            total_weight_sum1 += weights_for_model1.sum().item()
            total_weight_sum2 += weights_for_model2.sum().item()

            avg_loss1 = total_loss1 / len(data_loader)
            avg_loss2 = total_loss2 / len(data_loader)
            acc1 = correct1 / total
            acc2 = correct2 / total
            avg_weight1 = total_weight_sum1 / total
            avg_weight2 = total_weight_sum2 / total
            
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
    
def train_with_soft_co_teaching(args, device, use_adaptive=False):
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
        criterion = EarlyLearningRegularizationLoss(lambda_elr=3.0)
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
        weight_smoothing='sigmoid'
    )

    full_dataset = GraphDataset(args.train_path, transform=add_zeros)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(12))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_f1 = 0.0
    best_val_accuracy = 0.0
    patience = 5
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

    return model1 if val_f1_1 >= val_f1_2 else model2


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


class EarlyLearningRegularizationLoss(torch.nn.Module):
    def __init__(self, lambda_elr=3.0):
        super().__init__()
        self.lambda_elr = lambda_elr
        self.target_history = {}

    def forward(self, logits, targets, ids):
        probs = torch.nn.functional.softmax(logits, dim=1)
        current_targets = probs.detach()

        # Update target history using global sample IDs
        for i, sample_id in enumerate(ids):
            sample_id = sample_id.item()
            if sample_id not in self.target_history:
                self.target_history[sample_id] = current_targets[i].clone()
            else:
                self.target_history[sample_id] = (
                    0.9 * self.target_history[sample_id] + 0.1 * current_targets[i].clone()
                )

        q = torch.stack([self.target_history[sample_id.item()] for sample_id in ids])
        dot = (probs * q).sum(dim=1)
        elr_reg = self.lambda_elr * torch.log(1 - dot + 1e-4).mean()
        ce_loss = torch.nn.functional.cross_entropy(logits, targets)
        return ce_loss + elr_reg
    
from sklearn.metrics import f1_score

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
        preds, _, _, _ = result

    if save_csv_path and preds is not None:
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
    

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # criterion = torch.nn.CrossEntropyLoss()
    if args.baseline_mode == 2:
        criterion = NoisyCrossEntropyLoss(args.noise_prob)
    elif args.baseline_mode == 3:
        criterion = GeneralizedCrossEntropyLoss(q=0.7)
    elif args.baseline_mode == 4:
        criterion = SymmetricCrossEntropyLoss(alpha=1.0, beta=1.0)
    elif args.baseline_mode == 5:
        criterion = EarlyLearningRegularizationLoss(lambda_elr=3.0)
    else:
        criterion = torch.nn.CrossEntropyLoss()

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

    # Load pre-trained model for inference
    if os.path.exists(checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # If train_path is provided, train the model
    if args.train_path:
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        
        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        num_epochs = args.epochs
        best_val_accuracy = 0.0   

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        best_val_f1 = 0.0
        patience = 5
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )

            val_loss, val_acc, val_preds, val_labels = evaluate(val_loader, model, device, calculate_accuracy=True)
            val_f1 = f1_score(val_labels, val_preds, average='macro')

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model updated and saved at {checkpoint_path}")
                logging.info(f"New best F1: {best_val_f1:.4f} at epoch {epoch + 1}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement for {patience} consecutive epochs")
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))
        plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, "plotsVal"))

    if hasattr(args, 'use_co_teaching') and args.use_co_teaching:
            # Use co-teaching training
            best_model = train_with_soft_co_teaching(args, device, use_adaptive=True)
            model = best_model  # Use the best model for testing  


    # Load the best model Generate predictions for the test set using the best model
    model.load_state_dict(torch.load(checkpoint_path))

    # Evaluation on validation set (if available)
    if 'val_loader' in locals():
        run_evaluation(val_loader, model, device,
                    label_map=["class0", "class1", "class2", "class3", "class4", "class5"],
                    calculate_accuracy=True)

    # Evaluation on test set (no labels, just predictions)
    run_evaluation(test_loader, model, device,
                save_csv_path=os.path.join("submission", f"testset_{os.path.basename(os.path.dirname(args.test_path))}.csv"),
               calculate_accuracy=False)


def get_arguments():
    args = {}

    # Default argument values
    args['train_path'] = "datasets/B/0.4_train.json"
    args['test_path'] = "datasets/B/0.4_test.json"
    args['num_checkpoints'] = 3
    args['device'] = 0
    args['gnn'] = 'gcn-virtual'
    args['drop_ratio'] = 0.0
    args['num_layer'] = 5
    args['emb_dim'] = 256
    args['batch_size'] = 32
    args['epochs'] = 100
    args['baseline_mode'] = 4
    args['noise_prob'] = 0.2
    args['use_co_teaching'] = True

    return argparse.Namespace(**args)



def hyperparameter_search(base_args):
    search_space = {
        'gnn': ['gin', 'gcn', 'gin-virtual','gcn-virtual'],
        'num_layer': [2, 3, 5],
        'emb_dim': [128, 256],
        'baseline_mode': [2, 3, 4],
        'epochs': [20],
        'drop_ratio': [0.0],
        'batch_size': [32],
        'noise_prob': [0.2]
    }

    keys, values = zip(*search_space.items())
    best_accuracy = 0.0
    best_config = None

    for combo in product(*values):
        trial_args = copy.deepcopy(base_args)
        config = dict(zip(keys, combo))

        for key, value in config.items():
            setattr(trial_args, key, value)

        print(f"\nTrying configuration: {config}")
        try:
            main(trial_args)
        except Exception as e:
            print(f"Failed on config {config}: {e}")
            continue

        # Evaluate using the saved best model
        checkpoint_path = os.path.join("checkpoints", f"model_{os.path.basename(os.path.dirname(trial_args.test_path))}_best.pth")
        if os.path.exists(checkpoint_path):
            model = GNN(
                gnn_type=config['gnn'].replace('-virtual', ''),
                num_class=6,
                num_layer=config['num_layer'],
                emb_dim=config['emb_dim'],
                drop_ratio=config['drop_ratio'],
                virtual_node='virtual' in config['gnn']
            ).to(torch.device("cpu"))

            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
            val_dataset = GraphDataset(trial_args.train_path, transform=add_zeros)
            val_loader = DataLoader(val_dataset, batch_size=trial_args.batch_size)
            _, _, val_preds, val_labels = evaluate(val_loader, model, device='cpu', calculate_accuracy=True)

            val_f1 = f1_score(val_labels, val_preds, average='macro')

            if val_f1 > best_accuracy:
                best_accuracy = val_f1
                best_config = config

            # Memory cleanup
            del model
            del val_loader
            del val_dataset
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n Best configuration: {best_config} with validation F1 {best_accuracy:.4f}")


    # Save the best model under a unique name
    best_model_name = f"best_model_gnn_{best_config['gnn']}_layers_{best_config['num_layer']}_emb_{best_config['emb_dim']}_epochs_{best_config['epochs']}.pth"
    best_model_path = os.path.join("checkpoints", best_model_name)

    default_model_path = os.path.join("checkpoints", f"model_{os.path.basename(os.path.dirname(base_args.test_path))}_best.pth")
    if os.path.exists(default_model_path):
        shutil.copyfile(default_model_path, best_model_path)
        print(f"Best model saved as {best_model_path}")
    else:
        print("Warning: Could not find best model to copy.")



if __name__ == "__main__":

    args = get_arguments()

    main(args)
    #hyperparameter_search(args)