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
from src.models import GNN, GINEModelWithVirtualNode
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


import torch.nn as nn

import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy




# Set the random seed
set_seed()

def add_zeros(data):
    data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float32)
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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from src.loadData import GraphDataset
from src.models import GNN, GINEModelWithVirtualNode
from src.utils import set_seed
from main import ContinuousNodeEncoder, SymmetricCrossEntropyLoss, NCODPlusLoss, evaluate, add_enhanced_features_fast
from torch.optim.lr_scheduler import CosineAnnealingLR




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
        return min(self.forget_rate, 0.5)

    
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

      
    def entropy_regularization(logits):
        p = F.softmax(logits, dim=1)
        return -torch.mean(torch.sum(p * torch.log(p + 1e-6), dim=1))


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
        use_mixup_now = self.use_mixup and epoch >= getattr(self, "mixup_start_epoch", args.start_mixup)

        if self.use_mixup and epoch == getattr(self, "mixup_start_epoch", args.start_mixup):
            print(f"🔀 MixUp activated starting from epoch {epoch} with Beta(0.4, 0.4)")

        for data in tqdm(data_loader, desc=f"Epoch {epoch+1} - Training", unit="batch"):
            data = data.to(self.device)
            logits1, emb1 = self.model1(data)
            logits2, emb2 = self.model2(data)

            if use_mixup_now and not hasattr(self.criterion, 'update_centroids'):
                lam = np.random.beta(0.4, 0.4)
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

                        loss1 = self.criterion(emb1, logits1, data.y, alt_logits=logits2.detach(), indices=data.idx)
                        loss2 = self.criterion(emb2, logits2, data.y, alt_logits=logits1.detach(), indices=data.idx)
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

                    # ➕ Entropy regularization
                    entropy1 = self.entropy_regularization(logits1)
                    entropy2 = self.entropy_regularization(logits2)
                    loss1 += 0.01 * entropy1
                    loss2 += 0.01 * entropy2

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

    full_dataset = GraphDataset(args.train_path, transform = add_zeros)

    dataset_len = len(full_dataset)
    sample = full_dataset[0]
    input_feature_dim = sample.x.size(1)

    model1, model2 = create_co_teaching_models(args, device, input_feature_dim)


    for model in [model1, model2]:
        try:
            if hasattr(model, 'gnn_node') and hasattr(model.gnn_node, 'node_encoder'):
                emb_dim = model.gnn_node.node_encoder.embedding_dim
                model.gnn_node.node_encoder = ContinuousNodeEncoder(input_feature_dim, emb_dim)
                print(f"✅ Replaced node_encoder for co-teaching model.gnn_node: {input_feature_dim} -> {emb_dim}")
            elif hasattr(model, 'node_encoder'): # For GINEModelWithVirtualNode or similar
                emb_dim = model.node_encoder.embedding_dim if hasattr(model.node_encoder, 'embedding_dim') else args.emb_dim
                # Get the device of the parent model
                parent_model_device = next(model.parameters()).device
                model.node_encoder = ContinuousNodeEncoder(input_feature_dim, emb_dim).to(parent_model_device)
                print(f"✅ Replaced node_encoder for model: {input_feature_dim} -> {emb_dim} on device {parent_model_device}")
        except Exception as e:
            print(f"⚠️ Skipped encoder replacement: {e}")


    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=args.epochs, eta_min=1e-5)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.epochs, eta_min=1e-5)

    
    if args.baseline_mode == 4:
        criterion = SymmetricCrossEntropyLoss(alpha=0.5, beta=2.0)
    elif args.baseline_mode == 5:
        criterion = NCODPlusLoss(
            num_classes=6,
            emb_dim=args.emb_dim,
            num_samples=dataset_len,       # for ELR memory
            lambda_c=1.0,
            lambda_b=1.0,
            lambda_elr=3.0,
            ema_momentum=0.9
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()

    CoTeachingClass = AdaptiveSoftCoTeaching if use_adaptive else SoftCoTeaching
    co_teacher = CoTeachingClass(
        model1=model1,
        model2=model2, 
        optimizer1=optimizer1,
        optimizer2=optimizer2,
        criterion=criterion,
        forget_rate=args.noise_prob + 0.1,
        num_gradual=args.epochs // 3,
        device=device,
        temperature=1.0,
        weight_smoothing='sigmoid',
        use_mixup = True,
        mixup_start_epoch = args.start_mixup
    )

    current_criterion = criterion
        

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(12))

     # Assign idx after random split
    for i, data in enumerate(train_dataset):
        data.idx = i

    for i, data in enumerate(val_dataset):
        data.idx = len(train_dataset) + i  # so there's no overlap

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_f1 = 0.0
    best_val_accuracy = 0.0
    patience = args.patience
    patience_counter = 0

    for epoch in range(args.epochs):

        if epoch == args.switch_epoch:
            print("🔄 Switching to NCODPlusLoss + ELR")
            current_criterion = NCODPlusLoss(
                num_classes=6,
                emb_dim=args.emb_dim,
                num_samples=dataset_len,
                lambda_c=1.0,
                lambda_b=1.0,
                lambda_elr=3.0,
                ema_momentum=0.9
            )
            co_teacher.criterion = current_criterion

        loss1, loss2, acc1, acc2, avg_weight1, avg_weight2 = co_teacher.train_epoch(train_loader, epoch)

        val_loss1, val_acc1, val_preds1, val_labels1 = evaluate(val_loader, model1, device, calculate_accuracy=True)
        val_loss2, val_acc2, val_preds2, val_labels2 = evaluate(val_loader, model2, device, calculate_accuracy=True)

        val_f1_1 = f1_score(val_labels1, val_preds1, average='macro')
        val_f1_2 = f1_score(val_labels2, val_preds2, average='macro')

        val_f1 = max(val_f1_1, val_f1_2)
        best_model = model1 if val_f1_1 >= val_f1_2 else model2

        scheduler1.step()
        scheduler2.step()

        if epoch == 0:
            criterion.ema_momentum = 0.0 
        elif epoch == 1:
            criterion.ema_momentum = 0.9 

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


def create_co_teaching_models(args, device, input_dim):
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
    
    elif args.gnn == 'gat':
        model1 = GNN(gnn_type='gat', num_class=6, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False, input_dim=input_dim, residual=True).to(device)
        model2 = GNN(gnn_type='gat', num_class=6, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False, input_dim=input_dim,residual=True).to(device)
    elif args.gnn == 'gat-virtual':
        model1 = GNN(gnn_type='gat', num_class=6, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True,
                    input_dim=input_dim, residual=True).to(device)
        model2 = GNN(gnn_type='gat', num_class=6, num_layer=args.num_layer,
                    emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True,
                    input_dim=input_dim,residual=True).to(device)

    elif args.gnn == 'gine':
        model1 = GNN(gnn_type='gine', num_class=6, num_layer=args.num_layer,
                     emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                     virtual_node=False).to(device)
        model2 = GNN(gnn_type='gine', num_class=6, num_layer=args.num_layer,
                     emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                     virtual_node=False).to(device)

    elif args.gnn == 'gine-virtual':
        model1 = GNN(gnn_type='gine', num_class=6, num_layer=args.num_layer,
                     emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                     virtual_node=True).to(device)
        model2 = GNN(gnn_type='gine', num_class=6, num_layer=args.num_layer,
                     emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                     virtual_node=True).to(device)
        
    elif args.gnn == 'gine-virtualnode':
        model1 = GINEModelWithVirtualNode(num_features=input_dim, num_classes=6, num_layers=args.num_layer,
                                        emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)
        model2 = GINEModelWithVirtualNode(num_features=input_dim, num_classes=6, num_layers=args.num_layer,
                                        emb_dim=args.emb_dim, drop_ratio=args.drop_ratio).to(device)

    else:
        raise ValueError("Unsupported GNN type. Supported types: 'gin-virtual', 'gcn-virtual', 'transformer'")

    return model1, model2



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


class NCODPlusLoss(nn.Module):
    def __init__(self, num_classes, emb_dim, lambda_c=1.0, lambda_b=1.0, lambda_elr=3.0, ema_momentum=0.9, num_samples=20000):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.lambda_c = lambda_c
        self.lambda_b = lambda_b
        self.lambda_elr = lambda_elr
        self.ema_momentum = ema_momentum

        self.register_buffer("centroids", torch.zeros(num_classes, emb_dim))
        self.class_counts = torch.zeros(num_classes, dtype=torch.long)
        self.u = nn.Parameter(torch.ones(1))

        # ELR memory buffer
        self.register_buffer("target_probs", torch.zeros(num_samples, num_classes))

    def update_centroids(self, embeddings, targets):
        for i in range(self.num_classes):
            mask = (targets == i)
            if mask.any():
                self.centroids[i] = embeddings[mask].mean(dim=0).detach()
                self.class_counts[i] = mask.sum().item()

    def forward(self, embeddings, logits, targets, alt_logits=None, indices=None):
        device = logits.device
        ce_loss = F.cross_entropy(logits, targets)

        # NCOD: similarity-based soft labels
        sim = F.cosine_similarity(embeddings.unsqueeze(1), self.centroids.to(device).unsqueeze(0), dim=2)
        soft_labels = F.softmax(sim, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).float().to(device)

        similarity_to_target = (soft_labels * one_hot_targets).sum(dim=1)
        discount = 1.0 - similarity_to_target
        reg_term = discount.mean() * self.u.to(device)

        mse_term = F.mse_loss(soft_labels, one_hot_targets.detach())
        loss = ce_loss + mse_term + reg_term

        # NCOD+ ensemble regularization
        if alt_logits is not None:
            p1 = F.softmax(logits, dim=1)
            p2 = F.softmax(alt_logits, dim=1)
            avg_pred = (p1 + p2) / 2
            uniform_dist = torch.full_like(avg_pred, 1.0 / self.num_classes).to(device)

            kl_c = F.kl_div(p1.log(), p2, reduction='batchmean') + F.kl_div(p2.log(), p1, reduction='batchmean')
            kl_b = F.kl_div(avg_pred.log(), uniform_dist, reduction='batchmean')
            loss += self.lambda_c * kl_c + self.lambda_b * kl_b

        # 🔁 ELR Regularization
        if indices is not None:
            indices_cpu = indices.cpu()  # Ensure safe indexing into CPU buffer

            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=1)
                self.target_probs[indices_cpu] = (
                    self.ema_momentum * self.target_probs[indices_cpu] +
                    (1 - self.ema_momentum) * probs.cpu()
                )

            q_i = self.target_probs[indices_cpu].to(device)  # Also index using CPU version
            elr_reg = -torch.mean(torch.sum(F.softmax(logits, dim=1) * torch.log(1.0 - q_i + 1e-6), dim=1))
            loss += self.lambda_elr * elr_reg

        return loss




# Updated ContinuousNodeEncoder with proper initialization
class ContinuousNodeEncoder(nn.Module):
    """Encoder for continuous node features instead of discrete embeddings"""
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        self.embedding_dim = embedding_dim
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass for continuous node encoder.
        Automatically reshapes 1D or transposed input tensors and ensures correct dtype and device.
        
        Args:
            x (torch.Tensor): Input tensor of shape [num_nodes, num_features] or [1, num_nodes] (transposed).
            
        Returns:
            torch.Tensor: Output tensor of shape [num_nodes, embedding_dim]
        """
        # Ensure tensor is float
        if x.dtype != torch.float32:
            x = x.float()

        # Fix transposed inputs [1, N] -> [N, 1]
        if x.dim() == 2 and x.shape[0] == 1 and x.shape[1] > 1000:
            x = x.T

        # Handle 1D case [N] -> [N, 1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Move to the same device as the model parameters
        x = x.to(self.linear[0].weight.device)

        return self.linear(x)


    
def normalize_tensor(tensor):
    min_val = tensor.min(dim=0, keepdim=True)[0]
    max_val = tensor.max(dim=0, keepdim=True)[0]
    norm = (tensor - min_val) / (max_val - min_val + 1e-6)
    return norm

def add_enhanced_features(data):


    # Convert PyG data to undirected NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Compute structural features using NetworkX
    clustering_dict = nx.clustering(G)
    closeness_dict = nx.closeness_centrality(G)
    betweenness_dict = nx.betweenness_centrality(G, normalized=True)

    # Degree from PyG edge index
    degree = data.edge_index[0].bincount(minlength=data.num_nodes).float().unsqueeze(1)

    # Convert NetworkX features to torch tensors (Nx1)
    clustering = torch.tensor([clustering_dict[i] for i in range(data.num_nodes)], dtype=torch.float32).unsqueeze(1)
    closeness = torch.tensor([closeness_dict[i] for i in range(data.num_nodes)], dtype=torch.float32).unsqueeze(1)
    betweenness = torch.tensor([betweenness_dict[i] for i in range(data.num_nodes)], dtype=torch.float32).unsqueeze(1)

    # Combine and normalize
    features = torch.cat([degree, clustering, closeness, betweenness], dim=1)
    data.x = normalize_tensor(features)



    return data


def add_enhanced_features_fast(data, k=5):
    import scipy
    import scipy.sparse
    import scipy.sparse.linalg
    from torch_geometric.utils import to_networkx

    # Basic structural features
    deg = data.edge_index[0].bincount(minlength=data.num_nodes).float().unsqueeze(1)
    clustering = torch.rand(data.num_nodes, 1)  # placeholder
    centrality = torch.rand(data.num_nodes, 1)  # placeholder

    # Laplacian Positional Encoding
    G = to_networkx(data, to_undirected=True)
    laplacian = nx.normalized_laplacian_matrix(G)

    try:
        eigval, eigvec = scipy.sparse.linalg.eigsh(laplacian, k=min(k, G.number_of_nodes() - 2), which='SM')
        pe = torch.from_numpy(eigvec).float()
    except Exception:
        # Fallback: if eigsh fails (e.g. tiny graph)
        pe = torch.zeros(data.num_nodes, k)

    # Pad if not enough eigenvectors
    if pe.size(1) < k:
        pad = torch.zeros(data.num_nodes, k - pe.size(1))
        pe = torch.cat([pe, pad], dim=1)

    # Final feature matrix
    data.x = torch.cat([deg, clustering, centrality, pe], dim=1)
    return data


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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Setup paths and logging
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    os.makedirs(os.path.join(script_dir, "checkpoints", test_dir_name), exist_ok=True)

    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    os.makedirs(logs_folder, exist_ok=True)
    log_file = os.path.join(logs_folder, "training.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # Select transform function
    transform = add_zeros # or add_enhanced_features

    # Dynamically determine input features
    feature_probe_path = args.train_path if args.train_path else args.test_path
    sample = GraphDataset(feature_probe_path, transform=transform)[0]
    input_feature_dim = sample.x.size(1)

    # Initialize model
    if args.gnn == 'gine-virtualnode':
        model = GINEModelWithVirtualNode(num_features=input_feature_dim, num_classes=6,
                                        num_layers=args.num_layer, emb_dim=args.emb_dim,
                                        drop_ratio=args.drop_ratio).to(device)
    else:
        model = GNN(
            gnn_type=args.gnn.replace('-virtual', ''),
            num_class=6,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node='virtual' in args.gnn,
            input_dim=input_feature_dim,
            residual=True
        ).to(device)


    # Replace node encoder before loading weights
    try:
        if hasattr(model, 'node_encoder'):
            # Determine the correct embedding dimension for the new continuous encoder
            if isinstance(model.node_encoder, torch.nn.Embedding):
                embedding_dim = model.node_encoder.embedding_dim
            elif hasattr(model.node_encoder, 'embedding_dim'): # For already replaced ContinuousNodeEncoder
                 embedding_dim = model.node_encoder.embedding_dim
            else: # Fallback or if it's an unexpected encoder type
                embedding_dim = args.emb_dim

            model_device = next(model.parameters()).device # Get model's current device
            model.node_encoder = ContinuousNodeEncoder(input_feature_dim, embedding_dim).to(model_device)
            print(f"Replaced node encoder: {input_feature_dim} -> {embedding_dim} on device {model_device}")
    except Exception as e:
        print(f"Error replacing node encoder: {e}")

    model = model.to(device)

    # Load checkpoint model weights
    if args.start_from_base:
        base_ckpt = os.path.join(script_dir, "checkpoints", "model_base.pth")
        if os.path.exists(base_ckpt):
            state_dict = torch.load(base_ckpt, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"🔁 Started from base checkpoint: {base_ckpt}")
        else:
            raise FileNotFoundError(f"Missing base checkpoint: {base_ckpt}")
    elif os.path.exists(checkpoint_path) and not args.train_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        except RuntimeError as e:
            print(f"State dict mismatch: {e}")
            raise
        print(f"Loaded best model from {checkpoint_path}")

    # Load test dataset
    test_dataset = GraphDataset(args.test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Assign indices to test graphs (for ELR/NCOD if needed)
    if args.train_path:
        train_dataset_for_len = GraphDataset(args.train_path, transform=transform)
        offset = len(train_dataset_for_len) + 10000
        for i, data in enumerate(test_dataset):
            data.idx = offset + i
        del train_dataset_for_len
        torch.cuda.empty_cache()
        gc.collect()

    # Train model if training path is provided
    if args.train_path and getattr(args, 'use_co_teaching', False):
        model, val_loader = train_with_soft_co_teaching(args, device, use_adaptive=True, checkpoint_path=checkpoint_path)

    # Load best model after training
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Validation Evaluation
    if 'val_loader' in locals():
        run_evaluation(val_loader, model, device,
                       label_map=["class0", "class1", "class2", "class3", "class4", "class5"],
                       calculate_accuracy=True)

    # Final Test Evaluation
    run_evaluation(test_loader, model, device,
                   label_map=["class0", "class1", "class2", "class3", "class4", "class5"],
                   save_csv_path=os.path.join("submission", f"testset_{test_dir_name}.csv"),
                   calculate_accuracy=True)

    if 'val_loader' in locals():
        del val_loader
        torch.cuda.empty_cache()
        gc.collect()



def get_arguments():
    args = {}

    # Default argument values
    args['train_path'] = "datasets/A/0.2_train.json"
    args['test_path'] = "datasets/A/0.2_test.json"
    args['num_checkpoints'] = 5
    args['device'] = 0
    args['gnn'] = 'gine-virtual' #gin gin-virtual gcn gcn-virtual gine gine-virtual gine-virtualnode
    args['drop_ratio'] = 0.0
    args['num_layer'] = 2
    args['emb_dim'] = 300   #300 to load base model
    args['batch_size'] = 32
    args['epochs'] = 200
    args['baseline_mode'] = 4 #starting loss
    args['noise_prob'] = 0.4
    args['use_co_teaching'] = True
    args['switch_epoch'] = 0    #Switches to NCOD+ after this number of epochs
    args['patience'] = 10   #Early Stopping Patience
    args['start_from_base'] = True  #start from model trained on all datasets
    args['start_mixup'] = 300

    return argparse.Namespace(**args)



if __name__ == "__main__":

    args = get_arguments()
    main(args)
    #hyperparameter_search(args)