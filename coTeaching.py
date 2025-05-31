import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from encoder import ContinuousNodeEncoder, add_enhanced_features_fast, add_enhanced_features
from losses import SymmetricCrossEntropyLoss, NCODPlusLoss
from util import evaluate, add_zeros
from src.loadData import GraphDataset
from src.models import GNN
from src.utils import set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import plot_training_progress
import logging




class SoftCoTeaching:
    def __init__(self, model1, model2, optimizer1, optimizer2, criterion, 
                 forget_rate=0.2, num_gradual=10, device='cuda', 
                 temperature=1.0, weight_smoothing='sigmoid', use_mixup=False, mixup_start_epoch=10, args = {}):
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
        self.args = args
        
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
        use_mixup_now = self.use_mixup and epoch >= getattr(self, "mixup_start_epoch", self.args.start_mixup)

        if self.use_mixup and epoch == getattr(self, "mixup_start_epoch", self.args.start_mixup):
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
            elif hasattr(model, 'node_encoder'):
                emb_dim = model.node_encoder.embedding_dim if hasattr(model.node_encoder, 'embedding_dim') else args.emb_dim
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
        mixup_start_epoch = args.start_mixup,
        args = args
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

        # Logging         
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch {epoch + 1}/{args.epochs}")            
            logging.info(f"Train Losses: Model1={loss1:.4f}, Model2={loss2:.4f}")
            logging.info(f"Train Accuracies: Model1={acc1:.4f}, Model2={acc2:.4f}")            
            logging.info(f"Val F1: Model1={val_f1_1:.4f}, Model2={val_f1_2:.4f}")
            if use_adaptive:                
                logging.info(f"Temperature: {co_teacher.temperature:.4f}")

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
        
    else:
        raise ValueError("Unsupported GNN type. Supported types: 'gin-virtual', 'gcn-virtual', 'transformer'")

    return model1, model2