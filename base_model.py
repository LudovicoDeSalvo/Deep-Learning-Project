from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout

# Load utility functions from cloned repository
from src.loadData import GraphDataset
from src.utils import set_seed
from src.models import GNN
import argparse

from util import add_zeros, evaluate, save_predictions, plot_training_progress


# %%
class GINEModel(torch.nn.Module):
    """GINE model using GINEConv layers"""
    def __init__(self, num_features=1, num_classes=6, num_layers=5, emb_dim=300, drop_ratio=0.0, edge_dim=7):
        super(GINEModel, self).__init__()
        
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        
        # Input embedding
        self.node_encoder = torch.nn.Embedding(num_features, emb_dim)
        self.edge_encoder = torch.nn.Linear(edge_dim, emb_dim)
        
        # GINEConv layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for i in range(num_layers):
            mlp = Sequential(
                Linear(emb_dim, emb_dim),
                BatchNorm1d(emb_dim),
                ReLU(),
                Linear(emb_dim, emb_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.batch_norms.append(BatchNorm1d(emb_dim))
        
        # Classifier
        self.classifier = Sequential(
            Linear(emb_dim, emb_dim // 2),
            ReLU(),
            Dropout(drop_ratio),
            Linear(emb_dim // 2, num_classes)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Handle edge attributes - create if they don't exist
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr.float()
        else:
            edge_attr = torch.ones(edge_index.size(1), self.edge_dim, device=edge_index.device)
        
        # Encode node and edge features
        x = self.node_encoder(x.squeeze(-1) if x.dim() > 1 else x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Apply GINEConv layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_ratio, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        return self.classifier(x)




# %%
def get_user_input(prompt, default=None, required=False, type_cast=str):

    while True:
        user_input = input(f"{prompt} [{default}]: ")
        
        if user_input == "" and required:
            print("This field is required. Please enter a value.")
            continue
        
        if user_input == "" and default is not None:
            return default
        
        if user_input == "" and not required:
            return None
        
        try:
            return type_cast(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a valid {type_cast.__name__}.")

# %%
# def get_arguments():
#     args = {}
#     args['train_path'] = get_user_input("Path to the training dataset (optional)")
#     args['test_path'] = get_user_input("Path to the test dataset", required=True)
#     args['num_checkpoints'] = get_user_input("Number of checkpoints to save during training", type_cast=int)
#     args['device'] = get_user_input("Which GPU to use if any", default=1, type_cast=int)
#     args['gnn'] = get_user_input("GNN type (gin, gin-virtual, gcn, gcn-virtual)", default='gin')
#     args['drop_ratio'] = get_user_input("Dropout ratio", default=0.0, type_cast=float)
#     args['num_layer'] = get_user_input("Number of GNN message passing layers", default=5, type_cast=int)
#     args['emb_dim'] = get_user_input("Dimensionality of hidden units in GNNs", default=300, type_cast=int)
#     args['batch_size'] = get_user_input("Input batch size for training", default=32, type_cast=int)
#     args['epochs'] = get_user_input("Number of epochs to train", default=10, type_cast=int)
#     args['baseline_mode'] = get_user_input("Baseline mode: 1 (CE), 2 (Noisy CE)", default=1, type_cast=int)
#     args['noise_prob'] = get_user_input("Noise probability p (used if baseline_mode=2)", default=0.2, type_cast=float)

    
#     return argparse.Namespace(**args)


# Noise Probability
# A : 0.2
# B : 0.4
# C : 0.2
# D : 0.2

def get_arguments():
    return argparse.Namespace(
        train_path='datasets/A/train.json.gz',
        test_path='datasets/A/test.json.gz',
        num_checkpoints=5,
        device=0,
        gnn='gine',   
        drop_ratio=0.3,     # Increased dropout
        num_layer=2,        # Increased layers from 3 to 5
        emb_dim=300,        # Increased embedding size from 300 to 512
        batch_size=32,      # Reduced batch size for better gradients
        epochs=80,         # More epochs
        baseline_mode=2,     
        noise_prob=0.2,     # Reduced noise (was 0.15)
    )

# %%
def populate_args(args):
    print("Arguments received:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
args = get_arguments()
populate_args(args)

# %%
class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
# STEP 1: Better Learning Rate Schedule
# Replace your optimizer in get_arguments() with:
def create_optimizer_with_scheduler(model, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return optimizer, scheduler

# %%
script_dir = os.getcwd() 

# %%
import argparse

args = get_arguments()
device = args.device


# %%
script_dir = os.getcwd() 
train_dir_name = os.path.basename(os.path.dirname(args.train_path))
test_dir_name = os.path.basename(os.path.dirname(args.test_path))
logs_folder = os.path.join(script_dir, "logs", test_dir_name)
log_file = os.path.join(logs_folder, "training.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
os.makedirs(checkpoints_folder, exist_ok=True)


# %%
from torch_geometric.nn import GINEConv

# %%
from torch.nn import Dropout

# %%
def create_model_and_criterion(args, device):
    if args.gnn == 'gine':
            model = GINEModel(
                num_features=1, 
                num_classes=6, 
                num_layers=args.num_layer, 
                emb_dim=args.emb_dim, 
                drop_ratio=args.drop_ratio
        ).to(device)
    elif args.gnn == 'gin':
        model = GNN(gnn_type='gin', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True).to(device)
    else:
        raise ValueError(f"Invalid GNN type: {args.gnn}")

    # Define criterion
    if args.baseline_mode == 2:
        criterion = NoisyCrossEntropyLoss(args.noise_prob)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    return model, criterion


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def load_and_train(dataset_name, args, model, optimizer, criterion, device, script_dir):
    print(f"\n➡️ Training on dataset {dataset_name}")
    args.train_path = f'datasets/{dataset_name}/train.json.gz'

    if os.path.exists(args.train_path):
        full_dataset = GraphDataset(args.train_path, transform=add_zeros)
        val_size = int(0.15 * len(full_dataset))
        train_size = len(full_dataset) - val_size

        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        best_val_f1 = 0.0
        epochs_without_improvement = 0
        patience = getattr(args, 'patience', 8)  # Early stopping patience
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.2, patience=1, verbose=True, min_lr=1e-6
        )

        for epoch in range(args.epochs):
            train_loss, train_acc = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=False, checkpoint_path=None, current_epoch=epoch
            )
            val_loss, val_acc, val_f1 = enhanced_evaluate(val_loader, model, device, calculate_accuracy=True, calculate_f1=True)
            
            # Step the scheduler
            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Val F1 = {val_f1:.4f}, LR = {current_lr:.2e}")
        
            checkpoint_dir = os.path.join(script_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_{dataset_name}_best.pth")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_f1': best_val_f1,
                    'scheduler_state_dict': scheduler.state_dict()
                }, checkpoint_path)
                print(f"💾 Saved improved checkpoint at epoch {epoch+1} with val F1 {val_f1:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"⚠️ No improvement for {epochs_without_improvement} epochs")
                
                # Early stopping
                if epochs_without_improvement >= patience:
                    print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
                    print(f"📈 Best validation F1: {best_val_f1:.4f}")
                    break

        # Load best model before returning
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded best model with F1: {best_val_f1:.4f}")

    else:
        print(f"⚠️ Dataset not found: {args.train_path}")

# %%
def load_combined_dataset(dataset_names, args):
    combined_data = []
    for name in dataset_names:
        path = f'datasets/{name}/train.json.gz'
        if os.path.exists(path):
            dataset = GraphDataset(path, transform=add_zeros)
            combined_data.extend(dataset)
        else:
            print(f"⚠️ Dataset not found: {path}")
    return combined_data

# %%
def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    # Ensure criterion is also on the correct device
    criterion = criterion.to(device)
    
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
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
    
    return total_loss / len(data_loader), correct / total

# %%
def train_base_model_on_all(args, model, criterion, device, script_dir):
    dataset_names = ['A', 'B', 'C', 'D']
    print(f"📦 Loading and combining datasets: {dataset_names}")
    combined_data = load_combined_dataset(dataset_names, args)

    val_size = int(0.15 * len(combined_data))
    train_size = len(combined_data) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(combined_data, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Enhanced optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    patience = getattr(args, 'patience', 8)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, model, optimizer, criterion, device, 
                                    save_checkpoints=False, checkpoint_path=None, current_epoch=epoch)
        val_loss, val_acc, val_f1 = enhanced_evaluate(val_loader, model, device, calculate_accuracy=True, calculate_f1=True)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[Base Epoch {epoch+1}] Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Val F1 = {val_f1:.4f}, LR = {current_lr:.2e}")

        base_checkpoint_path = os.path.join(script_dir, "checkpoints/model_base.pth")
        os.makedirs(os.path.dirname(base_checkpoint_path), exist_ok=True)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': best_val_f1,
                'scheduler_state_dict': scheduler.state_dict()
            }, base_checkpoint_path)
            print(f"💾 Saved base model checkpoint with val F1 {val_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epochs")
            
            if epochs_without_improvement >= patience:
                print(f"🛑 Early stopping triggered for base model training")
                break

# %%
def create_layered_optimizer(model, base_lr=0.0001):
    """Create optimizer with different learning rates for different layers"""
    params = []
    
    # Lower learning rate for earlier layers (feature extraction)
    for name, param in model.named_parameters():
        if 'conv' in name.lower() or 'gnn' in name.lower():
            params.append({'params': param, 'lr': base_lr * 0.1})
        elif 'classifier' in name.lower() or 'fc' in name.lower():
            # Higher learning rate for classification layers
            params.append({'params': param, 'lr': base_lr * 2})
        else:
            params.append({'params': param, 'lr': base_lr})
    
    return torch.optim.AdamW(params, weight_decay=1e-4)

# %%
def enhanced_evaluate(loader, model, device, calculate_accuracy=True, calculate_f1=False):
    """Enhanced evaluation function with F1 score calculation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data)
            
            if hasattr(data, 'y'):
                labels = data.y
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if calculate_f1:
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    if calculate_f1 and len(all_preds) > 0:
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return avg_loss, accuracy, f1
    
    return avg_loss, accuracy

# %%

def apply_class_weights(criterion, train_loader, device):
    """Calculate and apply class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    # Extract all labels from training data
    all_labels = []
    for data in train_loader:
        if hasattr(data, 'y'):
            all_labels.extend(data.y.cpu().numpy())
    
    # Compute class weights
    unique_classes = np.unique(all_labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # Apply weights to criterion
    if hasattr(criterion, 'weight'):
        criterion.weight = class_weights_tensor
    
    print(f"📊 Applied class weights: {dict(zip(unique_classes, class_weights))}")
    return criterion


# %%
# Step 1: Train on all data
model, criterion = create_model_and_criterion(args, device)
train_base_model_on_all(args, model, criterion, device, script_dir) 


