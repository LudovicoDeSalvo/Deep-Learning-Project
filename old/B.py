import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
# Load utility functions from cloned repository
from torch_geometric.utils import degree
from src.loadData import GraphDataset
from src.utils import set_seed
from src.models import GNN
import argparse

# Set the random seed
set_seed()

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

from sklearn.metrics import f1_score
import torch
from tqdm import tqdm

def evaluate(data_loader, model, device, calculate_accuracy=False, calculate_f1=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    ground_truths = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            if calculate_accuracy or calculate_f1:
                predictions.extend(pred.cpu().numpy())
                ground_truths.extend(data.y.cpu().numpy())

            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
            elif not calculate_f1:
                predictions.extend(pred.cpu().numpy())

    if calculate_accuracy:
        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        if calculate_f1:
            f1 = f1_score(ground_truths, predictions, average='macro')
            return avg_loss, accuracy, f1
        return avg_loss, accuracy

    if calculate_f1:
        f1 = f1_score(ground_truths, predictions, average='macro')
        return f1

    return predictions

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
        gnn='gcn-virtual',   
        drop_ratio=0.2,     
        num_layer=3,       
        emb_dim=256,        
        batch_size=32,       
        epochs=120,          
        baseline_mode=2,     
        noise_prob=0.2,     
    )

def populate_args(args):
    print("Arguments received:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

args = get_arguments()
populate_args(args)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SymmetricNoiseCrossEntropyLoss(nn.Module):
    """
    Symmetric noise: flip probability is the same for all classes
    P(observed_label = j | true_label = i) = noise_prob / (num_classes - 1) for i != j
    P(observed_label = i | true_label = i) = 1 - noise_prob
    """
    def __init__(self, noise_prob, num_classes=6, temperature=1.0):
        super().__init__()
        self.noise_prob = noise_prob
        self.num_classes = num_classes
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
        # Build noise transition matrix for symmetric noise
        self.register_buffer('noise_matrix', self._build_symmetric_noise_matrix())
        
    def _build_symmetric_noise_matrix(self):
        """Build symmetric noise transition matrix"""
        matrix = torch.zeros(self.num_classes, self.num_classes)
        
        # Diagonal elements (correct label probability)
        matrix.fill_diagonal_(1.0 - self.noise_prob)
        
        # Off-diagonal elements (uniform noise to other classes)
        off_diag_prob = self.noise_prob / (self.num_classes - 1)
        matrix = matrix + off_diag_prob
        matrix.fill_diagonal_(1.0 - self.noise_prob)  # Restore diagonal
        
        return matrix
    
    def forward(self, logits, targets, epoch=0):
        # Temperature scaling
        logits = logits / self.temperature
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Calculate confidence-based weighting
        confidence = torch.max(probs, dim=1)[0]
        
        # Standard CE loss
        losses = self.ce(logits, targets)
        
        # Dynamic noise adaptation based on confidence and epoch
        epoch_factor = min(1.0, epoch / 50.0)  # Gradually increase robustness
        dynamic_noise_prob = self.noise_prob * (1 - 0.3 * confidence.detach() * epoch_factor)
        
        # Weight calculation for symmetric noise
        # Use the noise matrix to get the probability of observing the target given true label
        target_probs = self.noise_matrix[targets, targets]  # P(observed=target|true=target)
        weights = target_probs + (1 - target_probs) * (1 - dynamic_noise_prob)
        
        return (losses * weights).mean()

class AsymmetricNoiseCrossEntropyLoss(nn.Module):
    """
    Asymmetric noise: different flip probabilities for different class pairs
    More realistic noise model where certain classes are more likely to be confused
    """
    def __init__(self, noise_probs_dict, num_classes=6, temperature=1.0):
        super().__init__()
        self.noise_probs_dict = noise_probs_dict  # e.g., {(0,1): 0.2, (2,3): 0.4}
        self.num_classes = num_classes
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
        # Build noise transition matrix for asymmetric noise
        self.register_buffer('noise_matrix', self._build_asymmetric_noise_matrix())
        
    def _build_asymmetric_noise_matrix(self):
        """Build asymmetric noise transition matrix"""
        matrix = torch.eye(self.num_classes)  # Start with identity
        
        # Apply specific noise probabilities
        for (from_class, to_class), prob in self.noise_probs_dict.items():
            if from_class != to_class:
                matrix[from_class, to_class] = prob
                matrix[from_class, from_class] -= prob  # Ensure rows sum to 1
        
        # Ensure no negative probabilities
        matrix = torch.clamp(matrix, min=0.0)
        
        # Renormalize rows to sum to 1
        row_sums = matrix.sum(dim=1, keepdim=True)
        matrix = matrix / (row_sums + 1e-8)
        
        return matrix
    
    def forward(self, logits, targets, epoch=0):
        # Temperature scaling
        logits = logits / self.temperature
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Calculate confidence
        confidence = torch.max(probs, dim=1)[0]
        
        # Standard CE loss
        losses = self.ce(logits, targets)
        
        # Get noise probabilities for each sample based on their target class
        noise_probs = []
        for target in targets:
            # Get the probability of noise for this class (1 - diagonal element)
            class_noise_prob = 1.0 - self.noise_matrix[target, target].item()
            noise_probs.append(class_noise_prob)
        
        noise_probs = torch.tensor(noise_probs, device=logits.device)
        
        # Dynamic adaptation based on confidence and epoch
        epoch_factor = min(1.0, epoch / 50.0)
        dynamic_noise_probs = noise_probs * (1 - 0.3 * confidence.detach() * epoch_factor)
        
        # Weight calculation for asymmetric noise
        weights = (1 - dynamic_noise_probs) + dynamic_noise_probs * confidence.detach()
        
        return (losses * weights).mean()
    

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
        # Ensure input is float
        if x.dtype != torch.float32:
            x = x.float()
        return self.linear(x)

# Replace your model initialization section with this:
script_dir = os.getcwd() 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3

# Input feature dimension: 3 for add_enhanced_features, 2 for add_simple_degree_features
input_feature_dim = 3  # Change to 2 if using add_simple_degree_features

# Create the base model first
if args.gnn == 'gin':
    model = GNN(gnn_type='gin', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False)
elif args.gnn == 'gin-virtual':
    model = GNN(gnn_type='gin', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True)
elif args.gnn == 'gcn':
    model = GNN(gnn_type='gcn', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=False)
elif args.gnn == 'gcn-virtual':
    model = GNN(gnn_type='gcn', num_class=6, num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.drop_ratio, virtual_node=True)
else:
    raise ValueError('Invalid GNN type')

# Now replace the node encoder BEFORE moving to device
try:
    # Get the embedding dimension from the original encoder
    if hasattr(model.gnn_node, 'node_encoder'):
        embedding_dim = model.gnn_node.node_encoder.embedding_dim
        # Replace with continuous encoder
        model.gnn_node.node_encoder = ContinuousNodeEncoder(input_feature_dim, embedding_dim)
        print(f"Successfully replaced node encoder: {input_feature_dim} -> {embedding_dim}")
    else:
        print("Warning: Could not find node_encoder in model structure")
        # Print model structure to debug
        print("Model structure:")
        print(model)
except Exception as e:
    print(f"Error replacing node encoder: {e}")
    print("Model structure:")
    print(model)

# Move model to device AFTER replacing the encoder
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if args.baseline_mode == 2:
    # A and B symetric noise 
    criterion = SymmetricNoiseCrossEntropyLoss()
    # D and C assymetric noise 
    #criterion = AsymmetricNoiseCrossEntropyLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()

test_dir_name = os.path.basename(os.path.dirname(args.test_path))
logs_folder = os.path.join(script_dir, "logs", test_dir_name)
log_file = os.path.join(logs_folder, "training.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
os.makedirs(checkpoints_folder, exist_ok=True)

if os.path.exists(checkpoint_path) and not args.train_path:
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded best model from {checkpoint_path}")

# Replace your transform function with this corrected version
import torch
from torch_geometric.utils import degree

def add_enhanced_features(data):
    """Add meaningful node features with correct data types"""
    num_nodes = data.num_nodes
    
    # Calculate degree centrality
    edge_index = data.edge_index
    deg = degree(edge_index[0], num_nodes, dtype=torch.float)  # Changed to float
    
    # Normalize degree
    deg_normalized = deg / (deg.max() + 1e-8)
    
    # Create position-based features (if nodes have implicit ordering)
    position_features = torch.arange(num_nodes, dtype=torch.float) / num_nodes  # Changed to float
    
    # Combine features - all as float tensors
    node_features = torch.stack([
        deg_normalized,
        position_features,
        torch.ones(num_nodes, dtype=torch.float) * num_nodes / 1000.0,  # Changed to float
    ], dim=1)
    
    data.x = node_features
    return data

if args.train_path:
    full_dataset = GraphDataset(args.train_path, transform=add_enhanced_features)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(12)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    num_epochs = args.epochs
    best_val_f1 = 0.0   
    patience = 10
    patience_counter = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    if num_checkpoints > 1:
        checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
    else:
        checkpoint_intervals = [num_epochs]
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(
            train_loader, model, optimizer, criterion, device,
            save_checkpoints=(epoch + 1 in checkpoint_intervals),
            checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
            current_epoch=epoch
        )
        val_loss, val_acc, val_f1 = evaluate(val_loader, model, device, calculate_accuracy=True, calculate_f1=True)
        
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
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement for {patience} consecutive epochs")
            logging.info(f"Early stopping triggered after {epoch + 1} epochs due to no improvement for {patience} consecutive epochs")
            break
    
    plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))
    plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, "plotsVal"))

test_dataset = GraphDataset(args.test_path, transform=add_enhanced_features)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

model.load_state_dict(torch.load(checkpoint_path))
predictions = evaluate(test_loader, model, device, calculate_accuracy=False)
save_predictions(predictions, args.test_path)