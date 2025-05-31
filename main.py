import os
import torch
import logging
from torch_geometric.loader import DataLoader

from encoder import ContinuousNodeEncoder
from util import run_evaluation, add_zeros
from encoder import add_enhanced_features_fast, add_enhanced_features
from coTeaching import train_with_soft_co_teaching

from src.loadData import GraphDataset
from src.utils import set_seed
from src.models import GNN

import argparse
import gc


# Set the random seed
set_seed(42)


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

    # Initialize the model
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
    if args.start_from_base and args.train_path is not None:
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
    if args.train_path is not None and getattr(args, 'use_co_teaching', False):
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
    parser = argparse.ArgumentParser(description="Train or evaluate a GNN model.")

    parser.add_argument('--train_path', type=str, default=None, help='Path to training dataset (set to None for inference only)')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--num_checkpoints', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--gnn', type=str, default='gine-virtual', choices=['gin', 'gin-virtual', 'gcn', 'gcn-virtual', 'gine', 'gine-virtual', 'gine-virtualnode'])
    parser.add_argument('--drop_ratio', type=float, default=0.1)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--emb_dim', type=int, default=300) #300 to load base model
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--baseline_mode', type=int, default=4) #starting loss
    parser.add_argument('--noise_prob', type=float, default=0.4)
    parser.add_argument('--use_co_teaching', type=bool, default=True, help='Always use soft co-teaching')
    parser.add_argument('--switch_epoch', type=int, default=0) #Switches to NCOD+ after this number of epochs
    parser.add_argument('--patience', type=int, default=10) #Early Stopping Patience
    parser.add_argument('--start_from_base', type=bool, default=True, help='Start from base pretrained model') #Start training from the model trained on all datasets
    parser.add_argument('--start_mixup', type=int, default=300)

    return parser.parse_args()
## IF ERROR WHILE RUNNING CODE TRY SETTING UP EVERYTHIONG IN THE MAIN LIKE THIS:
# def get_arguments():
#     parser = argparse.ArgumentParser(description="Train or evaluate a GNN model.")

#     parser.add_argument('--train_path', type=str, default='datasets/A/train.json.gz', help='Path to training dataset (set to None for inference only)')
#     parser.add_argument('--test_path', type=str,  default='datasets/A/test.json.gz', required=False, help='Path to test dataset')
#     parser.add_argument('--num_checkpoints', type=int, default=5)
#     parser.add_argument('--device', type=int, default=0)
#     parser.add_argument('--gnn', type=str, default='gine-virtual', choices=['gin', 'gin-virtual', 'gcn', 'gcn-virtual', 'gine', 'gine-virtual', 'gine-virtualnode'])
#     parser.add_argument('--drop_ratio', type=float, default=0.1)
#     parser.add_argument('--num_layer', type=int, default=2)
#     parser.add_argument('--emb_dim', type=int, default=300) #300 to load base model
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--epochs', type=int, default=80)
#     parser.add_argument('--baseline_mode', type=int, default=4) #starting loss
#     parser.add_argument('--noise_prob', type=float, default=0.2)
#     parser.add_argument('--use_co_teaching', type=bool, default=True, help='Always use soft co-teaching')
#     parser.add_argument('--switch_epoch', type=int, default=0) #Switches to NCOD+ after this number of epochs
#     parser.add_argument('--patience', type=int, default=10) #Early Stopping Patience
#     parser.add_argument('--start_from_base', type=bool, default=True, help='Start from base pretrained model') #Start training from the model trained on all datasets
#     parser.add_argument('--start_mixup', type=int, default=300)

#     return parser.parse_args()


if __name__ == "__main__":

    args = get_arguments()
    main(args)
    #hyperparameter_search(args)
