import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from encoder import ContinuousNodeEncoder
from losses import SymmetricCrossEntropyLoss, NCODPlusLoss
from util import evaluate, run_evaluation, add_zeros
from coTeaching import train_with_soft_co_teaching

from src.loadData import GraphDataset
from src.utils import set_seed
from src.models import GNN, GINEModelWithVirtualNode

import argparse
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import numpy as np

import copy
import shutil
import gc
import torch.nn.functional as F
import torch.nn as nn


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
    args['drop_ratio'] = 0.1
    args['num_layer'] = 2
    args['emb_dim'] = 300   #300 to load base model
    args['batch_size'] = 32
    args['epochs'] = 200
    args['baseline_mode'] = 4 #starting loss
    args['noise_prob'] = 0.2
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