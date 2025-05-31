# Deep Learning Project

This repository implements a **robust graph classification pipeline** using **Graph Neural Networks (GNNs)**, tailored to **learning from noisy labels**. The system integrates several state-of-the-art methods for label noise robustness and feature encoding.

## Features

- **NCOD+ Loss** with centroid-based regularization and outlier discounting
- **ELR (Early Learning Regularization)** for robust learning during memorization phase
- **Soft Co-Teaching** with adaptive forget rate and dynamic sample weighting
- **GINE Convolutional Layers** with edge features
- **MixUp Augmentation** on graph embeddings
- **Early Stopping** based on macro-F1 score
- **Fine-Tuning** from a base model trained on all datasets
- **Enhanced node features** (clustering, closeness, betweenness, Laplacian PE)

---


## Training the Model

The main entry point is `main.py`:

```bash
python3 main.py --train_path datasets/B/0.4_train.json --test_path datasets/B/0.4_test.json
```

For memory reason we used unzipped .json datasets, but the script also accepts .gz files.

### Arguments & Configuration

Key arguments from `get_arguments()`:

```python
args = {
    'train_path': "datasets/C/0.2_asy_train.json",   # Path to training data (or None for inference only)
    'test_path': "datasets/C/0.2_asy_test.json",     # Path to test data

    'gnn': 'gine-virtual',        # GNN type: gin, gcn, gat (+ optional -virtual). Default: gine-virtual (best performing)
    'drop_ratio': 0.0,           # Dropout ratio
    'num_layer': 2,              # Number of GNN layers (2 has shown best performance)
    'emb_dim': 300,              # Node embedding dimension

    'batch_size': 32,            # Batch size
    'epochs': 200,               # Maximum number of epochs

    'baseline_mode': 4,          # Loss: 4 = SymmetricCE (start), 5 = NCOD+ (switched at switch_epoch)
    'noise_prob': 0.2,           # Noise rate (used in co-teaching weighting)

    'use_co_teaching': True,     # Must be True — enables dual-network training with noise-aware learning
    'switch_epoch': 0,           # Epoch to switch from baseline loss to NCOD+. 
                                 # Note: NCOD+ is unstable early, but ELR makes early use feasible.

    'patience': 10,              # Early stopping patience (monitored on macro-F1)

    'start_from_base': True,     # If True, initializes from `checkpoints/model_base.pth` (pretrained on all datasets)

    'start_mixup': 300           # Epoch to activate MixUp (set high, as early MixUp worsens performance in this architecture)
}
```


## Inference Only

To disable training and perform inference:

```bash
python3 main.py --test_path datasets/B/0.4_test.json
```

This will load the best model checkpoint and output predictions to `submission/`.

⚠️ Attention: to do inference up to 67GiB of VRAM are required

---

## Repository Structure

```
main.py               # Entry point for training and evaluation
base_model.py         # Pretraining script used to build model_base.pth
coTeaching.py         # Soft co-teaching with NCOD+, ELR, and adaptive weighting
losses.py             # NCODPlusLoss, SymmetricCrossEntropy, ELR logic
encoder.py            # ContinuousNodeEncoder and enhanced feature extractors
util.py               # Evaluation, plotting, metrics, MixUp, and saving
src/
  ├── conv.py         # GNN layers (GIN, GCN, GINE, GAT)
  └── models.py       # Full GNN model with pooling and classification
```

## Disclaimer

The logging system was implemented only near the end of the project. While it is functional, logs were not generated throughout the development, except for the final phase specifically for dataset B.