# Robust Graph Neural Networks with Noisy Labels

This repository implements a **robust graph classification pipeline** using **Graph Neural Networks (GNNs)**, tailored for learning under **noisy labels**. The system integrates advanced components such as:

- **NCOD+ Loss**: Centroid-aware outlier discounting with KL regularization
- **ELR (Early Learning Regularization)** to prevent memorization
- **Soft Co-Teaching** with adaptive weighting
- **GINE Convolutional layers** with edge features
- **MixUp data augmentation**
- **Early stopping** on macro-F1
- **Fine-tuning** and continuous node feature encoding

---

## Getting Started

### 🔧 Training

To train the model, run:

```bash
python main.py
```

This will automatically:
- Initialize the model (e.g., GINE, GCN, GAT, Transformer)
- Train using co-teaching with NCOD+ and ELR
- Use enhanced features and a continuous node encoder
- Save checkpoints with early stopping on validation F1

### Inference Mode

To perform inference only (no training), simply set:

```python
args.train_path = None
```

The script will then load the best model and evaluate on the test set.

---

## File Structure

```
main.py               # Entry point (training & inference)
coTeaching.py         # Co-teaching with NCOD+/ELR and early stopping
losses.py             # NCODPlusLoss, SymmetricCrossEntropyLoss, ELR
encoder.py            # Feature encoders and normalization
util.py               # Evaluation, metrics, plotting
src/
  ├── conv.py         # GNN layers (GIN, GINE, GCN, GAT, Transformer)
  └── models.py       # GNN wrapper with pooling and classification
```

---

```
# Deep-Learning-Project
