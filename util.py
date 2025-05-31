import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns


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

            # Check if ground truth labels exist
            has_labels = hasattr(data, 'y') and data.y is not None and data.y.numel() > 0

            if has_labels:
                logits, _ = model(data, return_embedding=True)  # validation mode
            else:
                logits = model(data, return_embedding=False)     # test mode

            pred = logits.argmax(dim=1)
            predictions += pred.cpu().tolist()

            if has_labels:
                true_labels += data.y.cpu().tolist()

                if calculate_accuracy:
                    correct += (pred == data.y).sum().item()
                    total += data.y.size(0)
                    total_loss += criterion(logits, data.y).item()

    if calculate_accuracy and true_labels:
        accuracy = correct / total
        return total_loss / len(data_loader), accuracy, predictions, true_labels

    return None, None, predictions, true_labels



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
