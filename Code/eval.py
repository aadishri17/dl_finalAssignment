"""
EVALUATE: Compute Accuracy, Precision, Recall, F1 using saved weights
"""

import json
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

from data import get_loaders
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best configs from bench.py
with open("config.json") as f:
    config = json.load(f)

datasets_cfg = config["DATASETS"]
defaults = config["DEFAULTS"]

results = []

print(f"{'='*60}")
print("EVALUATION METRICS (Using Best Weights)")
print(f"{'='*60}\n")

for dataset_name, dataset_cfg in datasets_cfg.items():
    print(f"Evaluating {dataset_name}...")

    # Load test data
    _, _, test_loader = get_loaders(
        data=dataset_name,
        data_path=defaults["DATA_PATH"],
        batch_size=64
    )

    # Create model
    Model = getattr(models, dataset_cfg["best_model"])
    model = Model(
    in_channels=dataset_cfg["channels"],
    num_classes=dataset_cfg["classes"],
    drop_rate=dataset_cfg["dropout"]
    ).to(device)
    
    # Load best weights (saved during bench.py)
    weight_path = f"weights/{dataset_name}_{dataset_cfg['best_model']}_{dataset_cfg['dropout']}_{dataset_cfg['lr']}.pt"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Get predictions on test set
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    results.append({
        "Dataset": dataset_name,
        "Model": dataset_cfg["best_model"],
        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4)
    })

    print(f"  ✓ Acc: {accuracy:.2f}% | Prec: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n")

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("eval.csv", index=False)

print(f"{'='*60}")
print("Results saved to: eval.csv")
print(f"{'='*60}\n")
print(results_df.to_string(index=False))
                                        