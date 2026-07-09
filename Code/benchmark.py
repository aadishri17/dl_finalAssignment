"""
BENCHMARK: Find best hyperparameters for each dataset
- 4 datasets (cells, chest, lesions, orgs)
- 3 models (AlexNet, VGG16, ResNet18)
- 2 dropout rates (0.2, 0.3, 0.5)
- 2 learning rates (0.0001, 0.00001)
- Early stopping to prevent overfitting
- Saves best weights for each experiment
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path

from data import get_loaders
import models

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nRunning on: {device}\n")

# Dataset configuration
DATASETS = {
    "cells": {"channels": 3, "classes": 8, "target": 90},
    "chest": {"channels": 1, "classes": 2, "target": 87},
    "lesions": {"channels": 3, "classes": 7, "target": 67},
    "orgs": {"channels": 1, "classes": 11, "target": 83},
}

# Hyperparameters to test
MODELS = ["AlexNet", "VGG16", "ResNet18"]
DROPOUTS = [0.2, 0.3, 0.5]
LEARNING_RATES = [1e-4, 1e-5]
BATCH_SIZE = 64  # Fixed

# Load config
with open("config.json") as f:
    config = json.load(f)

# Create directories
os.makedirs("weights", exist_ok=True)

# Store results
results = []

print("="*80)
print("STARTING BENCHMARK")
print("="*80)

# Run all combinations
for dataset_name, dataset_info in DATASETS.items():
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"{'='*80}")

    for model_name in MODELS:
        for dropout in DROPOUTS:
            for lr in LEARNING_RATES:

                print(f"\n{model_name} | Dropout={dropout} | LR={lr}")
                print("-" * 80)

                # Load data
                train_loader, val_loader, test_loader = get_loaders(
                    data=dataset_name,
                    data_path=config["DATA_PATH"],
                    batch_size=BATCH_SIZE
                )
                # Create model
                Model = getattr(models, model_name)
                model = Model(
                    in_channels=dataset_info["channels"],
                    num_classes=dataset_info["classes"],
                    drop_rate=dropout
                ).to(device)

                # Setup training
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                # Early stopping variables
                best_val_loss = float('inf')
                patience = 5  # Stop if no improvement for 5 epochs
                no_improve_count = 0
                best_epoch = 0

                # Print header for epoch table
                print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
                print("-" * 80)

                # Train with early stopping
                for epoch in range(100):  # Max 100 epochs

                    # Train
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0

                    for images, labels in train_loader:
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()

                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item() * images.size(0)
                        _, predicted = outputs.max(1)
                        train_total += labels.size(0)
                        train_correct += predicted.eq(labels).sum().item()

                    train_loss /= train_total
                    train_acc = (train_correct / train_total) * 100

                    # Validate
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            loss = criterion(outputs, labels)

                            val_loss += loss.item() * images.size(0)
                            _, predicted = outputs.max(1)
                            val_total += labels.size(0)
                            val_correct += predicted.eq(labels).sum().item()

                    val_loss /= val_total
                    val_acc = (val_correct / val_total) * 100

                    # Print epoch results
                    print(f"{epoch+1:<12} {train_loss:<10.4f} {train_acc:>10.2f}% {val_loss:>13.4f} {val_acc:>14.2f}%")

                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        best_epoch = epoch + 1

                        # Save best weights
                        weight_path = f"weights/{dataset_name}_{model_name}_{dropout}_{lr}.pt"
                        torch.save(model.state_dict(), weight_path)
                    else:
                        no_improve_count += 1

                    # Stop if no improvement
                    if no_improve_count >= patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break

                print("-" * 80)

                # Test with best weights
                model.load_state_dict(torch.load(
                    f"weights/{dataset_name}_{model_name}_{dropout}_{lr}.pt",
                    map_location=device
                ))
                model.eval()
                test_correct = 0
                test_total = 0

                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        test_correct += predicted.eq(labels).sum().item()
                        test_total += labels.size(0)

                test_acc = (test_correct / test_total) * 100

                # Store result
                results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Dropout": dropout,
                    "LR": lr,
                    "Accuracy": round(test_acc, 2),
                    "BestEpoch": best_epoch,
                    "MetTarget": "✓" if test_acc >= dataset_info["target"] else "✗"
                })

                # Print final summary
                print(f"Acc: {test_acc:.2f}% | Target: {dataset_info['target']}% | Epochs: {best_epoch} | Status: {results[-1]['MetTarget']}\n")

# Save all results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
results_df.to_csv("bench.csv", index=False)

print(f"\n{'='*80}")

print(f"\n{'='*80}")
print("BENCHMARK COMPLETE - Results saved to bench.csv")
print(f"{'='*80}\n")

# Print summary table (sorted by accuracy)
print("ALL EXPERIMENTS (Sorted by Accuracy):")
print("-"*90)
print(f"{'#':<4} {'Dataset':<10} {'Model':<12} {'Dropout':<10} {'LR':<12} {'Accuracy':<12} {'Target':<10} {'Status':<8}")
print("-"*90)

for idx, (_, row) in enumerate(results_df.iterrows(), 1):
    print(f"{idx:<4} {row['Dataset']:<10} {row['Model']:<12} {row['Dropout']:<10.1f} {row['LR']:<12.5f} {row['Accuracy']:<12.2f}% {row['MetTarget']:<10} {row['MetTarget']:<8}")

print("-"*90 + "\n")

# Best configuration per dataset
print("="*80)
print("BEST CONFIGURATION PER DATASET (Meeting Accuracy Targets)")
print("="*80 + "\n")

best_configs = {}

for dataset_name in DATASETS.keys():
    dataset_results = results_df[results_df["Dataset"] == dataset_name]
    target = DATASETS[dataset_name]["target"]
    meets_target = dataset_results[dataset_results["Accuracy"] >= target]

    if not meets_target.empty:
        best = meets_target.iloc[0]  # First one (highest accuracy)
        best_configs[dataset_name] = {
        "channels": DATASETS[dataset_name]["channels"],
        "classes": DATASETS[dataset_name]["classes"],
        "best_model": best["Model"],
        "dropout": float(best["Dropout"]),
        "lr": float(best["LR"]),
        "epochs": int(best["BestEpoch"])
        }
        print(f"{dataset_name.upper()}")
        print(f"  Model:          {best['Model']}")
        print(f"  Dropout:        {best['Dropout']}")
        print(f"  Learning Rate:  {best['LR']}")
        print(f"  Accuracy:       {best['Accuracy']:.2f}% (Target: {target}%)")
        print(f"  Best Epoch:     {best['BestEpoch']}")
        print()
    else:
        print(f"{dataset_name.upper()}")
        print(f"  ✗ NO CONFIG MEETS TARGET {target}%")
        print()

# Update config.json with best configurations
config["DATASETS"].update(best_configs)

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

print("="*80)
print("Best configs saved directly to: config.json")
print("Best weights saved to: weights/")
print("="*80 + "\n")