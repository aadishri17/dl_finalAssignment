"""
TRANSFER LEARNING: Organs Dataset (Scarce - 500 samples)
Approach 1: Train from scratch (random weights)
Approach 2: Transfer learning (pre-trained orgs)
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from data import get_loaders  
import models
from fit import Trainer

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Device: {device}\n")

# Load config
with open("config.json") as f:
    config = json.load(f)

orgs_best_config = config["DATASETS"]["orgs"]
pretrain_model = orgs_best_config["best_model"]
pretrain_dropout = orgs_best_config["dropout"]
pretrain_lr = orgs_best_config["lr"]

# Organs dataset info
organs = config["DATASETS"]["orgs"]

# Orgs pre-trained weights (source model)
orgs_weights_path = (f"weights/orgs_{pretrain_model}_{pretrain_dropout}_{pretrain_lr}.pt")

# Hyperparameters
hyperparams = {
    "batch_size": 16,      # Small batch for scarce data
    "lr": 0.00001,         # small learning rate for scarce data
    "epochs": 100,         # Max epochs (early stop limits this)
    "patience": 10,        # Early stopping patience
}

print("="*80)
print("TRANSFER LEARNING: Organs (Scarce - 500 samples)")
print("="*80 + "\n")

# Store results
results = []

# APPROACH 1: TRAIN FROM SCRATCH
print("APPROACH 1: TRAINING FROM SCRATCH")
print("\n")

# Load organs data (with normalization from data.py)
train_loader, val_loader, test_loader = get_loaders(
    data="organs",
    data_path=config["DEFAULTS"]["DATA_PATH"],
    batch_size=hyperparams["batch_size"]
)

# Create model (random weights)
print("Training...")
Model = getattr(models, pretrain_model)

model_scratch = Model(
    in_channels=organs["channels"],
    num_classes=organs["classes"],
    drop_rate=pretrain_dropout
).to(device)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=hyperparams["lr"])

# Train with early stopping
trainer_scratch = Trainer(model_scratch, criterion, optimizer_scratch, device)

print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
print("-" * 80)

best_val_loss = float('inf')
no_improve = 0
best_epoch_scratch = 0

for epoch in range(hyperparams["epochs"]):
    train_loss, train_acc = trainer_scratch.train_one_epoch(train_loader)
    val_loss, val_acc = trainer_scratch.evaluate(val_loader)

    print(f"{epoch+1:<12} {train_loss:<10.4f} {train_acc:>10.2f}% {val_loss:>13.4f} {val_acc:>14.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        best_epoch_scratch = epoch + 1
        torch.save(model_scratch.state_dict(), "weights/organs_scratch.pt")
    else:
        no_improve += 1

    if no_improve >= hyperparams["patience"]:
        print(f"\nEarly stopping at epoch {epoch + 1}\n")
        break

# Test from scratch
model_scratch.load_state_dict(torch.load("weights/organs_scratch.pt", map_location=device))
_, test_acc_scratch = trainer_scratch.evaluate(test_loader)

print(f"Test Accuracy (From Scratch): {test_acc_scratch:.2f}%\n")

results.append({
    "Mode": "From Scratch",
    "Accuracy_%": round(test_acc_scratch, 2),
    "Epochs": best_epoch_scratch,
})

# APPROACH 2: TRANSFER LEARNING
print("APPROACH 2: TRANSFER LEARNING (Pretrained ORGS)")
print("="*80 + "\n")

# Create model
model_transfer = Model(
    in_channels=organs["channels"],
    num_classes=organs["classes"],
    drop_rate=pretrain_dropout
).to(device)

# Load pretrained weights
model_transfer.load_state_dict(torch.load(orgs_weights_path, map_location=device))

print(f"Loaded pretrained weights from {orgs_weights_path}")

# Freezing all layers
for param in model_transfer.parameters():
    param.requires_grad = False

# Replacing classifier(fc head) with input features from pretrained model and output classes for organs dataset
model_transfer.classifier = nn.Linear(model_transfer.classifier.in_features, organs["classes"]).to(device)

# Optimizer (classifier only)
criterion = nn.CrossEntropyLoss()

optimizer_transfer = optim.Adam(model_transfer.classifier.parameters(), lr=1e-3)      # this 1e-3 controls learning of the classifier

trainer_transfer = Trainer(model_transfer, criterion, optimizer_transfer, device)

# Training
best_val_loss = float("inf")
patience = orgs_best_config["patience"]
no_improve = 0
best_epoch_transfer = 0

print(f"{'Epoch':<10} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<150}")
print("-"*80)

for epoch in range(orgs_best_config["epochs"]):

    train_loss, train_acc = trainer_transfer.train_one_epoch(train_loader)
    val_loss, val_acc = trainer_transfer.evaluate(val_loader)

    print(f"{epoch+1:<12} {train_loss:<10.4f} {train_acc:>10.2f}% {val_loss:>13.4f} {val_acc:>14.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        best_epoch_transfer = epoch + 1

        torch.save(model_transfer.state_dict(), "weights/organs_pretrained.pt")

    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# Test
model_transfer.load_state_dict(torch.load("weights/organs_pretrained.pt", map_location=device)
)

_, transfer_acc = trainer_transfer.evaluate(test_loader)

print(f"\nPretrained ORGS model test Accuracy: {transfer_acc:.2f}%")

results.append({
    "Mode": "Transfer Learning",
    "Accuracy_%": round(transfer_acc , 2),
    "Epochs": best_epoch_transfer,
})

# Comparison
print("\n" + "="*80)
print("COMPARISON: Training from Scratch vs Transfer Learning")
print("="*80 + "\n")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

improvement = transfer_acc - test_acc_scratch
improvement_pct = (improvement / test_acc_scratch * 100) if test_acc_scratch > 0 else 0

print(f"\n{'─'*80}")
print(f"Improvement:    {improvement*100:+.2f} percentage points ({improvement_pct:+.1f}%)")
print(f"Epochs saved:   {best_epoch_scratch - best_epoch_transfer} epochs")
print(f"{'─'*80}\n")

# Save result
results_df.to_csv("transfer.csv", index=False)
print("Results saved to: transfer.csv\n")
                                                                                                                  