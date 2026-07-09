"""
PROFILER: Comprehensive Green Initiative Analysis 
1. For each combination, takes the BEST hyperparameters from benchmark
3. Profiles BOTH full-size and downscaled models with those hyperparameters
4. Measures: Training time, Inference latency, Peak memory
5. Compares improvements
"""
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import pandas as pd

from data import get_loaders
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n Device: {device}\n")

# All combinations of datasets and models to profile
ALL_COMBINATIONS = {
    "cells_AlexNet": {"dataset": "cells", "model": "AlexNet"},
    "cells_VGG16": {"dataset": "cells", "model": "VGG16"},
    "cells_ResNet18": {"dataset": "cells", "model": "ResNet18"},

    "chest_AlexNet": {"dataset": "chest", "model": "AlexNet"},
    "chest_VGG16": {"dataset": "chest", "model": "VGG16"},
    "chest_ResNet18": {"dataset": "chest", "model": "ResNet18"},

    "lesions_AlexNet": {"dataset": "lesions", "model": "AlexNet"},
    "lesions_VGG16": {"dataset": "lesions", "model": "VGG16"},
    "lesions_ResNet18": {"dataset": "lesions", "model": "ResNet18"},

    "orgs_AlexNet": {"dataset": "orgs", "model": "AlexNet"},
    "orgs_VGG16": {"dataset": "orgs", "model": "VGG16"},
    "orgs_ResNet18": {"dataset": "orgs", "model": "ResNet18"},
}

DATASETS = {
    "cells": {"channels": 3, "classes": 8},
    "chest": {"channels": 1, "classes": 2},
    "lesions": {"channels": 3, "classes": 7},
    "orgs": {"channels": 1, "classes": 11},
}

TARGET_ACCURACY = {
    "cells": 90,
    "chest": 87,
    "lesions": 67,
    "orgs": 83
}

with open("config.json") as f:
    config = json.load(f)

# Load benchmark results to get best hyperparams for each combo
bench_df = pd.read_csv("bench.csv")


def get_best_hyperparams(dataset, model):
    """
    Get best hyperparameters from config.json
    """

    dataset_config = config["DATASETS"][dataset]

    return {
        "dropout": dataset_config["dropout"],
        "lr": dataset_config["lr"],
        "epochs": dataset_config["epochs"]
    }

def percentage_improvement(full_value, down_value):
    if full_value == 0:
        return 0.0
    return ((full_value - down_value) / full_value) * 100

def get_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def count_params(model):
    """Count total model parameters"""
    return sum(p.numel() for p in model.parameters())

def evaluate_accuracy(model, test_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

def profile_model(dataset_name, model_name, hyperparams, model_version):
    """
    Profile a single model

    dataset_name: "cells", "chest", etc.
    model_name: "AlexNet", "VGG16", "ResNet18"
    hyperparams: {"dropout": 0.3, "lr": 0.0001, "epochs": 20}
    model_version: "full" or "downscaled"

    Returns: {params, train_time, latency, mem_train, mem_infer, test_acc}
    """
    # Load data
    train_loader, _, test_loader = get_loaders(
    data=dataset_name,
    data_path=config["DEFAULTS"]["DATA_PATH"],
    batch_size=64
    )

    # Create model
    if model_version == "downscaled":
        model_class_name = "Downscaled" + model_name
    else:
        model_class_name = model_name

    Model = getattr(models, model_class_name)
    model = Model(
        in_channels=DATASETS[dataset_name]["channels"],
        num_classes=DATASETS[dataset_name]["classes"],
        drop_rate=hyperparams["dropout"]
    ).to(device)

    param_count = count_params(model)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_start = time.time()

    # Train for best number of epochs found in benchmark
    for epoch in range(hyperparams["epochs"]):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

    train_time = time.time() - train_start
    peak_mem_train = get_memory_mb()

    # Inference profiling
    model.eval()
    latencies = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            # Warm up GPU
            _ = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Measure inference time
            t0 = time.time()
            _ = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()

            # Convert to ms per sample
            batch_time_ms = (t1 - t0) * 1000
            latency_per_sample = batch_time_ms / len(images)
            latencies.append(latency_per_sample)

    avg_latency = sum(latencies) / len(latencies)
    peak_mem_infer = get_memory_mb()
    test_accuracy = evaluate_accuracy(model, test_loader)

    result = {
        "params": param_count,
        "train_time": train_time,
        "latency": avg_latency,
        "mem_train": peak_mem_train,
        "mem_infer": peak_mem_infer,
        "accuracy": test_accuracy
    }

    torch.cuda.empty_cache()
 
    return result

# Main profiling loop
results = []

print("="*130)
print("COMPREHENSIVE EFFICIENCY VERIFICATION")
print("="*130)

combination_count = 0
for combo_name, combo_info in ALL_COMBINATIONS.items():
    combination_count += 1

    dataset_name = combo_info["dataset"]
    model_name = combo_info["model"]

    # Get best hyperparameters for this specific combination
    hyperparams = get_best_hyperparams(dataset_name, model_name)

    print(f"{dataset_name.upper()} + {model_name}")
    print(f" Hyperparams: Dropout={hyperparams['dropout']}, LR={hyperparams['lr']}, Epochs={hyperparams['epochs']}")
    print(f"{'='*130}\n")

    # Calling full-size model
    full_metrics = profile_model(dataset_name, model_name, hyperparams, "full")

    # Calling downscaled model
    down_metrics = profile_model(dataset_name, model_name, hyperparams, "downscaled")

  
    # Calculate improvements
    param_reduction = percentage_improvement(full_metrics["params"], down_metrics["params"])
    time_speedup = percentage_improvement(full_metrics["train_time"], down_metrics["train_time"])
    latency_speedup = percentage_improvement(full_metrics["latency"], down_metrics["latency"])
    mem_train_reduction = percentage_improvement(full_metrics["mem_train"], down_metrics["mem_train"])
    mem_infer_reduction = percentage_improvement(full_metrics["mem_infer"], down_metrics["mem_infer"])

    # Display comparison for this combination
    print(f"  COMPARISON:\n")
    print(f"  {'Metric':<35} {'Full-Size':<22} {'Downscaled':<22} {'Improvement':<20}")
    print(f"  {'-'*125}")
    print(f"  {'Parameters':<35} {full_metrics['params']:>17,} {down_metrics['params']:>17,} {param_reduction:>18.1f}%")
    print(f"  {'Training Time (s)':<35} {full_metrics['train_time']:>22.3f} {down_metrics['train_time']:>22.3f} {time_speedup:>18.1f}% faster")
    print(f"  {'Inference Latency (ms/sample)':<35} {full_metrics['latency']:>22.4f} {down_metrics['latency']:>22.4f} {latency_speedup:>18.1f}% faster")
    print(f"  {'Training Memory (MB)':<35} {full_metrics['mem_train']:>22.2f} {down_metrics['mem_train']:>22.2f} {mem_train_reduction:>18.1f}% less")
    print(f"  {'Inference Memory (MB)':<35} {full_metrics['mem_infer']:>22.2f} {down_metrics['mem_infer']:>22.2f} {mem_infer_reduction:>18.1f}% less")
    print(f"  {'-'*125}\n")

    
    # Store result
    full_meets_target = full_metrics["accuracy"] >= TARGET_ACCURACY[dataset_name]
    down_meets_target = down_metrics["accuracy"] >= TARGET_ACCURACY[dataset_name]

    results.append({
        "Dataset": dataset_name,
        "Model": model_name,
        "Dropout": hyperparams["dropout"],
        "LR": hyperparams["lr"],
        "Epochs": hyperparams["epochs"],

        "Full_Params": full_metrics["params"],
        "Downscaled_Params": down_metrics["params"],
        "Param_Reduction_%": round(param_reduction, 1),

        "Full_Train_Time_s": round(full_metrics["train_time"], 3),
        "Downscaled_Train_Time_s": round(down_metrics["train_time"], 3),
        "Train_Speedup_%": round(time_speedup, 1),

        "Full_Latency_ms": round(full_metrics["latency"], 4),
        "Downscaled_Latency_ms": round(down_metrics["latency"], 4),
        "Latency_Speedup_%": round(latency_speedup, 1),

        "Full_Memory_Train_MB": round(full_metrics["mem_train"], 2),
        "Downscaled_Memory_Train_MB": round(down_metrics["mem_train"], 2),
        "Memory_Train_Reduction_%": round(mem_train_reduction, 1),

        "Full_Memory_Infer_MB": round(full_metrics["mem_infer"], 2),
        "Downscaled_Memory_Infer_MB": round(down_metrics["mem_infer"], 2),
        "Memory_Infer_Reduction_%": round(mem_infer_reduction, 1),

        "Full_Accuracy": round(full_metrics["accuracy"], 2),
        "Downscaled_Accuracy": round(down_metrics["accuracy"], 2),

        "Target_Accuracy": TARGET_ACCURACY[dataset_name],

        "Full_Meets_Target": full_meets_target,
        "Downscaled_Meets_Target": down_meets_target,
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("profiler.csv", index=False)
print("Results saved to profiler.csv")
print("\n" + "="*130)

# Summarize results
print(" EFFICIENCY VERIFICATION MATRIX:\n")

summary_cols = ["Dataset", "Model", "Param_Reduction_%", "Train_Speedup_%", "Latency_Speedup_%", "Memory_Train_Reduction_%"]
print(results_df[summary_cols].to_string(index=False))

print("\n" + "="*130)

# Aggregate analysis
print(" AGGREGATE GREEN INITIATIVE ANALYSIS:\n")

# Calculate averages by metric
avg_param_reduction = results_df["Param_Reduction_%"].mean()
avg_train_speedup = results_df["Train_Speedup_%"].mean()
avg_latency_speedup = results_df["Latency_Speedup_%"].mean()
avg_mem_train_reduction = results_df["Memory_Train_Reduction_%"].mean()
avg_mem_infer_reduction = results_df["Memory_Infer_Reduction_%"].mean()

print("OVERALL IMPROVEMENTS:")
print(f"Parameter Reduction:     {avg_param_reduction:.1f}%")
print(f"Training Speedup:        {avg_train_speedup:.1f}%")
print(f"Inference Speedup:       {avg_latency_speedup:.1f}%")
print(f"Training Memory Savings: {avg_mem_train_reduction:.1f}%")
print(f"Inference Memory Savings:{avg_mem_infer_reduction:.1f}%\n")

# Best and worst performers
print("BEST PERFORMERS:")
best_param = results_df.loc[results_df["Param_Reduction_%"].idxmax()]
best_speed = results_df.loc[results_df["Train_Speedup_%"].idxmax()]
best_mem = results_df.loc[results_df["Memory_Train_Reduction_%"].idxmax()]

print(f"Most Parameters Reduced: {best_param['Dataset']}-{best_param['Model']} ({best_param['Param_Reduction_%']:.1f}%)")
print(f"Fastest Training:        {best_speed['Dataset']}-{best_speed['Model']} ({best_speed['Train_Speedup_%']:.1f}% faster)")
print(f"Most Memory Saved:       {best_mem['Dataset']}-{best_mem['Model']} ({best_mem['Memory_Train_Reduction_%']:.1f}%)\n")

# By dataset analysis
print("ANALYSIS BY DATASET:\n")
for dataset in ["cells", "chest", "lesions", "orgs"]:
    dataset_results = results_df[results_df["Dataset"] == dataset]                       
    print(f"{dataset.upper()}:")                                                         
    print(f"Avg param reduction: {dataset_results['Param_Reduction_%'].mean():.1f}%")
    print(f"Avg train speedup:   {dataset_results['Train_Speedup_%'].mean():.1f}%")
    print(f"Avg memory savings:  {dataset_results['Memory_Train_Reduction_%'].mean():.1f}%")                                                                        