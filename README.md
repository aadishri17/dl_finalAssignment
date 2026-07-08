# MAI/IDL SS26 - Final Assignment: Medical Image Classification Pipeline

**Author**: Aadishri Soni & Suhasini Sharma 
**Date**: June 27 – July 8, 2026  
**Status**: Core bugs fixed & ready for training

---

## Overview

This is a PyTorch-based medical image classification pipeline supporting three deep learning models:
- **AlexNet** (Krizhevsky et al., 2012)
- **VGG16** (Simonyan & Zisserman, 2014)
- **ResNet18** (He et al., 2016)

The pipeline is tested on four medical imaging datasets and includes comprehensive bug fixes, configuration infrastructure, and documentation.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Dataset Specifications](#dataset-specifications)
7. [Accuracy Targets](#accuracy-targets)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python**: 3.8+
- **PyTorch**: 2.0+ (with CUDA 11.8+ for GPU support, or CPU)
- **Required packages**:
  - `torch`
  - `torchvision` (for transforms, optional)
  - `numpy`
  - `pathlib` (stdlib)

### GPU (Optional but Recommended)

If you have an NVIDIA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CPU-only Installation

```bash
pip install torch torchvision torchaudio
```

---

## Installation

### 1. Clone or Download the Repository

```bash
cd /path/to/project
```

### 2. Install Dependencies

```bash
pip install torch numpy
```

### 3. Verify Installation

```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

---

## Project Structure

```
.
├── train.py              # Main training script
├── models.py             # Model definitions (AlexNet, VGG16, ResNet18)
├── fit.py                # Trainer class with train_one_epoch() and evaluate()
├── data.py               # Data loading and preprocessing
├── benchmark.py          # Task 1 - Benchmark file to run and test models across all datasets
├── eval.py               # Task 1 - evaluating the performace metrics 
├── transfer.py           # Task 3 - transfer learning on orgs data set
├── config.json           # Configuration file (tunable hyperparameters)
├── requirements.txt      # all the necessary packages and libraries required
├── README.md             # This file
├── AUDIT_LOG.md          # Detailed bug report and fixes
├── REPORT.md             # Performance results and analysis
└── datasets/
    ├── cells.pt          # Medical image dataset
    ├── chest.pt          # Medical image dataset
    ├── lesions.pt        # Medical image dataset
    ├── organs.pt         # Medical image dataset
    └── orgs.pt           # Medical image dataset
```

---

## Configuration

### config.json Parameters

The pipeline is configured via `config.json`. Below is the complete parameter list:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DATA` | str | `"cells"` | Dataset name: `cells`, `chest`, `lesions`, `organs`, `orgs` |
| `DATA_PATH` | str | `/path/to/data` | Full path to dataset directory containing `.pt` files |
| `MODEL` | str | `"AlexNet"` | Model class: `AlexNet`, `VGG16`, or `ResNet18` |
| `CHANNELS` | int | 3 | Input image channels (3 for RGB, 1 for grayscale) |
| `NUM_CLASSES` | int | 8 | Number of classification classes |
| `BATCH_SIZE` | int | 32 | Training batch size |
| `LEARNING_RATE` | float | 0.001 | Adam optimizer learning rate |
| `EPOCHS` | int | 50 | Number of training epochs |
| `DROP_RATE` | float | 0.5 | Dropout rate (0.0–1.0) |


### Example Configurations

#### Training AlexNet on Cells Dataset

```json
{
  "DATA": "cells",
  "DATA_PATH": "/mnt/data",
  "MODEL": "AlexNet",
  "CHANNELS": 3,
  "NUM_CLASSES": 8,
  "BATCH_SIZE": 32,
  "LEARNING_RATE": 0.001,
  "EPOCHS": 100,
  "DROP_RATE": 0.5
}
```

#### Training VGG16 on Chest Dataset

```json
{
  "DATA": "chest",
  "DATA_PATH": "/mnt/data",
  "MODEL": "VGG16",
  "CHANNELS": 1,
  "NUM_CLASSES": 2,
  "BATCH_SIZE": 64,
  "LEARNING_RATE": 0.0005,
  "EPOCHS": 150,
  "DROP_RATE": 0.3
}
```

#### Training ResNet18 on Lesions Dataset

```json
{
  "DATA": "lesions",
  "DATA_PATH": "/mnt/data",
  "MODEL": "ResNet18",
  "CHANNELS": 3,
  "NUM_CLASSES": 7,
  "BATCH_SIZE": 32,
  "LEARNING_RATE": 0.001,
  "EPOCHS": 120,
  "DROP_RATE": 0.5
}
```

---

## Running the Pipeline

### Step 1: Update config.json

Edit `config.json` to point to your dataset directory and select model/dataset:

```bash
nano config.json
```

Ensure `DATA_PATH` points to the directory containing your `.pt` files.

### Step 2: Run Training

```bash
python train.py
```

**Expected Output**:
```
Training executing on device: cuda
 Starting Training Routine...
--------------------------------------------------
Epoch [01/50] | Train Loss: 2.0847 - Train Acc: 12.50% | Val Loss: 2.0751 - Val Acc: 12.50%
Epoch [02/50] | Train Loss: 1.9234 - Train Acc: 25.33% | Val Loss: 1.8456 - Val Acc: 28.12%
...
Epoch [50/50] | Train Loss: 0.1234 - Train Acc: 98.45% | Val Loss: 0.4567 - Val Acc: 92.34%
--------------------------------------------------
Training Complete!
```

### Step 3: Interpret Results

- **Train Loss**: Should decrease steadily each epoch
- **Train Acc**: Should increase steadily toward 100%
- **Val Loss**: Should decrease initially, may plateau or slightly increase (indicator of overfitting if gap widens)
- **Val Acc**: Target accuracy thresholds below

If loss flatlines or increases, check:
1. Learning rate (try 0.001 or 0.0005)
2. Batch size (try 16, 32, or 64)
3. Number of epochs (increase to 100+)
4. Dropout rate (reduce from 0.5 to 0.3)

---

## Dataset Specifications

| Dataset | Train Shape | Test Shape | Channels | Classes | Value Range | Size |
|---------|-------------|------------|----------|---------|-------------|------|
| **cells** | (13671, 3, 64, 64) | (3421, 3, 64, 64) | 3 (RGB) | 8 | [0.031, 1.0] | 801.25 MB |
| **chest** | (5232, 1, 64, 64) | (624, 1, 64, 64) | 1 (Gray) | 2 | [0.0, 1.0] | 91.52 MB |
| **lesions** | (8010, 3, 64, 64) | (2005, 3, 64, 64) | 3 (RGB) | 7 | [0.0, 1.0] | 469.49 MB |
| **organs** | (500, 1, 64, 64) | (200, 1, 64, 64) | 1 (Gray) | 11 | [0.0, 1.0] | 10.94 MB |
| **orgs** | (15367, 1, 64, 64) | (8216, 1, 64, 64) | 1 (Gray) | 11 | [0.0, 1.0] | 368.57 MB |

**Notes**:
- `orgs` and `organs` are two distinct datasets, not a typo: `orgs` is the original, larger organ-imaging dataset; `organs` is the new, low-sample dataset requiring a data-efficient / transfer-learning strategy (see the low-data integration task).
- All images are 64×64 pixels
- RGB datasets use 3 channels; grayscale use 1
- Data is normalized to [0, 1] range
- Validation split: **20%** of training data by default (`val_split=0.2` in `get_loaders()`, `data.py`), taken as the last 20% of the training samples. This is not currently exposed as a `config.json` parameter — to change it you must edit the `val_split` default or pass it explicitly in `train.py`.

---

## Accuracy Targets

### Minimum Test Accuracies (Task 1.3)

| Dataset | Target | Difficulty | Notes |
|---------|--------|-----------|-------|
| **cells** | ≥ 90% | High | 8-class RGB, 13.6K training samples |
| **chest** | ≥ 87% | Medium | 2-class binary, 5.2K training samples |
| **lesions** | ≥ 67% | Medium-High | 7-class RGB, 8K training samples, imbalanced |
| **orgs** | ≥ 83% | Medium | 11-class grayscale, 15.4K training samples |

### Strategy to Hit Targets

1. **Start with VGG16 or ResNet18** (more capacity than AlexNet)
2. **Tune learning rate**: Try 0.001, 0.0005, 0.0002
3. **Increase epochs**: 100–200 epochs for better convergence
4. **Reduce dropout**: 0.3–0.5 (not 0.99!)
5. **Larger batch size** for stable gradients: 32 or 64
6. **Monitor validation curve**: Stop if overfitting occurs

---

## Troubleshooting

### Issue: "FileNotFoundError: No such file or directory: 'config.json'"

**Solution**: Ensure `config.json` is in the same directory as `train.py`.

```bash
ls -la config.json
```

### Issue: "RuntimeError: mat1 and mat2 shapes cannot be multiplied" in the classifier

**Solution**: This was **BUG-005** (AlexNet) / **BUG-006** (VGG16) — the flattened feature dimension feeding the first `Linear` layer was miscalculated. Already corrected in the current `models.py` (AlexNet uses 3072, VGG16 uses 2048). If you see this again, check that you're on the fixed version.

### Issue: Model output is `None` / crash when computing loss or accuracy on ResNet18

**Solution**: This was **BUG-007** — `ResNet18.forward()` was missing its `return` statement. Already fixed in the current `models.py`.

### Issue: Loss does not decrease; accuracy remains at random baseline

**Possible causes**:
- **BUG-008**: Gradient accumulation from calling `optimizer.zero_grad()` after `loss.backward()` instead of before. Already fixed in `fit.py`.
- **BUG-011**: Incorrect padding on 1×1 convolutions in `VGGBlock`, corrupting the feature-map size progression. Already fixed in `models.py`.
- **BUG-012**: `VGGBlock` reusing the original `in_channels` for every conv layer in a multi-conv block instead of updating it after each layer. Already fixed in `models.py`.
- Learning rate too high/low. Try 0.001 or 0.0005.
- Batch size too small. Try 32 or 64.

### Issue: Validation/test accuracy suspiciously high, or doesn't generalize when re-run on fresh data

**Solution**: This was **BUG-002** — training and validation slices overlapped (data leakage) because the training slice wasn't cut off before the validation start index. Already fixed in `data.py`.

### Issue: `TypeError` when instantiating a model class

**Solution**: This was **BUG-009** — `drop_rate` was read from `config.json` but never passed into the model constructor. Already fixed in `train.py`.

### Issue: Works for one dataset but silently breaks on another (e.g., AlexNet on a dataset with more/fewer than 11 classes)

**Solution**: This was **BUG-013** — AlexNet's final classifier layer had `num_classes` hardcoded to 11. Already fixed to read from `config.json`.

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size in config.json:

```json
{
  "BATCH_SIZE": 16
}
```

### Issue: Training is very slow

**Solution**: Check device:

```python
import torch
print(torch.cuda.is_available())  # Should be True for GPU
print(torch.cuda.get_device_name(0))  # GPU model name
```

If CPU-only, consider:
- Using a machine with GPU
- Reducing batch size and epochs for faster iteration

---

## Model Details

### AlexNet
- **Layers**: 5 conv + 3 dense
- **Parameters**: ~11M
- **Best for**: Smaller datasets, quick prototyping
- **Recommended**: Cells, Chest

### VGG16
- **Layers**: 13 conv (C config) + 3 dense
- **Parameters**: ~138M
- **Best for**: Medium-to-large datasets
- **Recommended**: Lesions, Orgs

### ResNet18
- **Layers**: 18 residual blocks
- **Parameters**: ~11M
- **Best for**: Deep networks with skip connections, prevents vanishing gradients
- **Recommended**: All datasets (especially Orgs with 11 classes)

---

## Performance Reporting

After training, metrics are printed each epoch:
- **Train Loss / Acc**: Quality on training data
- **Val Loss / Acc**: Quality on held-out validation data
- **Test Accuracy**: Reported after retraining on train+val

For detailed analysis, see `REPORT.md`.

---

## Citation

```
@article{krizhevsky2012imagenet,
  title={ImageNet Classification with Deep Convolutional Neural Networks},
  author={Krizhevsky, A and Sutskever, I and Hinton, GE},
  journal={NeurIPS},
  year={2012}
}

@article{simonyan2014very,
  title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
  author={Simonyan, K and Zisserman, A},
  journal={ICLR},
  year={2015}
}

@article{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, K and Zhang, X and Ren, S and Sun, J},
  journal={CVPR},
  year={2016}
}
```

---

## Support

For bugs, errors, or questions:
1. Check `AUDIT_LOG.md` for known issues
2. Review console output for runtime errors
3. Consult `Troubleshooting` section above
4. Verify `config.json` is correctly formatted (valid JSON)

---

**Last Updated**: July 8, 2026
