"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.2):
    d_path = Path(data_path) / f"{data}.pt" # BUG-001: Corrected file name to match saved data
    data_dict = torch.load(d_path)

    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    #BUG-002: Ensure that the data is properly split into training and validation sets
    train_data = data_dict['train_images'][:val_start]
    train_labels = data_dict['train_labels'][:val_start].squeeze()  # BUG-003: Ensure labels are squeezed to correct shape
    val_data = data_dict['train_images'][val_start:]
    val_labels = data_dict['train_labels'][val_start:].squeeze()  # BUG-003: Ensure labels are squeezed to correct shape
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(data_dict['test_images'], data_dict['test_labels'].squeeze(1))  # BUG-003: Ensure labels are squeezed to correct shape
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader