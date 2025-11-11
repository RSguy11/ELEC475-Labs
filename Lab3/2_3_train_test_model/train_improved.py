"""
ELEC475 Lab 3 - Improved SMNet Training with Class Balancing and Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directories to path
sys.path.append('../2_2_Custom_SMNet')
sys.path.append('../2_1_Evaluate_Model')
from model import SMNet
from step1_local_voc import LocalVOCDataset, squeeze_and_long

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.
    Focuses learning on hard examples and reduces the weight of well-classified examples.
    """
    def __init__(self, alpha=1, gamma=2, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss with automatic class weight calculation.
    """
    def __init__(self, ignore_index=255, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.class_weights = None
    
    def calculate_class_weights(self, dataset_loader, num_classes=21, device='cpu'):
        """Calculate class weights based on inverse frequency."""
        print("Calculating class weights from dataset...")
        class_counts = torch.zeros(num_classes)
        total_pixels = 0
        
        for _, targets in tqdm(dataset_loader, desc="Analyzing dataset"):
            targets = targets.to(device)
            valid_mask = targets != self.ignore_index
            valid_targets = targets[valid_mask]
            
            for cls in range(num_classes):
                class_counts[cls] += (valid_targets == cls).sum().item()
            total_pixels += valid_mask.sum().item()
        
        # Calculate inverse frequency weights
        class_frequencies = class_counts / total_pixels
        # Add small epsilon to avoid division by zero
        class_weights = 1.0 / (class_frequencies + 1e-8)
        # Normalize weights so they sum to num_classes
        class_weights = class_weights / class_weights.sum() * num_classes
        
        print("Class weights calculated:")
        for i, weight in enumerate(class_weights):
            print(f"  Class {i}: {weight:.4f} (freq: {class_frequencies[i]:.4f})")
        
        self.class_weights = class_weights.to(device)
        return self.class_weights
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.class_weights, 
                             ignore_index=self.ignore_index, reduction=self.reduction)

def save_training_history(train_losses, val_losses, train_mious, val_mious, 
                         learning_rates, model_info, base_dim, num_epochs, suffix="improved"):
    """Save training history to JSON file."""
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mious': train_mious,
        'val_mious': val_mious,
        'learning_rates': learning_rates,
        'model_info': model_info,
        'epochs': num_epochs
    }
    
    filename = f'smnet_training_history_base{base_dim}_{suffix}.json'
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"[FILE] Training history saved: {filename}")

def calculate_miou(pred, target, num_classes=21, ignore_index=255):
    """Calculate mean IoU for segmentation with proper masking."""
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    
    # Mask out ignore_index pixels
    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union > 0:
            ious.append(intersection / union)
    
    return np.array(ious)

def calculate_batch_miou(predictions, targets, num_classes=21):
    """Calculate mIoU for a batch."""
    batch_ious = []
    for pred, target in zip(predictions, targets):
        ious = calculate_miou(pred, target, num_classes)
        if len(ious) > 0:
            batch_ious.append(np.mean(ious))
    return np.mean(batch_ious) if batch_ious else 0.0

def train_improved_smnet(base_dim=32, batch_size=16, num_epochs=50, learning_rate=1e-3, 
                        loss_type='focal', max_samples=None):
    """
    Train SMNet with improved techniques for better segmentation.
    
    Args:
        base_dim: Base dimension for SMNet (try 32 instead of 16)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        loss_type: 'focal', 'weighted', or 'standard'
        max_samples: Maximum samples for quick testing
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    print(f"Configuration: base_dim={base_dim}, loss={loss_type}, lr={learning_rate}")
    
    # Dataset path
    voc_root = r"../pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val"
    
    if not os.path.exists(voc_root):
        alt_paths = [
            r"./pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val",
            r"../pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val",
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                voc_root = alt_path
                break
        else:
            raise FileNotFoundError("PASCAL VOC 2012 dataset not found!")
    
    # Improved transforms with better augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Slightly larger for better detail
        transforms.RandomCrop((224, 224)),  # Random crop for augmentation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop((224, 224)),  # Match image transforms
        transforms.PILToTensor(),
        squeeze_and_long
    ])
    
    val_target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        squeeze_and_long
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='train',
        transform=train_transform,
        target_transform=target_transform,
        max_samples=max_samples
    )
    
    val_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='val',
        transform=val_transform,
        target_transform=val_target_transform,
        max_samples=max_samples//4 if max_samples else None
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize improved model with larger capacity
    model = SMNet(num_classes=21, base_dim=base_dim).to(device)
    model_info = model.get_model_info()
    print(f"Model initialized: {model_info['model_name']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    
    # Choose loss function based on type
    if loss_type == 'focal':
        print("Using Focal Loss for class imbalance handling")
        criterion = FocalLoss(alpha=1, gamma=2, ignore_index=255)
    elif loss_type == 'weighted':
        print("Using Weighted Cross Entropy Loss")
        criterion = WeightedCrossEntropyLoss(ignore_index=255)
        # Calculate class weights from training data
        weight_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
        criterion.calculate_class_weights(weight_loader, device=device)
    else:
        print("Using standard Cross Entropy Loss")
        criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Improved optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training history
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []
    learning_rates = []
    
    print(f"\nStarting improved training for {num_epochs} epochs...")
    print("-" * 80)
    
    best_val_miou = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_miou = 0.0
        train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                batch_miou = calculate_batch_miou(predictions, targets)
                
                train_loss += loss.item() * images.size(0)
                train_miou += batch_miou * images.size(0)
                train_samples += images.size(0)
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'mIoU': f'{batch_miou:.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # Calculate epoch averages
        avg_train_loss = train_loss / train_samples
        avg_train_miou = train_miou / train_samples
        current_lr = optimizer.param_groups[0]['lr']
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        val_samples = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, targets in val_pbar:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                predictions = torch.argmax(outputs, dim=1)
                batch_miou = calculate_batch_miou(predictions, targets)
                
                val_loss += loss.item() * images.size(0)
                val_miou += batch_miou * images.size(0)
                val_samples += images.size(0)
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'mIoU': f'{batch_miou:.4f}'
                })
        
        avg_val_loss = val_loss / val_samples
        avg_val_miou = val_miou / val_samples
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_mious.append(avg_train_miou)
        val_mious.append(avg_val_miou)
        learning_rates.append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train mIoU: {avg_train_miou:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val mIoU: {avg_val_miou:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            model_filename = f'best_smnet_model_base{base_dim}_{loss_type}.pth'
            torch.save(model.state_dict(), model_filename)
            print(f"  ✓ New best model saved! mIoU: {best_val_miou:.4f} -> {model_filename}")
        
        print("-" * 80)
    
    # Save training history
    save_training_history(train_losses, val_losses, train_mious, val_mious, 
                         learning_rates, model_info, base_dim, num_epochs, 
                         suffix=f"{loss_type}")
    
    print(f"\n[COMPLETED] Improved training finished!")
    print(f"Best validation mIoU: {best_val_miou:.4f}")
    print(f"Model saved as: best_smnet_model_base{base_dim}_{loss_type}.pth")

def main():
    parser = argparse.ArgumentParser(description='Train improved SMNet with class balancing')
    parser.add_argument('--base-dim', type=int, default=32,
                       help='Base dimension of the model. Default: 32 (increased from 16)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training. Default: 16')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs. Default: 50')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate. Default: 1e-3')
    parser.add_argument('--loss-type', choices=['focal', 'weighted', 'standard'], default='focal',
                       help='Loss function type. Default: focal')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples for quick testing. Default: None (use all)')
    
    args = parser.parse_args()
    
    train_improved_smnet(
        base_dim=args.base_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        loss_type=args.loss_type,
        max_samples=args.max_samples
    )

if __name__ == '__main__':
    main()