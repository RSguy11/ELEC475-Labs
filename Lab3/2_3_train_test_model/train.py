"""
ELEC475 Lab 3 - Step 2.3: Train SMNet Custom Segmentation Model
"""

import torch
import torch.nn as nn
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

def save_training_history(train_losses, val_losses, train_mious, val_mious, 
                         learning_rates, model_info, base_dim, num_epochs):
    """Save simple training history to JSON file."""
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mious': train_mious,
        'val_mious': val_mious,
        'learning_rates': learning_rates
    }
    
    with open(f'smnet_training_history_base{base_dim}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"[FILE] Training history saved: smnet_training_history_base{base_dim}.json")

def calculate_miou(pred, target, num_classes=21):
    """Calculate mean IoU for segmentation."""
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    ious = []
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union > 0:
            ious.append(intersection / union)
        else:
            # If class not present in either pred or target, ignore it
            continue
    
    return np.mean(ious) if ious else 0.0

def calculate_batch_miou(predictions, targets, num_classes=21):
    """Calculate mIoU for a batch of predictions."""
    batch_miou = 0.0
    valid_samples = 0
    
    for pred, target in zip(predictions, targets):
        miou = calculate_miou(pred, target, num_classes)
        if miou > 0:  # Only count samples with valid IoU
            batch_miou += miou
            valid_samples += 1
    
    return batch_miou / valid_samples if valid_samples > 0 else 0.0

def train_smnet(base_dim=16, num_epochs=50, batch_size=8, learning_rate=0.001, max_samples=None):
    """Train SMNet custom segmentation model."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training SMNet on device: {device}")
    
    # Print hyperparameters
    print(f"Hyperparameters:")
    print(f"  Base dimension: {base_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Max samples: {max_samples if max_samples else 'All'}")
    
    # Dataset path
    voc_root = r"../pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val"
    
    # Check dataset existence
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
    
    # Transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        target_transform=target_transform,
        max_samples=max_samples//4 if max_samples else None  # Smaller validation set
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = SMNet(num_classes=21, base_dim=base_dim).to(device)
    model_info = model.get_model_info()
    print(f"Model initialized: {model_info['model_name']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    
    # Loss function - Cross-entropy for segmentation
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore void class
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []
    learning_rates = []
    val_mious = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
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
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'mIoU': f'{batch_miou:.4f}'
            })
        
        # Calculate average training metrics
        avg_train_loss = train_loss / train_samples
        avg_train_miou = train_miou / train_samples
        train_losses.append(avg_train_loss)
        train_mious.append(avg_train_miou)
        
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
                
                # Calculate metrics
                predictions = torch.argmax(outputs, dim=1)
                batch_miou = calculate_batch_miou(predictions, targets)
                
                val_loss += loss.item() * images.size(0)
                val_miou += batch_miou * images.size(0)
                val_samples += images.size(0)
                
                # Update progress bar
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'mIoU': f'{batch_miou:.4f}'
                })
        
        avg_val_loss = val_loss / val_samples
        avg_val_miou = val_miou / val_samples
        val_losses.append(avg_val_loss)
        val_mious.append(avg_val_miou)
        
        # Store learning rate
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, Train mIoU: {avg_train_miou:.4f} | '
              f'Val Loss: {avg_val_loss:.4f}, Val mIoU: {avg_val_miou:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model based on validation mIoU
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            # Save to parent directory
            model_path = f'best_smnet_model_base{base_dim}.pth'
            torch.save(model.state_dict(), model_path)
            print(f'New best model saved! Val mIoU: {avg_val_miou:.4f}')
        
        print("-" * 80)
    
    # Save comprehensive training history
    save_training_history(train_losses, val_losses, train_mious, val_mious,
                         learning_rates, model_info, base_dim, num_epochs)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_mious, val_mious, base_dim)
    
    # Final evaluation
    print(f"\nTraining completed!")
    print(f"Best validation mIoU: {best_val_miou:.4f}")
    print(f"Final validation mIoU: {avg_val_miou:.4f}")
    
    return model, train_losses, val_losses, train_mious, val_mious

def plot_training_curves(train_losses, val_losses, train_mious, val_mious, base_dim):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title(f'SMNet Training Loss (Base Dim: {base_dim})', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # mIoU curves
    ax2.plot(epochs, train_mious, 'b-', label='Training mIoU', linewidth=2)
    ax2.plot(epochs, val_mious, 'r-', label='Validation mIoU', linewidth=2)
    ax2.set_title(f'SMNet Training mIoU (Base Dim: {base_dim})', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean IoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to plots directory
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/training_curves_base{base_dim}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[FILE] Training curves saved: plots/training_curves_base{base_dim}.png")

def main():
    """Main training function for SMNet."""
    
    parser = argparse.ArgumentParser(description='Train SMNet custom segmentation model')
    parser.add_argument('--base-dim', type=int, default=16,
                       help='Base dimension for model channels. Default: 16')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs. Default: 50')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training. Default: 8')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate. Default: 0.001')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples for quick testing. Default: None (use all)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SMNET CUSTOM SEGMENTATION MODEL TRAINING")
    print("="*80)
    print(f"Model Configuration: Base Dimension {args.base_dim}")
    print(f"Training Configuration: {args.epochs} epochs, batch size {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    if args.max_samples:
        print(f"Sample Limit: {args.max_samples} (for quick testing)")
    print("="*80)
    
    # Train the model
    try:
        model, train_losses, val_losses, train_mious, val_mious = train_smnet(
            base_dim=args.base_dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_samples=args.max_samples
        )
        
        print("\n[OK] Training completed successfully!")
        print(f"[FILE] Model saved: best_smnet_model_base{args.base_dim}.pth")
        print(f"[FILE] Training history: smnet_training_history_base{args.base_dim}.json") 
        print(f"[FILE] Training curves: plots/training_curves_base{args.base_dim}.png")
        print(f"[METRIC] Final validation mIoU: {val_mious[-1]:.4f}")
        print(f"[METRIC] Best validation mIoU: {max(val_mious):.4f}")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise

if __name__ == '__main__':
    main()