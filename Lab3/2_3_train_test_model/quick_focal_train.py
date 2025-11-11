"""
ELEC475 Lab 3 - Quick Focal Loss Training Script
Addresses class imbalance issue that's causing poor segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import sys
import os
from tqdm import tqdm
import argparse

# Add parent directories to path
sys.path.append('../2_2_Custom_SMNet')
sys.path.append('../2_1_Evaluate_Model')
from model import SMNet
from step1_local_voc import LocalVOCDataset, squeeze_and_long

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance - focuses on hard examples."""
    def __init__(self, alpha=1, gamma=2, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        
        # Calculate p_t
        pt = torch.exp(-ce_loss)
        
        # Focal loss formula: FL = -α(1-pt)^γ * log(pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

def quick_focal_training(base_dim=24, epochs=15):
    """Quick training with focal loss to improve segmentation."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Quick Focal Loss Training - Device: {device}")
    
    # Dataset setup
    voc_root = r"../pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val"
    
    # Better augmentation for segmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop((224, 224)),
        transforms.PILToTensor(),
        squeeze_and_long
    ])
    
    train_dataset = LocalVOCDataset(
        voc_root=voc_root, split='train', 
        transform=train_transform, target_transform=target_transform
    )
    
    val_dataset = LocalVOCDataset(
        voc_root=voc_root, split='val',
        transform=transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(), squeeze_and_long
        ])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=0)
    
    print(f"Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model setup
    model = SMNet(num_classes=21, base_dim=base_dim).to(device)
    
    # Try to load existing model for fine-tuning
    existing_model = f'best_smnet_model_base{base_dim}.pth'
    if os.path.exists(existing_model):
        model.load_state_dict(torch.load(existing_model, map_location=device, weights_only=True))
        print(f"Loaded existing model: {existing_model}")
    else:
        print("Training from scratch")
    
    # Focal Loss with strong gamma to focus on hard examples
    criterion = FocalLoss(alpha=1, gamma=3, ignore_index=255)  # Higher gamma = more focus on hard examples
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    print(f"Model: {model.get_model_info()['total_parameters']:,} parameters")
    print(f"Training for {epochs} epochs with Focal Loss (gamma=3)")
    print("-" * 60)
    
    best_val_miou = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            valid_mask = targets != 255
            correct = (predictions[valid_mask] == targets[valid_mask]).sum().item()
            total = valid_mask.sum().item()
            
            train_loss += loss.item()
            train_correct += correct
            train_total += total
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.1f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_ious = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                predictions = torch.argmax(outputs, dim=1)
                valid_mask = targets != 255
                correct = (predictions[valid_mask] == targets[valid_mask]).sum().item()
                total = valid_mask.sum().item()
                
                val_loss += loss.item()
                val_correct += correct
                val_total += total
                
                # Quick IoU calculation
                for pred, target in zip(predictions, targets):
                    valid = target != 255
                    if valid.sum() > 0:
                        pred_valid = pred[valid]
                        target_valid = target[valid]
                        intersection = (pred_valid == target_valid).sum().item()
                        all_ious.append(intersection / valid.sum().item())
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        val_miou = np.mean(all_ious)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%, mIoU: {val_miou:.4f}")
        
        # Save best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), f'best_smnet_model_base{base_dim}_focal.pth')
            print(f"  ✓ NEW BEST MODEL! mIoU: {val_miou:.4f}")
        
        print("-" * 60)
    
    print(f"Training completed! Best mIoU: {best_val_miou:.4f}")
    print(f"Model saved: best_smnet_model_base{base_dim}_focal.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dim', type=int, default=24, help='Base dimension')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs')
    args = parser.parse_args()
    
    quick_focal_training(args.base_dim, args.epochs)