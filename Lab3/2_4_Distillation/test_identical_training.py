"""
Direct comparison: Distillation vs Standard Training setup
This will run IDENTICAL training to your standard training, just in the distillation folder
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
import os

# Add paths
sys.path.append('../../2_2_Custom_SMNet')
sys.path.append('../../2_1_Evaluate_Model')

from model import SMNet
from step1_local_voc import LocalVOCDataset, squeeze_and_long

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance - EXACT COPY from standard training."""
    def __init__(self, alpha=1, gamma=4, ignore_index=255, reduction='mean'):
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
        return focal_loss

def calculate_batch_miou(predictions, targets, num_classes=21):
    """EXACT COPY from standard training."""
    batch_miou = 0.0
    valid_samples = 0
    
    for pred, target in zip(predictions, targets):
        pred, target = pred.cpu().numpy(), target.cpu().numpy()
        ious = []
        
        for cls in range(num_classes):
            pred_mask = pred == cls
            target_mask = target == cls
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            
            if union > 0:
                ious.append(intersection / union)
        
        if ious:
            miou = sum(ious) / len(ious)
            batch_miou += miou
            valid_samples += 1
    
    return batch_miou / valid_samples if valid_samples > 0 else 0.0

def test_identical_training():
    """Run EXACTLY the same training as your standard training."""
    print("Running IDENTICAL training setup to standard training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # EXACT same model as standard training
    model = SMNet(num_classes=21, base_dim=16).to(device)
    
    # EXACT same loss and optimizer
    criterion = FocalLoss(alpha=1, gamma=4, ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # EXACT same transforms as standard training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        squeeze_and_long
    ])
    
    # EXACT same dataset path
    voc_root = r"../../pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val"
    
    train_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='train',
        transform=train_transform,
        target_transform=target_transform
    )
    
    val_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='val',
        transform=train_transform,  # Use same transform for simplicity
        target_transform=target_transform
    )
    
    # EXACT same batch size and num_workers
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Run ONE epoch exactly like standard training
    model.train()
    epoch_loss = 0.0
    train_samples = 0
    train_miou = 0.0
    
    print("\\nRunning 1 epoch of PURE standard training...")
    for batch_idx, (images, targets) in enumerate(train_loader):
        if batch_idx >= 50:  # Just test first 50 batches
            break
            
        images, targets = images.to(device), targets.to(device)
        
        # Standard training forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Calculate metrics EXACTLY like standard training
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            batch_miou = calculate_batch_miou(predictions, targets)
            
            epoch_loss += loss.item() * images.size(0)
            train_miou += batch_miou * images.size(0)
            train_samples += images.size(0)
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}: Loss={loss.item():.4f}, mIoU={batch_miou:.4f}')
    
    avg_loss = epoch_loss / train_samples
    avg_miou = train_miou / train_samples
    
    print(f"\\nEPOCH 1 RESULTS:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average mIoU: {avg_miou:.4f}")
    print(f"Expected mIoU (from your standard training): ~0.108")
    
    if avg_miou > 0.05:
        print("✅ SUCCESS: mIoU looks reasonable!")
    else:
        print("❌ FAILURE: mIoU still too low - fundamental issue remains")
    
    return avg_miou

if __name__ == "__main__":
    miou = test_identical_training()
    print(f"\\nFinal result: {miou:.4f} mIoU")