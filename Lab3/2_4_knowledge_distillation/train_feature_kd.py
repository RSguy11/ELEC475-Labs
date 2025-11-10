"""
ELEC475 Lab 3 - Pure Feature-based Knowledge Distillation
Train SMNet using only feature matching (no soft targets)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models.segmentation as segmentation
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm

# Add parent directories to path
sys.path.append('../2_2_Custom_SMNet')
sys.path.append('../2_1_Evaluate_Model')
from model import SMNet
from step1_local_voc import LocalVOCDataset, squeeze_and_long

class FeatureDistillationLoss(nn.Module):
    """
    Pure Feature-based Knowledge Distillation Loss combining:
    1. Hard target supervision (ground truth labels)  
    2. Feature-based distillation (intermediate feature matching only)
    """
    
    def __init__(self, alpha=0.5, beta=0.5):
        """
        Args:
            alpha: Weight for hard target loss (ground truth)
            beta: Weight for feature distillation loss
        """
        super(FeatureDistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.hard_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_logits, teacher_features, student_features, targets):
        """
        Calculate pure feature-based distillation loss.
        
        Args:
            student_logits: Student model predictions [B, C, H, W]
            teacher_features: Teacher intermediate features [B, F, H, W]
            student_features: Student intermediate features [B, F, H, W]
            targets: Ground truth labels [B, H, W]
        """
        
        # 1. Hard target loss (standard cross-entropy with ground truth)
        hard_loss = self.hard_loss(student_logits, targets)
        
        # 2. Feature distillation loss
        if teacher_features.shape != student_features.shape:
            # Adapt student features to teacher feature size
            teacher_h, teacher_w = teacher_features.shape[2:]
            student_features = torch.nn.functional.interpolate(
                student_features, size=(teacher_h, teacher_w), 
                mode='bilinear', align_corners=False
            )
            
            # If channel dimensions don't match, use 1x1 conv adaptation
            if teacher_features.shape[1] != student_features.shape[1]:
                teacher_channels = teacher_features.shape[1]
                student_channels = student_features.shape[1]
                
                # Create adaptive layer on-the-fly 
                if not hasattr(self, 'channel_adapter'):
                    self.channel_adapter = nn.Conv2d(
                        student_channels, teacher_channels, 1, bias=False
                    ).to(teacher_features.device)
                    
                student_features = self.channel_adapter(student_features)
        
        # Calculate L2 feature matching loss
        feature_loss = self.mse_loss(student_features, teacher_features)
        
        # Combine losses (no soft target component)
        total_loss = self.alpha * hard_loss + self.beta * feature_loss
        
        return total_loss, hard_loss, feature_loss

def load_teacher_model(device):
    """Load pre-trained FCN-ResNet50 teacher model."""
    teacher = segmentation.fcn_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
    teacher.to(device)
    teacher.eval()
    
    print("Teacher model (FCN-ResNet50) loaded with COCO+VOC weights")
    total_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher parameters: {total_params:,}")
    
    return teacher

def get_feature_hook(features_dict, name):
    """Hook function to capture intermediate features."""
    def hook(module, input, output):
        features_dict[name] = output
    return hook

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
    
    return np.array(ious)

def train_feature_distillation(student_model, teacher_model, train_loader, val_loader, 
                              device, epochs, base_dim):
    """Train student model with pure feature-based knowledge distillation."""
    
    # Initialize pure feature distillation loss
    fd_criterion = FeatureDistillationLoss(alpha=0.5, beta=0.5)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5, verbose=True)
    
    # Training tracking
    best_val_miou = 0.0
    
    print(f"\\nStarting Pure Feature Distillation training for {epochs} epochs...")
    print(f"Loss weights: α=0.5 (hard), β=0.5 (feature)")
    
    # Set up feature hooks for teacher model
    teacher_features = {}
    teacher_hook = teacher_model.backbone.layer4.register_forward_hook(
        get_feature_hook(teacher_features, 'teacher_feat')
    )
    
    for epoch in range(epochs):
        # Training phase
        student_model.train()
        teacher_model.eval()  # Teacher always in eval mode
        
        epoch_loss = 0.0
        epoch_ious = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Feature KD Train]")
        
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                _ = teacher_model(images)['out']  # Just for feature extraction
                teacher_feat = teacher_features['teacher_feat']
            
            # Student forward pass with feature extraction
            student_features = {}
            student_hook = student_model.encoder4.register_forward_hook(
                get_feature_hook(student_features, 'student_feat')
            )
            
            student_output = student_model(images)
            student_feat = student_features['student_feat']
            
            # Remove hook
            student_hook.remove()
            
            # Calculate pure feature distillation loss (no soft targets)
            loss, hard_loss, feature_loss = fd_criterion(
                student_output, teacher_feat, student_feat, targets
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            
            # Calculate mIoU
            with torch.no_grad():
                predictions = torch.argmax(student_output, dim=1)
                batch_ious = []
                for pred, target in zip(predictions, targets):
                    ious = calculate_miou(pred, target)
                    if len(ious) > 0:
                        batch_ious.append(np.mean(ious))
                
                if batch_ious:
                    epoch_ious.extend(batch_ious)
                    avg_miou = np.mean(epoch_ious)
                else:
                    avg_miou = 0.0
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Hard': f'{hard_loss.item():.4f}',
                'Feat': f'{feature_loss.item():.4f}',
                'mIoU': f'{avg_miou:.4f}'
            })
        
        # Calculate epoch averages
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_miou = np.mean(epoch_ious) if epoch_ious else 0.0
        
        # Simple validation (just check improvement)
        student_model.eval()
        val_ious = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = student_model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                for pred, target in zip(predictions, targets):
                    ious = calculate_miou(pred, target)
                    if len(ious) > 0:
                        val_ious.append(np.mean(ious))
        
        avg_val_miou = np.mean(val_ious) if val_ious else 0.0
        
        # Learning rate step
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, "
              f"Train mIoU: {avg_train_miou:.4f} | Val mIoU: {avg_val_miou:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model based on validation mIoU
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            model_path = f'smnet_feature_kd_model_base{base_dim}.pth'
            torch.save(student_model.state_dict(), model_path)
            print(f'New best Feature KD model saved! Val mIoU: {avg_val_miou:.4f}')
        
        print("-" * 60)
    
    # Remove teacher hook
    teacher_hook.remove()
    
    print(f"\\nPure Feature Distillation training completed!")
    print(f"Best validation mIoU: {best_val_miou:.4f}")
    
    return best_val_miou

def main():
    """Main function for pure feature-based knowledge distillation."""
    
    parser = argparse.ArgumentParser(description='Train SMNet with Pure Feature Distillation')
    parser.add_argument('--base-dim', type=int, default=16,
                       help='Base dimension of student model. Default: 16')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs. Default: 20')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training. Default: 8')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples for quick testing. Default: None (use all)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Pure Feature Distillation on device: {device}")
    
    print("="*80)
    print("[ELEC475 LAB 3] PURE FEATURE-BASED KNOWLEDGE DISTILLATION")
    print("="*80)
    print(f"Student Model: SMNet (Base dimension {args.base_dim})")
    print(f"Teacher Model: FCN-ResNet50 (Pre-trained)")
    print(f"Training: {args.epochs} epochs, batch size {args.batch_size}")
    print(f"Method: Pure feature matching (no soft targets)")
    if args.max_samples:
        print(f"Sample Limit: {args.max_samples} (for quick testing)")
    print("="*80)
    
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
    
    # Load datasets
    print("Loading datasets...")
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
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
    
    # Create datasets
    train_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='train',
        transform=train_transform,
        target_transform=target_transform,
        max_samples=args.max_samples
    )
    
    val_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='val',
        transform=val_transform,
        target_transform=target_transform,
        max_samples=min(100, args.max_samples) if args.max_samples else 100  # Quick val
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Load models
    print("\\nLoading models...")
    
    # Load teacher model (FCN-ResNet50)
    teacher_model = load_teacher_model(device)
    
    # Initialize student model (SMNet)
    student_model = SMNet(num_classes=21, base_dim=args.base_dim).to(device)
    student_info = student_model.get_model_info()
    print(f"Student model initialized: {student_info['model_name']}")
    print(f"Student parameters: {student_info['total_parameters']:,}")
    
    # Check if pre-trained student model exists
    pretrained_path = f'../2_3_train_test_model/best_smnet_model_base{args.base_dim}.pth'
    if os.path.exists(pretrained_path):
        student_model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"Loaded pre-trained student model from: {pretrained_path}")
    else:
        print("No pre-trained student model found. Starting from random initialization.")
    
    # Train with pure feature distillation
    best_miou = train_feature_distillation(
        student_model, teacher_model, train_loader, val_loader,
        device, args.epochs, args.base_dim
    )
    
    print(f"\\n[OK] Pure feature distillation completed!")
    print(f"[FILE] Best Feature KD model saved as: smnet_feature_kd_model_base{args.base_dim}.pth")
    print(f"[METRIC] Best validation mIoU: {best_miou:.4f}")

if __name__ == '__main__':
    main()