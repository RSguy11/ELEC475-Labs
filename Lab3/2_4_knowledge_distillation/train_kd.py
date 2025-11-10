"""
ELEC475 Lab 3 - Step 2.4: Knowledge Distillation Training
Implement response-based and feature-based knowledge distillation
with FCN-ResNet50 (teacher) and SMNet (student)
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
import matplotlib.pyplot as plt
import time

# Add parent directories to path
sys.path.append('../2_2_Custom_SMNet')
sys.path.append('../2_1_Evaluate_Model')
from model import SMNet
from step1_local_voc import LocalVOCDataset, squeeze_and_long

class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    1. Response-based distillation (soft targets from teacher logits)
    2. Feature-based distillation (intermediate feature matching)
    3. Hard target supervision (ground truth labels)
    """
    
    def __init__(self, alpha=0.3, beta=0.4, gamma=0.3, temperature=4.0):
        """
        Args:
            alpha: Weight for hard target loss (ground truth)
            beta: Weight for soft target loss (teacher predictions)
            gamma: Weight for feature distillation loss
            temperature: Temperature for softening distributions
        """
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.temperature = temperature
        self.hard_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_logits, teacher_logits, teacher_features, student_features, targets):
        """
        Calculate combined knowledge distillation loss.
        
        Args:
            student_logits: Student model predictions [B, C, H, W]
            teacher_logits: Teacher model predictions [B, C, H, W] 
            teacher_features: Teacher intermediate features [B, F, H, W]
            student_features: Student intermediate features [B, F, H, W]
            targets: Ground truth labels [B, H, W]
        """
        
        # 1. Hard target loss (standard cross-entropy with ground truth)
        hard_loss = self.hard_loss(student_logits, targets)
        
        # 2. Soft target loss (distillation from teacher logits)
        # Apply temperature scaling
        teacher_soft = torch.softmax(teacher_logits / self.temperature, dim=1)
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 3. Feature distillation loss
        # Ensure feature dimensions match via adaptive pooling if needed
        if teacher_features.shape != student_features.shape:
            # Adapt student features to teacher feature size
            teacher_h, teacher_w = teacher_features.shape[2:]
            student_features = torch.nn.functional.interpolate(
                student_features, size=(teacher_h, teacher_w), 
                mode='bilinear', align_corners=False
            )
            
            # If channel dimensions don't match, use 1x1 conv adaptation
            if teacher_features.shape[1] != student_features.shape[1]:
                # Simple channel adaptation - project to teacher dimension
                teacher_channels = teacher_features.shape[1]
                student_channels = student_features.shape[1]
                
                # Create adaptive layer on-the-fly (for simplicity in this demo)
                # In practice, this should be a learned parameter
                if not hasattr(self, 'channel_adapter'):
                    self.channel_adapter = nn.Conv2d(
                        student_channels, teacher_channels, 1, bias=False
                    ).to(teacher_features.device)
                    
                student_features = self.channel_adapter(student_features)
        
        # Calculate L2 feature matching loss
        feature_loss = self.mse_loss(student_features, teacher_features)
        
        # Combine all losses
        total_loss = (self.alpha * hard_loss + 
                      self.beta * soft_loss + 
                      self.gamma * feature_loss)
        
        return total_loss, hard_loss, soft_loss, feature_loss

def load_teacher_model(device):
    """Load pre-trained FCN-ResNet50 teacher model."""
    # Load teacher model
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

def validate_model(student_model, teacher_model, val_loader, device, kd_criterion):
    """Validate the student model with knowledge distillation loss."""
    student_model.eval()
    teacher_model.eval()
    
    total_loss = 0.0
    total_hard_loss = 0.0
    total_soft_loss = 0.0
    total_feature_loss = 0.0
    all_ious = []
    
    # Set up feature hooks for teacher model
    teacher_features = {}
    # Get features from teacher backbone (ResNet-50)
    teacher_hook = teacher_model.backbone.layer4.register_forward_hook(
        get_feature_hook(teacher_features, 'teacher_feat')
    )
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            
            # Teacher forward pass
            teacher_output = teacher_model(images)['out']
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
            
            # Calculate losses
            loss, hard_loss, soft_loss, feature_loss = kd_criterion(
                student_output, teacher_output, teacher_feat, student_feat, targets
            )
            
            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            total_feature_loss += feature_loss.item()
            
            # Calculate mIoU
            predictions = torch.argmax(student_output, dim=1)
            batch_ious = []
            for pred, target in zip(predictions, targets):
                ious = calculate_miou(pred, target)
                if len(ious) > 0:
                    batch_ious.append(np.mean(ious))
            
            if batch_ious:
                all_ious.extend(batch_ious)
    
    # Remove teacher hook
    teacher_hook.remove()
    
    avg_loss = total_loss / len(val_loader)
    avg_hard_loss = total_hard_loss / len(val_loader)
    avg_soft_loss = total_soft_loss / len(val_loader)
    avg_feature_loss = total_feature_loss / len(val_loader)
    avg_miou = np.mean(all_ious) if all_ious else 0.0
    
    return avg_loss, avg_hard_loss, avg_soft_loss, avg_feature_loss, avg_miou

def train_with_knowledge_distillation(student_model, teacher_model, train_loader, val_loader, 
                                     device, epochs, base_dim, distillation_config):
    """Train student model with knowledge distillation."""
    
    # Initialize knowledge distillation loss
    kd_criterion = KnowledgeDistillationLoss(
        alpha=distillation_config['alpha'],
        beta=distillation_config['beta'], 
        gamma=distillation_config['gamma'],
        temperature=distillation_config['temperature']
    )
    
    # Optimizer and scheduler
    optimizer = optim.Adam(student_model.parameters(), lr=distillation_config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5, verbose=True)
    
    # Training tracking
    train_losses = []
    train_hard_losses = []
    train_soft_losses = []
    train_feature_losses = []
    train_mious = []
    val_losses = []
    val_mious = []
    
    best_val_miou = 0.0
    
    print(f"\\nStarting Knowledge Distillation training for {epochs} epochs...")
    print(f"KD Configuration: α={distillation_config['alpha']}, β={distillation_config['beta']}, γ={distillation_config['gamma']}, T={distillation_config['temperature']}")
    print(f"Learning Rate: {distillation_config['learning_rate']}")
    
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
        epoch_hard_loss = 0.0
        epoch_soft_loss = 0.0
        epoch_feature_loss = 0.0
        epoch_ious = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_output = teacher_model(images)['out']
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
            
            # Calculate knowledge distillation loss
            loss, hard_loss, soft_loss, feature_loss = kd_criterion(
                student_output, teacher_output, teacher_feat, student_feat, targets
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_loss += loss.item()
            epoch_hard_loss += hard_loss.item()
            epoch_soft_loss += soft_loss.item() 
            epoch_feature_loss += feature_loss.item()
            
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
                'Soft': f'{soft_loss.item():.4f}',
                'Feat': f'{feature_loss.item():.4f}',
                'mIoU': f'{avg_miou:.4f}'
            })
        
        # Calculate epoch averages
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_hard = epoch_hard_loss / len(train_loader)
        avg_train_soft = epoch_soft_loss / len(train_loader)
        avg_train_feat = epoch_feature_loss / len(train_loader)
        avg_train_miou = np.mean(epoch_ious) if epoch_ious else 0.0
        
        train_losses.append(avg_train_loss)
        train_hard_losses.append(avg_train_hard)
        train_soft_losses.append(avg_train_soft)
        train_feature_losses.append(avg_train_feat)
        train_mious.append(avg_train_miou)
        
        # Validation phase
        val_loss, val_hard, val_soft, val_feat, avg_val_miou = validate_model(
            student_model, teacher_model, val_loader, device, kd_criterion
        )
        
        val_losses.append(val_loss)
        val_mious.append(avg_val_miou)
        
        # Learning rate step
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} (H:{avg_train_hard:.3f}, S:{avg_train_soft:.3f}, F:{avg_train_feat:.3f}), "
              f"Train mIoU: {avg_train_miou:.4f} | Val Loss: {val_loss:.4f}, Val mIoU: {avg_val_miou:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model based on validation mIoU
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            model_path = f'smnet_kd_model_base{base_dim}.pth'
            torch.save(student_model.state_dict(), model_path)
            print(f'New best KD model saved! Val mIoU: {avg_val_miou:.4f}')
        
        print("-" * 80)
    
    # Remove teacher hook
    teacher_hook.remove()
    
    # Plot training curves
    plot_kd_training_curves(train_losses, train_hard_losses, train_soft_losses, 
                           train_feature_losses, train_mious, val_losses, val_mious, base_dim)
    
    print(f"\\nKnowledge Distillation training completed!")
    print(f"Best validation mIoU: {best_val_miou:.4f}")
    
    return best_val_miou

def plot_kd_training_curves(train_losses, train_hard_losses, train_soft_losses, 
                           train_feature_losses, train_mious, val_losses, val_mious, base_dim):
    """Plot comprehensive knowledge distillation training curves."""
    
    os.makedirs('kd_results', exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss components
    ax1.plot(epochs, train_losses, 'b-', label='Total Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Total Val Loss', linewidth=2)
    ax1.set_title('Knowledge Distillation - Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Loss breakdown
    ax2.plot(epochs, train_hard_losses, 'g-', label='Hard Loss (α)', linewidth=2)
    ax2.plot(epochs, train_soft_losses, 'b-', label='Soft Loss (β)', linewidth=2) 
    ax2.plot(epochs, train_feature_losses, 'orange', label='Feature Loss (γ)', linewidth=2)
    ax2.set_title('Knowledge Distillation - Loss Components')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # mIoU progress
    ax3.plot(epochs, train_mious, 'b-', label='Train mIoU', linewidth=2)
    ax3.plot(epochs, val_mious, 'r-', label='Val mIoU', linewidth=2)
    ax3.set_title('Knowledge Distillation - mIoU Progress')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mIoU')
    ax3.legend()
    ax3.grid(True)
    
    # Combined view
    ax4_twin = ax4.twinx()
    ax4.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax4_twin.plot(epochs, val_mious, 'r-', label='Val mIoU', linewidth=2)
    ax4.set_title('Knowledge Distillation - Loss vs mIoU')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='b')
    ax4_twin.set_ylabel('mIoU', color='r')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f'kd_results/smnet_kd_training_curves_base{base_dim}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[FILE] KD training plots saved: {plot_path}")

def main():
    """Main function for knowledge distillation training."""
    
    parser = argparse.ArgumentParser(description='Train SMNet with Knowledge Distillation')
    parser.add_argument('--base-dim', type=int, default=16,
                       help='Base dimension of student model. Default: 16')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs. Default: 30')
    parser.add_argument('--batch-size', type=int, default=6,
                       help='Batch size for training. Default: 6')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate. Default: 0.001')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Weight for hard target loss. Default: 0.3')
    parser.add_argument('--beta', type=float, default=0.4, 
                       help='Weight for soft target loss. Default: 0.4')
    parser.add_argument('--gamma', type=float, default=0.3,
                       help='Weight for feature distillation loss. Default: 0.3')
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='Temperature for knowledge distillation. Default: 4.0')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples for quick testing. Default: None (use all)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Knowledge Distillation on device: {device}")
    
    # Validate loss weights
    total_weight = args.alpha + args.beta + args.gamma
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: Loss weights sum to {total_weight:.3f}, not 1.0")
        print(f"Normalizing weights: α={args.alpha/total_weight:.3f}, β={args.beta/total_weight:.3f}, γ={args.gamma/total_weight:.3f}")
        args.alpha /= total_weight
        args.beta /= total_weight  
        args.gamma /= total_weight
    
    distillation_config = {
        'alpha': args.alpha,
        'beta': args.beta,
        'gamma': args.gamma,
        'temperature': args.temperature,
        'learning_rate': args.lr
    }
    
    print("="*80)
    print("[ELEC475 LAB 3] KNOWLEDGE DISTILLATION TRAINING")
    print("="*80)
    print(f"Student Model: SMNet (Base dimension {args.base_dim})")
    print(f"Teacher Model: FCN-ResNet50 (Pre-trained)")
    print(f"Training: {args.epochs} epochs, batch size {args.batch_size}")
    print(f"KD Config: α={args.alpha:.3f}, β={args.beta:.3f}, γ={args.gamma:.3f}, T={args.temperature}")
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
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Target transforms (same for both)
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
        max_samples=args.max_samples
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
    
    # Train with knowledge distillation
    best_miou = train_with_knowledge_distillation(
        student_model, teacher_model, train_loader, val_loader,
        device, args.epochs, args.base_dim, distillation_config
    )
    
    print(f"\\n[OK] Knowledge distillation completed!")
    print(f"[FILE] Best KD model saved as: smnet_kd_model_base{args.base_dim}.pth")
    print(f"[METRIC] Best validation mIoU: {best_miou:.4f}")

if __name__ == '__main__':
    main()