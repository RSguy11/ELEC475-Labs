"""
Response-Based Knowledge Distillation Training
Pure response-based distillation using temperature scaling and KL divergence.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path for imports
sys.path.append('../../2_2_Custom_SMNet')
sys.path.append('../../2_1_Evaluate_Model')

from model import SMNet
import torchvision.models as models

class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance - focuses on hard examples."""
    def __init__(self, alpha=1, gamma=4, ignore_index=255, reduction='mean'):
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

class LocalVOCDataset(Dataset):
    """Local PASCAL VOC 2012 dataset."""
    
    def __init__(self, voc_root, split='val', transform=None, target_transform=None, max_samples=None):
        self.voc_root = voc_root
        self.transform = transform
        self.target_transform = target_transform
        
        print(f"Initializing dataset with root: {voc_root}")
        
        # Paths
        self.images_dir = os.path.join(voc_root, 'JPEGImages')
        self.masks_dir = os.path.join(voc_root, 'SegmentationClass')
        split_file = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{split}.txt')
        
        # Check if paths exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # Load image IDs
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # Limit samples if specified
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]
        
        print(f"Loaded {len(self.image_ids)} images from {split} set")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image and mask
        img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
        mask_path = os.path.join(self.masks_dir, f'{img_id}.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask

def squeeze_and_long(x):
    """Transform function to squeeze and convert to long - needed for multiprocessing."""
    return x.squeeze(0).long()

class ResponseBasedDistillation:
    def __init__(self, temperature=3.0, device='cuda'):  # Reduced temperature for sharper learning
        self.temperature = temperature
        self.device = device
        
        # Initialize models
        self.teacher_model = self._load_teacher()
        self.student_model = SMNet(num_classes=21, base_dim=16).to(device)  # Random initialization
        
        # Loss function and optimizer - EXACTLY MATCH WORKING STANDARD TRAINING
        self.criterion = FocalLoss(alpha=1, gamma=4, ignore_index=255)  # Exact match
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=0.001)  # Exact match
        
        # Tracking
        self.train_losses = []
        self.best_miou = 0.0
        
        print(f"Teacher parameters: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
        print(f"Student parameters: {sum(p.numel() for p in self.student_model.parameters()):,}")
        print("🎯 Student model initialized with random weights (no pre-training)")
        print("📚 Pure knowledge distillation learning from teacher model")
    
    def _load_teacher(self):
        """Load pre-trained FCN-ResNet50 teacher model."""
        teacher = models.segmentation.fcn_resnet50(weights='DEFAULT')
        teacher.to(self.device)
        teacher.eval()
        
        # Freeze teacher parameters
        for param in teacher.parameters():
            param.requires_grad = False
        
        return teacher
    
    def knowledge_distillation_loss(self, student_logits, teacher_logits, hard_targets):
        """
        Pure response-based knowledge distillation loss.
        
        Args:
            student_logits: Student model predictions [B, C, H, W]
            teacher_logits: Teacher model predictions [B, C, H, W]
            hard_targets: Ground truth targets [B, H, W]
        
        Returns:
            Combined loss (soft targets + hard targets)
        """
        # Soft targets loss (KL divergence with temperature scaling)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distillation_loss = F.kl_div(
            student_soft, 
            teacher_soft, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard targets loss
        segmentation_loss = self.criterion(student_logits, hard_targets)
        
        # Combine losses - FIXED weights for response-based KD
        # KL divergence is naturally much larger than segmentation loss
        # Use small weight for distillation: 10% distill + 90% segmentation
        total_loss = 0.1 * distillation_loss + 0.9 * segmentation_loss
        
        return total_loss, distillation_loss, segmentation_loss
    
    def calculate_miou(self, model, dataloader):
        """Calculate mean Intersection over Union (mIoU) for the model - FIXED to match standard training."""
        model.eval()
        total_intersection = torch.zeros(21, device=self.device)
        total_union = torch.zeros(21, device=self.device)
        
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                for cls in range(21):
                    pred_cls = (predictions == cls)
                    target_cls = (targets == cls)
                    
                    intersection = (pred_cls & target_cls).sum()
                    union = (pred_cls | target_cls).sum()
                    
                    total_intersection[cls] += intersection
                    total_union[cls] += union
        
        # Calculate IoU for each class - ONLY INCLUDE CLASSES WITH UNION > 0
        valid_ious = []
        for cls in range(21):
            if total_union[cls] > 0:
                iou = (total_intersection[cls] / total_union[cls]).item()
                valid_ious.append(iou)
        
        miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
        
        model.train()
        return miou
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.student_model.train()
        epoch_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_seg_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Teacher predictions (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(images)['out']
            
            # Student predictions
            student_outputs = self.student_model(images)
            
            # Calculate loss
            total_loss, distill_loss, seg_loss = self.knowledge_distillation_loss(
                student_outputs, teacher_outputs, targets
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Track losses
            epoch_loss += total_loss.item()
            epoch_distill_loss += distill_loss.item()
            epoch_seg_loss += seg_loss.item()
            
            if batch_idx % 10 == 0:
                # Show raw KL divergence before temperature scaling for better insight
                raw_kl = distill_loss.item() / (self.temperature ** 2)
                print(f'Batch {batch_idx}/{len(train_loader)}: '
                      f'Total={total_loss.item():.4f}, '
                      f'Distill={distill_loss.item():.4f} (raw_KL={raw_kl:.4f}), '
                      f'Seg={seg_loss.item():.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        avg_distill = epoch_distill_loss / len(train_loader)
        avg_seg = epoch_seg_loss / len(train_loader)
        
        return avg_loss, avg_distill, avg_seg
    
    def train(self, train_loader, val_loader, epochs=75):
        """Main training loop."""
        print(f"\n🚀 Starting Response-Based Knowledge Distillation Training")
        print(f"🎯 Student model: Random initialization → Learning from teacher")
        print(f"🌡️ Temperature: {self.temperature}")
        print(f"🎛️ Loss weights: 10% distillation + 90% segmentation (corrected for KL scale)")
        print(f"📊 Training for {epochs} epochs with incremental plots every 5 epochs")
        print("=" * 70)
        
        loss_history = {
            'epochs': [],
            'total': [],
            'distillation': [],
            'segmentation': [],
            'miou': []
        }
        
        for epoch in range(epochs):
            current_epoch = epoch + 1
            print(f'\nEpoch {current_epoch}/{epochs}')
            
            # Train
            avg_loss, avg_distill, avg_seg = self.train_epoch(train_loader)
            
            # Validation mIoU
            val_miou = self.calculate_miou(self.student_model, val_loader)
            
            # Track losses
            loss_history['epochs'].append(current_epoch)
            loss_history['total'].append(avg_loss)
            loss_history['distillation'].append(avg_distill)
            loss_history['segmentation'].append(avg_seg)
            loss_history['miou'].append(val_miou)
            
            print(f'Training Loss: {avg_loss:.4f} '
                  f'(Distill: {avg_distill:.4f}, Seg: {avg_seg:.4f})')
            print(f'Validation mIoU: {val_miou:.4f}')
            
            # Save best model
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'miou': val_miou
                }, 'response_based_model.pth')
                print(f'✅ New best model saved! mIoU: {val_miou:.4f}')
            
            # Save incremental plots every 5 epochs
            if current_epoch % 5 == 0:
                self.save_incremental_plots(loss_history, current_epoch)
        
        # Final plot
        self.save_incremental_plots(loss_history, epochs)
        self.save_loss_plots(loss_history)
        return loss_history
    
    def save_incremental_plots(self, loss_history, current_epoch):
        """Save training plots with current progress."""
        if len(loss_history['epochs']) == 0:
            return
            
        plt.figure(figsize=(20, 10))
        
        # Loss plots
        plt.subplot(2, 2, 1)
        plt.plot(loss_history['epochs'], loss_history['total'], 'b-', linewidth=2, label='Total Loss')
        plt.title('Total Training Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(loss_history['epochs'], loss_history['distillation'], 'r-', linewidth=2, label='Distillation Loss')
        plt.plot(loss_history['epochs'], loss_history['segmentation'], 'g-', linewidth=2, label='Segmentation Loss')
        plt.title('Component Losses', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # mIoU plot
        plt.subplot(2, 2, 3)
        plt.plot(loss_history['epochs'], loss_history['miou'], 'purple', linewidth=2, label='Validation mIoU')
        plt.title('Validation mIoU Progress', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Combined view
        plt.subplot(2, 2, 4)
        # Normalize for combined plotting
        if len(loss_history['miou']) > 0:
            import numpy as np
            norm_miou = np.array(loss_history['miou']) * 10  # Scale up for visibility
            plt.plot(loss_history['epochs'], norm_miou, 'purple', linewidth=2, label='mIoU x10')
        plt.plot(loss_history['epochs'], loss_history['total'], 'b-', linewidth=2, label='Total Loss')
        plt.title('Training Overview', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Normalized Values')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.suptitle('Response-Based Knowledge Distillation Training', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save with timestamp to avoid overwriting
        plot_name = f'response_training_progress_epoch_{current_epoch}.png'
        plt.savefig(f'results_images/{plot_name}', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Progress plot saved: results_images/{plot_name}")
    
    def save_loss_plots(self, loss_history):
        """Save final training loss plots."""
        plt.figure(figsize=(15, 5))
        
        # Total loss
        plt.subplot(1, 3, 1)
        plt.plot(loss_history['epochs'], loss_history['total'], 'b-', linewidth=2)
        plt.title('Total Training Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Distillation and segmentation losses
        plt.subplot(1, 3, 2)
        plt.plot(loss_history['epochs'], loss_history['distillation'], 'r-', linewidth=2, label='Distillation')
        plt.plot(loss_history['epochs'], loss_history['segmentation'], 'g-', linewidth=2, label='Segmentation')
        plt.title('Component Losses', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # mIoU
        plt.subplot(1, 3, 3)
        plt.plot(loss_history['epochs'], loss_history['miou'], 'purple', linewidth=2)
        plt.title('Validation mIoU', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('Response-Based Knowledge Distillation Training', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results_images/response_based_training_losses.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Final loss plots saved: results_images/response_based_training_losses.png")

def main():
    """Main training function."""
    # Data transforms - MATCH SUCCESSFUL STANDARD TRAINING
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
        squeeze_and_long  # Use named function instead of lambda
    ])
    
    # Dataset path - use the same path as original distillation pipeline
    voc_root = r"../../pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val"
    
    # Check if dataset exists
    if not os.path.exists(voc_root):
        print(f"Dataset not found at: {voc_root}")
        print("Current directory:", os.getcwd())
        return
    
    # Create datasets - USE TRAINING TRANSFORMS
    train_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='train',
        transform=train_transform,
        target_transform=target_transform
    )
    
    val_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='val',
        transform=val_transform,
        target_transform=target_transform
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create trainer and train
    trainer = ResponseBasedDistillation(temperature=3.0)
    loss_history = trainer.train(train_loader, val_loader, epochs=75)
    
    print(f"\n🎉 Response-Based Training Complete!")
    print(f"Best validation mIoU: {trainer.best_miou:.4f}")
    print(f"Model saved: response_based_model.pth")

if __name__ == "__main__":
    main()