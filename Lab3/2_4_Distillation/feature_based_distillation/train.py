"""
Feature-Based Knowledge Distillation Training
Pure feature-based distillation using cosine similarity loss on intermediate feature maps.
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

class FeatureBasedDistillation:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize models
        self.teacher_model = self._load_teacher()
        self.student_model = SMNet(num_classes=21, base_dim=16).to(device)  # Random initialization
        
        # Feature adaptation layers (to match dimensions)
        self.adaptation_layers = self._create_adaptation_layers()
        
        # Loss function and optimizer - USE FOCAL LOSS for class imbalance
        self.criterion = FocalLoss(alpha=1, gamma=4, ignore_index=255)
        self.optimizer = optim.Adam(
            list(self.student_model.parameters()) + list(self.adaptation_layers.parameters()), 
            lr=0.001
        )
        
        # Tracking
        self.train_losses = []
        self.best_miou = 0.0
        
        print(f"Teacher parameters: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
        print(f"Student parameters: {sum(p.numel() for p in self.student_model.parameters()):,}")
        print(f"Adaptation parameters: {sum(p.numel() for p in self.adaptation_layers.parameters()):,}")
        print("🎯 Student model initialized with random weights (no pre-training)")
        print("📚 Pure feature-based distillation learning from teacher model")
    
    def _load_teacher(self):
        """Load pre-trained FCN-ResNet50 teacher model."""
        teacher = models.segmentation.fcn_resnet50(weights='DEFAULT')
        teacher.to(self.device)
        teacher.eval()
        
        # Freeze teacher parameters
        for param in teacher.parameters():
            param.requires_grad = False
        
        return teacher
    
    def _create_adaptation_layers(self):
        """Create adaptation layers to match teacher-student feature dimensions."""
        # Student SMNet features: base_dim=16
        # layer1: 32 channels (base_dim*2)
        # layer2: 48 channels (base_dim*3)  
        # layer3: 64 channels (base_dim*4)
        # Teacher FCN-ResNet50 intermediate features have different dimensions
        adaptation_layers = nn.ModuleDict({
            'adapt_1': nn.Conv2d(32, 256, 1),    # Student layer1 (32) -> Teacher early (256)
            'adapt_2': nn.Conv2d(48, 512, 1),    # Student layer2 (48) -> Teacher mid (512)
            'adapt_3': nn.Conv2d(64, 1024, 1),   # Student layer3 (64) -> Teacher late (1024)
        }).to(self.device)
        
        return adaptation_layers
    
    def cosine_similarity_loss(self, student_features, teacher_features):
        """
        Calculate cosine similarity loss between feature maps.
        
        Args:
            student_features: Student feature maps [B, C_s, H, W]
            teacher_features: Teacher feature maps [B, C_t, H, W]
        
        Returns:
            Cosine similarity loss (1 - cosine_similarity)
        """
        # Flatten spatial dimensions
        B, C_s, H, W = student_features.shape
        B, C_t, H_t, W_t = teacher_features.shape
        
        # Resize if needed
        if H != H_t or W != W_t:
            teacher_features = F.interpolate(teacher_features, size=(H, W), mode='bilinear', align_corners=False)
        
        # Flatten to [B, C, H*W]
        student_flat = student_features.view(B, C_s, -1)
        teacher_flat = teacher_features.view(B, C_t, -1)
        
        # Normalize features
        student_norm = F.normalize(student_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)
        
        # Calculate cosine similarity for each spatial location
        cosine_sim = torch.sum(student_norm * teacher_norm, dim=1)  # [B, H*W]
        
        # Convert to loss (1 - similarity)
        cosine_loss = 1 - cosine_sim.mean()
        
        return cosine_loss
    
    def extract_teacher_features(self, x):
        """Extract intermediate features from teacher ResNet backbone."""
        features = {}
        
        # Access ResNet backbone
        backbone = self.teacher_model.backbone
        
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        
        x = backbone.layer1(x)
        features['layer1'] = x  # Early features
        
        x = backbone.layer2(x)
        features['layer2'] = x  # Mid features
        
        x = backbone.layer3(x)
        features['layer3'] = x  # Late features
        
        return features
    
    def extract_student_features(self, x):
        """Extract intermediate features from student SMNet."""
        features = {}
        
        # Follow SMNet encoder path (matching the forward pass)
        stem_feat = self.student_model.stem(x)  # H x W, base_dim (16)
        
        enc1 = self.student_model.encoder1(stem_feat)   # H/2, base_dim*2 (32)
        features['layer1'] = enc1
        
        enc2 = self.student_model.encoder2(enc1)        # H/4, base_dim*3 (48)
        features['layer2'] = enc2
        
        enc3 = self.student_model.encoder3(enc2)        # H/8, base_dim*4 (64)
        features['layer3'] = enc3
        
        return features
    
    def knowledge_distillation_loss(self, images, hard_targets):
        """
        Pure feature-based knowledge distillation loss using cosine similarity.
        
        Args:
            images: Input images [B, 3, H, W]
            hard_targets: Ground truth targets [B, H, W]
        
        Returns:
            Combined loss (feature matching + hard targets)
        """
        # Extract features
        with torch.no_grad():
            teacher_features = self.extract_teacher_features(images)
        
        student_features = self.extract_student_features(images)
        
        # Student final output for segmentation loss
        student_output = self.student_model(images)
        
        # Feature matching losses
        feature_losses = []
        for layer in ['layer1', 'layer2', 'layer3']:
            # Adapt student features to match teacher dimensions
            adapted_student = self.adaptation_layers[f'adapt_{layer[-1]}'](student_features[layer])
            
            # Cosine similarity loss
            cosine_loss = self.cosine_similarity_loss(adapted_student, teacher_features[layer])
            feature_losses.append(cosine_loss)
        
        # Combined feature loss
        total_feature_loss = sum(feature_losses) / len(feature_losses)
        
        # Hard targets loss
        segmentation_loss = self.criterion(student_output, hard_targets)
        
        # Combine losses - PRIORITIZE SEGMENTATION for actual performance 
        # Fixed weights: 30% feature distillation + 70% segmentation task
        total_loss = 0.3 * total_feature_loss + 0.7 * segmentation_loss
        
        return total_loss, total_feature_loss, segmentation_loss
    
    def calculate_miou(self, model, dataloader):
        """Calculate mean Intersection over Union (mIoU) for the model."""
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
        
        # Calculate IoU for each class
        ious = total_intersection / (total_union + 1e-8)
        miou = ious.mean().item()
        
        model.train()
        return miou
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.student_model.train()
        epoch_loss = 0.0
        epoch_feature_loss = 0.0
        epoch_seg_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Calculate loss
            total_loss, feature_loss, seg_loss = self.knowledge_distillation_loss(images, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Track losses
            epoch_loss += total_loss.item()
            epoch_feature_loss += feature_loss.item()
            epoch_seg_loss += seg_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}: '
                      f'Total={total_loss.item():.4f}, '
                      f'Feature={feature_loss.item():.4f}, '
                      f'Seg={seg_loss.item():.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        avg_feature = epoch_feature_loss / len(train_loader)
        avg_seg = epoch_seg_loss / len(train_loader)
        
        return avg_loss, avg_feature, avg_seg
    
    def train(self, train_loader, val_loader, epochs=150):
        """Main training loop."""
        print(f"\n🚀 Starting Feature-Based Knowledge Distillation Training")
        print(f"🎯 Student model: Random initialization → Learning from teacher")
        print(f"Feature matching method: Cosine Similarity Loss")
        print(f"Loss weights: 30% feature matching + 70% segmentation (corrected weights)")
        print(f"🔄 Extended training: {epochs} epochs (knowledge distillation needs longer)")
        print("=" * 60)
        
        loss_history = {
            'total': [],
            'feature': [],
            'segmentation': []
        }
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # Train
            avg_loss, avg_feature, avg_seg = self.train_epoch(train_loader)
            
            # Validation mIoU
            val_miou = self.calculate_miou(self.student_model, val_loader)
            
            # Track losses
            loss_history['total'].append(avg_loss)
            loss_history['feature'].append(avg_feature)
            loss_history['segmentation'].append(avg_seg)
            
            print(f'Training Loss: {avg_loss:.4f} '
                  f'(Feature: {avg_feature:.4f}, Seg: {avg_seg:.4f})')
            print(f'Validation mIoU: {val_miou:.4f}')
            
            # Save best model
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student_model.state_dict(),
                    'adaptation_layers': self.adaptation_layers.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'miou': val_miou
                }, 'feature_based_model.pth')
                print(f'✅ New best model saved! mIoU: {val_miou:.4f}')
        
        self.save_loss_plots(loss_history)
        return loss_history
    
    def save_loss_plots(self, loss_history):
        """Save training loss plots."""
        plt.figure(figsize=(15, 5))
        
        # Total loss
        plt.subplot(1, 3, 1)
        plt.plot(loss_history['total'], 'b-', linewidth=2)
        plt.title('Total Training Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Feature loss
        plt.subplot(1, 3, 2)
        plt.plot(loss_history['feature'], 'r-', linewidth=2)
        plt.title('Feature Matching Loss (Cosine)', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity Loss')
        plt.grid(True, alpha=0.3)
        
        # Segmentation loss
        plt.subplot(1, 3, 3)
        plt.plot(loss_history['segmentation'], 'g-', linewidth=2)
        plt.title('Hard Target Segmentation Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('Feature-Based Knowledge Distillation Training', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results_images/feature_based_training_losses.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Loss plots saved: results_images/feature_based_training_losses.png")

def main():
    """Main training function."""
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    
    # Create datasets
    train_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='train',
        transform=transform,
        target_transform=target_transform
    )
    
    val_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='val',
        transform=transform,
        target_transform=target_transform
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create trainer and train
    trainer = FeatureBasedDistillation()
    loss_history = trainer.train(train_loader, val_loader, epochs=150)
    
    print(f"\n🎉 Feature-Based Training Complete!")
    print(f"Best validation mIoU: {trainer.best_miou:.4f}")
    print(f"Model saved: feature_based_model.pth")

if __name__ == "__main__":
    main()