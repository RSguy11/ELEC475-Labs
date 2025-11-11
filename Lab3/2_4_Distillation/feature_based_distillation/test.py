"""
Feature-Based Knowledge Distillation Testing and Evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.append('../../2_2_Custom_SMNet')
sys.path.append('../../2_1_Evaluate_Model')

from model import SMNet
import torchvision.models as models

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

class FeatureBasedTester:
    def __init__(self, model_path='feature_based_model.pth', device='cuda'):
        self.device = device
        
        # Load models
        self.teacher_model = self._load_teacher()
        self.student_model, self.adaptation_layers = self._load_student(model_path)
        
        # VOC class names for visualization
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor'
        ]
        
        # VOC colormap for visualization
        self.colors = self._create_colormap()
    
    def _load_teacher(self):
        """Load pre-trained teacher model."""
        teacher = models.segmentation.fcn_resnet50(weights='DEFAULT')
        teacher.to(self.device)
        teacher.eval()
        return teacher
    
    def _load_student(self, model_path):
        """Load trained student model and adaptation layers."""
        student = SMNet(num_classes=21, base_dim=16).to(self.device)
        
        # Adaptation layers - CORRECTED dimensions to match SMNet
        # SMNet base_dim=16: layer1=32, layer2=48, layer3=64
        adaptation_layers = nn.ModuleDict({
            'adapt_1': nn.Conv2d(32, 256, 1),    # Student layer1 (32) -> Teacher early (256)
            'adapt_2': nn.Conv2d(48, 512, 1),    # Student layer2 (48) -> Teacher mid (512)
            'adapt_3': nn.Conv2d(64, 1024, 1),   # Student layer3 (64) -> Teacher late (1024)
        }).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            student.load_state_dict(checkpoint['model_state_dict'])
            adaptation_layers.load_state_dict(checkpoint['adaptation_layers'])
            print(f"✅ Loaded student model from {model_path}")
            print(f"Training mIoU: {checkpoint.get('miou', 'N/A'):.4f}")
        else:
            print(f"⚠️ Model file {model_path} not found. Using random initialization.")
            print("🎯 Train the model first using: python train.py")
        
        student.eval()
        adaptation_layers.eval()
        return student, adaptation_layers
    
    def _create_colormap(self):
        """Create VOC colormap for visualization."""
        colors = np.zeros((21, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]      # background
        colors[1] = [128, 0, 0]    # aeroplane
        colors[2] = [0, 128, 0]    # bicycle
        colors[3] = [128, 128, 0]  # bird
        colors[4] = [0, 0, 128]    # boat
        colors[5] = [128, 0, 128]  # bottle
        colors[6] = [0, 128, 128]  # bus
        colors[7] = [128, 128, 128] # car
        colors[8] = [64, 0, 0]     # cat
        colors[9] = [192, 0, 0]    # chair
        colors[10] = [64, 128, 0]  # cow
        colors[11] = [192, 128, 0] # diningtable
        colors[12] = [64, 0, 128]  # dog
        colors[13] = [192, 0, 128] # horse
        colors[14] = [64, 128, 128] # motorbike
        colors[15] = [192, 128, 128] # person
        colors[16] = [0, 64, 0]    # pottedplant
        colors[17] = [128, 64, 0]  # sheep
        colors[18] = [0, 192, 0]   # sofa
        colors[19] = [128, 192, 0] # train
        colors[20] = [0, 64, 128]  # tvmonitor
        return colors
    
    def colorize_segmentation(self, segmentation_mask):
        """Convert segmentation mask to colored image."""
        h, w = segmentation_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(21):
            mask = (segmentation_mask == class_id)
            colored[mask] = self.colors[class_id]
        
        return colored
    
    def calculate_miou(self, model, dataloader, model_name="Model"):
        """Calculate detailed mIoU metrics."""
        model.eval()
        total_intersection = torch.zeros(21, device=self.device)
        total_union = torch.zeros(21, device=self.device)
        
        print(f"\n📊 Calculating mIoU for {model_name}...")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                if 'teacher' in model_name.lower():
                    outputs = model(images)['out']
                else:
                    outputs = model(images)
                
                predictions = torch.argmax(outputs, dim=1)
                
                for cls in range(21):
                    pred_cls = (predictions == cls)
                    target_cls = (targets == cls)
                    
                    intersection = (pred_cls & target_cls).sum()
                    union = (pred_cls | target_cls).sum()
                    
                    total_intersection[cls] += intersection
                    total_union[cls] += union
                
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx}/{len(dataloader)} batches")
        
        # Calculate IoU for each class
        ious = total_intersection / (total_union + 1e-8)
        miou = ious.mean().item()
        
        # Print per-class results
        print(f"\n{model_name} Results:")
        print("-" * 40)
        for i, (class_name, iou) in enumerate(zip(self.class_names, ious)):
            print(f"{class_name:12}: {iou:.3f}")
        print("-" * 40)
        print(f"Mean IoU: {miou:.4f}")
        
        return miou, ious
    
    def analyze_feature_similarity(self, dataloader):
        """Analyze feature similarity between teacher and student."""
        self.teacher_model.eval()
        self.student_model.eval()
        
        cosine_similarities = {'layer1': [], 'layer2': [], 'layer3': []}
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                if batch_idx >= 20:  # Analyze first 20 batches
                    break
                
                images = images.to(self.device)
                
                # Extract teacher features
                teacher_features = self._extract_teacher_features(images)
                student_features = self._extract_student_features(images)
                
                # Calculate cosine similarity for each layer
                for layer in ['layer1', 'layer2', 'layer3']:
                    # Adapt student features
                    adapted_student = self.adaptation_layers[f'adapt_{layer[-1]}'](student_features[layer])
                    
                    # Calculate similarity
                    similarity = self._cosine_similarity(adapted_student, teacher_features[layer])
                    cosine_similarities[layer].append(similarity.item())
        
        # Average similarities
        avg_similarities = {
            layer: np.mean(sims) for layer, sims in cosine_similarities.items()
        }
        
        return avg_similarities
    
    def _extract_teacher_features(self, x):
        """Extract teacher features."""
        features = {}
        backbone = self.teacher_model.backbone
        
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        
        x = backbone.layer1(x)
        features['layer1'] = x
        
        x = backbone.layer2(x)
        features['layer2'] = x
        
        x = backbone.layer3(x)
        features['layer3'] = x
        
        return features
    
    def _extract_student_features(self, x):
        """Extract student features matching SMNet structure."""
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
    
    def _cosine_similarity(self, student_features, teacher_features):
        """Calculate cosine similarity between features."""
        B, C_s, H, W = student_features.shape
        B, C_t, H_t, W_t = teacher_features.shape
        
        # Resize if needed
        if H != H_t or W != W_t:
            teacher_features = F.interpolate(teacher_features, size=(H, W), mode='bilinear', align_corners=False)
        
        # Flatten and normalize
        student_flat = student_features.view(B, C_s, -1)
        teacher_flat = teacher_features.view(B, C_t, -1)
        
        student_norm = F.normalize(student_flat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)
        
        # Calculate similarity
        cosine_sim = torch.sum(student_norm * teacher_norm, dim=1)
        return cosine_sim.mean()
    
    def create_prediction_comparison(self, dataloader, num_samples=6):
        """Create visual comparison of predictions."""
        self.teacher_model.eval()
        self.student_model.eval()
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 3))
        
        with torch.no_grad():
            for idx, (images, targets) in enumerate(dataloader):
                if idx >= num_samples:
                    break
                
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Get predictions
                teacher_output = self.teacher_model(images)['out']
                student_output = self.student_model(images)
                
                teacher_pred = torch.argmax(teacher_output, dim=1)
                student_pred = torch.argmax(student_output, dim=1)
                
                # Convert to numpy
                image = images[0].cpu()
                target = targets[0].cpu().numpy()
                teacher_mask = teacher_pred[0].cpu().numpy()
                student_mask = student_pred[0].cpu().numpy()
                
                # Denormalize image
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                image = image * std.view(3, 1, 1) + mean.view(3, 1, 1)
                image = torch.clamp(image, 0, 1)
                image = image.permute(1, 2, 0).numpy()
                
                # Original image
                axes[idx, 0].imshow(image)
                axes[idx, 0].set_title('Original Image')
                axes[idx, 0].axis('off')
                
                # Ground truth
                axes[idx, 1].imshow(self.colorize_segmentation(target))
                axes[idx, 1].set_title('Ground Truth')
                axes[idx, 1].axis('off')
                
                # Teacher prediction
                axes[idx, 2].imshow(self.colorize_segmentation(teacher_mask))
                axes[idx, 2].set_title('Teacher (FCN-ResNet50)')
                axes[idx, 2].axis('off')
                
                # Student prediction
                axes[idx, 3].imshow(self.colorize_segmentation(student_mask))
                axes[idx, 3].set_title('Student (Feature-Based)')
                axes[idx, 3].axis('off')
        
        plt.suptitle('Feature-Based Knowledge Distillation: Prediction Comparison', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results_images/feature_based_predictions.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print("\n📸 Prediction comparison saved: results_images/feature_based_predictions.png")
    
    def create_performance_summary(self, teacher_miou, student_miou, feature_similarities):
        """Create performance summary visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # mIoU comparison
        models = ['Teacher\n(FCN-ResNet50)', 'Student\n(Feature-Based)']
        mious = [teacher_miou, student_miou]
        colors = ['blue', 'purple']
        
        bars = ax1.bar(models, mious, color=colors, alpha=0.7)
        ax1.set_title('Model Performance Comparison', fontweight='bold')
        ax1.set_ylabel('mIoU')
        ax1.set_ylim(0, max(mious) * 1.1)
        
        for bar, miou in zip(bars, mious):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{miou:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Knowledge transfer analysis
        knowledge_transfer = student_miou / teacher_miou * 100
        remaining_gap = 100 - knowledge_transfer
        
        labels = ['Knowledge\nTransferred', 'Performance\nGap']
        sizes = [knowledge_transfer, remaining_gap]
        colors2 = ['green', 'orange']
        
        ax2.pie(sizes, labels=labels, colors=colors2, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Knowledge Transfer Effectiveness', fontweight='bold')
        
        # Feature similarity analysis
        layers = list(feature_similarities.keys())
        similarities = list(feature_similarities.values())
        colors3 = ['red', 'green', 'blue']
        
        bars3 = ax3.bar(layers, similarities, color=colors3, alpha=0.7)
        ax3.set_title('Feature Similarity (Cosine)', fontweight='bold')
        ax3.set_ylabel('Cosine Similarity')
        ax3.set_ylim(0, 1)
        
        for bar, sim in zip(bars3, similarities):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{sim:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Method breakdown
        method_weights = [0.7, 0.3]
        method_labels = ['Feature\nMatching\n(70%)', 'Hard Target\nTraining\n(30%)']
        colors4 = ['purple', 'red']
        
        ax4.pie(method_weights, labels=method_labels, colors=colors4, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Training Method Composition', fontweight='bold')
        
        plt.suptitle('Feature-Based Knowledge Distillation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results_images/feature_based_summary.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print("\n📊 Performance summary saved: results_images/feature_based_summary.png")
        
        return knowledge_transfer

def main():
    """Main testing function."""
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
    
    # Create test dataset
    test_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='val',
        transform=transform,
        target_transform=target_transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create tester
    tester = FeatureBasedTester()
    
    # Calculate mIoU for both models
    teacher_miou, teacher_ious = tester.calculate_miou(tester.teacher_model, test_loader, "Teacher")
    student_miou, student_ious = tester.calculate_miou(tester.student_model, test_loader, "Student (Feature-Based)")
    
    # Analyze feature similarities
    feature_similarities = tester.analyze_feature_similarity(test_loader)
    
    # Create visualizations
    tester.create_prediction_comparison(test_loader)
    knowledge_transfer = tester.create_performance_summary(teacher_miou, student_miou, feature_similarities)
    
    print(f"\n🎉 Feature-Based Testing Complete!")
    print("=" * 50)
    print(f"📈 Teacher mIoU:      {teacher_miou:.4f}")
    print(f"📈 Student mIoU:      {student_miou:.4f}")
    print(f"📊 Knowledge Transfer: {knowledge_transfer:.1f}%")
    print(f"📊 Feature Similarities:")
    for layer, sim in feature_similarities.items():
        print(f"   {layer}: {sim:.3f}")
    print("=" * 50)

if __name__ == "__main__":
    main()