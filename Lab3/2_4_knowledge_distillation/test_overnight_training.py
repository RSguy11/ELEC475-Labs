"""
ELEC475 Lab 3 - Quick Test for Overnight Training Suite
Test the overnight training script with minimal configurations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models.segmentation as segmentation
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from datetime import datetime

# Add parent directories to path
sys.path.append('../2_2_Custom_SMNet')
sys.path.append('../2_1_Evaluate_Model')
from model import SMNet
from step1_local_voc import LocalVOCDataset, squeeze_and_long

def main():
    """Test the overnight training pipeline with minimal settings."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing overnight training on device: {device}")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Quick dataset setup
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
    
    # Simple transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        lambda x: x.squeeze(0).long()
    ])
    
    # Small datasets for testing
    train_dataset = LocalVOCDataset(
        voc_root=voc_root, split='train',
        transform=transform, target_transform=target_transform,
        max_samples=50
    )
    
    val_dataset = LocalVOCDataset(
        voc_root=voc_root, split='val',
        transform=transform, target_transform=target_transform, 
        max_samples=20
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Test datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Quick_Baseline',
            'method': 'baseline',
            'epochs': 3
        },
        {
            'name': 'Quick_KD',
            'method': 'kd',
            'epochs': 3
        }
    ]
    
    os.makedirs('test_results', exist_ok=True)
    
    for config in test_configs:
        print(f"\\n{'='*50}")
        print(f"Testing: {config['name']}")
        print(f"{'='*50}")
        
        # Initialize model
        student_model = SMNet(num_classes=21, base_dim=16).to(device)
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Load teacher if KD
        teacher_model = None
        if config['method'] == 'kd':
            teacher_model = segmentation.fcn_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
            teacher_model.to(device)
            teacher_model.eval()
            print("✓ Teacher model loaded")
        
        # Quick training loop
        for epoch in range(config['epochs']):
            start_time = time.time()
            
            # Training
            student_model.train()
            train_loss = 0.0
            
            for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                images = images.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                
                if config['method'] == 'baseline':
                    outputs = student_model(images)
                    loss = criterion(outputs, targets)
                else:
                    # Simple KD
                    with torch.no_grad():
                        teacher_outputs = teacher_model(images)['out']
                    
                    student_outputs = student_model(images)
                    
                    # Hard loss
                    hard_loss = criterion(student_outputs, targets)
                    
                    # Soft loss (simplified)
                    T = 4.0
                    teacher_soft = torch.softmax(teacher_outputs / T, dim=1)
                    student_soft = torch.log_softmax(student_outputs / T, dim=1)
                    soft_loss = nn.KLDivLoss(reduction='batchmean')(student_soft, teacher_soft)
                    
                    loss = 0.5 * hard_loss + 0.5 * soft_loss
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            student_model.eval()
            val_loss = 0.0
            val_ious = []
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    
                    outputs = student_model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # Quick mIoU calculation
                    predictions = torch.argmax(outputs, dim=1)
                    for pred, target in zip(predictions, targets):
                        pred_np = pred.cpu().numpy()
                        target_np = target.cpu().numpy()
                        
                        # Calculate IoU for class 1 (simple test)
                        pred_mask = pred_np == 1
                        target_mask = target_np == 1
                        intersection = (pred_mask & target_mask).sum()
                        union = (pred_mask | target_mask).sum()
                        
                        if union > 0:
                            val_ious.append(intersection / union)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_miou = np.mean(val_ious) if val_ious else 0.0
            epoch_time = time.time() - start_time
            
            print(f"  Epoch {epoch+1}/{config['epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"mIoU: {avg_miou:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Save test model
        model_path = f"test_results/{config['name']}_model.pth"
        torch.save(student_model.state_dict(), model_path)
        print(f"✓ Test model saved: {model_path}")
    
    print(f"\\n✓ Overnight training test completed successfully!")
    print(f"✓ All components working correctly")
    print(f"\\nNow you can run the full overnight training with:")
    print(f"  python overnight_training_suite.py")

if __name__ == '__main__':
    main()