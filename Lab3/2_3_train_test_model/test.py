"""
ELEC475 Lab 3 - Step 2.3: Test SMNet Custom Segmentation Model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import json
from datetime import datetime

# Add parent directories to path
sys.path.append('../2_2_Custom_SMNet')
sys.path.append('../2_1_Evaluate_Model')
from model import SMNet
from step1_local_voc import LocalVOCDataset, squeeze_and_long

# PASCAL VOC class names
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def load_training_history(base_dim):
    """Load training history if available."""
    possible_paths = [
        f'smnet_training_history_base{base_dim}.json',
        f'training_history_base{base_dim}.json'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    return None

def plot_simple_loss_curves(history, test_loss, base_dim):
    """Create simple loss and mIoU plots."""
    if history is None:
        print("[WARNING] No training history found - cannot create training plots")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.axhline(y=test_loss, color='green', linestyle='--', label=f'Test Loss ({test_loss:.4f})')
    ax1.set_title('Loss Over Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # mIoU plot (if available)
    if 'val_mious' in history:
        ax2.plot(epochs, history['val_mious'], 'r-', label='Validation mIoU')
        if 'train_mious' in history:
            ax2.plot(epochs, history['train_mious'], 'b-', label='Training mIoU')
        ax2.set_title('mIoU Over Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'mIoU data not available', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'plots/loss_miou_plots_base{base_dim}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[FILE] Loss and mIoU plots saved: plots/loss_miou_plots_base{base_dim}.png")

def load_smnet_model(model_path, base_dim, device):
    """Load the trained SMNet model."""
    model = SMNet(num_classes=21, base_dim=base_dim).to(device)
    
    # Try current directory first, then the model path
    if os.path.exists(f'best_smnet_model_base{base_dim}.pth'):
        model_path = f'best_smnet_model_base{base_dim}.pth'
    elif not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"SMNet model loaded from {model_path}")
    model.eval()
    return model

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

def calculate_per_class_iou(predictions, targets, num_classes=21):
    """Calculate per-class IoU across all test samples."""
    class_ious = []
    
    for cls in range(num_classes):
        class_intersection = 0
        class_union = 0
        
        for pred, target in zip(predictions, targets):
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            
            pred_mask = pred_np == cls
            target_mask = target_np == cls
            
            class_intersection += (pred_mask & target_mask).sum()
            class_union += (pred_mask | target_mask).sum()
        
        if class_union > 0:
            class_ious.append(class_intersection / class_union)
        else:
            class_ious.append(0.0)  # Class not present
    
    return np.array(class_ious)

def test_model_accuracy(model, test_loader, device):
    """Test model and return predictions, targets, and metrics."""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    sample_count = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    print("Testing model...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            sample_count += images.size(0)
            
            # Store predictions and targets
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    avg_loss = total_loss / sample_count
    
    return all_predictions, all_targets, avg_loss

def calculate_inference_speed(model, test_loader, device, num_batches=10):
    """Calculate inference speed in ms per image."""
    model.eval()
    
    import time
    times = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break
                
            images = images.to(device)
            
            # Warmup
            if i == 0:
                for _ in range(3):
                    _ = model(images)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
            
            # Timing
            start_time = time.time()
            _ = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            per_image_time = batch_time / images.size(0)
            times.append(per_image_time)
    
    return np.mean(times)

def print_detailed_results(class_ious, overall_miou, avg_loss, inference_speed, model_info):
    """Print comprehensive test results."""
    print(f"\n[RESULTS] SMNET CUSTOM SEGMENTATION MODEL TESTING")
    print("="*80)
    
    print(f"[MODEL] {model_info['model_name']}")
    print(f"[ARCHITECTURE] {model_info['architecture_type']}")
    print(f"[PARAMETERS] {model_info['total_parameters']:,} parameters")
    print(f"[BASE_DIM] {model_info['base_dimension']}")
    
    print(f"\n[METRICS] Overall Performance:")
    print(f"   Mean IoU (mIoU):     {overall_miou:.4f}")
    print(f"   Test Loss:           {avg_loss:.4f}")
    print(f"   Inference Speed:     {inference_speed:.2f} ms/image")
    
    print(f"\n[CLASS_METRICS] Per-Class IoU Scores:")
    valid_classes = 0
    for i, (class_name, iou) in enumerate(zip(VOC_CLASSES, class_ious)):
        if iou > 0:  # Only show classes that appear in test set
            print(f"   {class_name:<15}: {iou:.4f}")
            valid_classes += 1
        else:
            print(f"   {class_name:<15}: N/A (not present)")
    
    print(f"\n[SUMMARY] Performance Summary:")
    print(f"   Classes evaluated:   {valid_classes}/21")
    print(f"   Best performing:     {VOC_CLASSES[np.argmax(class_ious)]} ({np.max(class_ious):.4f})")
    print(f"   Worst performing:    {VOC_CLASSES[np.argmax(class_ious[class_ious > 0])]} ({np.min(class_ious[class_ious > 0]):.4f})")

def create_visualization(model, test_dataset, device, base_dim):
    """Create segmentation visualization with exactly 4 examples."""
    model.eval()
    
    # Ensure we don't go beyond dataset size
    dataset_size = len(test_dataset)
    sample_indices = []
    for i in range(4):
        idx = min(i * (dataset_size // 4), dataset_size - 1)
        sample_indices.append(idx)
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    for i, sample_idx in enumerate(sample_indices):
        # Get sample
        image, target = test_dataset[sample_idx]
        
        # Denormalize image for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_vis = image.permute(1, 2, 0).numpy()
        img_vis = std * img_vis + mean
        img_vis = np.clip(img_vis, 0, 1)
        
        # Get prediction
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu()
        
        # Plot
        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target.numpy(), cmap='tab20', vmin=0, vmax=20)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(prediction.numpy(), cmap='tab20', vmin=0, vmax=20)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'SMNet Base-{base_dim} Segmentation Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to visualization folder
    plt.savefig(f'visualizations/segmentation_examples_base{base_dim}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[FILE] 4 segmentation examples saved: visualizations/segmentation_examples_base{base_dim}.png")

def main():
    """Main testing function for SMNet."""
    
    parser = argparse.ArgumentParser(description='Test SMNet custom segmentation model')
    parser.add_argument('--base-dim', type=int, default=16,
                       help='Base dimension of the model to test. Default: 16')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for testing. Default: 8')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples for quick testing. Default: None (use all)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create segmentation visualizations')
    parser.add_argument('--plot-training', action='store_true', default=True,
                       help='Create training curve plots. Default: True')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing SMNet on device: {device}")
    print(f"Model configuration: Base dimension {args.base_dim}")
    
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
    
    # Load test dataset
    print("Loading test dataset...")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        squeeze_and_long
    ])
    
    test_dataset = LocalVOCDataset(
        voc_root=voc_root,
        split='val',  # Use validation set for testing
        transform=test_transform,
        target_transform=target_transform,
        max_samples=args.max_samples
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    model_path = f'best_smnet_model_base{args.base_dim}.pth'
    model = load_smnet_model(model_path, args.base_dim, device)
    
    if model is None:
        print(f"No trained SMNet model found! Please train the model first.")
        print(f"Expected model file: {model_path}")
        return
    
    model_info = model.get_model_info()
    
    # Test the model
    predictions, targets, avg_loss = test_model_accuracy(model, test_loader, device)
    
    # Calculate per-class IoU
    class_ious = calculate_per_class_iou(predictions, targets)
    overall_miou = np.mean(class_ious[class_ious > 0])  # Mean of valid classes
    
    # Calculate inference speed
    inference_speed = calculate_inference_speed(model, test_loader, device)
    
    # Print results
    print_detailed_results(class_ious, overall_miou, avg_loss, inference_speed, model_info)
    
    # Create simple loss and mIoU plots
    if args.plot_training:
        training_history = load_training_history(args.base_dim)
        plot_simple_loss_curves(training_history, avg_loss, args.base_dim)
    
    # Create visualizations if requested
    if args.visualize:
        create_visualization(model, test_dataset, device, args.base_dim)
    
    print(f"\n[OK] SMNet testing completed!")

if __name__ == '__main__':
    main()