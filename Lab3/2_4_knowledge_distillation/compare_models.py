"""
ELEC475 Lab 3 - Step 2.4: Knowledge Distillation Testing & Comparison
Compare baseline SMNet vs Knowledge Distillation trained SMNet
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models.segmentation as segmentation
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import time

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

def load_model(model_path, base_dim, device, model_type="baseline"):
    """Load a trained SMNet model."""
    model = SMNet(num_classes=21, base_dim=base_dim).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"{model_type} SMNet model loaded from {model_path}")
        model.eval()
        return model
    else:
        print(f"Model file {model_path} not found!")
        return None

def load_teacher_model(device):
    """Load FCN-ResNet50 teacher model for comparison."""
    teacher = segmentation.fcn_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
    teacher.to(device)
    teacher.eval()
    print("Teacher model (FCN-ResNet50) loaded")
    return teacher

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

def test_model_comprehensive(model, test_loader, device, model_name):
    """Comprehensive model testing with detailed metrics."""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    sample_count = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    print(f"Testing {model_name}...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=f"Testing {model_name}"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Handle teacher model output format
            if model_name == "Teacher (FCN-ResNet50)":
                outputs = model(images)['out']
            else:
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

def calculate_inference_speed(model, test_loader, device, model_name, num_batches=10):
    """Calculate inference speed in ms per image."""
    model.eval()
    
    times = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break
                
            images = images.to(device)
            
            # Warmup
            if i == 0:
                for _ in range(3):
                    if model_name == "Teacher (FCN-ResNet50)":
                        _ = model(images)['out']
                    else:
                        _ = model(images)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
            
            # Timing
            start_time = time.time()
            if model_name == "Teacher (FCN-ResNet50)":
                _ = model(images)['out'] 
            else:
                _ = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            per_image_time = batch_time / images.size(0)
            times.append(per_image_time)
    
    return np.mean(times)

def print_comparison_results(results_dict):
    """Print comprehensive comparison of all models."""
    print(f"\\n{'='*100}")
    print(f"[ELEC475 LAB 3] KNOWLEDGE DISTILLATION COMPARISON RESULTS")
    print(f"{'='*100}")
    
    # Header
    print(f"{'Model':<25} {'Parameters':<12} {'mIoU':<8} {'Loss':<8} {'Speed (ms)':<12} {'Improvement'}")
    print(f"{'-'*100}")
    
    # Calculate improvements relative to baseline
    baseline_miou = results_dict.get('Baseline SMNet', {}).get('miou', 0)
    teacher_miou = results_dict.get('Teacher (FCN-ResNet50)', {}).get('miou', 0)
    
    for model_name, metrics in results_dict.items():
        params = f"{metrics['parameters']:,}" if metrics['parameters'] > 0 else "N/A"
        miou = metrics['miou']
        loss = metrics['loss']
        speed = metrics['speed']
        
        # Calculate improvement
        if model_name == "Baseline SMNet":
            improvement = "baseline"
        elif model_name == "Teacher (FCN-ResNet50)":
            improvement = "teacher"
        else:
            if baseline_miou > 0:
                improvement = f"+{((miou - baseline_miou) / baseline_miou * 100):+.1f}%"
            else:
                improvement = "N/A"
        
        print(f"{model_name:<25} {params:<12} {miou:.4f}   {loss:.4f}   {speed:8.2f}     {improvement}")
    
    print(f"{'-'*100}")
    
    # Best model summary
    best_model = max(results_dict.keys(), 
                     key=lambda x: results_dict[x]['miou'] if x != "Teacher (FCN-ResNet50)" else 0)
    best_miou = results_dict[best_model]['miou']
    
    print(f"\\n[SUMMARY] Best Student Model: {best_model} (mIoU: {best_miou:.4f})")
    
    # Knowledge transfer analysis
    if teacher_miou > 0 and baseline_miou > 0:
        teacher_gap = teacher_miou - baseline_miou
        print(f"[ANALYSIS] Teacher-Student Gap: {teacher_gap:.4f} mIoU")
        
        if 'KD SMNet' in results_dict:
            kd_miou = results_dict['KD SMNet']['miou']
            kd_improvement = kd_miou - baseline_miou
            gap_closed = (kd_improvement / teacher_gap * 100) if teacher_gap > 0 else 0
            print(f"[ANALYSIS] Knowledge Distillation closed {gap_closed:.1f}% of the teacher-student gap")

def create_detailed_comparison_visualization(models_data, test_dataset, device, num_samples=4):
    """Create detailed comparison visualization."""
    
    fig, axes = plt.subplots(num_samples, len(models_data) + 2, 
                            figsize=(3*(len(models_data)+2), 3*num_samples))
    
    # Column headers
    col_names = ['Input', 'Ground Truth'] + list(models_data.keys())
    
    for i, col_name in enumerate(col_names):
        if num_samples == 1:
            axes[i].set_title(col_name, fontweight='bold')
        else:
            axes[0, i].set_title(col_name, fontweight='bold')
    
    for sample_idx in range(num_samples):
        # Get sample (ensure we don't exceed dataset bounds)
        safe_idx = min(sample_idx * 100, len(test_dataset) - 1)
        image, target = test_dataset[safe_idx] 
        
        # Denormalize image for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_vis = image.permute(1, 2, 0).numpy()
        img_vis = std * img_vis + mean
        img_vis = np.clip(img_vis, 0, 1)
        
        # Plot input image
        ax_idx = (sample_idx, 0) if num_samples > 1 else 0
        axes[ax_idx].imshow(img_vis)
        axes[ax_idx].axis('off')
        
        # Plot ground truth
        ax_idx = (sample_idx, 1) if num_samples > 1 else 1
        axes[ax_idx].imshow(target.numpy(), cmap='tab20', vmin=0, vmax=20)
        axes[ax_idx].axis('off')
        
        # Plot model predictions
        for model_idx, (model_name, model) in enumerate(models_data.items()):
            model.eval()
            with torch.no_grad():
                image_batch = image.unsqueeze(0).to(device)
                
                if model_name == "Teacher (FCN-ResNet50)":
                    output = model(image_batch)['out']
                else:
                    output = model(image_batch)
                    
                prediction = torch.argmax(output, dim=1).squeeze(0).cpu()
            
            ax_idx = (sample_idx, model_idx + 2) if num_samples > 1 else model_idx + 2
            axes[ax_idx].imshow(prediction.numpy(), cmap='tab20', vmin=0, vmax=20)
            axes[ax_idx].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('comparison_results', exist_ok=True)
    plt.savefig('comparison_results/knowledge_distillation_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[FILE] Comparison visualization saved: comparison_results/knowledge_distillation_comparison.png")

def create_performance_charts(results_dict):
    """Create performance comparison charts."""
    
    os.makedirs('comparison_results', exist_ok=True)
    
    # Extract data for plotting
    models = list(results_dict.keys())
    mious = [results_dict[model]['miou'] for model in models]
    params = [results_dict[model]['parameters']/1000 for model in models]  # Convert to K
    speeds = [results_dict[model]['speed'] for model in models]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # mIoU comparison
    colors = ['blue' if 'Baseline' in model else 'green' if 'KD' in model else 'red' 
              for model in models]
    bars1 = ax1.bar(models, mious, color=colors, alpha=0.7)
    ax1.set_title('Model Performance Comparison - mIoU')
    ax1.set_ylabel('mIoU')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, miou in zip(bars1, mious):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{miou:.4f}', ha='center', va='bottom')
    
    # Parameters vs mIoU
    scatter_colors = ['blue' if 'Baseline' in model else 'green' if 'KD' in model else 'red' 
                     for model in models]
    ax2.scatter(params, mious, c=scatter_colors, s=100, alpha=0.7)
    for i, model in enumerate(models):
        ax2.annotate(model, (params[i], mious[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_title('Efficiency vs Performance')
    ax2.set_xlabel('Parameters (K)')
    ax2.set_ylabel('mIoU')
    ax2.grid(True, alpha=0.3)
    
    # Inference speed comparison  
    bars3 = ax3.bar(models, speeds, color=colors, alpha=0.7)
    ax3.set_title('Inference Speed Comparison')
    ax3.set_ylabel('Inference Time (ms/image)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, speed in zip(bars3, speeds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.1f}', ha='center', va='bottom')
    
    # Performance improvement radar chart (simplified)
    baseline_miou = next((results_dict[model]['miou'] for model in models if 'Baseline' in model), 0)
    improvements = [(results_dict[model]['miou'] - baseline_miou) / baseline_miou * 100 
                   if baseline_miou > 0 and 'Baseline' not in model else 0 
                   for model in models]
    
    bars4 = ax4.bar([model for model in models if 'Baseline' not in model], 
                   [imp for imp in improvements if imp != 0], 
                   color=['green' if 'KD' in model else 'red' for model in models if 'Baseline' not in model], 
                   alpha=0.7)
    ax4.set_title('Performance Improvement over Baseline (%)')
    ax4.set_ylabel('Improvement (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, imp in zip(bars4, [imp for imp in improvements if imp != 0]):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('comparison_results/performance_comparison_charts.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[FILE] Performance charts saved: comparison_results/performance_comparison_charts.png")

def main():
    """Main function for knowledge distillation comparison."""
    
    parser = argparse.ArgumentParser(description='Compare baseline vs Knowledge Distillation SMNet')
    parser.add_argument('--base-dim', type=int, default=16,
                       help='Base dimension of models to compare. Default: 16')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for testing. Default: 8')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples for quick testing. Default: None (use all)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create detailed comparison visualizations')
    parser.add_argument('--include-teacher', action='store_true',
                       help='Include teacher model in comparison')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Knowledge Distillation comparison on device: {device}")
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
    print("\\nLoading test dataset...")
    
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
    
    # Load models for comparison
    models_to_compare = {}
    results_dict = {}
    
    # 1. Baseline SMNet
    baseline_path = f'../2_3_train_test_model/best_smnet_model_base{args.base_dim}.pth'
    baseline_model = load_model(baseline_path, args.base_dim, device, "Baseline")
    if baseline_model is not None:
        models_to_compare['Baseline SMNet'] = baseline_model
    
    # 2. Knowledge Distillation SMNet
    kd_path = f'smnet_kd_model_base{args.base_dim}.pth'
    kd_model = load_model(kd_path, args.base_dim, device, "Knowledge Distillation")
    if kd_model is not None:
        models_to_compare['KD SMNet'] = kd_model
    
    # 3. Teacher model (optional)
    if args.include_teacher:
        teacher_model = load_teacher_model(device)
        models_to_compare['Teacher (FCN-ResNet50)'] = teacher_model
    
    if len(models_to_compare) == 0:
        print("No trained models found for comparison!")
        return
    
    print(f"\\nComparing {len(models_to_compare)} models...")
    
    # Test all models
    for model_name, model in models_to_compare.items():
        print(f"\\n{'-'*50}")
        
        # Get model info
        if model_name == "Teacher (FCN-ResNet50)":
            total_params = sum(p.numel() for p in model.parameters())
            model_info = {'total_parameters': total_params}
        else:
            model_info = model.get_model_info()
        
        # Test model
        predictions, targets, avg_loss = test_model_comprehensive(
            model, test_loader, device, model_name
        )
        
        # Calculate per-class IoU and overall mIoU
        class_ious = calculate_per_class_iou(predictions, targets)
        overall_miou = np.mean(class_ious[class_ious > 0])
        
        # Calculate inference speed
        inference_speed = calculate_inference_speed(model, test_loader, device, model_name)
        
        # Store results
        results_dict[model_name] = {
            'parameters': model_info['total_parameters'],
            'miou': overall_miou,
            'loss': avg_loss,
            'speed': inference_speed,
            'class_ious': class_ious
        }
        
        print(f"[{model_name}] mIoU: {overall_miou:.4f}, Loss: {avg_loss:.4f}, Speed: {inference_speed:.2f} ms/img")
    
    # Print comprehensive comparison
    print_comparison_results(results_dict)
    
    # Create visualizations if requested
    if args.visualize and len(models_to_compare) > 1:
        print("\\nGenerating comparison visualizations...")
        create_detailed_comparison_visualization(models_to_compare, test_dataset, device)
        create_performance_charts(results_dict)
    
    # Save detailed results
    os.makedirs('comparison_results', exist_ok=True)
    
    # Create detailed report
    with open('comparison_results/detailed_comparison_report.txt', 'w') as f:
        f.write("ELEC475 Lab 3 - Knowledge Distillation Comparison Report\\n")
        f.write("="*60 + "\\n\\n")
        
        f.write(f"Test Configuration:\\n")
        f.write(f"  Device: {device}\\n")
        f.write(f"  Base Dimension: {args.base_dim}\\n")
        f.write(f"  Test Samples: {len(test_dataset)}\\n")
        f.write(f"  Batch Size: {args.batch_size}\\n\\n")
        
        for model_name, metrics in results_dict.items():
            f.write(f"{model_name}:\\n")
            f.write(f"  Parameters: {metrics['parameters']:,}\\n")
            f.write(f"  Mean IoU: {metrics['miou']:.6f}\\n")
            f.write(f"  Test Loss: {metrics['loss']:.6f}\\n")
            f.write(f"  Inference Speed: {metrics['speed']:.3f} ms/image\\n")
            
            f.write(f"  Per-class IoU:\\n")
            for i, (class_name, iou) in enumerate(zip(VOC_CLASSES, metrics['class_ious'])):
                if iou > 0:
                    f.write(f"    {class_name}: {iou:.6f}\\n")
            f.write("\\n")
    
    print(f"\\n[FILE] Detailed report saved: comparison_results/detailed_comparison_report.txt")
    print(f"[OK] Knowledge distillation comparison completed!")

if __name__ == '__main__':
    main()