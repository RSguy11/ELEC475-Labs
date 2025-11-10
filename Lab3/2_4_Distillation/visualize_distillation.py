import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys

# Add parent directories to path
sys.path.append('../2_2_Custom_SMNet')
sys.path.append('../2_1_Evaluate_Model')

def plot_distillation_losses(loss_history):
    """Plot the distillation training losses."""
    epochs = range(1, len(loss_history['total']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Total loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss_history['total'], 'b-', label='Total Loss')
    plt.title('Total Distillation Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Individual loss components
    plt.subplot(1, 3, 2)
    plt.plot(epochs, loss_history['response'], 'r-', label='Response Loss')
    plt.plot(epochs, loss_history['hard'], 'g-', label='Hard Target Loss')
    plt.plot(epochs, loss_history['feature'], 'm-', label='Feature Loss')
    plt.title('Loss Components')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Loss ratios
    plt.subplot(1, 3, 3)
    total = np.array(loss_history['total'])
    response_ratio = np.array(loss_history['response']) / total
    hard_ratio = np.array(loss_history['hard']) / total
    feature_ratio = np.array(loss_history['feature']) / total
    
    plt.plot(epochs, response_ratio, 'r-', label='Response %')
    plt.plot(epochs, hard_ratio, 'g-', label='Hard Target %')
    plt.plot(epochs, feature_ratio, 'm-', label='Feature %')
    plt.title('Loss Component Ratios')
    plt.xlabel('Batch')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('image_results/distillation_losses.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_predictions(images, targets, teacher_preds, student_orig_preds, student_dist_preds, 
                         teacher_miou, student_orig_miou, student_dist_miou, num_samples=4):
    """Visualize prediction comparisons for successes and failures."""
    
    # Calculate IoU for each sample to find best/worst
    sample_ious = []
    for i in range(len(images)):
        # Calculate IoU for this sample
        target = targets[i]
        teacher_pred = teacher_preds[i]
        student_orig_pred = student_orig_preds[i]
        student_dist_pred = student_dist_preds[i]
        
        # Simple IoU calculation per sample
        teacher_iou = calculate_sample_iou(teacher_pred, target)
        orig_iou = calculate_sample_iou(student_orig_pred, target)
        dist_iou = calculate_sample_iou(student_dist_pred, target)
        
        sample_ious.append({
            'idx': i,
            'teacher_iou': teacher_iou,
            'student_orig_iou': orig_iou,
            'student_dist_iou': dist_iou,
            'improvement': dist_iou - orig_iou
        })
    
    # Sort by improvement (distilled vs original)
    sample_ious.sort(key=lambda x: x['improvement'], reverse=True)
    
    # Select best and worst cases
    best_cases = sample_ious[:2]  # Top 2 improvements
    worst_cases = sample_ious[-2:]  # Bottom 2 improvements
    selected_cases = best_cases + worst_cases
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for row, case in enumerate(selected_cases):
        idx = case['idx']
        
        # Denormalize image for display
        img = images[idx].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Original image
        axes[row, 0].imshow(img.permute(1, 2, 0))
        axes[row, 0].set_title(f'Original Image\n{"Success" if case["improvement"] > 0 else "Failure"}')
        axes[row, 0].axis('off')
        
        # Ground truth
        axes[row, 1].imshow(targets[idx].cpu(), cmap='tab20')
        axes[row, 1].set_title('Ground Truth')
        axes[row, 1].axis('off')
        
        # Teacher prediction
        axes[row, 2].imshow(teacher_preds[idx].cpu(), cmap='tab20')
        axes[row, 2].set_title(f'Teacher (FCN)\nIoU: {case["teacher_iou"]:.3f}')
        axes[row, 2].axis('off')
        
        # Original student prediction
        axes[row, 3].imshow(student_orig_preds[idx].cpu(), cmap='tab20')
        axes[row, 3].set_title(f'Student Original\nIoU: {case["student_orig_iou"]:.3f}')
        axes[row, 3].axis('off')
        
        # Distilled student prediction
        axes[row, 4].imshow(student_dist_preds[idx].cpu(), cmap='tab20')
        axes[row, 4].set_title(f'Student Distilled\nIoU: {case["student_dist_iou"]:.3f}\n(Δ {case["improvement"]:+.3f})')
        axes[row, 4].axis('off')
    
    plt.suptitle(f'Knowledge Distillation Results\nTeacher: {teacher_miou:.3f} | Student Original: {student_orig_miou:.3f} | Student Distilled: {student_dist_miou:.3f}')
    plt.tight_layout()
    plt.savefig('image_results/distillation_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

def calculate_sample_iou(pred, target, num_classes=21):
    """Calculate IoU for a single sample."""
    iou_per_class = []
    for class_id in range(num_classes):
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        if union > 0:
            iou_per_class.append(intersection.float() / union.float())
    return torch.stack(iou_per_class).mean() if iou_per_class else torch.tensor(0.0)

def create_visualizations_folder():
    """Create visualizations folder if it doesn't exist."""
    if not os.path.exists('image_results'):
        os.makedirs('image_results')
        print("Created image_results folder")

if __name__ == "__main__":
    create_visualizations_folder()
    print("Visualization functions ready!")