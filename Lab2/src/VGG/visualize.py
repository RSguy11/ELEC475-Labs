import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Add parent directory to path
sys.path.append('..')
from dataset import CustomImageDataset
from model import VGG16Snout

def load_vgg16_model(model_path, device):
    model = VGG16Snout(pretrained=False).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"VGG16 model loaded from {model_path}")
        model.eval()
        return model
    else:
        print(f"Model file {model_path} not found!")
        return None

def test_model_accuracy(model, test_loader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, target_coords in test_loader:
            images = images.to(device)
            target_coords = target_coords.to(device)
            
            pred_coords = model(images)
            
            predictions.extend(pred_coords.cpu().numpy())
            targets.extend(target_coords.cpu().numpy())
    
    return np.array(predictions), np.array(targets)

def calculate_statistics(predictions, targets):
    euclidean_distances = np.sqrt(np.sum((predictions - targets)**2, axis=1))
    x_errors = predictions[:, 0] - targets[:, 0]
    y_errors = predictions[:, 1] - targets[:, 1]
    
    stats = {
        'euclidean': {
            'mean': np.mean(euclidean_distances),
            'std': np.std(euclidean_distances),
            'median': np.median(euclidean_distances),
            'q25': np.percentile(euclidean_distances, 25),
            'q75': np.percentile(euclidean_distances, 75),
            'min': np.min(euclidean_distances),
            'max': np.max(euclidean_distances)
        },
        'x_error': {
            'mean': np.mean(np.abs(x_errors)),
            'bias': np.mean(x_errors),
            'std': np.std(x_errors)
        },
        'y_error': {
            'mean': np.mean(np.abs(y_errors)),
            'bias': np.mean(y_errors),
            'std': np.std(y_errors)
        },
        'mse': np.mean((predictions - targets)**2),
        'mae': np.mean(euclidean_distances)
    }
    
    return stats, euclidean_distances, x_errors, y_errors

def plot_error_distributions(euclidean_distances, x_errors, y_errors, predictions, targets, model_condition):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Euclidean distance histogram
    axes[0, 0].hist(euclidean_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(euclidean_distances), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(euclidean_distances):.2f}')
    axes[0, 0].axvline(np.median(euclidean_distances), color='green', linestyle='--', 
                      label=f'Median: {np.median(euclidean_distances):.2f}')
    axes[0, 0].set_xlabel('Euclidean Distance (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'VGG16 {model_condition} - Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # X-error histogram
    axes[0, 1].hist(x_errors, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].axvline(np.mean(x_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(x_errors):.2f}')
    axes[0, 1].set_xlabel('X-coordinate Error (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('X-coordinate Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Y-error histogram
    axes[0, 2].hist(y_errors, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].axvline(np.mean(y_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(y_errors):.2f}')
    axes[0, 2].set_xlabel('Y-coordinate Error (pixels)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Y-coordinate Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Scatter plot: predicted vs true
    axes[1, 0].scatter(targets[:, 0], predictions[:, 0], alpha=0.6, s=20, color='blue', label='X-coord')
    axes[1, 0].scatter(targets[:, 1], predictions[:, 1], alpha=0.6, s=20, color='green', label='Y-coord')
    axes[1, 0].plot([0, 227], [0, 227], 'r--', label='Perfect Prediction')
    axes[1, 0].set_xlabel('True Coordinates')
    axes[1, 0].set_ylabel('Predicted Coordinates')
    axes[1, 0].set_title('Predicted vs True Coordinates')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error vs distance from center
    center = np.array([113.5, 113.5])
    distances_from_center = np.sqrt(np.sum((targets - center)**2, axis=1))
    axes[1, 1].scatter(distances_from_center, euclidean_distances, alpha=0.6, s=20, color='purple')
    axes[1, 1].set_xlabel('Distance from Image Center (pixels)')
    axes[1, 1].set_ylabel('Prediction Error (pixels)')
    axes[1, 1].set_title('Error vs Distance from Center')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Box plot summary
    error_data = [euclidean_distances, np.abs(x_errors), np.abs(y_errors)]
    axes[1, 2].boxplot(error_data, tick_labels=['Euclidean', 'X-Error', 'Y-Error'])
    axes[1, 2].set_ylabel('Error (pixels)')
    axes[1, 2].set_title('Error Distribution Summary')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    
    # Save to Results_Images folder
    folder_name = "Augmented" if "augmented" in model_condition.lower() else "Baseline"
    results_dir = f"Results_Images/{folder_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f'vgg16_test_error_analysis_{model_condition.lower()}.png'
    plt.savefig(f'{results_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure instead of showing to prevent stalling during automation

def visualize_best_worst_predictions(model, test_dataset, device, euclidean_distances, model_condition, num_samples=6):
    """Visualize best and worst predictions"""
    # Get indices of best and worst predictions
    sorted_indices = np.argsort(euclidean_distances)
    best_indices = sorted_indices[:num_samples//2]
    worst_indices = sorted_indices[-num_samples//2:]
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 8))
    
    model.eval()
    with torch.no_grad():
        # Plot best predictions
        for i, idx in enumerate(best_indices):
            image, true_coords = test_dataset[idx]
            filename = test_dataset.data[idx][0]
            
            # Get prediction
            image_batch = image.unsqueeze(0).to(device)
            pred_coords = model(image_batch).cpu().squeeze()
            
            # Convert image for display
            img_display = image.permute(1, 2, 0).numpy()
            
            axes[0, i].imshow(img_display)
            axes[0, i].scatter(true_coords[0], true_coords[1], c='green', s=100, marker='o', 
                             label='True', linewidths=2)
            axes[0, i].scatter(pred_coords[0], pred_coords[1], c='red', s=100, marker='x', 
                             label='Predicted', linewidths=3)
            
            error = euclidean_distances[idx]
            axes[0, i].set_title(f'BEST {i+1}\n{filename}\nError: {error:.1f}px')
            axes[0, i].legend()
            axes[0, i].axis('off')
        
        # Plot worst predictions
        for i, idx in enumerate(worst_indices):
            image, true_coords = test_dataset[idx]
            filename = test_dataset.data[idx][0]
            
            # Get prediction
            image_batch = image.unsqueeze(0).to(device)
            pred_coords = model(image_batch).cpu().squeeze()
            
            # Convert image for display
            img_display = image.permute(1, 2, 0).numpy()
            
            axes[1, i].imshow(img_display)
            axes[1, i].scatter(true_coords[0], true_coords[1], c='green', s=100, marker='o', 
                             label='True', linewidths=2)
            axes[1, i].scatter(pred_coords[0], pred_coords[1], c='red', s=100, marker='x', 
                             label='Predicted', linewidths=3)
            
            error = euclidean_distances[idx]
            axes[1, i].set_title(f'WORST {i+1}\n{filename}\nError: {error:.1f}px')
            axes[1, i].legend()
            axes[1, i].axis('off')
    
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    
    # Save to Results_Images folder
    folder_name = "Augmented" if "augmented" in model_condition.lower() else "Baseline"
    results_dir = f"Results_Images/{folder_name}"
    
    filename = f'vgg16_best_worst_predictions_{model_condition.lower()}.png'
    plt.savefig(f'{results_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure instead of showing to prevent stalling during automation

def visualize_four_examples(model, test_dataset, device, euclidean_distances, model_condition):
    """Create four specific visualization examples for lab report"""
    
    # Select diverse examples based on error range
    sorted_indices = np.argsort(euclidean_distances)
    
    # Select 4 strategic examples:
    # 1. Best case (lowest error)
    # 2. Good case (25th percentile) 
    # 3. Challenging case (75th percentile)
    # 4. Worst case (highest error)
    
    n_samples = len(euclidean_distances)
    selected_indices = [
        sorted_indices[0],                          # Best
        sorted_indices[int(0.25 * n_samples)],     # Good
        sorted_indices[int(0.75 * n_samples)],     # Challenging
        sorted_indices[-1]                          # Worst
    ]
    
    case_labels = ['Best Case', 'Good Case', 'Challenging Case', 'Worst Case']
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'SnoutNet-V {model_condition} - Four Representative Examples\nFor Lab Report Visualization', 
                fontsize=18, fontweight='bold')
    
    model.eval()
    
    for i, (idx, case_label) in enumerate(zip(selected_indices, case_labels)):
        image, true_coords = test_dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_coords = model(image).cpu().numpy()[0]
        
        # Convert to displayable format
        image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
        filename = test_dataset.data[idx][0]
        
        # Top row: Original image with predictions
        if image_np.shape[2] == 1:  # Grayscale
            image_np = image_np.squeeze()
            axes[0, i].imshow(image_np, cmap='gray')
        else:  # Color
            image_np = np.clip(image_np, 0, 1)
            axes[0, i].imshow(image_np)
        
        # Add prediction overlays with enhanced visibility
        axes[0, i].scatter(true_coords[0], true_coords[1], c='lime', s=200, marker='o', 
                         label='Ground Truth', linewidths=3, edgecolors='black')
        axes[0, i].scatter(pred_coords[0], pred_coords[1], c='red', s=200, marker='x', 
                         label='Prediction', linewidths=4)
        
        # Draw connecting line to show error
        axes[0, i].plot([true_coords[0], pred_coords[0]], [true_coords[1], pred_coords[1]], 
                       'yellow', linewidth=2, linestyle='--', alpha=0.8)
        
        error = euclidean_distances[idx]
        axes[0, i].set_title(f'{case_label}\nError: {error:.2f} pixels', 
                           fontsize=12, fontweight='bold')
        axes[0, i].legend(loc='upper right', fontsize=10)
        axes[0, i].axis('off')
        
        # Bottom row: Zoomed view around nose area
        # Create zoomed view (50x50 pixel region around true nose)
        zoom_size = 50
        center_x, center_y = int(true_coords[0]), int(true_coords[1])
        
        # Calculate zoom bounds with boundary checking
        x_min = max(0, center_x - zoom_size//2)
        x_max = min(image_np.shape[1] if len(image_np.shape) > 2 else image_np.shape[0], 
                   center_x + zoom_size//2)
        y_min = max(0, center_y - zoom_size//2)
        y_max = min(image_np.shape[0], center_y + zoom_size//2)
        
        if len(image_np.shape) == 3:
            zoomed_img = image_np[y_min:y_max, x_min:x_max, :]
        else:
            zoomed_img = image_np[y_min:y_max, x_min:x_max]
        
        if len(image_np.shape) == 2:
            axes[1, i].imshow(zoomed_img, cmap='gray')
        else:
            axes[1, i].imshow(zoomed_img)
        
        # Adjust coordinates for zoomed view
        true_x_zoom = true_coords[0] - x_min
        true_y_zoom = true_coords[1] - y_min
        pred_x_zoom = pred_coords[0] - x_min
        pred_y_zoom = pred_coords[1] - y_min
        
        axes[1, i].scatter(true_x_zoom, true_y_zoom, c='lime', s=300, marker='o', 
                         linewidths=4, edgecolors='black')
        axes[1, i].scatter(pred_x_zoom, pred_y_zoom, c='red', s=300, marker='x', 
                         linewidths=5)
        axes[1, i].plot([true_x_zoom, pred_x_zoom], [true_y_zoom, pred_y_zoom], 
                       'yellow', linewidth=3, linestyle='--', alpha=0.9)
        
        axes[1, i].set_title(f'Zoomed View\n{filename}', fontsize=10, fontweight='bold')
        axes[1, i].axis('off')
    
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    
    # Save to Results_Images folder
    folder_name = "Augmented" if "augmented" in model_condition.lower() else "Baseline"
    results_dir = f"Results_Images/{folder_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f'vgg16_four_examples_{model_condition.lower()}.png'
    plt.savefig(f'{results_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    
    return selected_indices, case_labels

def main():
    """Main visualization function for VGG16"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize VGG16-SnoutNet model results')
    parser.add_argument('-t', '--type', choices=['baseline', 'augmented', 'auto'], 
                       default='auto', help='Model type to visualize (default: auto-detect)')
    parser.add_argument('--error-analysis', action='store_true', 
                       help='Generate comprehensive error analysis plots')
    parser.add_argument('--best-worst', action='store_true', 
                       help='Generate best/worst prediction examples')
    parser.add_argument('--all', action='store_true', 
                       help='Generate all visualization types')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Visualizing VGG16-SnoutNet results on device: {device}")
    print(f"Model type preference: {args.type}")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = CustomImageDataset(
        annotations_file="../oxford-iiit-pet-noses/test_noses.txt",
        img_dir='../oxford-iiit-pet-noses/images-original/images/'
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Build model paths based on user preference
    if args.type == 'baseline':
        model_paths = ['best_vgg16_model_baseline.pth']
    elif args.type == 'augmented':
        model_paths = ['best_vgg16_model_augmented.pth']
    else:  # auto-detect
        model_paths = [
            'best_vgg16_model_baseline.pth',
            'best_vgg16_model_augmented.pth'
        ]
    
    model = None
    used_model_path = None
    
    for model_path in model_paths:
        model = load_vgg16_model(model_path, device)
        if model is not None:
            used_model_path = model_path
            break
    
    if model is None:
        print(f"No trained VGG16 model found! Please train the model first.")
        return
    
    print(f"Using model: {used_model_path}")
    
    # Determine model condition
    model_condition = "Augmented" if "augmented" in used_model_path.lower() else "Baseline"
    
    # Test the model to get predictions
    print("Getting model predictions...")
    predictions, targets = test_model_accuracy(model, test_loader, device)
    
    # Calculate statistics for visualization
    stats, euclidean_distances, x_errors, y_errors = calculate_statistics(predictions, targets)
    
    # Generate visualizations based on arguments
    if args.all or args.error_analysis or (not args.best_worst and not args.error_analysis):
        print("Generating error analysis plots...")
        plot_error_distributions(euclidean_distances, x_errors, y_errors, predictions, targets, model_condition)
        print(f"Error analysis plot saved to Results_Images/{model_condition}/")
    
    if args.all or args.best_worst:
        print("Generating best/worst prediction examples...")
        visualize_best_worst_predictions(model, test_dataset, device, euclidean_distances, model_condition)
        print(f"Best/worst predictions plot saved to Results_Images/{model_condition}/")
    
    # Always generate four examples for lab report
    print("Generating four representative examples for lab report...")
    selected_indices, case_labels = visualize_four_examples(model, test_dataset, device, euclidean_distances, model_condition)
    print(f"Four examples plot saved to Results_Images/{model_condition}/")
    
    print(f"\n[OK] VGG16 visualization completed! Check Results_Images/ for detailed analysis plots.")

if __name__ == '__main__':
    main()