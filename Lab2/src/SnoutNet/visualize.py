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
from model import SnoutNet

def load_snoutnet_model(model_path, device):
    """Load the trained SnoutNet model"""
    model = SnoutNet().to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"SnoutNet model loaded from {model_path}")
        model.eval()
        return model
    else:
        print(f"Model file {model_path} not found!")
        return None

def test_model_accuracy(model, test_loader, device):
    """Test model and return predictions and targets"""
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
    """Calculate comprehensive statistics"""
    euclidean_distances = np.sqrt(np.sum((predictions - targets)**2, axis=1))
    x_errors = predictions[:, 0] - targets[:, 0]
    y_errors = predictions[:, 1] - targets[:, 1]
    
    # Calculate comprehensive statistics
    stats = {
        'euclidean': {
            'min': np.min(euclidean_distances),
            'max': np.max(euclidean_distances),
            'mean': np.mean(euclidean_distances),
            'std': np.std(euclidean_distances),
            'median': np.median(euclidean_distances)
        },
        'x_error': {
            'min': np.min(np.abs(x_errors)),
            'max': np.max(np.abs(x_errors)),
            'mean': np.mean(np.abs(x_errors)),
            'std': np.std(x_errors),
            'median': np.median(np.abs(x_errors))
        },
        'y_error': {
            'min': np.min(np.abs(y_errors)),
            'max': np.max(np.abs(y_errors)),
            'mean': np.mean(np.abs(y_errors)),
            'std': np.std(y_errors),
            'median': np.median(np.abs(y_errors))
        }
    }
    
    return stats, euclidean_distances, x_errors, y_errors

def plot_error_distributions(euclidean_distances, x_errors, y_errors, predictions, targets, model_condition):
    """Plot comprehensive error analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Euclidean distance histogram
    axes[0, 0].hist(euclidean_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(euclidean_distances), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(euclidean_distances):.2f}')
    axes[0, 0].axvline(np.median(euclidean_distances), color='green', linestyle='--', 
                      label=f'Median: {np.median(euclidean_distances):.2f}')
    axes[0, 0].set_xlabel('Euclidean Distance (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'SnoutNet {model_condition} - Error Distribution')
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
    
    filename = f'snoutnet_visualization_error_analysis_{model_condition.lower()}.png'
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
    
    filename = f'snoutnet_visualization_best_worst_predictions_{model_condition.lower()}.png'
    plt.savefig(f'{results_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure instead of showing to prevent stalling during automation

def main():
    """Main visualization function for SnoutNet"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize SnoutNet model results')
    parser.add_argument('-t', '--type', choices=['baseline', 'augmented', 'auto'], 
                       default='auto', help='Model type to visualize (default: auto-detect)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Visualizing SnoutNet on device: {device}")
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
        model_paths = ['best_snoutnet_model_baseline.pth']
    elif args.type == 'augmented':
        model_paths = ['best_snoutnet_model_augmented.pth']
    else:  # auto-detect
        model_paths = [
            'best_snoutnet_model_baseline.pth',
            'best_snoutnet_model_augmented.pth'
        ]
    
    # Test available models and generate visualizations
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\n[VISUALIZATION] SnoutNet model: {model_path}")
            
            # Determine model condition from filename
            model_condition = "Baseline" if "baseline" in model_path else "Augmented"
            
            # Load model
            model = load_snoutnet_model(model_path, device)
            if model is None:
                continue
            
            print(f"Using model: {model_path}")
            
            # Get predictions
            predictions, targets = test_model_accuracy(model, test_loader, device)
            
            # Calculate statistics
            stats, euclidean_distances, x_errors, y_errors = calculate_statistics(predictions, targets)
            
            # Print basic statistics
            print(f"\n[STATISTICS] {model_condition} Model Performance:")
            print(f"  Euclidean Distance - Mean: {stats['euclidean']['mean']:.2f}px, Std: {stats['euclidean']['std']:.2f}px")
            print(f"  Min: {stats['euclidean']['min']:.2f}px, Max: {stats['euclidean']['max']:.2f}px")
            
            # Generate visualizations
            print(f"[VISUALIZING] Generating error analysis plots...")
            plot_error_distributions(euclidean_distances, x_errors, y_errors, predictions, targets, model_condition)
            
            print(f"[VISUALIZING] Generating best/worst prediction examples...")
            visualize_best_worst_predictions(model, test_dataset, device, euclidean_distances, model_condition)
            
            print(f"[SUCCESS] Visualizations saved to Results_Images/{model_condition}/")
        else:
            print(f"[SKIP] Model not found: {model_path}")
    
    print(f"\n[COMPLETE] SnoutNet visualization complete!")

if __name__ == "__main__":
    main()