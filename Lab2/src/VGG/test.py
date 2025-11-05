import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import sys

# Add parent directory to path
sys.path.append('..')
from dataset import CustomImageDataset
from model import VGG16Snout

def load_vgg16_model(model_path, device):
    """Load the trained VGG16 model"""
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
    
    stats = {
        'euclidean': {
            'mean': np.mean(euclidean_distances),
            'median': np.median(euclidean_distances),
            'std': np.std(euclidean_distances),
            'min': np.min(euclidean_distances),
            'max': np.max(euclidean_distances),
            'q25': np.percentile(euclidean_distances, 25),
            'q75': np.percentile(euclidean_distances, 75)
        },
        'x_error': {
            'mean': np.mean(np.abs(x_errors)),
            'std': np.std(x_errors),
            'bias': np.mean(x_errors)
        },
        'y_error': {
            'mean': np.mean(np.abs(y_errors)),
            'std': np.std(y_errors),
            'bias': np.mean(y_errors)
        },
        'mse': np.mean((predictions - targets)**2),
        'mae': np.mean(np.abs(predictions - targets))
    }
    
    return stats, euclidean_distances, x_errors, y_errors

def print_statistics(stats, model_condition):
    """Print comprehensive statistics"""
    print(f"\n[RESULTS] VGG16-SNOUTNET {model_condition.upper()} MODEL TESTING RESULTS")
    print("="*70)
    
    print(f"[TARGET] Euclidean Distance Statistics:")
    print(f"   Mean Error:   {stats['euclidean']['mean']:.4f} pixels")
    print(f"   Median Error: {stats['euclidean']['median']:.4f} pixels")
    print(f"   Std Error:    {stats['euclidean']['std']:.4f} pixels")
    print(f"   Min Error:    {stats['euclidean']['min']:.4f} pixels")
    print(f"   Max Error:    {stats['euclidean']['max']:.4f} pixels")
    print(f"   25th %ile:    {stats['euclidean']['q25']:.4f} pixels")
    print(f"   75th %ile:    {stats['euclidean']['q75']:.4f} pixels")
    
    print(f"\n[LOCATION] Component-wise Error Analysis:")
    print(f"   X-coordinate MAE: {stats['x_error']['mean']:.4f} pixels (bias: {stats['x_error']['bias']:.4f})")
    print(f"   Y-coordinate MAE: {stats['y_error']['mean']:.4f} pixels (bias: {stats['y_error']['bias']:.4f})")
    
    print(f"\n[CHART] Overall Metrics:")
    print(f"   MSE: {stats['mse']:.4f}")
    print(f"   MAE: {stats['mae']:.4f} pixels")

def calculate_accuracy_at_thresholds(euclidean_distances):
    """Calculate accuracy at different pixel thresholds"""
    thresholds = [5, 10, 15, 20, 25, 30, 40, 50]
    
    print(f"\n[TARGET] Accuracy at Different Thresholds:")
    for threshold in thresholds:
        accuracy = np.mean(euclidean_distances <= threshold) * 100
        print(f"   Within {threshold:2d} pixels: {accuracy:5.1f}%")

def main():
    """Main testing function for VGG16"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test VGG16-SnoutNet model')
    parser.add_argument('-t', '--type', choices=['baseline', 'augmented', 'auto'], 
                       default='auto', help='Model type to test (default: auto-detect)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing VGG16-SnoutNet on device: {device}")
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
    
    # Test the model
    predictions, targets = test_model_accuracy(model, test_loader, device)
    
    # Calculate statistics
    stats, euclidean_distances, x_errors, y_errors = calculate_statistics(predictions, targets)
    
    # Print results
    print_statistics(stats, model_condition)
    
    # Calculate accuracy at thresholds
    calculate_accuracy_at_thresholds(euclidean_distances)
    
    print(f"\n[OK] VGG16 testing completed!")

if __name__ == '__main__':
    main()