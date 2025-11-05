import torch
import numpy as np
import sys
import os

# Simple script to get just the 4 best and 4 worst values
def get_best_worst_from_model(model_path, model_class):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add parent directory to path for dataset
    sys.path.append('..')
    from dataset import CustomImageDataset
    from torch.utils.data import DataLoader
    
    # Load model
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test dataset
    test_dataset = CustomImageDataset(
        annotations_file="../oxford-iiit-pet-noses/test_noses.txt",
        img_dir='../oxford-iiit-pet-noses/images-original/images/'
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Get predictions
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, target_coords in test_loader:
            images = images.to(device)
            target_coords = target_coords.to(device)
            
            pred_coords = model(images)
            
            predictions.extend(pred_coords.cpu().numpy())
            targets.extend(target_coords.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate euclidean distances
    euclidean_distances = np.sqrt(np.sum((predictions - targets)**2, axis=1))
    
    # Get 4 best and 4 worst indices
    sorted_indices = np.argsort(euclidean_distances)
    best_4_indices = sorted_indices[:4]
    worst_4_indices = sorted_indices[-4:]
    
    # Extract the actual error values
    best_4_errors = euclidean_distances[best_4_indices]
    worst_4_errors = euclidean_distances[worst_4_indices]
    
    # Calculate statistics
    best_4_stats = {
        'min': np.min(best_4_errors),
        'max': np.max(best_4_errors),
        'mean': np.mean(best_4_errors),
        'std': np.std(best_4_errors)
    }
    
    worst_4_stats = {
        'min': np.min(worst_4_errors),
        'max': np.max(worst_4_errors),
        'mean': np.mean(worst_4_errors),
        'std': np.std(worst_4_errors)
    }
    
    overall_stats = {
        'min': np.min(euclidean_distances),
        'max': np.max(euclidean_distances),
        'mean': np.mean(euclidean_distances),
        'std': np.std(euclidean_distances)
    }
    
    return overall_stats, best_4_stats, worst_4_stats, best_4_errors, worst_4_errors

# Test with SnoutNet
if __name__ == "__main__":
    from model import SnoutNet
    
    print("SnoutNet Baseline:")
    overall, best_4, worst_4, best_vals, worst_vals = get_best_worst_from_model(
        "best_snoutnet_model_baseline.pth", SnoutNet)
    
    print(f"Overall - min: {overall['min']:.2f}, max: {overall['max']:.2f}, mean: {overall['mean']:.2f}, std: {overall['std']:.2f}")
    print(f"Best 4 - min: {best_4['min']:.2f}, max: {best_4['max']:.2f}, mean: {best_4['mean']:.2f}, std: {best_4['std']:.2f}")
    print(f"Worst 4 - min: {worst_4['min']:.2f}, max: {worst_4['max']:.2f}, mean: {worst_4['mean']:.2f}, std: {worst_4['std']:.2f}")
    print(f"Best 4 values: {[f'{v:.2f}' for v in best_vals]}")
    print(f"Worst 4 values: {[f'{v:.2f}' for v in worst_vals]}")
    
    print("\nSnoutNet Augmented:")
    overall, best_4, worst_4, best_vals, worst_vals = get_best_worst_from_model(
        "best_snoutnet_model_augmented.pth", SnoutNet)
    
    print(f"Overall - min: {overall['min']:.2f}, max: {overall['max']:.2f}, mean: {overall['mean']:.2f}, std: {overall['std']:.2f}")
    print(f"Best 4 - min: {best_4['min']:.2f}, max: {best_4['max']:.2f}, mean: {best_4['mean']:.2f}, std: {best_4['std']:.2f}")
    print(f"Worst 4 - min: {worst_4['min']:.2f}, max: {worst_4['max']:.2f}, mean: {worst_4['mean']:.2f}, std: {worst_4['std']:.2f}")
    print(f"Best 4 values: {[f'{v:.2f}' for v in best_vals]}")
    print(f"Worst 4 values: {[f'{v:.2f}' for v in worst_vals]}")