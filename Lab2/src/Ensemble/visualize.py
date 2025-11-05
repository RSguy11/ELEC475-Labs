import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from PIL import Image
import random

# Add parent directory to path
sys.path.append('..')

from dataset import CustomImageDataset
from torch.utils.data import DataLoader
from model import EnsembleSnoutNet

def evaluate_ensemble_model(model, test_loader, device):
    """Evaluate ensemble model performance"""
    model.eval()
    total_loss = 0
    total_samples = 0
    predictions = []
    targets_list = []
    individual_predictions = {'snoutnet': [], 'alexnet': [], 'vgg16': []}
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            
            # Get ensemble prediction
            ensemble_outputs = model(images)
            
            # Get individual model predictions for comparison
            individual_preds = model.get_individual_predictions(images)
            
            loss = criterion(ensemble_outputs, targets)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            predictions.extend(ensemble_outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            
            # Store individual predictions
            individual_predictions['snoutnet'].extend(individual_preds['snoutnet'].cpu().numpy())
            individual_predictions['alexnet'].extend(individual_preds['alexnet'].cpu().numpy())
            individual_predictions['vgg16'].extend(individual_preds['vgg16'].cpu().numpy())
    
    avg_loss = total_loss / total_samples
    return avg_loss, np.array(predictions), np.array(targets_list), individual_predictions

def calculate_ensemble_metrics(predictions, targets, individual_predictions, model_condition):
    """Calculate comprehensive ensemble metrics"""
    # Ensemble metrics
    euclidean_distances = np.sqrt(np.sum((predictions - targets)**2, axis=1))
    
    print(f"\n[METRICS] {model_condition} Performance:")
    print(f"  Mean distance: {np.mean(euclidean_distances):.3f} ± {np.std(euclidean_distances):.3f} pixels")
    print(f"  Median distance: {np.median(euclidean_distances):.3f} pixels")
    print(f"  95th percentile: {np.percentile(euclidean_distances, 95):.3f} pixels")
    
    # Calculate individual model metrics
    individual_distances = {}
    for model_type, preds in individual_predictions.items():
        preds = np.array(preds)
        distances = np.sqrt(np.sum((preds - targets)**2, axis=1))
        individual_distances[model_type] = distances
        
        print(f"  {model_type.upper()} mean: {np.mean(distances):.3f} pixels")
    
    # Calculate improvements
    print(f"\n[IMPROVEMENT] Ensemble vs Individual Models:")
    for model_type, distances in individual_distances.items():
        improvement = np.mean(distances) - np.mean(euclidean_distances)
        improvement_pct = (improvement / np.mean(distances)) * 100
        print(f"  vs {model_type.upper()}: {improvement:+.2f} pixels ({improvement_pct:+.1f}%)")
    
    return euclidean_distances, individual_distances

def visualize_ensemble_comparison(model, test_dataset, device, ensemble_distances, 
                                individual_distances, model_condition, num_samples=6):
    """Visualize ensemble vs individual model predictions"""
    
    # Get indices of best and worst ensemble predictions
    sorted_indices = np.argsort(ensemble_distances)
    best_indices = sorted_indices[:num_samples//2]
    worst_indices = sorted_indices[-num_samples//2:]
    all_indices = np.concatenate([best_indices, worst_indices])
    
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 8))
    axes = axes.flatten()
    
    model.eval()
    
    for i, idx in enumerate(all_indices):
        image, target = test_dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            ensemble_pred = model(image_tensor).cpu().numpy()[0]
            individual_preds = model.get_individual_predictions(image_tensor)
            individual_preds = {k: v.cpu().numpy()[0] for k, v in individual_preds.items()}
        
        # Convert image for display
        if image.shape[0] == 3:  # RGB
            display_image = image.permute(1, 2, 0).numpy()
            display_image = np.clip(display_image, 0, 1)
        
        # Plot
        axes[i].imshow(display_image)
        axes[i].scatter(target[0], target[1], c='red', s=100, marker='x', linewidth=3, label='Ground Truth')
        axes[i].scatter(ensemble_pred[0], ensemble_pred[1], c='gold', s=80, marker='o', label='Ensemble')
        axes[i].scatter(individual_preds['snoutnet'][0], individual_preds['snoutnet'][1], 
                       c='blue', s=60, marker='^', alpha=0.7, label='SnoutNet')
        axes[i].scatter(individual_preds['alexnet'][0], individual_preds['alexnet'][1], 
                       c='green', s=60, marker='s', alpha=0.7, label='AlexNet')
        axes[i].scatter(individual_preds['vgg16'][0], individual_preds['vgg16'][1], 
                       c='purple', s=60, marker='D', alpha=0.7, label='VGG16')
        
        ensemble_dist = ensemble_distances[idx]
        category = "Best" if i < num_samples//2 else "Worst"
        axes[i].set_title(f'{category} #{i%3+1}\nEnsemble: {ensemble_dist:.1f}px')
        axes[i].axis('off')
        
        if i == 0:
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.suptitle(f'Ensemble vs Individual Model Predictions\n{model_condition} Model', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    folder_name = "Augmented" if "augmented" in model_condition.lower() else "Baseline"
    results_dir = f"Results_Images/{folder_name}"
    
    filename = f'ensemble_comparison_{model_condition.lower()}.png'
    plt.savefig(f'{results_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure instead of showing to prevent stalling during automation

def visualize_ensemble_error_analysis(ensemble_distances, individual_distances, model_condition):
    """Create comprehensive error analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distance comparison histogram
    axes[0, 0].hist(ensemble_distances, bins=30, alpha=0.7, label='Ensemble', color='gold')
    for model_type, distances in individual_distances.items():
        axes[0, 0].hist(distances, bins=30, alpha=0.5, label=model_type.title())
    axes[0, 0].set_xlabel('Euclidean Distance (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distance Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    all_distances = [ensemble_distances] + list(individual_distances.values())
    labels = ['Ensemble'] + [m.title() for m in individual_distances.keys()]
    axes[0, 1].boxplot(all_distances, labels=labels)
    axes[0, 1].set_ylabel('Euclidean Distance (pixels)')
    axes[0, 1].set_title('Distance Distribution Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Improvement scatter plot
    axes[1, 0].scatter(individual_distances['snoutnet'], ensemble_distances, 
                      alpha=0.6, s=20, label='vs SnoutNet')
    axes[1, 0].scatter(individual_distances['alexnet'], ensemble_distances, 
                      alpha=0.6, s=20, label='vs AlexNet')
    axes[1, 0].scatter(individual_distances['vgg16'], ensemble_distances, 
                      alpha=0.6, s=20, label='vs VGG16')
    
    # Add diagonal line
    max_dist = max(np.max(ensemble_distances), 
                   max(np.max(d) for d in individual_distances.values()))
    axes[1, 0].plot([0, max_dist], [0, max_dist], 'k--', alpha=0.5, label='No improvement')
    axes[1, 0].set_xlabel('Individual Model Distance (pixels)')
    axes[1, 0].set_ylabel('Ensemble Distance (pixels)')
    axes[1, 0].set_title('Ensemble vs Individual Model Performance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    thresholds = np.arange(1, 31)
    ensemble_acc = [np.mean(ensemble_distances <= t) * 100 for t in thresholds]
    
    axes[1, 1].plot(thresholds, ensemble_acc, linewidth=3, label='Ensemble', color='gold')
    for model_type, distances in individual_distances.items():
        acc = [np.mean(distances <= t) * 100 for t in thresholds]
        axes[1, 1].plot(thresholds, acc, linewidth=2, label=model_type.title(), alpha=0.8)
    
    axes[1, 1].set_xlabel('Distance Threshold (pixels)')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Accuracy vs Threshold Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Ensemble Model Error Analysis - {model_condition}', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    folder_name = "Augmented" if "augmented" in model_condition.lower() else "Baseline"
    results_dir = f"Results_Images/{folder_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f'ensemble_error_analysis_{model_condition.lower()}.png'
    plt.savefig(f'{results_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure instead of showing to prevent stalling during automation

def main():
    """Main visualization function for Ensemble model"""
    
    parser = argparse.ArgumentParser(description='Visualize Ensemble SnoutNet Model Results')
    parser.add_argument('-t', '--test_type', type=str, default='baseline',
                       choices=['baseline', 'augmented', 'auto'],
                       help='Type of model to visualize')
    parser.add_argument('--method', type=str, default='weighted',
                       choices=['weighted'],
                       help='Ensemble combination method (only weighted available)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for prediction generation')
    parser.add_argument('--error-analysis', action='store_true', 
                       help='Generate comprehensive error analysis plots')
    parser.add_argument('--comparison', action='store_true', 
                       help='Generate ensemble vs individual model comparison')
    parser.add_argument('--all', action='store_true', 
                       help='Generate all visualization types')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[START] Ensemble SnoutNet Visualization")
    print(f"  Device: {device}")
    print(f"  Method: {args.method}")
    print(f"  Test type: {args.test_type}")
    
    # Determine model paths to visualize
    test_configs = []
    if args.test_type == 'auto':
        # Test both baseline and augmented if available
        for test_type in ['baseline', 'augmented']:
            model_path = f'best_ensemble_{args.method}_{test_type}.pth'
            if os.path.exists(model_path):
                test_configs.append((test_type, model_path))
    else:
        model_path = f'best_ensemble_{args.method}_{args.test_type}.pth'
        if os.path.exists(model_path):
            test_configs.append((args.test_type, model_path))
    
    if not test_configs:
        print(f"[ERROR] No trained ensemble models found for method '{args.method}'")
        print("Please train the ensemble model first using train.py")
        return
    
    # Create test dataset
    test_dataset = CustomImageDataset(
        '../oxford-iiit-pet-noses/test_noses.txt',
        '../oxford-iiit-pet-noses/images-original/images/',
        augment=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  Test dataset size: {len(test_dataset)}")
    
    # Visualize each configuration
    for test_type, model_path in test_configs:
        print(f"\n[VISUALIZE] Processing {test_type} model: {model_path}")
        
        try:
            # Find pretrained individual models
            pretrained_paths = {}
            for model_name in ['snoutnet', 'alexnet', 'vgg16']:
                for suffix in ['_baseline', '_augmented']:
                    path = f'../{model_name.title() if model_name != "vgg16" else "VGG"}/best_{model_name}_model{suffix}.pth'
                    if os.path.exists(path):
                        pretrained_paths[model_name] = path
                        break
            
            # Load ensemble model
            model = EnsembleSnoutNet(combination_method=args.method, pretrained_paths=pretrained_paths)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            
            # Print model weights for weighted ensemble
            if args.method == 'weighted':
                weights = model.get_model_weights()
                print(f"  Learned model weights:")
                print(f"    SnoutNet: {weights['snoutnet']:.3f}")
                print(f"    AlexNet: {weights['alexnet']:.3f}")
                print(f"    VGG16: {weights['vgg16']:.3f}")
            
            # Evaluate model to get predictions
            print("Getting model predictions...")
            avg_loss, predictions, targets, individual_predictions = evaluate_ensemble_model(
                model, test_loader, device
            )
            
            # Calculate metrics
            model_condition = f"Ensemble {args.method.title()} {test_type.title()}"
            ensemble_distances, individual_distances = calculate_ensemble_metrics(
                predictions, targets, individual_predictions, model_condition
            )
            
            # Generate visualizations based on arguments
            if args.all or args.error_analysis or (not args.comparison and not args.error_analysis):
                print("Generating error analysis plots...")
                visualize_ensemble_error_analysis(ensemble_distances, individual_distances, model_condition)
                print(f"Error analysis plot saved to Results_Images/{test_type.title()}/")
            
            if args.all or args.comparison:
                print("Generating ensemble vs individual comparison...")
                visualize_ensemble_comparison(model, test_dataset, device, ensemble_distances, 
                                            individual_distances, model_condition)
                print(f"Comparison plot saved to Results_Images/{test_type.title()}/")
            
            print(f"\n[OK] Visualization completed for {test_type} model!")
            
        except Exception as e:
            print(f"[ERROR] Visualization failed for {test_type} model: {str(e)}")
    
    print(f"\n[OK] Ensemble visualization completed! Check Results_Images/ for detailed analysis plots.")

if __name__ == "__main__":
    main()