import torch
import torch.nn as nn
import numpy as np
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
    predictions = np.array(predictions)
    targets_list = np.array(targets_list)
    
    return avg_loss, predictions, targets_list, individual_predictions

def calculate_ensemble_metrics(predictions, targets, individual_predictions, model_name):
    """Calculate comprehensive metrics for ensemble model"""
    
    # Calculate Euclidean distances for ensemble
    euclidean_distances = np.sqrt(np.sum((predictions - targets) ** 2, axis=1))
    
    # Calculate individual model distances
    individual_distances = {}
    for model_type, preds in individual_predictions.items():
        preds = np.array(preds)
        individual_distances[model_type] = np.sqrt(np.sum((preds - targets) ** 2, axis=1))
    
    print(f"[RESULTS] ENSEMBLE-SNOUTNET {model_name.upper()} MODEL TESTING RESULTS")
    print("=" * 80)
    
    # Ensemble metrics
    print(f"[TARGET] Ensemble Distance Statistics:")
    print(f"  Mean Euclidean Distance: {np.mean(euclidean_distances):.2f} pixels")
    print(f"  Median Euclidean Distance: {np.median(euclidean_distances):.2f} pixels")
    print(f"  Std Euclidean Distance: {np.std(euclidean_distances):.2f} pixels")
    print(f"  Min Distance: {np.min(euclidean_distances):.2f} pixels")
    print(f"  Max Distance: {np.max(euclidean_distances):.2f} pixels")
    
    # Individual model comparison
    print(f"\n[LOCATION] Individual Model Performance Comparison:")
    for model_type, distances in individual_distances.items():
        print(f"  {model_type.upper()}:")
        print(f"    Mean Distance: {np.mean(distances):.2f} pixels")
        print(f"    Median Distance: {np.median(distances):.2f} pixels")
    
    # Component-wise analysis
    x_mae = np.mean(np.abs(predictions[:, 0] - targets[:, 0]))
    y_mae = np.mean(np.abs(predictions[:, 1] - targets[:, 1]))
    
    print(f"\n[LOCATION] Component-wise Error Analysis:")
    print(f"  X-coordinate MAE: {x_mae:.2f} pixels")
    print(f"  Y-coordinate MAE: {y_mae:.2f} pixels")
    
    # Overall metrics
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    print(f"\n[CHART] Overall Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.2f} pixels")
    print(f"  MAE: {mae:.2f} pixels")
    
    # Accuracy at different thresholds
    thresholds = [5, 10, 15, 20, 25]
    print(f"\n[TARGET] Accuracy at Different Thresholds:")
    for threshold in thresholds:
        accuracy = np.mean(euclidean_distances <= threshold) * 100
        print(f"  Within {threshold} pixels: {accuracy:.1f}%")
    
    # Improvement analysis
    print(f"\n[CHART] Ensemble Improvement Analysis:")
    for model_type, distances in individual_distances.items():
        improvement = np.mean(distances) - np.mean(euclidean_distances)
        improvement_pct = (improvement / np.mean(distances)) * 100
        print(f"  vs {model_type.upper()}: {improvement:+.2f} pixels ({improvement_pct:+.1f}%)")
    
    return euclidean_distances, individual_distances

def main():
    """Main testing function for Ensemble model"""
    
    parser = argparse.ArgumentParser(description='Test Ensemble SnoutNet Model')
    parser.add_argument('-t', '--test_type', type=str, default='baseline',
                       choices=['baseline', 'augmented', 'auto'],
                       help='Type of model to test')
    parser.add_argument('--method', type=str, default='weighted',
                       choices=['weighted'],
                       help='Ensemble combination method (only weighted available)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[START] Ensemble SnoutNet Testing")
    print(f"  Device: {device}")
    print(f"  Method: {args.method}")
    print(f"  Test type: {args.test_type}")
    
    # Determine model paths to test
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
    
    # Test each configuration
    for test_type, model_path in test_configs:
        print(f"\n[TEST] Testing {test_type} model: {model_path}")
        
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
            
            # Evaluate model
            avg_loss, predictions, targets, individual_predictions = evaluate_ensemble_model(
                model, test_loader, device
            )
            
            print(f"  Average test loss: {avg_loss:.4f}")
            
            # Calculate metrics
            model_condition = f"Ensemble {args.method.title()} {test_type.title()}"
            ensemble_distances, individual_distances = calculate_ensemble_metrics(
                predictions, targets, individual_predictions, model_condition
            )
            
            print(f"\n[OK] Testing completed for {test_type} model!")
            
        except Exception as e:
            print(f"[ERROR] Testing failed for {test_type} model: {str(e)}")
    
    print(f"\n[OK] Ensemble testing completed!")

if __name__ == "__main__":
    main()