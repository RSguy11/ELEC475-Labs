import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.append('..')

from dataset import CustomImageDataset
from torch.utils.data import DataLoader
from model import EnsembleSnoutNet

def train_ensemble_model(combination_method='weighted', pretrained_paths=None, 
                        use_augmentation=False, num_epochs=50, batch_size=32, 
                        learning_rate=0.001, device=None):
    """
    Train the ensemble model using weighted combination strategy.
    
    Args:
        combination_method: 'weighted' (only supported method)
        pretrained_paths: Dictionary with paths to pretrained models
        use_augmentation: Whether to use data augmentation
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to use for training
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[MODEL INFO] Training Ensemble Model ({combination_method}):")
    print(f"  Device: {device}")
    print(f"  Augmentation: {use_augmentation}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create datasets
    train_dataset = CustomImageDataset(
        '../oxford-iiit-pet-noses/train_noses.txt',
        '../oxford-iiit-pet-noses/images-original/images/',
        augment=use_augmentation
    )
    
    test_dataset = CustomImageDataset(
        '../oxford-iiit-pet-noses/test_noses.txt',
        '../oxford-iiit-pet-noses/images-original/images/',
        augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train dataset size: {len(train_dataset)}")
    print(f"  Test dataset size: {len(test_dataset)}")
    
    # Initialize model
    model = EnsembleSnoutNet(combination_method=combination_method, pretrained_paths=pretrained_paths)
    model = model.to(device)
    
    # Print model info
    info = model.get_model_info()
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    print(f"  Base models frozen: {info['base_models_frozen']}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training tracking
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    print(f"\n[TRAINING] Starting ensemble training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_test_loss = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                epoch_test_loss += loss.item()
                num_test_batches += 1
        
        avg_test_loss = epoch_test_loss / num_test_batches
        test_losses.append(avg_test_loss)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_suffix = "_augmented" if use_augmentation else "_baseline"
            torch.save(model.state_dict(), f'best_ensemble_{combination_method}{model_suffix}.pth')
        
        print(f"  Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Print model weights for weighted ensemble
        if combination_method == 'weighted' and (epoch + 1) % 10 == 0:
            weights = model.get_model_weights()
            print(f"    Model weights: SnoutNet: {weights['snoutnet']:.3f}, "
                  f"AlexNet: {weights['alexnet']:.3f}, VGG16: {weights['vgg16']:.3f}")
    
    total_time = time.time() - start_time
    print(f"\n[OK] Training completed in {total_time/60:.1f} minutes")
    print(f"  Best test loss: {best_test_loss:.4f}")
    
    # Final model weights
    if combination_method == 'weighted':
        final_weights = model.get_model_weights()
        print(f"  Final model weights:")
        print(f"    SnoutNet: {final_weights['snoutnet']:.3f}")
        print(f"    AlexNet: {final_weights['alexnet']:.3f}")
        print(f"    VGG16: {final_weights['vgg16']:.3f}")
    
    # Plot training curves
    plot_training_curves(train_losses, test_losses, combination_method, use_augmentation)
    
    return model, train_losses, test_losses

def plot_training_curves(train_losses, test_losses, combination_method, use_augmentation):
    """Plot and save training curves"""
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Ensemble ({combination_method.title()}) Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss comparison (zoomed)
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    plt.plot(test_losses, label='Validation Loss', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training Progress (Last 20 Epochs)')
    if len(train_losses) > 20:
        plt.xlim(len(train_losses) - 20, len(train_losses))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to Results_Images folder
    folder_name = "Augmented" if use_augmentation else "Baseline"
    results_dir = f"Results_Images/{folder_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    aug_suffix = "_augmented" if use_augmentation else "_baseline"
    filename = f'ensemble_{combination_method}_training_curves{aug_suffix}.png'
    plt.savefig(f'{results_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure instead of showing to prevent stalling during automation

def find_pretrained_models():
    """Find available pretrained models"""
    pretrained_paths = {}
    
    # Look for SnoutNet models
    snoutnet_paths = ['../SnoutNet/best_snoutnet_model_baseline.pth', 
                     '../SnoutNet/best_snoutnet_model_augmented.pth']
    for path in snoutnet_paths:
        if os.path.exists(path):
            pretrained_paths['snoutnet'] = path
            break
    
    # Look for AlexNet models
    alexnet_paths = ['../AlexNet/best_alexnet_model_baseline.pth',
                    '../AlexNet/best_alexnet_model_augmented.pth']
    for path in alexnet_paths:
        if os.path.exists(path):
            pretrained_paths['alexnet'] = path
            break
    
    # Look for VGG16 models
    vgg_paths = ['../VGG/best_vgg16_model_baseline.pth',
                '../VGG/best_vgg16_model_augmented.pth']
    for path in vgg_paths:
        if os.path.exists(path):
            pretrained_paths['vgg16'] = path
            break
    
    return pretrained_paths

def main():
    """Main training function for Ensemble model"""
    
    parser = argparse.ArgumentParser(description='Train Ensemble SnoutNet Model')
    parser.add_argument('-a', '--augmentation', type=str, default='false',
                       help='Use data augmentation (true/false)')
    parser.add_argument('--method', type=str, default='weighted',
                       choices=['weighted'],
                       help='Ensemble combination method (only weighted available)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Fine-tune base models (unfreeze them)')
    
    args = parser.parse_args()
    
    # Parse augmentation argument
    use_augmentation = args.augmentation.lower() == 'true'
    
    print(f"[START] Ensemble SnoutNet Training Session")
    print(f"  Method: {args.method}")
    print(f"  Augmentation: {use_augmentation}")
    print(f"  Fine-tuning: {args.fine_tune}")
    
    # Find pretrained models
    pretrained_paths = find_pretrained_models()
    print(f"\n[INFO] Found pretrained models:")
    for model_name, path in pretrained_paths.items():
        print(f"  {model_name}: {path}")
    
    if not pretrained_paths:
        print("[WARNING] No pretrained models found. Training from scratch may not work well.")
        print("Please train individual models first using their respective training scripts.")
        return
    
    # Train model
    try:
        trained_model, train_losses, test_losses = train_ensemble_model(
            combination_method=args.method,
            pretrained_paths=pretrained_paths,
            use_augmentation=use_augmentation,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        
        # Optionally fine-tune base models
        if args.fine_tune:
            print(f"\n[FINE-TUNE] Starting fine-tuning phase...")
            trained_model.unfreeze_base_models()
            
            # Reduce learning rate for fine-tuning
            fine_tune_model, ft_train_losses, ft_test_losses = train_ensemble_model(
                combination_method=args.method,
                pretrained_paths=None,  # Don't reload weights
                use_augmentation=use_augmentation,
                num_epochs=args.epochs // 2,  # Fewer epochs for fine-tuning
                batch_size=args.batch_size,
                learning_rate=args.lr * 0.1  # Lower learning rate
            )
        
        # Save final results
        model_suffix = "_augmented" if use_augmentation else "_baseline"
        model_filename = f'best_ensemble_{args.method}{model_suffix}.pth'
        plot_filename = f'ensemble_{args.method}_training_curves{model_suffix}.png'
        
        print(f"\n[OK] Ensemble training completed!")
        print(f"  Method: {args.method}")
        print(f"  Final training loss: {train_losses[-1]:.4f}")
        print(f"  Final test loss: {test_losses[-1]:.4f}")
        print(f"[FILE] Model saved as: {model_filename}")
        print(f"[FILE] Training plot saved as: Results_Images/{plot_filename}")
        print(f"\n[OK] Ensemble training completed! Check the generated model files and plots.")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        return

if __name__ == "__main__":
    main()