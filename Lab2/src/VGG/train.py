import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

# Add parent directory to path to import dataset
sys.path.append('..')
from dataset import CustomImageDataset
from model import VGG16Snout

def train_vgg16_model(use_augmentation=False, num_epochs=50, batch_size=16, learning_rate=0.0001):
    """
    Training script for VGG16-SnoutNet model.
    
    Args:
        use_augmentation (bool): Whether to apply data augmentation during training
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training VGG16-SnoutNet on device: {device}")
    
    # Hyperparameters
    print(f"Hyperparameters:")
    print(f"  Model: VGG16-SnoutNet")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Augmentation: {'Yes' if use_augmentation else 'No'}")
    
    # Create datasets
    print("Loading datasets...")
    
    train_dataset = CustomImageDataset(
        annotations_file="../oxford-iiit-pet-noses/train_noses.txt",
        img_dir='../oxford-iiit-pet-noses/images-original/images/',
        augment=use_augmentation
    )
    
    test_dataset = CustomImageDataset(
        annotations_file="../oxford-iiit-pet-noses/test_noses.txt",
        img_dir='../oxford-iiit-pet-noses/images-original/images/',
        augment=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = VGG16Snout(pretrained=True).to(device)
    
    # Print model info
    info = model.get_model_info()
    print(f"\n[MODEL INFO] Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    train_losses = []
    test_losses = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            if batch_idx % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_samples = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item() * images.size(0)
                test_samples += images.size(0)
        
        avg_test_loss = test_loss / test_samples
        test_losses.append(avg_test_loss)
        
        scheduler.step(avg_test_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        aug_suffix = "_augmented" if use_augmentation else "_baseline"
        model_filename = f'best_vgg16_model{aug_suffix}.pth'
        
        if epoch == 0 or avg_test_loss < min(test_losses[:-1]):
            torch.save(model.state_dict(), model_filename)
            print(f'New best model saved! Test Loss: {avg_test_loss:.4f}')
        
        print("-" * 60)
    
    # Plot training curves
    plot_training_curves(train_losses, test_losses, use_augmentation)
    
    # Evaluate final model performance
    evaluate_model(model, test_loader, device)
    
    print("Training completed!")
    return model, train_losses, test_losses

def plot_training_curves(train_losses, test_losses, use_augmentation=False):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss', linewidth=2)
    
    aug_suffix = " (With Augmentation)" if use_augmentation else " (Baseline)"
    plt.title(f'VGG16-SnoutNet Training Progress{aug_suffix}', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to Results_Images folder
    folder_name = "Augmented" if use_augmentation else "Baseline"
    results_dir = f"Results_Images/{folder_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    aug_suffix = "_augmented" if use_augmentation else "_baseline"
    filename = f'vgg16_training_curves{aug_suffix}.png'
    plt.savefig(f'{results_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure instead of showing to prevent stalling during automation

def evaluate_model(model, test_loader, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_samples = 0
    predictions = []
    targets_list = []
    
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    
    # Calculate additional metrics
    predictions = np.array(predictions)
    targets_list = np.array(targets_list)
    
    mae = np.mean(np.abs(predictions - targets_list))
    euclidean_errors = np.sqrt(np.sum((predictions - targets_list)**2, axis=1))
    mean_euclidean_error = np.mean(euclidean_errors)
    
    print(f"\nFinal VGG16 Model Evaluation:")
    print(f"Test MSE Loss: {avg_loss:.4f}")
    print(f"Mean Absolute Error: {mae:.4f} pixels")
    print(f"Mean Euclidean Error: {mean_euclidean_error:.4f} pixels")
    print(f"Max Euclidean Error: {np.max(euclidean_errors):.4f} pixels")
    print(f"Min Euclidean Error: {np.min(euclidean_errors):.4f} pixels")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train VGG16-SnoutNet model for pet nose localization')
    parser.add_argument('-a', '--augment', type=str, choices=['true', 'false'], default='false',
                       help='Enable data augmentation (true/false). Default: false')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs. Default: 50')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training. Default: 16')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate. Default: 0.0001')
    
    args = parser.parse_args()
    
    # Convert string to boolean
    use_augmentation = args.augment.lower() == 'true'
    
    print("="*80)
    print("VGG16-SNOUTNET TRAINING CONFIGURATION")
    print("="*80)
    print(f"Data Augmentation: {'ENABLED' if use_augmentation else 'DISABLED'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("="*80)
    
    # Train the model
    trained_model, train_losses, test_losses = train_vgg16_model(
        use_augmentation=use_augmentation,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print("\n[OK] VGG16 training completed!")
    
    aug_suffix = "_augmented" if use_augmentation else "_baseline"
    model_filename = f'best_vgg16_model{aug_suffix}.pth'
    plot_filename = f'vgg16_training_curves{aug_suffix}.png'
    
    print(f"[FILE] Model saved as: {model_filename}")
    print(f"[FILE] Training plot saved as: Results_Images/{plot_filename}")
    print(f"\n[OK] VGG16 training completed!")