
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
import sys

# Add parent directories to path
sys.path.append('../2_2_Custom_SMNet')
sys.path.append('../2_1_Evaluate_Model')
from model import SMNet
from step1_local_voc import LocalVOCDataset, squeeze_and_long

class LocalVOCDataset(Dataset):
    """Local PASCAL VOC 2012 dataset."""
    
    def __init__(self, voc_root, split='val', transform=None, target_transform=None, max_samples=None):
        self.voc_root = voc_root
        self.transform = transform
        self.target_transform = target_transform
        
        print(f"Initializing dataset with root: {voc_root}")
        
        # Paths
        self.images_dir = os.path.join(voc_root, 'JPEGImages')
        self.masks_dir = os.path.join(voc_root, 'SegmentationClass')
        split_file = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{split}.txt')
        
        # Check if paths exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # Load image IDs
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # Limit samples if specified
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]
        
        print(f"Loaded {len(self.image_ids)} images from {split} set")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image and mask
        img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
        mask_path = os.path.join(self.masks_dir, f'{img_id}.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask

def squeeze_and_long(x):
    """Transform function to squeeze and convert to long - needed for multiprocessing."""
    return x.squeeze(0).long()

def cosine_feature_loss(student_features, teacher_features):
    """Calculate cosine similarity loss between feature maps."""
    # Ensure features are valid and have the same batch size
    if student_features.shape[0] != teacher_features.shape[0]:
        raise ValueError(f"Batch size mismatch: student {student_features.shape[0]} vs teacher {teacher_features.shape[0]}")
    
    # Flatten feature maps
    student_flat = student_features.view(student_features.size(0), -1)
    teacher_flat = teacher_features.view(teacher_features.size(0), -1)
    
    # Make sure both have same number of features by interpolating if needed
    if student_flat.shape[1] != teacher_flat.shape[1]:
        # Reshape to match the smaller one
        min_features = min(student_flat.shape[1], teacher_flat.shape[1])
        if student_flat.shape[1] > min_features:
            student_flat = student_flat[:, :min_features]
        if teacher_flat.shape[1] > min_features:
            teacher_flat = teacher_flat[:, :min_features]
    
    # Check for NaN or infinite values
    if torch.isnan(student_flat).any() or torch.isinf(student_flat).any():
        print("Warning: NaN or Inf in student features, replacing with zeros")
        student_flat = torch.where(torch.isnan(student_flat) | torch.isinf(student_flat), 
                                   torch.zeros_like(student_flat), student_flat)
    
    if torch.isnan(teacher_flat).any() or torch.isinf(teacher_flat).any():
        print("Warning: NaN or Inf in teacher features, replacing with zeros")
        teacher_flat = torch.where(torch.isnan(teacher_flat) | torch.isinf(teacher_flat), 
                                   torch.zeros_like(teacher_flat), teacher_flat)
    
    # Normalize features with epsilon for numerical stability
    eps = 1e-8
    student_norm = F.normalize(student_flat + eps, p=2, dim=1)
    teacher_norm = F.normalize(teacher_flat + eps, p=2, dim=1)
    
    # Cosine similarity loss (1 - cosine_similarity)
    cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=1)
    loss = 1.0 - cosine_sim.mean()
    
    return loss

def calculate_miou(predictions, targets, num_classes=21):
    """Calculate mean Intersection over Union."""
    iou_per_class = []
    for class_id in range(num_classes):
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        if union > 0:
            iou_per_class.append(intersection.float() / union.float())
    return torch.stack(iou_per_class).mean() if iou_per_class else torch.tensor(0.0)

def evaluate_model(model, dataloader, device, model_name):
    """Evaluate model performance: mIoU, inference speed, parameters."""
    model.eval()
    total_miou = 0
    inference_times = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            # Handle different model output formats
            if isinstance(outputs, dict):
                outputs = outputs['out']  # FCN-ResNet50 format
            # SMNet returns tensor directly
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate mIoU
            predictions = outputs.argmax(dim=1)
            miou = calculate_miou(predictions, targets)
            total_miou += miou
    
    avg_miou = total_miou / len(dataloader)
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"\n{model_name} Performance:")
    print(f"  mIoU: {avg_miou:.4f}")
    print(f"  Inference Speed: {avg_inference_time:.2f} ms/batch")
    print(f"  Parameters: {param_count:,}")
    
    return avg_miou, avg_inference_time, param_count

def dataLoader():
    # Dataset path - works from 2_4_Distillation directory
    voc_root = r"../pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val"
    
    # Check if dataset exists
    if not os.path.exists(voc_root):
        print(f"Dataset not found at: {voc_root}")
        print("Current directory:", os.getcwd())
        print("Looking for dataset...")
        
        # Try alternative paths
        alt_paths = [
            r"../pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val",
            r"./pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val",
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                voc_root = alt_path
                print(f"Found dataset at: {alt_path}")
                break
        else:
            print("Dataset not found in any expected location!")
            print("Please ensure the dataset is in the Lab3 directory")
            return
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((520, 520), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        squeeze_and_long  # Use named function instead of lambda
    ])
    
    # Load dataset
    try:
        print(f"Using dataset path: {voc_root}")
        dataset = LocalVOCDataset(
            voc_root=voc_root,
            split='train',  # Use training data for knowledge distillation
            transform=transform,
            target_transform=target_transform,
            max_samples=None  # Full dataset now with lighter base16 model
        )
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
        print("Please check that the dataset is properly extracted")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Dataset path:", voc_root)
        print("Current directory:", os.getcwd())
        return
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)  # Back to batch_size=4 for full dataset
    return dataloader
    
def load_smnet_model(base_dim, device):
    """Load the trained SMNet model with error handling for architecture mismatches."""
    model = SMNet(num_classes=21, base_dim=base_dim).to(device)
    
    # Look for model in 2_3_train_test_model directory
    model_path = f'../2_3_train_test_model/best_smnet_model_base{base_dim}.pth'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found! Starting with random weights.")
        model.train()  # Set to training mode for distillation
        return model
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"SMNet model loaded from {model_path}")
    except RuntimeError as e:
        print(f"Warning: Cannot load weights from {model_path} due to architecture mismatch:")
        print(f"Error: {str(e)[:200]}...")
        print("Starting with randomly initialized weights instead.")
    
    model.train()  # Set to training mode for distillation
    return model

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Load a checkpoint and return the epoch, loss, and loss history.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Student model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load the checkpoint on
    
    Returns:
        tuple: (start_epoch, loss, loss_history)
    """
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        loss_history = checkpoint.get('loss_history', {'total': [], 'response': [], 'hard': [], 'feature': []})
        
        print(f"Loaded checkpoint from epoch {start_epoch} with loss {loss:.4f}")
        return start_epoch, loss, loss_history
    else:
        print(f"No checkpoint found at: {checkpoint_path}")
        return 0, float('inf'), {'total': [], 'response': [], 'hard': [], 'feature': []}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("Loading FCN-ResNet50...")
    teacher_model = fcn_resnet50(weights='DEFAULT').to(device)
    teacher_model.eval()
    print(f"Teacher model loaded: {sum(p.numel() for p in teacher_model.parameters()):,} parameters")

    dataloader = dataLoader()
    print("Dataset loaded successfully!")

    print("Loading SMNet student model...")
    student_model = load_smnet_model(base_dim=16, device=device)

    # Setup optimizer for distillation training
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    
    # Checkpoint saving setup
    import os
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for existing checkpoint to resume training
    resume_from_checkpoint = True  # Set to False to start fresh
    latest_checkpoint = os.path.join(checkpoint_dir, 'smnet_distilled_latest.pth')
    
    if resume_from_checkpoint and os.path.exists(latest_checkpoint):
        start_epoch, best_loss, loss_history = load_checkpoint(latest_checkpoint, student_model, optimizer, device)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_loss = float('inf')
        loss_history = {'total': [], 'response': [], 'hard': [], 'feature': []}
        print("Starting fresh training")
    
    print("Starting knowledge distillation training...")
    print(f"Best and latest checkpoints will be saved in '{checkpoint_dir}/' directory")
    
    # Train for 30 epochs for thorough knowledge distillation
    num_epochs = 30
    for epoch in range(start_epoch, num_epochs):
        epoch_losses = {'total': [], 'response': [], 'hard': [], 'feature': []}
        
        # Create progress bar for this epoch
        epoch_pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                         unit='batch', leave=True)
        
        for batch_idx, (images, targets) in enumerate(epoch_pbar):
            images, targets = images.to(device), targets.to(device)
            
            # Forward through student model (SMNet returns tensor directly)
            student_outputs = student_model(images)
            
            # Forward through teacher model (frozen)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)['out']
            
            # Simplified feature-based distillation - use outputs as features
            student_features = student_outputs
            teacher_features = teacher_outputs
            
            # Distillation loss parameters
            alpha = 0.5      # Weight for response-based distillation
            beta = 0.3       # Weight for hard target loss  
            gamma = 0.2      # Weight for feature-based distillation
            tau = 4.0        # Temperature for softmax
            
            # Clean targets - replace ignore_index (255) with 0 for loss computation
            clean_targets = targets.clone()
            clean_targets[clean_targets == 255] = 0  # Replace ignore_index with background class
            
            # Response-based distillation loss: L_response = α*H(σ(zs;τ), σ(zt;τ))
            student_soft = F.log_softmax(student_outputs / tau, dim=1)
            response_loss = F.kl_div(student_soft, F.softmax(teacher_outputs / tau, dim=1), reduction='batchmean')
            
            # Hard target loss: L_hard = β*H(σ(zs;1), y) with ignore_index
            student_hard = F.log_softmax(student_outputs, dim=1)
            hard_loss = F.nll_loss(student_hard, targets, ignore_index=255)
            
            # Feature-based distillation loss: L_feature = γ*cosine_loss(f_s, f_t)
            try:
                feature_loss = cosine_feature_loss(student_features, teacher_features)
            except Exception as e:
                print(f"Warning: Feature loss computation failed: {e}")
                feature_loss = torch.tensor(0.0, device=device)
            
            # Combined loss: L_total = α*L_response*(τ²) + β*L_hard + γ*L_feature
            total_loss = alpha * response_loss * (tau ** 2) + beta * hard_loss + gamma * feature_loss
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Store losses for plotting
            epoch_losses['response'].append(response_loss.item())
            epoch_losses['hard'].append(hard_loss.item())
            epoch_losses['feature'].append(feature_loss.item())
            epoch_losses['total'].append(total_loss.item())
            
            # Update progress bar with current losses
            epoch_pbar.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Response': f'{response_loss.item():.4f}',
                'Hard': f'{hard_loss.item():.4f}',
                'Feature': f'{feature_loss.item():.4f}'
            })
        
        # Close progress bar for this epoch
        epoch_pbar.close()
        
        # Print epoch summary
        if epoch % 5 == 0 or epoch == num_epochs - 1:  # Print every 5 epochs
            avg_total_loss = np.mean(epoch_losses['total'])
            avg_response_loss = np.mean(epoch_losses['response'])
            avg_hard_loss = np.mean(epoch_losses['hard'])
            avg_feature_loss = np.mean(epoch_losses['feature'])
            
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Avg Total Loss: {avg_total_loss:.4f}")
            print(f"  Avg Response Loss: {avg_response_loss:.4f}")
            print(f"  Avg Hard Loss: {avg_hard_loss:.4f}")
            print(f"  Avg Feature Loss: {avg_feature_loss:.4f}")
            print("-" * 50)
        
        # Store epoch averages for final plotting
        epoch_avg_loss = np.mean(epoch_losses['total'])
        loss_history['total'].append(epoch_avg_loss)
        loss_history['response'].append(np.mean(epoch_losses['response']))
        loss_history['hard'].append(np.mean(epoch_losses['hard']))
        loss_history['feature'].append(np.mean(epoch_losses['feature']))
        
        # Save checkpoints
        # 1. Save best model based on loss
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_model_path = os.path.join(checkpoint_dir, 'smnet_distilled_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'loss_history': loss_history
            }, best_model_path)
            print(f"New best model saved (epoch {epoch+1}, loss: {best_loss:.4f}): {best_model_path}")
        
        # 2. Save latest model (for resuming training) - overwrite each time
        latest_model_path = os.path.join(checkpoint_dir, 'smnet_distilled_latest.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_avg_loss,
            'loss_history': loss_history
        }, latest_model_path)
    
    # Save final distilled model (copy from best checkpoint)
    distilled_model_path = f'smnet_distilled_base{16}.pth'
    best_model_path = os.path.join(checkpoint_dir, 'smnet_distilled_best.pth')
    
    if os.path.exists(best_model_path):
        # Load best model state
        best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        torch.save(best_checkpoint['model_state_dict'], distilled_model_path)
        print(f"Final distilled model saved as: {distilled_model_path} (from best checkpoint at epoch {best_checkpoint['epoch']} with loss {best_checkpoint['loss']:.4f})")
    else:
        # Fallback to current model state
        torch.save(student_model.state_dict(), distilled_model_path)
        print(f"Final distilled model saved as: {distilled_model_path} (current state)")
    
    print(f"\nTraining completed!")
    print(f"Checkpoints saved in: {checkpoint_dir}/")
    print(f"- Latest: smnet_distilled_latest.pth (for resuming training)")
    print(f"- Best: smnet_distilled_best.pth (lowest loss model)")
    
    # Evaluate all models
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*50)
    
    # Evaluate teacher model
    evaluate_model(teacher_model, dataloader, device, "FCN-ResNet50 (Teacher)")
    
    # Evaluate original student model
    original_student = load_smnet_model(base_dim=16, device=device)
    original_student.eval()
    evaluate_model(original_student, dataloader, device, "SMNet Original (Student)")
    
    # Evaluate distilled student model
    evaluate_model(student_model, dataloader, device, "SMNet Distilled (Student)")
    
    print("="*50)
    
    # Create image_results folder
    if not os.path.exists('image_results'):
        os.makedirs('image_results')
        print("Created image_results folder")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot loss curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(loss_history['total'], 'b-', label='Total Loss')
    plt.title('Total Distillation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(loss_history['response'], 'r-', label='Response Loss')
    plt.plot(loss_history['hard'], 'g-', label='Hard Target Loss')
    plt.plot(loss_history['feature'], 'm-', label='Feature Loss')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    epochs = range(1, len(loss_history['total']) + 1)
    total = np.array(loss_history['total'])
    plt.plot(epochs, np.array(loss_history['response'])/total, 'r-', label='Response %')
    plt.plot(epochs, np.array(loss_history['hard'])/total, 'g-', label='Hard %')
    plt.plot(epochs, np.array(loss_history['feature'])/total, 'm-', label='Feature %')
    plt.title('Loss Ratios')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('image_results/distillation_losses.png', dpi=150, bbox_inches='tight')
    print("Loss plots saved as 'image_results/distillation_losses.png'")
    
    # Generate prediction comparisons on 4 sample images
    print("Generating prediction comparisons...")
    sample_images, sample_targets = next(iter(dataloader))
    sample_images, sample_targets = sample_images.to(device), sample_targets.to(device)
    
    with torch.no_grad():
        teacher_preds = teacher_model(sample_images[:4])['out'].argmax(dim=1)
        original_student.eval()
        student_orig_preds = original_student(sample_images[:4]).argmax(dim=1)  # SMNet returns tensor directly
        student_model.eval()
        student_dist_preds = student_model(sample_images[:4]).argmax(dim=1)  # SMNet returns tensor directly
    
    # Create comparison visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for i in range(4):
        # Denormalize image
        img = sample_images[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        axes[i, 0].imshow(img.permute(1, 2, 0))
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sample_targets[i].cpu(), cmap='tab20')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(teacher_preds[i].cpu(), cmap='tab20')
        axes[i, 2].set_title('Teacher (FCN)')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(student_orig_preds[i].cpu(), cmap='tab20')
        axes[i, 3].set_title('Student Original')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(student_dist_preds[i].cpu(), cmap='tab20')
        axes[i, 4].set_title('Student Distilled')
        axes[i, 4].axis('off')
    
    plt.suptitle('Knowledge Distillation: Prediction Comparisons')
    plt.tight_layout()
    plt.savefig('image_results/distillation_predictions.png', dpi=150, bbox_inches='tight')
    print("Prediction comparisons saved as 'image_results/distillation_predictions.png'")
    
    print("\nVisualization complete!")








if __name__ == '__main__':
    main()