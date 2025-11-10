"""
ELEC475 Lab 3 - Overnight Knowledge Distillation Training Suite
Comprehensive training with early stopping and convergence detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models.segmentation as segmentation
import numpy as np
import os
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add parent directories to path
sys.path.append('../2_2_Custom_SMNet')
sys.path.append('../2_1_Evaluate_Model')
from model import SMNet
from step1_local_voc import LocalVOCDataset, squeeze_and_long

class EarlyStopping:
    """Early stopping to stop training when validation loss plateaus."""
    
    def __init__(self, patience=10, min_delta=0.0001, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class ConvergenceMonitor:
    """Monitor convergence based on mIoU improvement."""
    
    def __init__(self, patience=15, min_improvement=0.001):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_miou = 0.0
        self.counter = 0
        self.history = []
        
    def __call__(self, val_miou):
        self.history.append(val_miou)
        
        if val_miou > self.best_miou + self.min_improvement:
            self.best_miou = val_miou
            self.counter = 0
        else:
            self.counter += 1
            
        # Check for convergence
        if self.counter >= self.patience:
            return True
            
        # Also check if we've plateaued for the last 10 epochs
        if len(self.history) >= 10:
            recent_variance = np.var(self.history[-10:])
            if recent_variance < 0.0001:  # Very small variance = plateau
                return True
                
        return False

class KnowledgeDistillationLoss(nn.Module):
    """Advanced Knowledge Distillation Loss with multiple components."""
    
    def __init__(self, alpha=0.3, beta=0.4, gamma=0.3, temperature=4.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.hard_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        self.feature_adapter = None
        
    def forward(self, student_logits, teacher_logits, teacher_features, student_features, targets):
        # Hard target loss
        hard_loss = self.hard_loss(student_logits, targets)
        
        # Soft target loss with temperature scaling
        teacher_soft = torch.softmax(teacher_logits / self.temperature, dim=1)
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Feature distillation loss
        if teacher_features.shape != student_features.shape:
            # Spatial alignment
            teacher_h, teacher_w = teacher_features.shape[2:]
            student_features = torch.nn.functional.interpolate(
                student_features, size=(teacher_h, teacher_w), 
                mode='bilinear', align_corners=False
            )
            
            # Channel alignment
            if teacher_features.shape[1] != student_features.shape[1]:
                if self.feature_adapter is None:
                    self.feature_adapter = nn.Sequential(
                        nn.Conv2d(student_features.shape[1], teacher_features.shape[1], 1),
                        nn.BatchNorm2d(teacher_features.shape[1]),
                        nn.ReLU()
                    ).to(teacher_features.device)
                
                student_features = self.feature_adapter(student_features)
        
        feature_loss = self.mse_loss(student_features, teacher_features)
        
        total_loss = self.alpha * hard_loss + self.beta * soft_loss + self.gamma * feature_loss
        
        return total_loss, hard_loss, soft_loss, feature_loss

def load_teacher_model(device):
    """Load pre-trained FCN-ResNet50 teacher model."""
    teacher = segmentation.fcn_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
    teacher.to(device)
    teacher.eval()
    return teacher

def get_feature_hook(features_dict, name):
    """Hook function to capture intermediate features."""
    def hook(module, input, output):
        features_dict[name] = output
    return hook

def calculate_miou(pred, target, num_classes=21):
    """Calculate mean IoU for segmentation."""
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    ious = []
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union > 0:
            ious.append(intersection / union)
    
    return np.array(ious)

def validate_model(student_model, teacher_model, val_loader, device, kd_criterion):
    """Validate model with comprehensive metrics."""
    student_model.eval()
    teacher_model.eval()
    
    total_loss = 0.0
    total_hard_loss = 0.0
    total_soft_loss = 0.0
    total_feature_loss = 0.0
    all_ious = []
    
    teacher_features = {}
    teacher_hook = teacher_model.backbone.layer4.register_forward_hook(
        get_feature_hook(teacher_features, 'teacher_feat')
    )
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Teacher forward pass
            teacher_output = teacher_model(images)['out']
            teacher_feat = teacher_features['teacher_feat']
            
            # Student forward pass
            student_features = {}
            student_hook = student_model.encoder4.register_forward_hook(
                get_feature_hook(student_features, 'student_feat')
            )
            
            student_output = student_model(images)
            student_feat = student_features['student_feat']
            student_hook.remove()
            
            # Calculate losses
            if kd_criterion is not None:
                loss, hard_loss, soft_loss, feature_loss = kd_criterion(
                    student_output, teacher_output, teacher_feat, student_feat, targets
                )
                total_loss += loss.item()
                total_hard_loss += hard_loss.item()
                total_soft_loss += soft_loss.item()
                total_feature_loss += feature_loss.item()
            else:
                # Just hard loss for baseline
                hard_loss = nn.CrossEntropyLoss(ignore_index=255)(student_output, targets)
                total_loss += hard_loss.item()
                total_hard_loss += hard_loss.item()
            
            # Calculate mIoU
            predictions = torch.argmax(student_output, dim=1)
            for pred, target in zip(predictions, targets):
                ious = calculate_miou(pred, target)
                if len(ious) > 0:
                    all_ious.append(np.mean(ious))
    
    teacher_hook.remove()
    
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_hard_loss = total_hard_loss / num_batches
    avg_soft_loss = total_soft_loss / num_batches if kd_criterion else 0
    avg_feature_loss = total_feature_loss / num_batches if kd_criterion else 0
    avg_miou = np.mean(all_ious) if all_ious else 0.0
    
    return avg_loss, avg_hard_loss, avg_soft_loss, avg_feature_loss, avg_miou

def train_model_with_config(config, train_loader, val_loader, device, max_epochs=100):
    """Train a model with a specific configuration until convergence."""
    
    print(f"\\n{'='*80}")
    print(f"Training Configuration: {config['name']}")
    print(f"{'='*80}")
    print(f"Method: {config['method']}")
    print(f"Base Dimension: {config['base_dim']}")
    print(f"Learning Rate: {config['lr']}")
    if 'kd_params' in config:
        kd = config['kd_params']
        print(f"KD Parameters: α={kd['alpha']:.3f}, β={kd['beta']:.3f}, γ={kd['gamma']:.3f}, T={kd['temperature']}")
    print(f"Max Epochs: {max_epochs}")
    print(f"{'='*80}")
    
    # Initialize model
    student_model = SMNet(num_classes=21, base_dim=config['base_dim']).to(device)
    
    # Load pre-trained baseline if available
    pretrained_path = f'../2_3_train_test_model/best_smnet_model_base{config["base_dim"]}.pth'
    if os.path.exists(pretrained_path):
        student_model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"✓ Loaded pre-trained baseline model")
    
    # Initialize teacher model if using KD
    teacher_model = None
    kd_criterion = None
    if config['method'] != 'baseline':
        teacher_model = load_teacher_model(device)
        kd_params = config['kd_params']
        kd_criterion = KnowledgeDistillationLoss(
            alpha=kd_params['alpha'],
            beta=kd_params['beta'],
            gamma=kd_params['gamma'],
            temperature=kd_params['temperature']
        )
        print(f"✓ Teacher model and KD criterion initialized")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(student_model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.5, verbose=True)
    
    # Monitoring
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    convergence_monitor = ConvergenceMonitor(patience=20, min_improvement=0.001)
    
    # Training history
    history = {
        'train_losses': [], 'train_mious': [],
        'val_losses': [], 'val_mious': [],
        'val_hard_losses': [], 'val_soft_losses': [], 'val_feature_losses': [],
        'learning_rates': []
    }
    
    best_val_miou = 0.0
    best_epoch = 0
    best_epoch_time = 0.0
    start_time = time.time()
    convergence_time = None
    
    # Set up teacher hooks if using KD
    teacher_features = {}
    teacher_hook = None
    if teacher_model:
        teacher_hook = teacher_model.backbone.layer4.register_forward_hook(
            get_feature_hook(teacher_features, 'teacher_feat')
        )
    
    print(f"\\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        student_model.train()
        if teacher_model:
            teacher_model.eval()
        
        train_loss = 0.0
        train_ious = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{max_epochs}", leave=False)
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            if config['method'] == 'baseline':
                # Baseline training
                student_output = student_model(images)
                loss = nn.CrossEntropyLoss(ignore_index=255)(student_output, targets)
            else:
                # Knowledge distillation training
                with torch.no_grad():
                    teacher_output = teacher_model(images)['out']
                    teacher_feat = teacher_features['teacher_feat']
                
                # Student forward pass
                student_features = {}
                student_hook = student_model.encoder4.register_forward_hook(
                    get_feature_hook(student_features, 'student_feat')
                )
                
                student_output = student_model(images)
                student_feat = student_features['student_feat']
                student_hook.remove()
                
                # Calculate KD loss
                loss, hard_loss, soft_loss, feature_loss = kd_criterion(
                    student_output, teacher_output, teacher_feat, student_feat, targets
                )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate mIoU for progress tracking
            if batch_idx % 10 == 0:  # Every 10 batches
                with torch.no_grad():
                    predictions = torch.argmax(student_output, dim=1)
                    batch_ious = []
                    for pred, target in zip(predictions, targets):
                        ious = calculate_miou(pred, target)
                        if len(ious) > 0:
                            batch_ious.append(np.mean(ious))
                    if batch_ious:
                        train_ious.extend(batch_ious)
            
            # Update progress bar
            current_miou = np.mean(train_ious) if train_ious else 0.0
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'mIoU': f'{current_miou:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
            })
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_miou = np.mean(train_ious) if train_ious else 0.0
        
        # Validation phase
        val_loss, val_hard, val_soft, val_feat, avg_val_miou = validate_model(
            student_model, teacher_model, val_loader, device, kd_criterion
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        history['train_losses'].append(avg_train_loss)
        history['train_mious'].append(avg_train_miou)
        history['val_losses'].append(val_loss)
        history['val_mious'].append(avg_val_miou)
        history['val_hard_losses'].append(val_hard)
        history['val_soft_losses'].append(val_soft)
        history['val_feature_losses'].append(val_feat)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        if config['method'] == 'baseline':
            print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                  f"Train: L={avg_train_loss:.4f}, mIoU={avg_train_miou:.4f} | "
                  f"Val: L={val_loss:.4f}, mIoU={avg_val_miou:.4f} | "
                  f"Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.1e}")
        else:
            print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                  f"Train: L={avg_train_loss:.4f}, mIoU={avg_train_miou:.4f} | "
                  f"Val: L={val_loss:.4f} (H:{val_hard:.3f}, S:{val_soft:.3f}, F:{val_feat:.3f}), mIoU={avg_val_miou:.4f} | "
                  f"Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        # Save best model
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            best_epoch = epoch + 1
            best_epoch_time = time.time() - start_time
            model_path = f"overnight_results/{config['name']}_best_model.pth"
            torch.save(student_model.state_dict(), model_path)
            print(f"    → New best model saved! mIoU: {avg_val_miou:.4f} at epoch {best_epoch} ({best_epoch_time/60:.1f}m)")
        
        # Check for early stopping
        if early_stopping(val_loss, student_model):
            convergence_time = time.time() - start_time
            print(f"    → Early stopping triggered after {epoch+1} epochs ({convergence_time/60:.1f}m)")
            break
        
        # Check for convergence
        if convergence_monitor(avg_val_miou):
            convergence_time = time.time() - start_time
            print(f"    → Convergence detected after {epoch+1} epochs ({convergence_time/60:.1f}m)")
            break
        
        # Memory cleanup
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
    
    # Cleanup
    if teacher_hook:
        teacher_hook.remove()
    
    total_time = time.time() - start_time
    if convergence_time is None:
        convergence_time = total_time  # Full training completed
    
    print(f"\\n✓ Training completed in {total_time/3600:.2f} hours")
    print(f"✓ Best validation mIoU: {best_val_miou:.6f} (epoch {best_epoch}, {best_epoch_time/60:.1f}m)")
    print(f"✓ Convergence time: {convergence_time/60:.1f} minutes")
    print(f"✓ Final epoch: {len(history['val_losses'])}")
    
    # Add timing info to history
    history['timing'] = {
        'total_time_seconds': total_time,
        'convergence_time_seconds': convergence_time,
        'best_epoch_time_seconds': best_epoch_time,
        'time_to_best_minutes': best_epoch_time / 60,
        'time_to_convergence_minutes': convergence_time / 60,
        'best_epoch': best_epoch
    }
    
    return history, best_val_miou, student_model

def plot_training_curves(histories, configs):
    """Plot comprehensive training curves for all configurations."""
    
    os.makedirs('overnight_results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    # Plot 1: Training Loss
    for i, (history, config) in enumerate(zip(histories, configs)):
        axes[0].plot(history['train_losses'], color=colors[i % len(colors)], 
                    label=f"{config['name']} (Train)", linestyle='-', alpha=0.7)
        axes[0].plot(history['val_losses'], color=colors[i % len(colors)], 
                    label=f"{config['name']} (Val)", linestyle='--', alpha=0.9)
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: mIoU Progress
    for i, (history, config) in enumerate(zip(histories, configs)):
        axes[1].plot(history['val_mious'], color=colors[i % len(colors)], 
                    label=config['name'], linewidth=2)
    axes[1].set_title('Validation mIoU Progress')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Schedule
    for i, (history, config) in enumerate(zip(histories, configs)):
        axes[2].plot(history['learning_rates'], color=colors[i % len(colors)], 
                    label=config['name'], alpha=0.8)
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: KD Loss Components (if available)
    kd_histories = [(h, c) for h, c in zip(histories, configs) if c['method'] != 'baseline']
    if kd_histories:
        for i, (history, config) in enumerate(kd_histories):
            axes[3].plot(history['val_hard_losses'], color=colors[i % len(colors)], 
                        linestyle='-', label=f"{config['name']} Hard")
            axes[3].plot(history['val_soft_losses'], color=colors[i % len(colors)], 
                        linestyle='--', label=f"{config['name']} Soft")
            axes[3].plot(history['val_feature_losses'], color=colors[i % len(colors)], 
                        linestyle=':', label=f"{config['name']} Feature")
    axes[3].set_title('Knowledge Distillation Loss Components')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Loss')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Final mIoU Comparison
    final_mious = [history['val_mious'][-1] for history in histories]
    config_names = [config['name'] for config in configs]
    bars = axes[4].bar(config_names, final_mious, color=colors[:len(configs)], alpha=0.7)
    axes[4].set_title('Final Validation mIoU Comparison')
    axes[4].set_ylabel('mIoU')
    axes[4].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, miou in zip(bars, final_mious):
        height = bar.get_height()
        axes[4].text(bar.get_x() + bar.get_width()/2., height,
                    f'{miou:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Training Efficiency (mIoU vs Epochs)
    for i, (history, config) in enumerate(zip(histories, configs)):
        epochs_to_best = np.argmax(history['val_mious']) + 1
        best_miou = max(history['val_mious'])
        axes[5].scatter(epochs_to_best, best_miou, color=colors[i % len(colors)], 
                       s=100, label=config['name'], alpha=0.8)
    axes[5].set_title('Training Efficiency (Epochs to Best mIoU)')
    axes[5].set_xlabel('Epochs to Best Performance')
    axes[5].set_ylabel('Best mIoU')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overnight_results/overnight_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[FILE] Training analysis saved: overnight_results/overnight_training_analysis.png")

def save_results_summary(histories, configs, best_mious, device_info):
    """Save comprehensive results summary."""
    
    results_summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'device': str(device_info),
            'total_configs': len(configs)
        },
        'configurations': [],
        'performance_ranking': []
    }
    
    # Collect detailed results
    for i, (history, config, best_miou) in enumerate(zip(histories, configs, best_mious)):
        config_result = {
            'name': config['name'],
            'method': config['method'],
            'base_dim': config['base_dim'],
            'learning_rate': config['lr'],
            'total_epochs': len(history['val_losses']),
            'best_val_miou': best_miou,
            'final_val_miou': history['val_mious'][-1],
            'best_epoch': np.argmax(history['val_mious']) + 1,
            'final_train_loss': history['train_losses'][-1],
            'final_val_loss': history['val_losses'][-1]
        }
        
        # Add timing information if available
        if 'timing' in history:
            timing = history['timing']
            config_result.update({
                'total_training_time_hours': timing['total_time_seconds'] / 3600,
                'time_to_convergence_minutes': timing['time_to_convergence_minutes'],
                'time_to_best_performance_minutes': timing['time_to_best_minutes'],
                'convergence_efficiency': timing['time_to_best_minutes'] / timing['time_to_convergence_minutes'] if timing['time_to_convergence_minutes'] > 0 else 1.0
            })
        
        if 'kd_params' in config:
            config_result['kd_parameters'] = config['kd_params']
        
        results_summary['configurations'].append(config_result)
    
    # Performance ranking
    ranked_configs = sorted(results_summary['configurations'], 
                          key=lambda x: x['best_val_miou'], reverse=True)
    
    for rank, config in enumerate(ranked_configs, 1):
        rank_info = {
            'rank': rank,
            'name': config['name'],
            'best_miou': config['best_val_miou'],
            'epochs_to_best': config['best_epoch']
        }
        
        # Add timing info to ranking
        if 'time_to_convergence_minutes' in config:
            rank_info['convergence_time_minutes'] = config['time_to_convergence_minutes']
            rank_info['time_to_best_minutes'] = config['time_to_best_performance_minutes']
            rank_info['efficiency_ratio'] = config['convergence_efficiency']
        
        results_summary['performance_ranking'].append(rank_info)
    
    # Save to JSON
    with open('overnight_results/experiment_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save human-readable summary
    with open('overnight_results/experiment_summary.txt', 'w') as f:
        f.write("ELEC475 Lab 3 - Overnight Knowledge Distillation Results\\n")
        f.write("="*60 + "\\n\\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"Device: {device_info}\\n")
        f.write(f"Total Configurations Tested: {len(configs)}\\n\\n")
        
        f.write("PERFORMANCE RANKING:\\n")
        f.write("-"*60 + "\\n")
        for rank_info in results_summary['performance_ranking']:
            line = f"{rank_info['rank']}. {rank_info['name']:<25} | mIoU: {rank_info['best_miou']:.6f} | Epochs: {rank_info['epochs_to_best']:<3d}"
            
            if 'convergence_time_minutes' in rank_info:
                line += f" | Conv: {rank_info['convergence_time_minutes']:.1f}m | Best: {rank_info['time_to_best_minutes']:.1f}m"
            
            f.write(line + "\\n")
        
        f.write("\\n\\nDETAILED RESULTS:\\n")
        f.write("="*60 + "\\n")
        for config in results_summary['configurations']:
            f.write(f"\\n{config['name']}:\\n")
            f.write(f"  Method: {config['method']}\\n")
            f.write(f"  Best mIoU: {config['best_val_miou']:.6f} (epoch {config['best_epoch']})\\n")
            f.write(f"  Total epochs: {config['total_epochs']}\\n")
            
            if 'time_to_convergence_minutes' in config:
                f.write(f"  Convergence time: {config['time_to_convergence_minutes']:.1f} minutes\\n")
                f.write(f"  Time to best: {config['time_to_best_performance_minutes']:.1f} minutes\\n")
                f.write(f"  Training efficiency: {config['convergence_efficiency']:.2f}\\n")
                f.write(f"  Total training time: {config['total_training_time_hours']:.2f} hours\\n")
            f.write(f"  Method: {config['method']}\\n")
            f.write(f"  Base Dimension: {config['base_dim']}\\n")
            f.write(f"  Learning Rate: {config['learning_rate']}\\n")
            f.write(f"  Best mIoU: {config['best_val_miou']:.6f} (epoch {config['best_epoch']})\\n")
            f.write(f"  Final mIoU: {config['final_val_miou']:.6f}\\n")
            f.write(f"  Total Epochs: {config['total_epochs']}\\n")
            f.write(f"  Final Losses: Train={config['final_train_loss']:.4f}, Val={config['final_val_loss']:.4f}\\n")
            
            if 'kd_parameters' in config:
                kd = config['kd_parameters']
                f.write(f"  KD Params: α={kd['alpha']:.3f}, β={kd['beta']:.3f}, γ={kd['gamma']:.3f}, T={kd['temperature']}\\n")
    
    print(f"[FILE] Results summary saved: overnight_results/experiment_summary.txt")
    print(f"[FILE] Results JSON saved: overnight_results/experiment_results.json")

def main():
    """Main overnight training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting overnight training on device: {device}")
    
    # Create results directory
    os.makedirs('overnight_results', exist_ok=True)
    
    print(f"\\n{'='*80}")
    print(f"ELEC475 LAB 3 - OVERNIGHT KNOWLEDGE DISTILLATION TRAINING SUITE")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Dataset setup
    voc_root = r"../pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val"
    
    if not os.path.exists(voc_root):
        alt_paths = [
            r"./pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val",
            r"../pascal-voc-2012-dataset/versions/1/VOC2012_train_val/VOC2012_train_val",
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                voc_root = alt_path
                break
        else:
            raise FileNotFoundError("PASCAL VOC 2012 dataset not found!")
    
    print(f"Dataset path: {voc_root}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        squeeze_and_long
    ])
    
    # Create datasets
    train_dataset = LocalVOCDataset(
        voc_root=voc_root, split='train',
        transform=train_transform, target_transform=target_transform
    )
    
    val_dataset = LocalVOCDataset(
        voc_root=voc_root, split='val', 
        transform=val_transform, target_transform=target_transform,
        max_samples=300  # Faster validation
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    
    # Training configurations
    configs = [
        {
            'name': 'Baseline_Optimized',
            'method': 'baseline',
            'base_dim': 16,
            'lr': 0.001
        },
        {
            'name': 'ResponseKD_Balanced',
            'method': 'response_kd',
            'base_dim': 16,
            'lr': 0.001,
            'kd_params': {'alpha': 0.3, 'beta': 0.4, 'gamma': 0.3, 'temperature': 4.0}
        },
        {
            'name': 'ResponseKD_SoftFocus',
            'method': 'response_kd',
            'base_dim': 16,
            'lr': 0.0008,
            'kd_params': {'alpha': 0.2, 'beta': 0.6, 'gamma': 0.2, 'temperature': 5.0}
        },
        {
            'name': 'FeatureKD_Pure',
            'method': 'feature_kd',
            'base_dim': 16,
            'lr': 0.001,
            'kd_params': {'alpha': 0.5, 'beta': 0.0, 'gamma': 0.5, 'temperature': 4.0}
        },
        {
            'name': 'ResponseKD_HighTemp',
            'method': 'response_kd',
            'base_dim': 16,
            'lr': 0.0012,
            'kd_params': {'alpha': 0.3, 'beta': 0.5, 'gamma': 0.2, 'temperature': 8.0}
        }
    ]
    
    print(f"\\nPlanned configurations: {len(configs)}")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config['name']} - {config['method']}")
    
    # Run experiments
    histories = []
    best_mious = []
    trained_models = []
    
    total_start_time = time.time()
    
    for i, config in enumerate(configs):
        print(f"\\n\\n{'#'*80}")
        print(f"EXPERIMENT {i+1}/{len(configs)}: {config['name']}")
        print(f"{'#'*80}")
        
        try:
            history, best_miou, model = train_model_with_config(
                config, train_loader, val_loader, device, max_epochs=150
            )
            
            histories.append(history)
            best_mious.append(best_miou)
            trained_models.append(model)
            
            print(f"\\n✓ {config['name']} completed successfully!")
            print(f"  Best mIoU: {best_miou:.6f}")
            print(f"  Epochs trained: {len(history['val_losses'])}")
            
        except Exception as e:
            print(f"\\n✗ {config['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Add dummy history for failed experiments
            histories.append({'train_losses': [0], 'val_losses': [0], 'val_mious': [0], 
                            'train_mious': [0], 'val_hard_losses': [0], 
                            'val_soft_losses': [0], 'val_feature_losses': [0], 
                            'learning_rates': [0]})
            best_mious.append(0.0)
            trained_models.append(None)
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    total_time = time.time() - total_start_time
    
    print(f"\\n\\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Generate comprehensive analysis
    print(f"\\nGenerating comprehensive analysis...")
    
    # Plot training curves
    valid_histories = [(h, c) for h, c in zip(histories, configs) if len(h['val_mious']) > 1]
    if valid_histories:
        valid_histories, valid_configs = zip(*valid_histories)
        plot_training_curves(valid_histories, valid_configs)
    
    # Save results summary
    save_results_summary(histories, configs, best_mious, device)
    
    # Print final ranking
    print(f"\\nFINAL PERFORMANCE RANKING:")
    print(f"{'-'*80}")
    config_results = list(zip(configs, best_mious, histories))
    config_results.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (config, miou, history) in enumerate(config_results, 1):
        line = f"  {rank}. {config['name']:<25} | mIoU: {miou:.6f}"
        
        if 'timing' in history:
            timing = history['timing']
            line += f" | Conv: {timing['time_to_convergence_minutes']:.1f}m | Best: {timing['time_to_best_minutes']:.1f}m"
        
        print(line)
    
    print(f"\\n[OK] Overnight training suite completed successfully!")
    print(f"[INFO] Check 'overnight_results/' folder for detailed analysis")
    print(f"[INFO] Best model files saved with '_best_model.pth' suffix")

if __name__ == '__main__':
    main()