# CLIP Ablation Study Runner - Fixed Version
import os
import sys
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F

# Add project paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CLIP_model_design.enhanced_clip_models import create_enhanced_model, enhanced_clip_loss
from Load_data_set.enhanced_load_data_set import create_enhanced_dataloaders

class AblationStudyRunner:
    """Runs comprehensive ablation study on CLIP model enhancements"""
    
    def __init__(self, device=None, results_dir='ablation_results'):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Enhancement configurations to test
        self.enhancement_configs = {
            'baseline': {'type': 'baseline', 'description': 'Standard CLIP model'},
            'enhanced_projection': {'type': 'enhanced_projection', 'description': 'Enhanced projection with residual connections'},
            'attention_pooling': {'type': 'attention_pooling', 'description': 'Attention-based pooling instead of global average'},
            'dropout_regularization': {'type': 'dropout_regularization', 'description': 'Enhanced dropout regularization'},
            'data_augmentation': {'type': 'data_augmentation', 'description': 'Advanced data augmentation techniques'}
        }
        
        # Training configuration
        self.train_config = {
            'learning_rate': 1e-4,
            'epochs': 3,  # Short for ablation study
            'batch_size': 16,
            'max_samples': 1000,  # Reduced for faster experiments
            'warmup_steps': 50
        }
        
        self.results = {}
        
    def compute_metrics(self, model, dataloader, max_batches=None):
        """Compute evaluation metrics for the model"""
        model.eval()
        all_image_embeds = []
        all_text_embeds = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, text_embeddings) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                    
                images = images.to(self.device)
                text_embeddings = text_embeddings.to(self.device)
                
                # Forward pass
                logits_per_image, logits_per_text = model(images, text_embeddings)
                
                # Compute loss
                loss = enhanced_clip_loss(logits_per_image, logits_per_text)
                total_loss += loss.item()
                
                # Store embeddings for recall calculation
                image_embeds = model.encode_image(images)
                all_image_embeds.append(image_embeds.cpu())
                all_text_embeds.append(F.normalize(text_embeddings, p=2, dim=1).cpu())
                
                num_batches += 1
        
        # Compute recall metrics
        all_image_embeds = torch.cat(all_image_embeds, dim=0)
        all_text_embeds = torch.cat(all_text_embeds, dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(all_image_embeds, all_text_embeds.t())
        
        # Compute recall@k metrics
        recall_1_i2t, recall_1_t2i = self.recall_at_k(sim_matrix, k=1)
        recall_5_i2t, recall_5_t2i = self.recall_at_k(sim_matrix, k=5)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return {
            'loss': avg_loss,
            'recall_1_i2t': recall_1_i2t,
            'recall_1_t2i': recall_1_t2i,
            'recall_5_i2t': recall_5_i2t,
            'recall_5_t2i': recall_5_t2i,
            'avg_recall_1': (recall_1_i2t + recall_1_t2i) / 2,
            'avg_recall_5': (recall_5_i2t + recall_5_t2i) / 2
        }
    
    def recall_at_k(self, sim_matrix, k=1):
        """Compute recall@k for both directions"""
        N = sim_matrix.size(0)
        
        # Image to Text recall
        i2t_ranks = torch.argsort(sim_matrix, dim=1, descending=True)
        i2t_correct = torch.arange(N).unsqueeze(1).expand(-1, k) == i2t_ranks[:, :k]
        recall_i2t = i2t_correct.any(dim=1).float().mean().item()
        
        # Text to Image recall  
        t2i_ranks = torch.argsort(sim_matrix, dim=0, descending=True)
        t2i_correct = torch.arange(N).unsqueeze(0).expand(k, -1) == t2i_ranks[:k, :]
        recall_t2i = t2i_correct.any(dim=0).float().mean().item()
        
        return recall_i2t, recall_t2i
    
    def train_model(self, model, enhancement_type, enhancement_name):
        """Train a model variant"""
        print(f"\\n🚀 Training {enhancement_name} model...")
        
        # Create dataloaders with appropriate transforms for this enhancement
        print(f"📊 Creating dataloaders for {enhancement_type} enhancement...")
        train_loader, val_loader = create_enhanced_dataloaders(
            max_samples=self.train_config['max_samples'],
            batch_size=self.train_config['batch_size'],
            enhancement_type=enhancement_type
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=self.train_config['learning_rate'])
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * self.train_config['epochs']
        warmup_steps = self.train_config['warmup_steps']
        
        def lr_schedule(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.1, (total_steps - step) / (total_steps - warmup_steps))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
        
        train_losses = []
        val_metrics_history = []
        
        for epoch in range(self.train_config['epochs']):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Progress bar for training
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.train_config["epochs"]}')
            
            for batch_idx, (images, text_embeddings) in enumerate(pbar):
                images = images.to(self.device)
                text_embeddings = text_embeddings.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits_per_image, logits_per_text = model(images, text_embeddings)
                
                # Compute loss
                loss = enhanced_clip_loss(logits_per_image, logits_per_text, label_smoothing=0.1)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
            
            avg_epoch_loss = epoch_loss / num_batches
            train_losses.append(avg_epoch_loss)
            
            # Validation metrics
            val_metrics = self.compute_metrics(model, val_loader, max_batches=10)
            val_metrics_history.append(val_metrics)
            
            print(f'Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}, Val_Recall@1={val_metrics["avg_recall_1"]:.4f}')
        
        return {
            'train_losses': train_losses,
            'val_metrics_history': val_metrics_history,
            'final_metrics': val_metrics_history[-1] if val_metrics_history else {},
            'val_loader': val_loader  # Return for final evaluation
        }
    
    def run_ablation_study(self):
        """Run complete ablation study"""
        print("🔬 Starting CLIP Ablation Study...")
        print(f"Device: {self.device}")
        print(f"Configuration: {self.train_config}")
        
        # Run experiments for each enhancement
        for enhancement_name, config in self.enhancement_configs.items():
            print(f"\\n{'='*60}")
            print(f"🧪 Testing Enhancement: {enhancement_name}")
            print(f"📝 Description: {config['description']}")
            print(f"{'='*60}")
            
            try:
                # Create model
                model = create_enhanced_model(enhancement_type=config['type'])
                model = model.to(self.device)
                
                print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
                
                # Train model
                training_results = self.train_model(model, config['type'], enhancement_name)
                
                # Final evaluation
                print("\\n🔍 Final evaluation...")
                final_metrics = self.compute_metrics(model, training_results['val_loader'])
                
                # Store results
                self.results[enhancement_name] = {
                    'config': config,
                    'training_results': training_results,
                    'final_metrics': final_metrics,
                    'model_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
                }
                
                print(f"✅ {enhancement_name} completed!")
                print(f"📊 Final Results:")
                print(f"   - Loss: {final_metrics['loss']:.4f}")
                print(f"   - Recall@1 (I2T): {final_metrics['recall_1_i2t']:.4f}")
                print(f"   - Recall@1 (T2I): {final_metrics['recall_1_t2i']:.4f}")
                print(f"   - Avg Recall@1: {final_metrics['avg_recall_1']:.4f}")
                
                # Save checkpoint
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'enhancement_type': config['type'],
                    'final_metrics': final_metrics
                }, os.path.join(self.results_dir, f'{enhancement_name}_model.pth'))
                
            except Exception as e:
                print(f"❌ Error with {enhancement_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                self.results[enhancement_name] = {'error': str(e)}
        
        # Generate comprehensive report
        self.generate_report()
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive ablation study report"""
        print("\\n" + "="*80)
        print("📊 ABLATION STUDY RESULTS")
        print("="*80)
        
        # Create results summary
        summary_data = []
        
        for enhancement_name, results in self.results.items():
            if 'error' in results:
                print(f"❌ {enhancement_name}: ERROR - {results['error']}")
                continue
                
            metrics = results['final_metrics']
            summary_data.append({
                'Enhancement': enhancement_name,
                'Description': results['config']['description'],
                'Loss': metrics['loss'],
                'Recall@1 (I2T)': metrics['recall_1_i2t'],
                'Recall@1 (T2I)': metrics['recall_1_t2i'],
                'Avg Recall@1': metrics['avg_recall_1'],
                'Recall@5 (I2T)': metrics['recall_5_i2t'],
                'Recall@5 (T2I)': metrics['recall_5_t2i'],
                'Avg Recall@5': metrics['avg_recall_5'],
                'Parameters': results['model_params']
            })
        
        # Sort by average recall@1 (descending)
        summary_data.sort(key=lambda x: x['Avg Recall@1'], reverse=True)
        
        # Print summary table
        print(f"\\n{'Rank':<4} {'Enhancement':<20} {'Avg R@1':<10} {'Avg R@5':<10} {'Loss':<8} {'Params':<10}")
        print("-" * 70)
        
        for rank, data in enumerate(summary_data, 1):
            print(f"{rank:<4} {data['Enhancement']:<20} {data['Avg Recall@1']:<10.4f} {data['Avg Recall@5']:<10.4f} {data['Loss']:<8.4f} {data['Parameters']:<10,}")
        
        # Find best performing enhancement
        if summary_data:
            best_enhancement = summary_data[0]
            baseline_recall = next((x['Avg Recall@1'] for x in summary_data if x['Enhancement'] == 'baseline'), 0)
            improvement = best_enhancement['Avg Recall@1'] - baseline_recall
            
            print(f"\\n🏆 Best Enhancement: {best_enhancement['Enhancement']}")
            print(f"📈 Improvement over baseline: {improvement:.4f}")
        
        # Save detailed results
        results_file = os.path.join(self.results_dir, f'ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        # Convert results for JSON serialization
        json_results = {}
        for name, results in self.results.items():
            if 'error' not in results:
                # Remove val_loader from training_results for JSON serialization
                training_results_clean = results['training_results'].copy()
                if 'val_loader' in training_results_clean:
                    del training_results_clean['val_loader']
                
                json_results[name] = {
                    'config': results['config'],
                    'final_metrics': results['final_metrics'],
                    'model_params': results['model_params'],
                    'training_results': training_results_clean
                }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\\n💾 Detailed results saved to: {results_file}")
        
        # Generate visualization
        self.plot_results(summary_data)
    
    def plot_results(self, summary_data):
        """Create visualization of ablation study results"""
        if not summary_data:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        enhancements = [d['Enhancement'] for d in summary_data]
        recalls_1 = [d['Avg Recall@1'] for d in summary_data]
        recalls_5 = [d['Avg Recall@5'] for d in summary_data]
        losses = [d['Loss'] for d in summary_data]
        params = [d['Parameters'] for d in summary_data]
        
        # Recall@1 comparison
        bars1 = ax1.bar(enhancements, recalls_1, color='skyblue', alpha=0.8)
        ax1.set_title('Average Recall@1 by Enhancement')
        ax1.set_ylabel('Recall@1')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, recalls_1):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Recall@5 comparison
        bars2 = ax2.bar(enhancements, recalls_5, color='lightcoral', alpha=0.8)
        ax2.set_title('Average Recall@5 by Enhancement')
        ax2.set_ylabel('Recall@5')
        ax2.tick_params(axis='x', rotation=45)
        
        # Loss comparison
        bars3 = ax3.bar(enhancements, losses, color='lightgreen', alpha=0.8)
        ax3.set_title('Final Loss by Enhancement')
        ax3.set_ylabel('Loss')
        ax3.tick_params(axis='x', rotation=45)
        
        # Parameters comparison
        bars4 = ax4.bar(enhancements, params, color='gold', alpha=0.8)
        ax4.set_title('Model Parameters by Enhancement')
        ax4.set_ylabel('Parameters')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.results_dir, f'ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Visualization saved to: {plot_file}")

def main():
    """Run the ablation study"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP Ablation Study")
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum samples to use')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--results_dir', type=str, default='ablation_results', help='Results directory')
    
    args = parser.parse_args()
    
    # Create and run ablation study
    runner = AblationStudyRunner(results_dir=args.results_dir)
    
    # Update configuration with command line arguments
    runner.train_config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'max_samples': args.max_samples,
        'learning_rate': args.learning_rate
    })
    
    # Run the study
    results = runner.run_ablation_study()
    
    print("\\n🎉 Ablation study completed!")
    return results

if __name__ == "__main__":
    main()