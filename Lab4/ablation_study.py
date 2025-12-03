# CLIP Ablation Study Runner
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

from CLIP_model_design.enhanced_clip_models import create_enhanced_model, enhanced_clip_loss, get_enhanced_transforms
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
        print(f"\n🚀 Training {enhancement_name} model...")
        
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
        
        for epoch in range(self.train_config['epochs']):\n            model.train()\n            epoch_loss = 0.0\n            num_batches = 0\n            \n            # Progress bar for training\n            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.train_config[\"epochs\"]}')\n            \n            for batch_idx, (images, text_embeddings) in enumerate(pbar):\n                images = images.to(self.device)\n                text_embeddings = text_embeddings.to(self.device)\n                \n                optimizer.zero_grad()\n                \n                # Forward pass\n                logits_per_image, logits_per_text = model(images, text_embeddings)\n                \n                # Compute loss\n                loss = enhanced_clip_loss(logits_per_image, logits_per_text, label_smoothing=0.1)\n                \n                # Backward pass\n                loss.backward()\n                \n                # Gradient clipping for stability\n                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n                \n                optimizer.step()\n                scheduler.step()\n                \n                epoch_loss += loss.item()\n                num_batches += 1\n                \n                # Update progress bar\n                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})\n            \n            avg_epoch_loss = epoch_loss / num_batches\n            train_losses.append(avg_epoch_loss)\n            \n            # Validation metrics\n            val_metrics = self.compute_metrics(model, val_loader, max_batches=10)\n            val_metrics_history.append(val_metrics)\n            \n            print(f'Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}, Val_Recall@1={val_metrics[\"avg_recall_1\"]:.4f}')\n        \n        return {\n            'train_losses': train_losses,\n            'val_metrics_history': val_metrics_history,\n            'final_metrics': val_metrics_history[-1] if val_metrics_history else {}\n        }\n    \n    def run_ablation_study(self):\n        \"\"\"Run complete ablation study\"\"\"\n        print(\"🔬 Starting CLIP Ablation Study...\")\n        print(f\"Device: {self.device}\")\n        print(f\"Configuration: {self.train_config}\")\n        \n        # Create dataloaders\n        print(\"\\n📊 Loading dataset...\")\n        train_loader, val_loader = create_dataloaders(\n            max_samples=self.train_config['max_samples'],\n            batch_size=self.train_config['batch_size']\n        )\n        \n        # Run experiments for each enhancement\n        for enhancement_name, config in self.enhancement_configs.items():\n            print(f\"\\n{'='*60}\")\n            print(f\"🧪 Testing Enhancement: {enhancement_name}\")\n            print(f\"📝 Description: {config['description']}\")\n            print(f\"{'='*60}\")\n            \n            try:\n                # Create model\n                model = create_enhanced_model(enhancement_type=config['type'])\n                model = model.to(self.device)\n                \n                print(f\"📊 Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\")\n                \n                # Train model\n                training_results = self.train_model(model, train_loader, val_loader, enhancement_name)\n                \n                # Final evaluation\n                print(\"\\n🔍 Final evaluation...\")\n                final_metrics = self.compute_metrics(model, val_loader)\n                \n                # Store results\n                self.results[enhancement_name] = {\n                    'config': config,\n                    'training_results': training_results,\n                    'final_metrics': final_metrics,\n                    'model_params': sum(p.numel() for p in model.parameters() if p.requires_grad)\n                }\n                \n                print(f\"✅ {enhancement_name} completed!\")\n                print(f\"📊 Final Results:\")\n                print(f\"   - Loss: {final_metrics['loss']:.4f}\")\n                print(f\"   - Recall@1 (I2T): {final_metrics['recall_1_i2t']:.4f}\")\n                print(f\"   - Recall@1 (T2I): {final_metrics['recall_1_t2i']:.4f}\")\n                print(f\"   - Avg Recall@1: {final_metrics['avg_recall_1']:.4f}\")\n                \n                # Save checkpoint\n                torch.save({\n                    'model_state_dict': model.state_dict(),\n                    'enhancement_type': config['type'],\n                    'final_metrics': final_metrics\n                }, os.path.join(self.results_dir, f'{enhancement_name}_model.pth'))\n                \n            except Exception as e:\n                print(f\"❌ Error with {enhancement_name}: {str(e)}\")\n                self.results[enhancement_name] = {'error': str(e)}\n        \n        # Generate comprehensive report\n        self.generate_report()\n        \n        return self.results\n    \n    def generate_report(self):\n        \"\"\"Generate comprehensive ablation study report\"\"\"\n        print(\"\\n\" + \"=\"*80)\n        print(\"📊 ABLATION STUDY RESULTS\")\n        print(\"=\"*80)\n        \n        # Create results summary\n        summary_data = []\n        \n        for enhancement_name, results in self.results.items():\n            if 'error' in results:\n                print(f\"❌ {enhancement_name}: ERROR - {results['error']}\")\n                continue\n                \n            metrics = results['final_metrics']\n            summary_data.append({\n                'Enhancement': enhancement_name,\n                'Description': results['config']['description'],\n                'Loss': metrics['loss'],\n                'Recall@1 (I2T)': metrics['recall_1_i2t'],\n                'Recall@1 (T2I)': metrics['recall_1_t2i'],\n                'Avg Recall@1': metrics['avg_recall_1'],\n                'Recall@5 (I2T)': metrics['recall_5_i2t'],\n                'Recall@5 (T2I)': metrics['recall_5_t2i'],\n                'Avg Recall@5': metrics['avg_recall_5'],\n                'Parameters': results['model_params']\n            })\n        \n        # Sort by average recall@1 (descending)\n        summary_data.sort(key=lambda x: x['Avg Recall@1'], reverse=True)\n        \n        # Print summary table\n        print(f\"\\n{'Rank':<4} {'Enhancement':<20} {'Avg R@1':<10} {'Avg R@5':<10} {'Loss':<8} {'Params':<10}\")\n        print(\"-\" * 70)\n        \n        for rank, data in enumerate(summary_data, 1):\n            print(f\"{rank:<4} {data['Enhancement']:<20} {data['Avg Recall@1']:<10.4f} {data['Avg Recall@5']:<10.4f} {data['Loss']:<8.4f} {data['Parameters']:<10,}\")\n        \n        # Find best performing enhancement\n        if summary_data:\n            best_enhancement = summary_data[0]\n            print(f\"\\n🏆 Best Enhancement: {best_enhancement['Enhancement']}\")\n            print(f\"📈 Improvement over baseline: {best_enhancement['Avg Recall@1'] - next(x['Avg Recall@1'] for x in summary_data if x['Enhancement'] == 'baseline'):.4f}\")\n        \n        # Save detailed results\n        results_file = os.path.join(self.results_dir, f'ablation_results_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.json')\n        \n        # Convert torch tensors to lists for JSON serialization\n        json_results = {}\n        for name, results in self.results.items():\n            if 'error' not in results:\n                json_results[name] = {\n                    'config': results['config'],\n                    'final_metrics': results['final_metrics'],\n                    'model_params': results['model_params']\n                }\n        \n        with open(results_file, 'w') as f:\n            json.dump(json_results, f, indent=2)\n        \n        print(f\"\\n💾 Detailed results saved to: {results_file}\")\n        \n        # Generate visualization\n        self.plot_results(summary_data)\n    \n    def plot_results(self, summary_data):\n        \"\"\"Create visualization of ablation study results\"\"\"\n        if not summary_data:\n            return\n            \n        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))\n        \n        enhancements = [d['Enhancement'] for d in summary_data]\n        recalls_1 = [d['Avg Recall@1'] for d in summary_data]\n        recalls_5 = [d['Avg Recall@5'] for d in summary_data]\n        losses = [d['Loss'] for d in summary_data]\n        params = [d['Parameters'] for d in summary_data]\n        \n        # Recall@1 comparison\n        bars1 = ax1.bar(enhancements, recalls_1, color='skyblue', alpha=0.8)\n        ax1.set_title('Average Recall@1 by Enhancement')\n        ax1.set_ylabel('Recall@1')\n        ax1.tick_params(axis='x', rotation=45)\n        \n        # Add value labels on bars\n        for bar, value in zip(bars1, recalls_1):\n            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,\n                    f'{value:.3f}', ha='center', va='bottom')\n        \n        # Recall@5 comparison\n        bars2 = ax2.bar(enhancements, recalls_5, color='lightcoral', alpha=0.8)\n        ax2.set_title('Average Recall@5 by Enhancement')\n        ax2.set_ylabel('Recall@5')\n        ax2.tick_params(axis='x', rotation=45)\n        \n        # Loss comparison\n        bars3 = ax3.bar(enhancements, losses, color='lightgreen', alpha=0.8)\n        ax3.set_title('Final Loss by Enhancement')\n        ax3.set_ylabel('Loss')\n        ax3.tick_params(axis='x', rotation=45)\n        \n        # Parameters comparison\n        bars4 = ax4.bar(enhancements, params, color='gold', alpha=0.8)\n        ax4.set_title('Model Parameters by Enhancement')\n        ax4.set_ylabel('Parameters')\n        ax4.tick_params(axis='x', rotation=45)\n        \n        plt.tight_layout()\n        \n        # Save plot\n        plot_file = os.path.join(self.results_dir, f'ablation_results_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.png')\n        plt.savefig(plot_file, dpi=300, bbox_inches='tight')\n        plt.show()\n        \n        print(f\"📈 Visualization saved to: {plot_file}\")\n\ndef main():\n    \"\"\"Run the ablation study\"\"\"\n    import argparse\n    \n    parser = argparse.ArgumentParser(description=\"CLIP Ablation Study\")\n    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')\n    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')\n    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum samples to use')\n    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')\n    parser.add_argument('--results_dir', type=str, default='ablation_results', help='Results directory')\n    \n    args = parser.parse_args()\n    \n    # Create and run ablation study\n    runner = AblationStudyRunner(results_dir=args.results_dir)\n    \n    # Update configuration with command line arguments\n    runner.train_config.update({\n        'epochs': args.epochs,\n        'batch_size': args.batch_size,\n        'max_samples': args.max_samples,\n        'learning_rate': args.learning_rate\n    })\n    \n    # Run the study\n    results = runner.run_ablation_study()\n    \n    print(\"\\n🎉 Ablation study completed!\")\n    return results\n\nif __name__ == \"__main__\":\n    main()