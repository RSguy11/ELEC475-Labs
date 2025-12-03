# Quick Ablation Study Runner
import os
import sys
import subprocess

def main():
    """Run the CLIP ablation study with reasonable defaults"""
    
    print("🚀 CLIP Enhancement Ablation Study")
    print("="*50)
    
    # Default configuration for quick testing
    config = {
        'epochs': 3,
        'batch_size': 16, 
        'max_samples': 1000,
        'learning_rate': 1e-4,
        'results_dir': 'ablation_results'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nThis will test the following enhancements:")
    print("  1. Baseline (standard CLIP)")
    print("  2. Enhanced Projection (residual connections)")
    print("  3. Attention Pooling (replace global avg pooling)")  
    print("  4. Dropout Regularization (enhanced dropout)")
    print("  5. Data Augmentation (advanced augmentations)")
    
    response = input("\nProceed with ablation study? (y/n): ")
    if response.lower() != 'y':
        print("Ablation study cancelled.")
        return
    
    # Run the ablation study
    try:
        cmd = [
            sys.executable, 
            'ablation_study_fixed.py',
            '--epochs', str(config['epochs']),
            '--batch_size', str(config['batch_size']),
            '--max_samples', str(config['max_samples']),
            '--learning_rate', str(config['learning_rate']),
            '--results_dir', config['results_dir']
        ]
        
        print(f"\nRunning command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print("\n🎉 Ablation study completed successfully!")
            print(f"📊 Results saved in: {config['results_dir']}/")
            
            # Show results directory contents
            results_dir = config['results_dir']
            if os.path.exists(results_dir):
                print(f"\nGenerated files:")
                for file in os.listdir(results_dir):
                    print(f"  - {file}")
        else:
            print(f"\n❌ Ablation study failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running ablation study: {str(e)}")
        
if __name__ == "__main__":
    main()