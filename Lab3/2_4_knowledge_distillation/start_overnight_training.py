"""
ELEC475 Lab 3 - Start Overnight Training Suite
User-friendly launcher for the comprehensive knowledge distillation experiments
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    """Main launcher function."""
    
    print("="*70)
    print("🌙 ELEC475 LAB 3 - OVERNIGHT TRAINING SUITE LAUNCHER")
    print("="*70)
    print("This will start comprehensive knowledge distillation training")
    print("with multiple configurations and automatic convergence detection.")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('overnight_training_suite.py'):
        print("❌ Error: overnight_training_suite.py not found!")
        print("   Please run this script from the 2_4_knowledge_distillation directory")
        return
    
    # Check for CUDA availability
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {device}")
        if device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("⚠️ PyTorch not available - please ensure environment is set up")
        return
    
    print()
    print("🔬 TRAINING CONFIGURATIONS:")
    print("   1. Baseline SMNet (no KD)")
    print("   2. Response-based KD (Conservative) - α=0.7, T=3.0")
    print("   3. Response-based KD (Balanced) - α=0.5, T=4.0")
    print("   4. Response-based KD (Aggressive) - α=0.3, T=5.0")
    print("   5. Feature-based KD - Pure feature matching")
    print()
    print("⏱️ ESTIMATED TIME:")
    print("   • Each config: 2-6 hours (depends on convergence)")
    print("   • Total: 8-12 hours overnight")
    print("   • Early stopping when models plateau")
    print()
    print("📊 OUTPUTS:")
    print("   • Performance ranking table")
    print("   • Training analysis plots")
    print("   • Best model checkpoints")
    print("   • Comprehensive comparison data")
    print()
    
    # Check if results directory already exists
    if os.path.exists('overnight_results'):
        print("⚠️ WARNING: overnight_results directory already exists!")
        print("   Previous results may be overwritten.")
        response = input("   Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("   Cancelled by user.")
            return
        print()
    
    # Final confirmation
    print("🚀 READY TO START TRAINING!")
    response = input("   Start overnight training suite? (y/N): ").strip().lower()
    
    if response != 'y':
        print("   Cancelled by user.")
        return
    
    print()
    print("="*70)
    print("🌙 STARTING OVERNIGHT TRAINING...")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   You can monitor progress with: python monitor_training.py")
    print("="*70)
    print()
    
    try:
        # Start the overnight training
        result = subprocess.run([
            sys.executable, 'overnight_training_suite.py'
        ], check=True, capture_output=False, text=True)
        
        print()
        print("="*70)
        print("✅ OVERNIGHT TRAINING COMPLETED SUCCESSFULLY!")
        print("   Check overnight_results/ for all outputs")
        print("="*70)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code: {e.returncode}")
        print("   Check error logs for details")
        
    except KeyboardInterrupt:
        print()
        print("⏹️ Training interrupted by user")
        print("   Partial results may be available in overnight_results/")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == '__main__':
    main()