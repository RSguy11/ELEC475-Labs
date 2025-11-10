"""
ELEC475 Lab 3 - Monitor Overnight Training Progress
Check the status and progress of the overnight training suite
"""

import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def check_training_status():
    """Check if training is currently running and show status."""
    
    print("="*60)
    print("ELEC475 LAB 3 - OVERNIGHT TRAINING MONITOR")
    print("="*60)
    print(f"Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if results directory exists
    if not os.path.exists('overnight_results'):
        print("❌ No overnight training results found.")
        print("   Training has not been started yet.")
        return
    
    # Check start time
    start_file = 'overnight_results/start_time.txt'
    if os.path.exists(start_file):
        with open(start_file, 'r') as f:
            start_info = f.read().strip()
        print("🚀 TRAINING STATUS:")
        print(f"   {start_info}")
    
    # Check for completion
    if os.path.exists('overnight_results/experiment_results.json'):
        print("✅ Training completed successfully!")
        show_final_results()
        return
    
    # Check for errors
    if os.path.exists('overnight_results/error_log.txt'):
        print("❌ Training failed!")
        with open('overnight_results/error_log.txt', 'r') as f:
            error_info = f.read().strip()
        print(f"   Error details: {error_info}")
        return
    
    if os.path.exists('overnight_results/interrupted.txt'):
        print("⏹️ Training was interrupted!")
        with open('overnight_results/interrupted.txt', 'r') as f:
            interrupt_info = f.read().strip()
        print(f"   Interruption details: {interrupt_info}")
        return
    
    # Check progress
    print("🔄 Training in progress...")
    show_current_progress()

def show_current_progress():
    """Show current training progress."""
    
    print("\\n📊 PROGRESS STATUS:")
    
    # Check for individual model files
    model_files = [f for f in os.listdir('overnight_results') if f.endswith('_best_model.pth')]
    
    if model_files:
        print(f"   ✅ Completed models: {len(model_files)}")
        for model_file in model_files:
            config_name = model_file.replace('_best_model.pth', '')
            print(f"      • {config_name}")
    else:
        print("   ⏳ No models completed yet...")
    
    # Check for any log files or temp results
    temp_files = [f for f in os.listdir('overnight_results') if f.endswith('.json') or f.endswith('.txt')]
    if temp_files:
        print(f"\\n📁 Current files in results:")
        for temp_file in temp_files:
            file_path = f'overnight_results/{temp_file}'
            file_size = os.path.getsize(file_path)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"      • {temp_file} ({file_size} bytes, {mod_time.strftime('%H:%M:%S')})")

def show_final_results():
    """Show final training results."""
    
    print("\\n🎉 FINAL RESULTS:")
    
    # Load results
    with open('overnight_results/experiment_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"   📅 Completed: {results['experiment_info']['timestamp']}")
    print(f"   🖥️ Device: {results['experiment_info']['device']}")
    print(f"   🔢 Configurations: {results['experiment_info']['total_configs']}")
    
    print("\\n🏆 PERFORMANCE RANKING:")
    for rank_info in results['performance_ranking']:
        print(f"   {rank_info['rank']}. {rank_info['name']:<25} | "
               f"mIoU: {rank_info['best_miou']:.6f} | "
               f"Epochs: {rank_info['epochs_to_best']}")
    
    # Show best configuration details
    best_config = results['performance_ranking'][0]
    best_details = next(c for c in results['configurations'] if c['name'] == best_config['name'])
    
    print(f"\\n🥇 BEST CONFIGURATION DETAILS:")
    print(f"   Name: {best_details['name']}")
    print(f"   Method: {best_details['method']}")
    print(f"   Best mIoU: {best_details['best_val_miou']:.6f}")
    print(f"   Achieved at epoch: {best_details['best_epoch']}")
    print(f"   Total epochs: {best_details['total_epochs']}")
    
    if 'kd_parameters' in best_details:
        kd = best_details['kd_parameters']
        print(f"   KD Parameters: α={kd['alpha']:.3f}, β={kd['beta']:.3f}, γ={kd['gamma']:.3f}, T={kd['temperature']}")
    
    # Check for analysis plots
    if os.path.exists('overnight_results/overnight_training_analysis.png'):
        print(f"\\n📈 Training analysis plot available:")
        print(f"   overnight_results/overnight_training_analysis.png")
    
    if os.path.exists('overnight_results/experiment_summary.txt'):
        print(f"\\n📄 Detailed summary available:")
        print(f"   overnight_results/experiment_summary.txt")

def plot_live_progress():
    """Create a live progress plot if data is available."""
    
    if not os.path.exists('overnight_results'):
        print("No results directory found.")
        return
    
    # Look for any progress data
    progress_files = []
    for file in os.listdir('overnight_results'):
        if file.endswith('.json') and 'progress' in file:
            progress_files.append(file)
    
    if not progress_files:
        print("No progress data available for plotting yet.")
        return
    
    print(f"Found {len(progress_files)} progress files")
    # Could implement live plotting here if needed

def main():
    """Main monitoring function."""
    
    try:
        check_training_status()
        
        # Offer to show live updates
        if os.path.exists('overnight_results') and not os.path.exists('overnight_results/experiment_results.json'):
            print("\\n" + "="*60)
            response = input("🔄 Show live monitoring? (y/N): ").strip().lower()
            
            if response == 'y':
                print("\\n📊 Live monitoring (Press Ctrl+C to stop)...")
                try:
                    while True:
                        print(f"\\r⏰ {datetime.now().strftime('%H:%M:%S')} - Checking progress...", end='', flush=True)
                        time.sleep(30)  # Check every 30 seconds
                        
                        if os.path.exists('overnight_results/experiment_results.json'):
                            print("\\n✅ Training completed!")
                            show_final_results()
                            break
                            
                except KeyboardInterrupt:
                    print("\\n⏹️ Live monitoring stopped.")
        
        print("\\n💡 TIP: Run this script again anytime to check progress!")
        
    except Exception as e:
        print(f"❌ Error checking training status: {e}")

if __name__ == '__main__':
    main()