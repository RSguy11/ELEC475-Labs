"""
Automated Knowledge Distillation Training & Testing Pipeline
Runs both response-based and feature-based distillation sequentially, then evaluates both.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(command, description, working_dir=None):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {command}")
    print(f"Directory: {working_dir if working_dir else os.getcwd()}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        if working_dir:
            result = subprocess.run(command, shell=True, cwd=working_dir, check=True, 
                                  capture_output=False, text=True)
        else:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=False, text=True)
        
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print(f"\n✅ {description} completed successfully!")
        print(f"⏱️  Time taken: {minutes}m {seconds}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print(f"\n❌ {description} failed!")
        print(f"⏱️  Time taken: {minutes}m {seconds}s")
        print(f"Error code: {e.returncode}")
        return False

def main():
    """Main pipeline execution."""
    pipeline_start = time.time()
    
    print("🎓 ELEC475 Lab 3: Knowledge Distillation Training Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    print("\n📋 PIPELINE OUTPUTS:")
    print("• Epoch-by-epoch training progress with loss values")
    print("• Validation mIoU computed each epoch") 
    print("• Best model checkpoints saved automatically")
    print("• Loss curves and performance charts generated")
    print("• Prediction comparison visualizations")
    
    # Get current directory
    base_dir = os.getcwd()
    response_dir = os.path.join(base_dir, "response_based_distillation")
    feature_dir = os.path.join(base_dir, "feature_based_distillation")
    
    # Check if directories exist
    if not os.path.exists(response_dir):
        print(f"❌ Directory not found: {response_dir}")
        return
    
    if not os.path.exists(feature_dir):
        print(f"❌ Directory not found: {feature_dir}")
        return
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Train Feature-Based Model (CHANGED ORDER)
    if run_command("python train.py", "Feature-Based Distillation Training (30 epochs)", feature_dir):
        success_count += 1
    else:
        print("\n🛑 Stopping pipeline due to feature-based training failure")
        return
    
    # Step 2: Train Response-Based Model (CHANGED ORDER)
    if run_command("python train.py", "Response-Based Distillation Training (30 epochs)", response_dir):
        success_count += 1
    else:
        print("\n🛑 Stopping pipeline due to response-based training failure")
        return
    
    # Step 3: Test Feature-Based Model (CHANGED ORDER)
    if run_command("python test.py", "Feature-Based Model Evaluation", feature_dir):
        success_count += 1
    else:
        print("\n⚠️ Feature-based testing failed, continuing...")
    
    # Step 4: Test Response-Based Model (CHANGED ORDER)
    if run_command("python test.py", "Response-Based Model Evaluation", response_dir):
        success_count += 1
    else:
        print("\n⚠️ Response-based testing failed, continuing...")
    
    # Pipeline summary
    total_time = time.time() - pipeline_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'='*60}")
    print("🎉 KNOWLEDGE DISTILLATION PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Completed steps: {success_count}/{total_steps}")
    print(f"⏱️  Total time: {hours}h {minutes}m {seconds}s")
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📊 RESULTS GENERATED:")
    print(f"Feature-based:")
    print(f"  - Model: feature_based_distillation/feature_based_model.pth") 
    print(f"  - Loss plots: feature_based_distillation/results_images/")
    print(f"Response-based:")
    print(f"  - Model: response_based_distillation/response_based_model.pth")
    print(f"  - Loss plots: response_based_distillation/results_images/")
    
    if success_count == total_steps:
        print(f"\n🎓 Lab 3 Knowledge Distillation: FULLY COMPLETE!")
    else:
        print(f"\n⚠️  Pipeline completed with {total_steps - success_count} failures")

if __name__ == "__main__":
    print("Starting automated knowledge distillation pipeline...")
    main()