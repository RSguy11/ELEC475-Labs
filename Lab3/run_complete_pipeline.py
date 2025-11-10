#!/usr/bin/env python3
"""
Lab 3 Complete Training Pipeline
Runs all training and testing components sequentially:
1. SMNet Training (2_3)
2. SMNet Testing (2_3) 
3. Knowledge Distillation Pipeline (2_4)
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import traceback

class LabTrainingPipeline:
    def __init__(self):
        self.lab3_root = os.path.dirname(os.path.abspath(__file__))
        self.start_time = time.time()
        self.log_file = os.path.join(self.lab3_root, "training_pipeline_log.txt")
        
        # Initialize log
        with open(self.log_file, 'w') as f:
            f.write(f"Lab 3 Complete Training Pipeline Started: {datetime.now()}\n")
            f.write("="*60 + "\n\n")

    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def run_script(self, script_path, script_name, working_dir=None):
        """Run a Python script and handle errors"""
        self.log(f"Starting {script_name}...")
        
        if working_dir:
            original_dir = os.getcwd()
            os.chdir(working_dir)
            self.log(f"Changed to directory: {working_dir}")
        
        try:
            # Run the script
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=7200)  # 2 hour timeout per script
            
            if result.returncode == 0:
                self.log(f"✅ {script_name} completed successfully!")
                self.log(f"Output: {result.stdout[-500:]}")  # Last 500 chars
            else:
                self.log(f"❌ {script_name} failed with return code {result.returncode}")
                self.log(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"⏰ {script_name} timed out after 2 hours")
            return False
        except Exception as e:
            self.log(f"💥 {script_name} crashed with exception: {str(e)}")
            self.log(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            if working_dir:
                os.chdir(original_dir)
                self.log(f"Returned to directory: {original_dir}")
        
        return True

    def run_pipeline(self):
        """Run the complete training pipeline"""
        self.log("🚀 Starting Lab 3 Complete Training Pipeline")
        self.log(f"Lab 3 Root Directory: {self.lab3_root}")
        
        # Step 1: SMNet Training (2_3)
        train_dir = os.path.join(self.lab3_root, "2_3_train_test_model")
        train_script = "train.py"
        
        if not self.run_script(train_script, "SMNet Training", train_dir):
            self.log("🛑 Pipeline stopped due to training failure")
            return False
        
        # Step 2: SMNet Testing (2_3)
        test_script = "test.py"
        
        if not self.run_script(test_script, "SMNet Testing", train_dir):
            self.log("🛑 Pipeline stopped due to testing failure")
            return False
        
        # Step 3: Knowledge Distillation Pipeline (2_4)
        distill_dir = os.path.join(self.lab3_root, "2_4_Distillation")
        distill_script = "distillation_pipeline.py"
        
        if not self.run_script(distill_script, "Knowledge Distillation Training", distill_dir):
            self.log("🛑 Pipeline stopped due to distillation failure")
            return False
        
        # Success!
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        self.log("🎉 Complete Training Pipeline Finished Successfully!")
        self.log(f"Total Time: {hours}h {minutes}m")
        self.log("All components completed:")
        self.log("  ✅ SMNet Training")
        self.log("  ✅ SMNet Testing") 
        self.log("  ✅ Knowledge Distillation Training")
        
        return True

    def check_prerequisites(self):
        """Check if all required files exist"""
        self.log("🔍 Checking prerequisites...")
        
        required_files = [
            "2_3_train_test_model/train.py",
            "2_3_train_test_model/test.py",
            "2_4_Distillation/distillation_pipeline.py",
            "2_2_Custom_SMNet/model.py",
            "pascal-voc-2012-dataset/versions/1/VOC2012_train_val"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = os.path.join(self.lab3_root, file_path)
            if not os.path.exists(full_path):
                missing_files.append(file_path)
        
        if missing_files:
            self.log("❌ Missing required files:")
            for file_path in missing_files:
                self.log(f"   - {file_path}")
            return False
        
        self.log("✅ All prerequisites found")
        return True

def main():
    """Main function to run the complete pipeline"""
    pipeline = LabTrainingPipeline()
    
    try:
        # Check prerequisites
        if not pipeline.check_prerequisites():
            pipeline.log("🛑 Cannot start pipeline - missing files")
            return
        
        # Run the complete pipeline
        success = pipeline.run_pipeline()
        
        if success:
            pipeline.log("🎯 Pipeline completed successfully! All deliverables ready.")
        else:
            pipeline.log("💥 Pipeline failed. Check logs for details.")
            
    except KeyboardInterrupt:
        pipeline.log("⛔ Pipeline interrupted by user")
    except Exception as e:
        pipeline.log(f"💥 Pipeline crashed: {str(e)}")
        pipeline.log(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()