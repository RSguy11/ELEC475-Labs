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
import threading
from datetime import datetime
import traceback

class LabTrainingPipeline:
    def __init__(self):
        self.lab3_root = os.path.dirname(os.path.abspath(__file__))
        self.start_time = time.time()
        self.log_file = os.path.join(self.lab3_root, "training_pipeline_log.txt")
        self.current_script = None
        self.heartbeat_active = False
        
        # Initialize log
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Lab 3 Complete Training Pipeline Started: {datetime.now()}\n")
            f.write("="*60 + "\n\n")

    def start_heartbeat(self, script_name):
        """Start heartbeat to show script is still running"""
        self.current_script = script_name
        self.heartbeat_active = True
        
        def heartbeat():
            count = 0
            while self.heartbeat_active:
                time.sleep(120)  # Every 2 minutes
                if self.heartbeat_active:
                    count += 1
                    elapsed = int(time.time() - self.start_time)
                    mins = elapsed // 60
                    secs = elapsed % 60
                    print(f"💓 [{self.current_script}] Still running... ({count*2} min elapsed, total: {mins}m{secs}s)")
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        
    def stop_heartbeat(self):
        """Stop heartbeat monitoring"""
        self.heartbeat_active = False

    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

    def run_script(self, script_path, script_name, working_dir=None):
        """Run a Python script and handle errors"""
        self.log(f"Starting {script_name}...")
        self.log(f"Expected duration: {self.get_expected_duration(script_name)}")
        
        if working_dir:
            original_dir = os.getcwd()
            os.chdir(working_dir)
            self.log(f"Changed to directory: {working_dir}")
        
        # Start heartbeat monitoring
        self.start_heartbeat(script_name)
        script_start_time = time.time()
        
        try:
            # Run the script with real-time output
            self.log(f"Executing: python {script_path}")
            process = subprocess.Popen([sys.executable, script_path], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True, 
                                     bufsize=1,
                                     universal_newlines=True)
            
            # Read output in real-time and show ALL important training info
            output_lines = []
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    output_lines.append(line)
                    
                    # Show ALL training-related output immediately
                    if line and any(keyword in line.lower() for keyword in [
                        'epoch', 'loss', 'batch', 'miou', 'accuracy', 'learning', 'training', 
                        'testing', 'validation', 'distillation', 'response', 'feature', 'hard',
                        'starting', 'loading', 'model', 'dataset', 'completed', 'error',
                        'saving', 'evaluation', 'inference', 'parameters']):
                        
                        elapsed = int(time.time() - script_start_time)
                        mins = elapsed // 60
                        secs = elapsed % 60
                        print(f"[{script_name} {mins:02d}:{secs:02d}] {line}")
                        
                        # Also log important messages 
                        if any(important in line.lower() for important in [
                            'epoch', 'loss', 'miou', 'completed', 'error', 'saving']):
                            self.log(f"[{script_name}] {line}")
            
            return_code = process.poll()
            script_duration = time.time() - script_start_time
            
            if return_code == 0:
                self.log(f"✅ {script_name} completed successfully!")
                self.log(f"Duration: {int(script_duration//60)}m {int(script_duration%60)}s")
                print(f"✅ {script_name} completed successfully!")
            else:
                self.log(f"❌ {script_name} failed with return code {return_code}")
                print(f"❌ {script_name} failed with return code {return_code}")
                # Show last 20 lines of output for debugging
                print("Last 20 lines of output:")
                for line in output_lines[-20:]:
                    print(f"[ERROR] {line}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"⏰ {script_name} timed out after 2 hours")
            print(f"⏰ {script_name} timed out after 2 hours")
            return False
        except Exception as e:
            self.log(f"💥 {script_name} crashed with exception: {str(e)}")
            print(f"💥 {script_name} crashed with exception: {str(e)}")
            return False
        finally:
            # Stop heartbeat
            self.stop_heartbeat()
            
            if working_dir:
                os.chdir(original_dir)
                self.log(f"Returned to directory: {original_dir}")
        
        return True
    
    def get_expected_duration(self, script_name):
        """Get expected duration for each script"""
        durations = {
            "SMNet Training": "30-60 minutes",
            "SMNet Testing": "5-10 minutes", 
            "Knowledge Distillation Training": "2-4 hours (50 epochs)"
        }
        return durations.get(script_name, "Unknown")

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