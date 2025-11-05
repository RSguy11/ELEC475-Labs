#!/usr/bin/env python3
"""
Overnight Training and Testing Script for SnoutNet Model Variants

This script automatically trains and tests all three SnoutNet model variants:
- Custom SnoutNet (baseline and augmented)
- SnoutNet-A (AlexNet-based, baseline and augmented)  
- SnoutNet-V (VGG16-based, baseline and augmented)

Total expected runtime: 4-8 hours depending on hardware
"""

import subprocess
import os
import sys
import time
from datetime import datetime, timedelta
import logging
import warnings

# Suppress common PIL/JPEG warnings to clean up output
warnings.filterwarnings("ignore", message="Corrupt JPEG data")
warnings.filterwarnings("ignore", message=".*extraneous bytes before marker.*")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('overnight_training_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)

class OvernightTrainer:
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {}
        self.current_dir = os.getcwd()
        
        # Ensure we're in the src directory
        if not os.path.basename(self.current_dir) == 'src':
            if os.path.exists('src'):
                os.chdir('src')
                self.current_dir = os.getcwd()
            else:
                logging.error("Please run this script from the Lab2 directory or Lab2/src directory")
                sys.exit(1)
        
        logging.info(f"Starting overnight training session from: {self.current_dir}")
        logging.info(f"Session started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_command(self, command, cwd, description, timeout=7200):  # 2 hour timeout
        """Run a command and capture its output with real-time streaming"""
        logging.info(f"\n{'='*60}")
        logging.info(f"STARTING: {description}")
        logging.info(f"Command: {command}")
        logging.info(f"Directory: {cwd}")
        logging.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Change to the specified directory
            original_cwd = os.getcwd()
            os.chdir(cwd)
            
            # Run the command with real-time output streaming
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    output_lines.append(line)
                    
                    # Filter out common JPEG corruption warnings to clean up output
                    if not ("Corrupt JPEG data" in line or "extraneous bytes before marker" in line):
                        # Log each line in real-time (excluding JPEG warnings)
                        logging.info(f"  {line}")
            
            # Wait for process to complete and get return code
            return_code = process.poll()
            
            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                process.kill()
                raise subprocess.TimeoutExpired(command, timeout)
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Combine all output
            full_output = '\n'.join(output_lines)
            
            if return_code == 0:
                logging.info(f"[SUCCESS]: {description}")
                logging.info(f"Duration: {duration/60:.1f} minutes")
                status = "SUCCESS"
            else:
                logging.error(f"[FAILED]: {description}")
                logging.error(f"Return code: {return_code}")
                status = "FAILED"
                
            return {
                'status': status,
                'duration': duration,
                'returncode': return_code,
                'stdout': full_output,
                'stderr': ''
            }
            
        except subprocess.TimeoutExpired:
            logging.error(f"[TIMEOUT] TIMEOUT: {description} (exceeded {timeout/60:.1f} minutes)")
            if 'process' in locals():
                process.kill()
            os.chdir(original_cwd)
            return {
                'status': 'TIMEOUT',
                'duration': timeout,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timed out'
            }
        except Exception as e:
            logging.error(f"[ERROR] ERROR: {description} - {str(e)}")
            if 'process' in locals():
                try:
                    process.kill()
                except:
                    pass
            os.chdir(original_cwd)
            return {
                'status': 'ERROR',
                'duration': time.time() - start_time,
                'returncode': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    def run_training_sequence(self):
        """Run the complete training and testing sequence"""
        
        # Define the complete sequence of operations
        operations = [
            # SnoutNet (Original)
            {
                'name': 'SnoutNet_Baseline_Train',
                'command': 'python train.py --epochs 50',
                'cwd': os.path.join(self.current_dir, 'SnoutNet'),
                'description': 'Training SnoutNet (Baseline)'
            },
            {
                'name': 'SnoutNet_Augmented_Train',
                'command': 'python train.py -a true --epochs 50',
                'cwd': os.path.join(self.current_dir, 'SnoutNet'),
                'description': 'Training SnoutNet (Augmented)'
            },
            {
                'name': 'SnoutNet_Test_Baseline',
                'command': 'python test.py -t baseline',
                'cwd': os.path.join(self.current_dir, 'SnoutNet'),
                'description': 'Testing SnoutNet (Baseline)'
            },
            {
                'name': 'SnoutNet_Test_Augmented',
                'command': 'python test.py -t augmented',
                'cwd': os.path.join(self.current_dir, 'SnoutNet'),
                'description': 'Testing SnoutNet (Augmented)'
            },
            
            # AlexNet-based SnoutNet
            {
                'name': 'SnoutNet_A_Baseline_Train',
                'command': 'python train.py --epochs 50',
                'cwd': os.path.join(self.current_dir, 'AlexNet'),
                'description': 'Training SnoutNet-A (Baseline)'
            },
            {
                'name': 'SnoutNet_A_Augmented_Train',
                'command': 'python train.py -a true --epochs 50',
                'cwd': os.path.join(self.current_dir, 'AlexNet'),
                'description': 'Training SnoutNet-A (Augmented)'
            },
            {
                'name': 'SnoutNet_A_Test_Baseline',
                'command': 'python test.py -t baseline',
                'cwd': os.path.join(self.current_dir, 'AlexNet'),
                'description': 'Testing SnoutNet-A (Baseline)'
            },
            {
                'name': 'SnoutNet_A_Test_Augmented',
                'command': 'python test.py -t augmented',
                'cwd': os.path.join(self.current_dir, 'AlexNet'),
                'description': 'Testing SnoutNet-A (Augmented)'
            },
            
            # VGG16-based SnoutNet
            {
                'name': 'SnoutNet_V_Baseline_Train',
                'command': 'python train.py --epochs 50',
                'cwd': os.path.join(self.current_dir, 'VGG'),
                'description': 'Training SnoutNet-V (Baseline)'
            },
            {
                'name': 'SnoutNet_V_Augmented_Train',
                'command': 'python train.py -a true --epochs 50',
                'cwd': os.path.join(self.current_dir, 'VGG'),
                'description': 'Training SnoutNet-V (Augmented)'
            },
            {
                'name': 'SnoutNet_V_Test_Baseline',
                'command': 'python test.py -t baseline',
                'cwd': os.path.join(self.current_dir, 'VGG'),
                'description': 'Testing SnoutNet-V (Baseline)'
            },
            {
                'name': 'SnoutNet_V_Test_Augmented',
                'command': 'python test.py -t augmented',
                'cwd': os.path.join(self.current_dir, 'VGG'),
                'description': 'Testing SnoutNet-V (Augmented)'
            },
            
            # Ensemble Models (trained after individual models)
            {
                'name': 'Ensemble_Weighted_Baseline_Train',
                'command': 'python train.py --method weighted -a false --epochs 30',
                'cwd': os.path.join(self.current_dir, 'Ensemble'),
                'description': 'Training Ensemble-Weighted (Baseline)'
            },
            {
                'name': 'Ensemble_Weighted_Augmented_Train',
                'command': 'python train.py --method weighted -a true --epochs 30',
                'cwd': os.path.join(self.current_dir, 'Ensemble'),
                'description': 'Training Ensemble-Weighted (Augmented)'
            },
            {
                'name': 'Ensemble_Meta_Baseline_Train',
                'command': 'python train.py --method meta_learner -a false --epochs 30',
                'cwd': os.path.join(self.current_dir, 'Ensemble'),
                'description': 'Training Ensemble-MetaLearner (Baseline)'
            },
            {
                'name': 'Ensemble_Meta_Augmented_Train',
                'command': 'python train.py --method meta_learner -a true --epochs 30',
                'cwd': os.path.join(self.current_dir, 'Ensemble'),
                'description': 'Training Ensemble-MetaLearner (Augmented)'
            },
            {
                'name': 'Ensemble_Weighted_Test',
                'command': 'python test.py --method weighted -t auto',
                'cwd': os.path.join(self.current_dir, 'Ensemble'),
                'description': 'Testing Ensemble-Weighted (Auto)'
            },
            {
                'name': 'Ensemble_Meta_Test',
                'command': 'python test.py --method meta_learner -t auto',
                'cwd': os.path.join(self.current_dir, 'Ensemble'),
                'description': 'Testing Ensemble-MetaLearner (Auto)'
            }
        ]
        
        logging.info(f"=== STARTING OVERNIGHT TRAINING SESSION ===")
        logging.info(f"Total operations to execute: {len(operations)}")
        
        # Estimate completion time with more accurate durations
        individual_train_ops = [op for op in operations if 'train' in op['name'].lower() and 'ensemble' not in op['name'].lower()]
        ensemble_train_ops = [op for op in operations if 'train' in op['name'].lower() and 'ensemble' in op['name'].lower()]
        test_ops = [op for op in operations if 'test' in op['name'].lower()]
        
        estimated_duration = len(individual_train_ops) * 60  # 60 min per individual model training
        estimated_duration += len(ensemble_train_ops) * 30   # 30 min per ensemble training
        estimated_duration += len(test_ops) * 5              # 5 min per test
        
        estimated_completion = self.start_time + timedelta(minutes=estimated_duration)
        logging.info(f"Estimated completion: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Estimated total duration: {estimated_duration/60:.1f} hours")
        logging.info(f"  Individual model training: {len(individual_train_ops)} ops × 60 min = {len(individual_train_ops) * 60} min")
        logging.info(f"  Ensemble training: {len(ensemble_train_ops)} ops × 30 min = {len(ensemble_train_ops) * 30} min")
        logging.info(f"  Testing: {len(test_ops)} ops × 5 min = {len(test_ops) * 5} min")
        
        # Execute each operation
        for i, operation in enumerate(operations, 1):
            logging.info(f"\n=== OPERATION {i}/{len(operations)} ===")
            
            # Check if directory exists
            if not os.path.exists(operation['cwd']):
                logging.error(f"[ERROR] Directory not found: {operation['cwd']}")
                self.results[operation['name']] = {
                    'status': 'FAILED',
                    'duration': 0,
                    'error': 'Directory not found'
                }
                continue
            
            # Run the operation with appropriate timeout
            if 'ensemble' in operation['name'].lower() and 'train' in operation['name'].lower():
                timeout = 3600  # 1 hour for ensemble training (shorter epochs)
            elif 'train' in operation['name'].lower():
                timeout = 7200  # 2 hours for individual model training
            else:
                timeout = 1800  # 30 minutes for testing
                
            result = self.run_command(
                operation['command'],
                operation['cwd'],
                operation['description'],
                timeout=timeout
            )
            
            self.results[operation['name']] = result
            
            # Log progress
            elapsed = datetime.now() - self.start_time
            logging.info(f"[TIME] Session elapsed time: {elapsed}")
            logging.info(f"[PROGRESS] Operation {i}/{len(operations)} completed")
            
            # Add a small delay between operations
            time.sleep(10)
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate a comprehensive final report"""
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        logging.info(f"\n{'='*80}")
        logging.info(f"=== OVERNIGHT TRAINING SESSION COMPLETED ===")
        logging.info(f"{'='*80}")
        logging.info(f"Session started:  {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Session ended:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Total duration:   {total_duration}")
        logging.info(f"Total operations: {len(self.results)}")
        
        # Count successes and failures
        successes = sum(1 for result in self.results.values() if result['status'] == 'SUCCESS')
        failures = sum(1 for result in self.results.values() if result['status'] in ['FAILED', 'ERROR', 'TIMEOUT'])
        
        logging.info(f"[SUCCESS] Successful:    {successes}")
        logging.info(f"[FAILED] Failed:        {failures}")
        logging.info(f"Success rate:     {successes/len(self.results)*100:.1f}%")
        
        # Detailed results
        logging.info(f"\n[DETAILS] DETAILED RESULTS:")
        logging.info(f"{'='*80}")
        
        for name, result in self.results.items():
            status_emoji = "[OK]" if result['status'] == 'SUCCESS' else "[FAIL]"
            duration_str = f"{result['duration']/60:.1f}min"
            logging.info(f"{status_emoji} {name:<35} | {result['status']:<8} | {duration_str}")
        
        # Training summary
        logging.info(f"\n[TRAINING] TRAINING SUMMARY:")
        logging.info(f"{'='*80}")
        training_results = {name: result for name, result in self.results.items() if 'train' in name.lower()}
        for name, result in training_results.items():
            model_name = name.replace('_Train', '').replace('_', ' ')
            status = "[COMPLETED]" if result['status'] == 'SUCCESS' else f"[{result['status']}]"
            duration = f"{result['duration']/60:.1f} minutes"
            logging.info(f"  {model_name:<30} | {status} | {duration}")
        
        # Testing summary  
        logging.info(f"\n=== TESTING SUMMARY ===")
        logging.info(f"{'='*80}")
        testing_results = {name: result for name, result in self.results.items() if 'test' in name.lower()}
        for name, result in testing_results.items():
            model_name = name.replace('_Test_', ' ').replace('_', ' ')
            status = "[COMPLETED]" if result['status'] == 'SUCCESS' else f"[{result['status']}]"
            duration = f"{result['duration']/60:.1f} minutes"
            logging.info(f"  {model_name:<30} | {status} | {duration}")
        
        # Generate file locations
        logging.info(f"\n=== GENERATED FILES ===")
        logging.info(f"{'='*80}")
        logging.info(f"[FILES] Model files (.pth):")
        logging.info(f"  SnoutNet/best_snoutnet_model_baseline.pth")
        logging.info(f"  SnoutNet/best_snoutnet_model_augmented.pth")
        logging.info(f"  AlexNet/best_alexnet_model_baseline.pth")
        logging.info(f"  AlexNet/best_alexnet_model_augmented.pth")
        logging.info(f"  VGG/best_vgg16_model_baseline.pth")
        logging.info(f"  VGG/best_vgg16_model_augmented.pth")
        
        logging.info(f"\n[CHART] Results and plots:")
        logging.info(f"  SnoutNet/Results_Images/Baseline/")
        logging.info(f"  SnoutNet/Results_Images/Augmented/")
        logging.info(f"  AlexNet/Results_Images/Baseline/")
        logging.info(f"  AlexNet/Results_Images/Augmented/")
        logging.info(f"  VGG/Results_Images/Baseline/")
        logging.info(f"  VGG/Results_Images/Augmented/")
        
        logging.info(f"\n[LOG] Session log: overnight_training_log.txt")
        
        # Recommendations
        logging.info(f"\n[INFO] NEXT STEPS:")
        logging.info(f"{'='*80}")
        if failures == 0:
            logging.info(f"[SUCCESS] All operations completed successfully!")
            logging.info(f"[ANALYSIS] Compare model performance by examining Results_Images folders")
            logging.info(f"[CHART] Review training curves and error analysis plots")
            logging.info(f"[WINNER] Identify the best performing model variant")
        else:
            logging.info(f"[WARNING] Some operations failed. Check the log for details.")
            logging.info(f"[RETRY] Consider re-running failed operations manually")
            failed_ops = [name for name, result in self.results.items() if result['status'] != 'SUCCESS']
            for op in failed_ops:
                logging.info(f"   - {op}")
        
        logging.info(f"\n=== SESSION COMPLETE! ===")

def main():
    """Main function to run the overnight training session"""
    print("[NIGHT] SnoutNet Overnight Training Session")
    print("=====================================")
    print("This script will train and test all SnoutNet model variants.")
    print("Expected duration: 4-8 hours depending on hardware.")
    print("Progress will be logged to 'overnight_training_log.txt'")
    
    # Ask for confirmation
    response = input("\nContinue with overnight training? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Create and run the trainer
    trainer = OvernightTrainer()
    try:
        trainer.run_training_sequence()
    except KeyboardInterrupt:
        logging.info("\n[INTERRUPTED] Training session interrupted by user (Ctrl+C)")
        logging.info("Partial results may be available in individual model folders")
    except Exception as e:
        logging.error(f"\n[ERROR] Unexpected error during training session: {str(e)}")
        logging.error("Check the log file for details")
    
    print(f"\n[LOG] Complete log saved to: overnight_training_log.txt")
    print(f"Session finished!")

if __name__ == "__main__":
    main()