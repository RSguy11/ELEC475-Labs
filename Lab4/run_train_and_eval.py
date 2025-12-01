# Script to run standard training and then evaluation for CLIP Lab 4
import subprocess
import sys

# Paths to scripts
TRAIN_SCRIPT = "c:/Users/naesl/ELEC475-Labs/Lab4/Training_/training_clip.py"
EVAL_SCRIPT = "c:/Users/naesl/ELEC475-Labs/Lab4/testing_evaluation/testing_evaluation.py"

def main():
    # Allow user to specify sample size as a command-line argument
    import argparse
    parser = argparse.ArgumentParser(description="Run CLIP training and evaluation with custom sample size and epochs.")
    parser.add_argument('--max_samples', type=int, default=100, help='Number of samples to use from the dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    args = parser.parse_args()

    print(f"Running standard training with max_samples={args.max_samples}, epochs={args.epochs}, batch_size={args.batch_size}...")
    train_result = subprocess.run([
        sys.executable, TRAIN_SCRIPT,
        '--max_samples', str(args.max_samples),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size)
    ])
    if train_result.returncode != 0:
        print("Training failed.")
        sys.exit(train_result.returncode)

    print("\nTraining complete. Running evaluation...")
    eval_result = subprocess.run([sys.executable, EVAL_SCRIPT, '--max_samples', str(args.max_samples)])
    if eval_result.returncode != 0:
        print("Evaluation failed.")
        sys.exit(eval_result.returncode)

    print("\nAll done!")

if __name__ == "__main__":
    main()
