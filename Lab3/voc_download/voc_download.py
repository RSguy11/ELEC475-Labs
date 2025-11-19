#pip install kagglehub
import kagglehub
import os
import shutil

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level to Lab3 directory
lab3_dir = os.path.dirname(script_dir)
# Target path in Lab3 directory
target_path = os.path.join(lab3_dir, "pascal-voc-2012-dataset")

# Download latest version
downloaded_path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset")

print("Downloaded to:", downloaded_path)

# If the downloaded path is not in our Lab3 directory, move it there
if not downloaded_path.startswith(lab3_dir):
    print(f"Moving dataset from {downloaded_path} to {target_path}")
    
    # Remove target if it exists
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    
    # Move the dataset
    shutil.move(downloaded_path, target_path)
    path = target_path
else:
    path = downloaded_path

print("Dataset available at:", path)
