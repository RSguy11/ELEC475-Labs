#pip install kagglehub
import kagglehub
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up one level to Lab3 directory
lab3_dir = os.path.dirname(script_dir)
# Set the download path to Lab3 directory
download_path = os.path.join(lab3_dir, "pascal-voc-2012-dataset")

# Ensure the download directory exists
os.makedirs(download_path, exist_ok=True)

print(f"Downloading PASCAL VOC 2012 dataset to: {download_path}")

# Download latest version to specific path
path = kagglehub.dataset_download("gopalbhattrai/pascal-voc-2012-dataset", path=download_path)

print("Dataset downloaded successfully!")
print("Path to dataset files:", path)