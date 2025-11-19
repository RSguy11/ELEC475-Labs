import kagglehub
import os
import shutil

# Get the current directory (Lab4)
current_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(current_dir, "coco-2014-dataset")

print(f"Downloading COCO 2014 dataset to: {target_path}")

# Download the dataset
try:
    downloaded_path = kagglehub.dataset_download("jeffaudi/coco-2014-dataset-for-yolov3")
    print(f"Downloaded to: {downloaded_path}")
    
    # If the downloaded path is not in our Lab4 directory, move it there
    if not downloaded_path.startswith(current_dir):
        print(f"Moving dataset from {downloaded_path} to {target_path}")
        
        # Remove target if it exists
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        
        # Move the dataset
        shutil.move(downloaded_path, target_path)
        final_path = target_path
    else:
        final_path = downloaded_path
    
    print(f"Dataset available at: {final_path}")
    
    # List the contents of the dataset
    print("\nDataset contents:")
    for item in os.listdir(final_path):
        item_path = os.path.join(final_path, item)
        if os.path.isdir(item_path):
            print(f"📁 {item}/")
        else:
            print(f"📄 {item}")
            
except Exception as e:
    print(f"❌ Download failed: {str(e)}")
    print("Please check your Kaggle API credentials and internet connection.")