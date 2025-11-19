import kagglehub
import os
import shutil
import time

# Get the current directory (Lab4)
current_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(current_dir, "coco-2014-dataset")

print(f"Downloading COCO 2014 dataset to: {target_path}")
print("⚠️  This is a large dataset (~25GB) - download may take a while")

# Download the dataset with retry logic
max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"📥 Attempt {attempt + 1}/{max_retries}: Starting download...")
        
        # Configure kagglehub for large downloads
        os.environ['KAGGLE_TIMEOUT'] = '3600'  # 1 hour timeout
        
        downloaded_path = kagglehub.dataset_download("jeffaudi/coco-2014-dataset-for-yolov3")
        print(f"✅ Downloaded successfully to: {downloaded_path}")
        
        # If the downloaded path is not in our Lab4 directory, move it there
        if not downloaded_path.startswith(current_dir):
            print(f"📁 Moving dataset from {downloaded_path} to {target_path}")
            
            # Remove target if it exists
            if os.path.exists(target_path):
                print("🗑️  Removing existing dataset directory...")
                shutil.rmtree(target_path)
            
            # Move the dataset
            print("🚚 Moving dataset files...")
            shutil.move(downloaded_path, target_path)
            final_path = target_path
        else:
            final_path = downloaded_path
        
        print(f"🎉 Dataset available at: {final_path}")
        
        # List the contents of the dataset
        print("\n📂 Dataset contents:")
        for item in os.listdir(final_path):
            item_path = os.path.join(final_path, item)
            if os.path.isdir(item_path):
                # Count files in directory
                try:
                    file_count = len(os.listdir(item_path))
                    print(f"📁 {item}/ ({file_count} items)")
                except:
                    print(f"📁 {item}/")
            else:
                # Show file size
                try:
                    size_mb = os.path.getsize(item_path) / (1024 * 1024)
                    print(f"📄 {item} ({size_mb:.1f} MB)")
                except:
                    print(f"📄 {item}")
        
        print("\n✅ Download completed successfully!")
        break
        
    except Exception as e:
        print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
        
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 30  # Progressive backoff: 30s, 60s, 90s
            print(f"⏳ Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        else:
            print("\n💡 All attempts failed. Suggestions:")
            print("1. Check your internet connection")
            print("2. Try using Kaggle CLI directly: kaggle datasets download -d jeffaudi/coco-2014-dataset-for-yolov3")
            print("3. Download manually from: https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3")
            print(f"4. Extract manually to: {target_path}")
            print("5. Try again later when network conditions are better")