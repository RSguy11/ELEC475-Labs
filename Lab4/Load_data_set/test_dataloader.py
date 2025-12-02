# Test the SimpleCOCODataset and dataloader
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Load_data_set.load_data_set import test_dataset

if __name__ == "__main__":
    print("Testing SimpleCOCODataset and dataloaders...")
    test_dataset()
