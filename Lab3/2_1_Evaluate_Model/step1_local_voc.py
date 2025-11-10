"""
ELEC475 Lab 3 - Step 1: FCN-ResNet50 evaluation on local PASCAL VOC 2012
"""

import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

class LocalVOCDataset(Dataset):
    """Local PASCAL VOC 2012 dataset."""
    
    def __init__(self, voc_root, split='val', transform=None, target_transform=None, max_samples=None):
        self.voc_root = voc_root
        self.transform = transform
        self.target_transform = target_transform
        
        print(f"Initializing dataset with root: {voc_root}")
        
        # Paths
        self.images_dir = os.path.join(voc_root, 'JPEGImages')
        self.masks_dir = os.path.join(voc_root, 'SegmentationClass')
        split_file = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{split}.txt')
        
        # Check if paths exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # Load image IDs
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # Limit samples if specified
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]
        
        print(f"Loaded {len(self.image_ids)} images from {split} set")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Load image and mask
        img_path = os.path.join(self.images_dir, f'{img_id}.jpg')
        mask_path = os.path.join(self.masks_dir, f'{img_id}.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask

def squeeze_and_long(x):
    """Transform function to squeeze and convert to long - needed for multiprocessing."""
    return x.squeeze(0).long()

def calculate_miou(pred, target):
    """Calculate mean IoU."""
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    ious = []
    
    for cls in range(21):  # PASCAL VOC has 21 classes
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        ious.append(intersection / union if union > 0 else float('nan'))
    
    return np.nanmean(ious)

def main():
    print("ELEC475 Lab 3 - Step 1: FCN-ResNet50 Evaluation")
    print("Using Local PASCAL VOC 2012 Dataset")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset path - works from Lab3 directory
    voc_root = r"pascal-voc-2012-dataset\versions\1\VOC2012_train_val\VOC2012_train_val"
    
    # Check if dataset exists
    if not os.path.exists(voc_root):
        print(f"Dataset not found at: {voc_root}")
        print("Current directory:", os.getcwd())
        print("Looking for dataset...")
        
        # Try alternative paths
        alt_paths = [
            r".\pascal-voc-2012-dataset\versions\1\VOC2012_train_val\VOC2012_train_val",
            r"..\pascal-voc-2012-dataset\versions\1\VOC2012_train_val\VOC2012_train_val",
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                voc_root = alt_path
                print(f"Found dataset at: {alt_path}")
                break
        else:
            print("Dataset not found in any expected location!")
            print("Please ensure the dataset is in the Lab3 directory")
            return
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((520, 520), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        squeeze_and_long  # Use named function instead of lambda
    ])
    
    # Load dataset
    try:
        print(f"Using dataset path: {voc_root}")
        dataset = LocalVOCDataset(
            voc_root=voc_root,
            split='val',
            transform=transform,
            target_transform=target_transform,
            max_samples=100  # Limit for faster evaluation
        )
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
        print("Please check that the dataset is properly extracted")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Dataset path:", voc_root)
        print("Current directory:", os.getcwd())
        return
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)  # num_workers=0 fixes Windows multiprocessing issues
    print("Dataset loaded successfully!")
    
    # Load model
    print("Loading FCN-ResNet50...")
    model = fcn_resnet50(weights='DEFAULT').to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Evaluate
    total_miou = 0
    count = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)['out']
            predictions = torch.argmax(outputs, dim=1)
            
            for pred, target in zip(predictions, targets):
                miou = calculate_miou(pred, target)
                if not np.isnan(miou):
                    total_miou += miou
                    count += 1
    
    avg_miou = total_miou / count if count > 0 else 0
    print(f"\nResults: mIoU = {avg_miou:.4f}")
    print("✓ Step 1 completed successfully!")
    
    # PASCAL VOC class names for reference
    print(f"\nEvaluated on {count} images from PASCAL VOC 2012 validation set")
    print("Classes: background, aeroplane, bicycle, bird, boat, bottle, bus, car,")
    print("        cat, chair, cow, diningtable, dog, horse, motorbike, person,")
    print("        pottedplant, sheep, sofa, train, tvmonitor")

if __name__ == "__main__":
    main()