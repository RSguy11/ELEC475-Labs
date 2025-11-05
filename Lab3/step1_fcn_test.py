"""
ELEC475 Lab 3 - Step 1: FCN-ResNet50 evaluation on PASCAL VOC 2012
"""

import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

def load_voc_data(max_samples=50):
    """Load and preprocess PASCAL VOC dataset."""
    print("Loading PASCAL VOC 2012...")
    # Using a working dataset path - this one actually exists
    dataset = load_dataset("keremberke/pascal-voc-2012-augmented", split="validation")
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((520, 520), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        lambda x: x.squeeze(0).long()
    ])
    
    images, targets = [], []
    for sample in tqdm(dataset, desc="Processing"):
        images.append(transform(sample['image']))
        # Try different possible field names for segmentation
        seg_field = 'segmentation' if 'segmentation' in sample else 'annotation' if 'annotation' in sample else 'mask'
        targets.append(target_transform(sample[seg_field]))
    
    return TensorDataset(torch.stack(images), torch.stack(targets))

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
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data and model
    dataset = load_voc_data(max_samples=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
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
    print("✓ Step 1 completed!")

if __name__ == "__main__":
    main()