"""
ELEC475 Lab 4: Direct COCO Dataset Loader for CLIP Fine-tuning
NO CACHING - Just use the damn dataset directly!
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import json
import os
from typing import Optional

# ImageNet constants for ResNet50
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class SimpleCOCODataset(Dataset):
    def __init__(self, data_root: str = "coco2014", split: str = 'train', max_samples: Optional[int] = None):
        self.data_root = data_root
        self.split = split
        self.max_samples = max_samples
        
        # Paths
        self.images_dir = os.path.join(data_root, "images", f"{split}2014")
        self.instances_file = os.path.join(data_root, "annotations", f"instances_{split}2014.json")
        
        # Setup transforms - use ImageNet normalization for ResNet50
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Load CLIP text encoder for on-the-fly encoding
        print("Loading CLIP text encoder...")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder.to(self.device)
        self.text_encoder.eval()
        
        # Load data pairs directly from COCO annotations
        self._load_data_from_coco()
        
        print(f"Loaded {len(self.data_pairs)} samples for {split}")
    
    def _load_data_from_coco(self):
        """Load data directly from COCO annotations - NO CACHING!"""
        print(f"Loading COCO annotations from {self.instances_file}...")
        
        with open(self.instances_file, 'r') as f:
            data = json.load(f)
        
        # Create mappings
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        
        # Group objects by image
        image_objects = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            cat_name = categories.get(ann['category_id'], 'object')
            
            if img_id not in image_objects:
                image_objects[img_id] = []
            image_objects[img_id].append(cat_name)
        
        # Create simple captions
        self.data_pairs = []
        
        for img_id, objects in image_objects.items():
            if img_id in id_to_filename:
                filename = id_to_filename[img_id]
                image_path = os.path.join(self.images_dir, filename)
                
                if os.path.exists(image_path):
                    # Create simple caption
                    unique_objects = list(dict.fromkeys(objects))  # Remove duplicates
                    if len(unique_objects) == 1:
                        caption = f"A photo of a {unique_objects[0]}"
                    elif len(unique_objects) <= 3:
                        caption = f"A photo of {', '.join(unique_objects)}"
                    else:
                        caption = f"A photo of {', '.join(unique_objects[:3])} and other objects"
                    
                    self.data_pairs.append({
                        'image_path': image_path,
                        'caption': caption,
                        'image_id': img_id
                    })
                    
                    # Stop if we hit max_samples
                    if self.max_samples and len(self.data_pairs) >= self.max_samples:
                        break
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        pair = self.data_pairs[idx]
        
        # Load and transform image
        image = Image.open(pair['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Encode text on-the-fly (fast enough)
        with torch.no_grad():
            inputs = self.tokenizer(pair['caption'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Use pooler_output for proper CLIP text features and normalize
            text_output = self.text_encoder(**inputs).pooler_output
            text_embedding = torch.nn.functional.normalize(text_output, p=2, dim=1).squeeze().cpu()
        
        return {
            'image': image,
            'text_embedding': text_embedding,
            'text': pair['caption'],
            'image_id': pair['image_id']
        }


def create_dataloaders(data_root: str = "coco2014", batch_size: int = 16, max_samples: int = 100):
    """Create simple train and val dataloaders"""
    
    train_dataset = SimpleCOCODataset(data_root, 'train', max_samples)
    val_dataset = SimpleCOCODataset(data_root, 'val', max_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def test_dataset():
    """Simple test function"""
    try:
        train_loader, val_loader = create_dataloaders(max_samples=10)
        
        # Test a batch
        batch = next(iter(train_loader))
        print(f"✅ Success! Batch shape: {batch['image'].shape}")
        print(f"Text sample: {batch['text'][0]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_dataset()