"""
ELEC475 Lab 4: Simple COCO Dataset Loader for CLIP Fine-tuning
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import json
import os
from typing import Optional
import random

# CLIP constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class SimpleCOCODataset(Dataset):
    def __init__(self, data_root: str = "coco2014", split: str = 'train', max_samples: Optional[int] = None):
        self.data_root = data_root
        self.split = split
        
        # Paths
        self.images_dir = os.path.join(data_root, "images", f"{split}2014")
        self.instances_file = os.path.join(data_root, "annotations", f"instances_{split}2014.json")
        
        # Load CLIP text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])
        
        # Load data
        self._load_data()
        
        # Limit samples
        if max_samples and len(self.data_pairs) > max_samples:
            self.data_pairs = self.data_pairs[:max_samples]
        
        print(f"Loaded {len(self.data_pairs)} samples for {split}")
    
    def _load_data(self):
        """Load annotations and create image-text pairs"""
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
        
        # Create pairs
        self.data_pairs = []
        for img_id, objects in image_objects.items():
            if img_id in id_to_filename:
                filename = id_to_filename[img_id]
                image_path = os.path.join(self.images_dir, filename)
                
                if os.path.exists(image_path):
                    # Simple description
                    unique_objects = list(set(objects))[:3]  # Max 3 objects
                    description = f"An image with {', '.join(unique_objects)}"
                    
                    self.data_pairs.append({
                        'image_path': image_path,
                        'text': description,
                        'filename': filename
                    })
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        pair = self.data_pairs[idx]
        # Load image
        image = Image.open(pair['image_path']).convert('RGB')
        image_tensor = self.transform(image)
        # Encode text (not used in training, but kept for compatibility)
        with torch.no_grad():
            inputs = self.tokenizer(pair['text'], return_tensors="pt", padding=True, truncation=True)
            text_embedding = self.text_encoder(**inputs).last_hidden_state[:, 0, :].squeeze()
        return {
            'image': image_tensor,
            'text_embedding': text_embedding,
            'text': pair['text'],
            'image_path': pair['image_path']
        }

def create_dataloaders(data_root: str = "coco2014", batch_size: int = 16, max_samples: int = 100):
    """Create simple train and val dataloaders"""
    
    train_dataset = SimpleCOCODataset(data_root, 'train', max_samples)
    val_dataset = SimpleCOCODataset(data_root, 'val', max_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def test_dataset():
    """Simple test function"""
    try:
        train_loader, val_loader = create_dataloaders(max_samples=10)
        
        # Test a batch
        batch = next(iter(train_loader))
        print(f"✅ Success! Batch shape: {batch['image'].shape}")
        print(f"Text embedding shape: {batch['text_embedding'].shape}")
        print(f"Sample text: {batch['text'][0]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_dataset()