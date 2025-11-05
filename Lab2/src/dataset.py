import os
import torch
import random
from torchvision.io import read_image
from torchvision.transforms import Resize, ColorJitter
from torch.utils.data import Dataset
import torch.nn.functional as F

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, augment=False):
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        
        # Initialize augmentation transforms
        if self.augment:
            self.color_jitter = ColorJitter(
                brightness=0.2,    # +/-20% brightness
                contrast=0.2,      # +/-20% contrast  
                saturation=0.2,    # +/-20% saturation
                hue=0.1           # +/-10% hue
            )
        
        # Parse the annotation file
        self.data = []
        with open(annotations_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    #Parsing to get the filename and coordinates
                    filename, coords = line.split(',', 1)
                    coords = coords.strip().strip('"()')  # Remove quotes and parentheses
                    x, y = map(int, coords.split(', '))
                    self.data.append((filename, [x, y]))
        
        self.resize = Resize((227, 227))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, coords = self.data[idx]
        
        img_path = os.path.join(self.img_dir, filename)
        image = read_image(img_path) 
        
        if image.shape[0] == 1: 
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3, :, :]
        
        original_height, original_width = image.shape[1], image.shape[2]
        
        image = image.float() / 255.0
        
        # Resize to 227x227
        image = self.resize(image)
        
        # Scale coordinates to match the resized image
        scale_x = 227.0 / original_width
        scale_y = 227.0 / original_height
        
        scaled_coords = [
            coords[0] * scale_x,
            coords[1] * scale_y
        ]
        
        if self.augment:
            # 1. Horizontal Flipping (50% chance)
            if random.random() > 0.5:
                image = torch.flip(image, dims=[2])  # Flip horizontally
                scaled_coords[0] = 227 - scaled_coords[0]  # Adjust x-coordinate
            # 2. Color Jittering (always applied when augment=True)
            image = self.color_jitter(image)

        
        coords = torch.tensor(scaled_coords, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, coords