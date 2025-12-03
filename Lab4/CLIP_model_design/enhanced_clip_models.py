# Enhanced CLIP Model Variants for Ablation Study
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

class EnhancedCLIPImageEncoder(nn.Module):
    """Enhanced CLIP Image Encoder with multiple improvements"""
    
    def __init__(self, embed_dim=512, enhancement_type='baseline'):
        super(EnhancedCLIPImageEncoder, self).__init__()
        
        # Base ResNet50 backbone
        self.resnet50 = models.resnet50(weights='DEFAULT')
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])  # Remove classifier
        
        # Enhancement-specific modifications
        self.enhancement_type = enhancement_type
        
        if enhancement_type == 'enhanced_projection':
            # Enhanced projection with residual connection and better normalization
            self.projection = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            # Add residual connection path
            self.residual_proj = nn.Linear(2048, embed_dim)
            
        elif enhancement_type == 'attention_pooling':
            # Replace average pooling with attention-based pooling
            self.attention_pool = nn.MultiheadAttention(embed_dim=2048, num_heads=8, batch_first=True)
            self.projection = nn.Sequential(
                nn.Linear(2048, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            )
            
        elif enhancement_type == 'dropout_regularization':
            # Enhanced dropout regularization
            self.projection = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.3),  # Increased dropout
                nn.Linear(1024, embed_dim),
                nn.Dropout(0.2),  # Additional dropout
                nn.LayerNorm(embed_dim)
            )
            
        else:  # baseline
            # Standard projection (same as original)
            self.projection = nn.Sequential(
                nn.Linear(2048, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Extract features from ResNet50
        if self.enhancement_type == 'attention_pooling':
            # Get feature maps before global pooling
            features = self.resnet50[:-1](x)  # [batch, 2048, 7, 7]
            batch_size, channels, h, w = features.shape
            features = features.view(batch_size, channels, h*w).permute(0, 2, 1)  # [batch, 49, 2048]
            
            # Apply attention pooling
            attn_output, _ = self.attention_pool(features, features, features)
            features = attn_output.mean(dim=1)  # [batch, 2048]
        else:
            features = self.resnet50(x)
            features = features.view(features.size(0), -1)  # Flatten
        
        # Apply projection based on enhancement type
        if self.enhancement_type == 'enhanced_projection':
            projected = self.projection(features)
            residual = self.residual_proj(features)
            embeddings = projected + residual  # Residual connection
        else:
            embeddings = self.projection(features)
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class EnhancedCLIPModel(nn.Module):
    """Enhanced CLIP Model with configurable improvements"""
    
    def __init__(self, embed_dim=512, temperature=0.07, enhancement_type='baseline'):
        super(EnhancedCLIPModel, self).__init__()
        
        self.image_encoder = EnhancedCLIPImageEncoder(embed_dim, enhancement_type)
        self.text_encoder = CLIPTextEncoder()
        self.enhancement_type = enhancement_type
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
        # Enhancement-specific parameters
        if enhancement_type == 'enhanced_projection':
            # Additional learnable parameters for enhanced projection
            self.adaptive_temperature = nn.Parameter(torch.ones([]) * 0.1)
        
    def encode_image(self, images):
        return self.image_encoder(images)
    
    def encode_text(self, texts):
        return self.text_encoder(texts)
        
    def forward(self, images, text_embeddings=None, texts=None):
        # Get image embeddings
        image_embeddings = self.encode_image(images)
        
        if text_embeddings is not None:
            # Normalize text embeddings
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        else:
            # Encode raw text
            text_embeddings = self.encode_text(texts)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        
        # Compute similarity with learnable temperature
        temperature = self.temperature.exp()
        if self.enhancement_type == 'enhanced_projection':
            # Use adaptive temperature for enhanced projection
            temperature = temperature + self.adaptive_temperature
            
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.t()) * temperature
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

class CLIPTextEncoder(nn.Module):
    """Standard CLIP Text Encoder (frozen)"""
    
    def __init__(self):
        super(CLIPTextEncoder, self).__init__()
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        device = next(self.text_encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            return outputs.pooler_output

# Enhanced Data Augmentation (for enhancement_type='data_augmentation')
import torchvision.transforms as transforms

def get_enhanced_transforms(enhancement_type='baseline'):
    """Get appropriate transforms based on enhancement type"""
    
    base_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if enhancement_type == 'data_augmentation':
        enhanced_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
        return enhanced_transforms
    
    return base_transforms

def create_enhanced_model(enhancement_type='baseline', embed_dim=512, temperature=0.07):
    """Factory function to create enhanced CLIP models"""
    return EnhancedCLIPModel(embed_dim=embed_dim, temperature=temperature, enhancement_type=enhancement_type)

# Enhanced Loss Function with Label Smoothing
def enhanced_clip_loss(logits_per_image, logits_per_text, label_smoothing=0.1):
    """Enhanced InfoNCE loss with label smoothing"""
    batch_size = logits_per_image.shape[0]
    
    # Create labels (diagonal elements are positive pairs)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    # Compute cross-entropy loss with label smoothing
    loss_image = F.cross_entropy(logits_per_image, labels, label_smoothing=label_smoothing)
    loss_text = F.cross_entropy(logits_per_text, labels, label_smoothing=label_smoothing)
    
    # Return average loss
    return (loss_image + loss_text) / 2