import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

class CLIPImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Use CLIP's own vision encoder for better alignment
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Unfreeze last few layers for fine-tuning
        for param in self.vision_model.parameters():
            param.requires_grad = False  # Freeze everything first
        
        # Unfreeze last transformer layers
        for layer in self.vision_model.vision_model.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Simple projection to match dimensions
        self.projection = nn.Sequential(
            nn.Linear(768, 512),  # CLIP vision output is 768
            nn.LayerNorm(512),
            nn.GELU()
        )
        # Better initialization 
        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)
    
    def forward(self, x):
        vision_outputs = self.vision_model(x)
        features = vision_outputs.pooler_output  # [batch_size, 768]
        return self.projection(features)


class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = CLIPImageEncoder()
        self.text_encoder = CLIPTextEncoder()
        
        # Learnable temperature parameter - start much lower for better initial recall
        self.logit_scale = nn.Parameter(torch.ones([]) * 1.6094)  # ln(1/0.2) = 5x multiplier
    
    def encode_image(self, images):
        return self.image_encoder(images)
    
    def encode_text(self, texts):
        return self.text_encoder(texts)
    
    def forward(self, images, text_embeddings=None, texts=None):
        """
        Forward pass for CLIP model
        Args:
            images: batch of images
            text_embeddings: pre-computed text embeddings (if available)
            texts: raw text strings (if text_embeddings not provided)
        Returns:
            image_features: normalized image features
            logit_scale: learnable temperature parameter
        """
        image_features = self.encode_image(images)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        if text_embeddings is not None:
            # Always normalize text embeddings for consistency
            text_features = F.normalize(text_embeddings, p=2, dim=1)
        else:
            # Encode raw text
            text_features = self.encode_text(texts)
            text_features = F.normalize(text_features, p=2, dim=1)
        
        return image_features, self.logit_scale


class CLIPTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        device = next(self.text_encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            # Use pooler_output for proper CLIP text features instead of CLS token
            return outputs.pooler_output
