import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import CLIPTextModel, CLIPTokenizer

class CLIPImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.resnet50 = models.resnet50(weights='DEFAULT')
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])  # Remove classifier
        
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )
    
    def forward(self, x):
        features = self.resnet50(x)
        features = features.view(features.size(0), -1)  # Flatten
        return self.projection(features)


class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = CLIPImageEncoder()
        self.text_encoder = CLIPTextEncoder()
    
    def encode_image(self, images):
        return self.image_encoder(images)
    
    def encode_text(self, texts):
        return self.text_encoder(texts)
    
    def forward(self, images, texts):
        image_embeds = F.normalize(self.encode_image(images), p=2, dim=1)
        text_embeds = F.normalize(self.encode_text(texts), p=2, dim=1)
        logits = torch.matmul(image_embeds, text_embeds.T)
        return {'image_embeds': image_embeds, 'text_embeds': text_embeds, 'logits': logits}


class CLIPTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            return outputs.last_hidden_state[:, 0, :]
