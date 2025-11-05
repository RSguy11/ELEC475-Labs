import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for importing models
sys.path.append('..')

from SnoutNet.model import SnoutNet
from AlexNet.model import AlexNetSnout
from VGG.model import VGG16Snout

class EnsembleSnoutNet(nn.Module):
    """
    Ensemble model that combines predictions from SnoutNet, AlexNet-based, and VGG16-based models.
    
    This ensemble uses weighted averaging with learned weights.
    """
    
    def __init__(self, combination_method='weighted', pretrained_paths=None):
        super(EnsembleSnoutNet, self).__init__()
        
        if combination_method != 'weighted':
            raise ValueError("Only 'weighted' combination method is supported")
            
        self.combination_method = combination_method
        
        # Initialize individual models
        self.snoutnet = SnoutNet()
        self.alexnet = AlexNetSnout(pretrained=True)
        self.vgg16 = VGG16Snout(pretrained=True)
        
        # Load pretrained weights if provided
        if pretrained_paths:
            self._load_pretrained_weights(pretrained_paths)
            
        # Freeze individual model parameters during ensemble training
        self._freeze_base_models()
        
        # Ensemble combination layers - only weighted averaging
        # Learnable weights for weighted average
        self.model_weights = nn.Parameter(torch.ones(3) / 3.0)
    
    def _load_pretrained_weights(self, pretrained_paths):
        """Load pretrained weights for individual models"""
        if 'snoutnet' in pretrained_paths:
            try:
                self.snoutnet.load_state_dict(torch.load(pretrained_paths['snoutnet'], map_location='cpu'))
                print(f"[OK] Loaded SnoutNet weights from {pretrained_paths['snoutnet']}")
            except Exception as e:
                print(f"[WARNING] Could not load SnoutNet weights: {e}")
                
        if 'alexnet' in pretrained_paths:
            try:
                self.alexnet.load_state_dict(torch.load(pretrained_paths['alexnet'], map_location='cpu'))
                print(f"[OK] Loaded AlexNet weights from {pretrained_paths['alexnet']}")
            except Exception as e:
                print(f"[WARNING] Could not load AlexNet weights: {e}")
                
        if 'vgg16' in pretrained_paths:
            try:
                self.vgg16.load_state_dict(torch.load(pretrained_paths['vgg16'], map_location='cpu'))
                print(f"[OK] Loaded VGG16 weights from {pretrained_paths['vgg16']}")
            except Exception as e:
                print(f"[WARNING] Could not load VGG16 weights: {e}")
    
    def _freeze_base_models(self):
        """Freeze parameters of base models to prevent training"""
        for param in self.snoutnet.parameters():
            param.requires_grad = False
        for param in self.alexnet.parameters():
            param.requires_grad = False
        for param in self.vgg16.parameters():
            param.requires_grad = False
    
    def unfreeze_base_models(self):
        """Unfreeze base models for fine-tuning"""
        for param in self.snoutnet.parameters():
            param.requires_grad = True
        for param in self.alexnet.parameters():
            param.requires_grad = True
        for param in self.vgg16.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """Forward pass through ensemble using weighted averaging"""
        # Get predictions from all models
        with torch.no_grad() if self._base_models_frozen() else torch.enable_grad():
            snoutnet_pred = self.snoutnet(x)
            alexnet_pred = self.alexnet(x)
            vgg16_pred = self.vgg16(x)
        
        # Weighted averaging with learnable weights
        weights = F.softmax(self.model_weights, dim=0)
        ensemble_pred = (weights[0] * snoutnet_pred + 
                       weights[1] * alexnet_pred + 
                       weights[2] * vgg16_pred)
        
        return ensemble_pred
    
    def _base_models_frozen(self):
        """Check if base models are frozen"""
        return not next(self.snoutnet.parameters()).requires_grad
    
    def get_individual_predictions(self, x):
        """Get predictions from individual models for analysis"""
        with torch.no_grad():
            return {
                'snoutnet': self.snoutnet(x),
                'alexnet': self.alexnet(x),
                'vgg16': self.vgg16(x)
            }
    
    def get_model_weights(self):
        """Get current model weights (for weighted ensemble)"""
        if self.combination_method == 'weighted':
            weights = F.softmax(self.model_weights, dim=0)
            return {
                'snoutnet': weights[0].item(),
                'alexnet': weights[1].item(),
                'vgg16': weights[2].item()
            }
        return None
    
    def get_model_info(self):
        """Return ensemble model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': f'EnsembleSnoutNet ({self.combination_method})',
            'combination_method': self.combination_method,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'individual_models': ['SnoutNet', 'AlexNet-based', 'VGG16-based'],
            'base_models_frozen': self._base_models_frozen()
        }

def test_ensemble_model():
    """Test the ensemble model with dummy data"""
    print("[TEST] Testing EnsembleSnoutNet model...")
    
    # Test only weighted ensemble method
    method = 'weighted'
    print(f"\n[MODEL INFO] Testing {method} ensemble:")
    
    model = EnsembleSnoutNet(combination_method=method)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(4, 3, 227, 227)  # Batch of 4 images
    
    try:
        # Forward pass
        output = model(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
        
        # Get model info
        info = model.get_model_info()
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Trainable parameters: {info['trainable_parameters']:,}")
        
        # Get model weights for weighted ensemble
        weights = model.get_model_weights()
        print(f"  Model weights: {weights}")
        
        print(f"  [OK] {method} ensemble test passed!")
        
    except Exception as e:
        print(f"  [FAIL] {method} ensemble test failed: {e}")

if __name__ == "__main__":
    test_ensemble_model()