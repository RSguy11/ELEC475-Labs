import torch
import torch.nn as nn
import torchvision.models as models

class AlexNetSnout(nn.Module):
    """
    AlexNet-based model for pet nose localization.
    Uses pretrained AlexNet backbone with modified classifier for regression.
    """
    
    def __init__(self, pretrained=True):
        super(AlexNetSnout, self).__init__()
        
        # Load pretrained AlexNet
        self.backbone = models.alexnet(pretrained=pretrained)
        
        # Replace classifier with regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)  # Output: (x, y) coordinates
        )
        
        # Initialize regression layers
        self._initialize_regression_layers()
        
    def _initialize_regression_layers(self):
        """Initialize the regression layers with appropriate weights"""
        for module in self.backbone.classifier.modules():
            if isinstance(module, nn.Linear):
                if module.out_features != 2:  # Not the final layer
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
                else:  # Final regression layer
                    nn.init.normal_(module.weight, 0, 0.01)
                    nn.init.constant_(module.bias, 113.5)  # Initialize to image center
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_model_info(self):
        """Return model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'AlexNet-SnoutNet',
            'backbone': 'AlexNet (pretrained)',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 227, 227),
            'output_size': 2
        }

if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("[TEST] Testing AlexNet-SnoutNet model...")
    model = AlexNetSnout(pretrained=True)
    model = model.to(device)
    
    # Print model info
    info = model.get_model_info()
    print(f"\n[MODEL INFO] Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 227, 227).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n[OK] Forward pass successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")