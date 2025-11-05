import torch
import torch.nn as nn
import torchvision.models as models

class VGG16Snout(nn.Module):

    def __init__(self, pretrained=True):
        super(VGG16Snout, self).__init__()
        
        # Load pretrained VGG16
        self.backbone = models.vgg16(pretrained=pretrained)
        
        # Replace classifier with regression head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # Output: (x, y) coordinates
        )
        
        # Initialize regression layers
        self._initialize_regression_layers()
        
    def _initialize_regression_layers(self):
        for module in self.backbone.classifier.modules():
            if isinstance(module, nn.Linear):
                if module.out_features != 2:  # Not the final layer
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
                else:  # Final regression layer
                    nn.init.normal_(module.weight, 0, 0.01)
                    nn.init.constant_(module.bias, 113.5)  # Initialize to image center
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'VGG16-SnoutNet',
            'backbone': 'VGG16 (pretrained)',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 227, 227),
            'output_size': 2
        }

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("[TEST] Testing VGG16-SnoutNet model")
    model = VGG16Snout(pretrained=True)
    model = model.to(device)
