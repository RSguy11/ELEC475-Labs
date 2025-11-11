import torch 
import torch.nn as nn
import torch.nn.functional as F

class CustomConvBlock(nn.Module):
    """Simple custom convolution block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_residual=False):
        super(CustomConvBlock, self).__init__()
        
        self.use_residual = use_residual and in_channels == out_channels and stride == 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        
        # Residual connection if applicable
        if self.use_residual:
            out += identity
            
        out = self.activation(out)
        return out

class SimpleUpsampler(nn.Module):
    """Simple upsampler"""
    def __init__(self, high_ch, low_ch, out_ch):
        super().__init__()
        
        self.fuse = nn.Sequential(
            nn.Conv2d(high_ch + low_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, high_feat, low_feat):
        up = F.interpolate(high_feat, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([up, low_feat], dim=1))

class SMNet(nn.Module):
    """
    Custom Segmentation Model (SMNet) - Simplified and Lightweight
    Architecture: Simple encoder-decoder with skip connections
    """
    def __init__(self, num_classes=21, base_dim=16):
        super(SMNet, self).__init__()
        
        # Simple initial feature extraction
        self.stem = CustomConvBlock(3, base_dim, kernel_size=3, stride=1)
        
        # Simple Encoder
        self.encoder1 = CustomConvBlock(base_dim, base_dim*2, stride=2, use_residual=False)
        self.encoder2 = CustomConvBlock(base_dim*2, base_dim*3, stride=2, use_residual=True)
        self.encoder3 = CustomConvBlock(base_dim*3, base_dim*4, stride=2, use_residual=True)
        self.encoder4 = CustomConvBlock(base_dim*4, base_dim*5, stride=2, use_residual=True)
        
        # Simple Bottleneck
        self.bottleneck = CustomConvBlock(base_dim*5, base_dim*5, use_residual=True)
        
        # Simple Decoder
        self.decoder4 = SimpleUpsampler(base_dim*5, base_dim*4, base_dim*4)
        self.decoder3 = SimpleUpsampler(base_dim*4, base_dim*3, base_dim*3)
        self.decoder2 = SimpleUpsampler(base_dim*3, base_dim*2, base_dim*2)
        self.decoder1 = SimpleUpsampler(base_dim*2, base_dim, base_dim)
        
        # Simple segmentation head
        self.seg_head = nn.Sequential(
            CustomConvBlock(base_dim, base_dim//2, kernel_size=3),
            nn.Conv2d(base_dim//2, num_classes, 1)
        )
        
        self.base_dim = base_dim
        
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Stem processing
        stem_feat = self.stem(x)  # H x W
        
        # Encoder path - extract multi-level features  
        enc1 = self.encoder1(stem_feat)   # H/2
        enc2 = self.encoder2(enc1)        # H/4
        enc3 = self.encoder3(enc2)        # H/8 
        enc4 = self.encoder4(enc3)        # H/16
        
        # Simple bottleneck processing
        bottleneck = self.bottleneck(enc4)  # H/16
        
        # Decoder path with simple upsampling and feature fusion
        dec4 = self.decoder4(bottleneck, enc3)  # H/8 (fuse with enc3)
        dec3 = self.decoder3(dec4, enc2)        # H/4 (fuse with enc2)
        dec2 = self.decoder2(dec3, enc1)        # H/2 (fuse with enc1)  
        dec1 = self.decoder1(dec2, stem_feat)   # H   (fuse with stem)
        
        # Simple segmentation prediction
        segmentation = self.seg_head(dec1)
        
        # Ensure output matches input resolution
        if segmentation.shape[-2:] != input_size:
            segmentation = F.interpolate(segmentation, size=input_size, 
                                       mode='bilinear', align_corners=False)
        
        return segmentation
    
    def get_model_info(self):
        """Get custom model information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'SMNet (Simplified Custom Segmentation Model)',
            'base_dimension': self.base_dim,
            'total_parameters': total_params,
            'custom_features': [
                'Simple encoder-decoder architecture', 
                'Basic skip connections',
                'Residual connections',
                'Progressive dimension scaling (2x, 3x, 4x, 5x)'
            ],
            'architecture_type': 'Simple Custom Encoder-Decoder'
        }