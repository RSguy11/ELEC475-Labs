import torch 
import torch.nn as nn
import torch.nn.functional as F

class CustomConvBlock(nn.Module):
    """Simple custom convolution block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_residual=False):
        super(CustomConvBlock, self).__init__()
        
        self.use_residual = use_residual and in_channels == out_channels and stride == 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)  # 1x1 conv
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.GELU()
        
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        # Refinement
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection if applicable
        if self.use_residual:
            out += identity
            
        out = self.activation(out)
        return out

class SimpleUpsampler(nn.Module):
    def __init__(self, high_ch, low_ch, out_ch, skip_ch=32):
        super().__init__()

        self.lateral = nn.Conv2d(low_ch, skip_ch, 1, bias=False) 

        self.fuse    = nn.Sequential(
            nn.Conv2d(high_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, high_feat, low_feat):
        up = F.interpolate(high_feat, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)
        skip = self.lateral(low_feat)

        return self.fuse(torch.cat([up, skip], dim=1))

class SMNet(nn.Module):
    """
    Custom Segmentation Model (SMNet) - Simplified and Lightweight
    Architecture: Simple encoder-decoder with skip connections
    """
    def __init__(self, num_classes=21, base_dim=16):
        super(SMNet, self).__init__()
        
        # Initial feature extraction
        self.stem = CustomConvBlock(3, base_dim, kernel_size=7, stride=1)
        
        # Simple Encoder - Progressive feature extraction
        self.encoder1 = CustomConvBlock(base_dim, base_dim*2, stride=2, use_residual=False)      # H/2
        self.encoder2 = CustomConvBlock(base_dim*2, base_dim*3, stride=2, use_residual=True)     # H/4
        self.encoder3 = CustomConvBlock(base_dim*3, base_dim*4, stride=2, use_residual=True)     # H/8
        self.encoder4 = CustomConvBlock(base_dim*4, base_dim*5, stride=2, use_residual=True)     # H/16
        
        # Simple Bottleneck - just additional processing
        self.bottleneck = CustomConvBlock(base_dim*5, base_dim*5, use_residual=True)
        
        # Simple Decoder with progressive upsampling
        self.decoder4 = SimpleUpsampler(base_dim*5, base_dim*4, base_dim*4)   # H/8
        self.decoder3 = SimpleUpsampler(base_dim*4, base_dim*3, base_dim*3)   # H/4  
        self.decoder2 = SimpleUpsampler(base_dim*3, base_dim*2, base_dim*2)   # H/2
        self.decoder1 = SimpleUpsampler(base_dim*2, base_dim, base_dim)       # H
        
        # Simple segmentation head
        self.seg_head = nn.Sequential(
            CustomConvBlock(base_dim, base_dim//2, kernel_size=3),
            nn.Conv2d(base_dim//2, num_classes, 1)  # Final 1x1 classif ier
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
                'Simple encoder-decoder structure', 
                'Basic upsampling with skip connections',
                'Residual connections in encoder',
                'Lightweight design',
                'Progressive dimension scaling (2x, 3x, 4x, 5x)'
            ],
            'architecture_type': 'Simplified Custom Encoder-Decoder'
        }