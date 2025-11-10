"""
Vision Transformer Patch Encoder - Optimal Depth Analysis
ELEC475 Lab 3 - Step 2.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class ConvPatchEncoder(nn.Module):
    """
    Convolutional patch encoder for Vision Transformer.
    Tests different depths, kernel sizes, and stride configurations.
    """
    
    def __init__(self, patch_size=16, embed_dim=768, config=None, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.config = config
        
        # Use config for layer specifications
        if config is None:
            # Default config - simple 3x3 layers
            config = {
                'name': 'default_3x3',
                'layers': [
                    {'in_ch': 3, 'out_ch': 64, 'kernel': 3, 'stride': 2, 'padding': 1},
                    {'in_ch': 64, 'out_ch': 128, 'kernel': 3, 'stride': 2, 'padding': 1},
                    {'in_ch': 128, 'out_ch': 256, 'kernel': 3, 'stride': 1, 'padding': 1},
                    {'in_ch': 256, 'out_ch': 512, 'kernel': 3, 'stride': 1, 'padding': 1},
                ]
            }
        
        self.config_name = config['name']
        
        # Build conv layers from config
        self.conv_layers = nn.ModuleList()
        for layer_spec in config['layers']:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(layer_spec['in_ch'], layer_spec['out_ch'], 
                             kernel_size=layer_spec['kernel'], 
                             stride=layer_spec['stride'], 
                             padding=layer_spec['padding']),
                    nn.BatchNorm2d(layer_spec['out_ch']),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Get final channel count
        final_channels = config['layers'][-1]['out_ch']
        
        # Safe patch projection using unfold + linear instead of direct conv
        self.use_unfold = True
        if self.use_unfold:
            self.proj = nn.Linear(final_channels * patch_size * patch_size, embed_dim)
        else:
            # Fallback: traditional conv projection
            self.proj = nn.Conv2d(final_channels, embed_dim, 
                                 kernel_size=patch_size, stride=patch_size)
        
        # Calculate output stride for reporting
        self.output_stride = 1
        for layer_spec in config['layers']:
            self.output_stride *= layer_spec['stride']
            
    def _pad_to_patch_multiple(self, x, patch_size):
        """Pad feature map to ensure divisible by patch size"""
        B, C, H, W = x.shape
        
        # Calculate padding needed
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            
        return x
        
    def forward(self, x):
        """
        Args:
            x: Input image [B, C, H, W]
        Returns:
            patches: Encoded patches [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        
        # Apply conv layers for feature extraction
        features = x
        for conv_layer in self.conv_layers:
            features = conv_layer(features)
            
        # Ensure features are divisible by patch size
        features = self._pad_to_patch_multiple(features, self.patch_size)
        B, C_feat, H_feat, W_feat = features.shape
        
        # Calculate number of patches after padding
        num_patches_h = H_feat // self.patch_size
        num_patches_w = W_feat // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Check minimum patch grid
        if num_patches < 1:
            raise ValueError(f"Feature map too small ({H_feat}x{W_feat}) for patch size {self.patch_size}")
        if num_patches_h < 1 or num_patches_w < 1:
            raise ValueError(f"Invalid patch grid: {num_patches_h}x{num_patches_w}")
            
        if self.use_unfold:
            # Safe projection using unfold + linear
            patches = F.unfold(features, kernel_size=self.patch_size, stride=self.patch_size)
            # patches: [B, C*patch_size*patch_size, num_patches]
            patches = patches.transpose(1, 2)  # [B, num_patches, C*patch_size*patch_size]
            patches = self.proj(patches)  # [B, num_patches, embed_dim]
        else:
            # Traditional conv projection
            patches = self.proj(features)  # [B, embed_dim, H_p, W_p]
            patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return patches, (num_patches_h, num_patches_w)
    
def calculate_receptive_field(config):
    """Calculate theoretical receptive field for a given configuration"""
    rf = 1
    jump = 1
    
    for layer_spec in config['layers']:
        kernel_size = layer_spec['kernel']
        stride = layer_spec['stride']
        rf = rf + (kernel_size - 1) * jump
        jump = jump * stride
    
    return rf

def get_encoder_configurations():
    """Define lightweight encoder configurations targeting ~2M parameters"""
    configs = []
    
    # Config 1: Ultra-lightweight 3x3 layers
    configs.append({
        'name': 'ultra_light_3x3',
        'layers': [
            {'in_ch': 3, 'out_ch': 8, 'kernel': 3, 'stride': 2, 'padding': 1},
            {'in_ch': 8, 'out_ch': 16, 'kernel': 3, 'stride': 2, 'padding': 1},
            {'in_ch': 16, 'out_ch': 32, 'kernel': 3, 'stride': 1, 'padding': 1},
            {'in_ch': 32, 'out_ch': 64, 'kernel': 3, 'stride': 1, 'padding': 1},
        ]
    })
    
    # Config 2: Lightweight with larger kernels
    configs.append({
        'name': 'light_large_kernel',
        'layers': [
            {'in_ch': 3, 'out_ch': 16, 'kernel': 7, 'stride': 2, 'padding': 3},
            {'in_ch': 16, 'out_ch': 32, 'kernel': 5, 'stride': 2, 'padding': 2},
            {'in_ch': 32, 'out_ch': 64, 'kernel': 3, 'stride': 1, 'padding': 1},
        ]
    })
    
    # Config 3: 2M parameter target
    configs.append({
        'name': '2M_param_target',
        'layers': [
            {'in_ch': 3, 'out_ch': 12, 'kernel': 5, 'stride': 2, 'padding': 2},
            {'in_ch': 12, 'out_ch': 24, 'kernel': 3, 'stride': 2, 'padding': 1},
            {'in_ch': 24, 'out_ch': 48, 'kernel': 3, 'stride': 1, 'padding': 1},
            {'in_ch': 48, 'out_ch': 64, 'kernel': 3, 'stride': 1, 'padding': 1},
        ]
    })
    
    # Config 4: Ultra-aggressive 2 layer
    configs.append({
        'name': 'ultra_aggressive_2layer',
        'layers': [
            {'in_ch': 3, 'out_ch': 24, 'kernel': 7, 'stride': 4, 'padding': 3},
            {'in_ch': 24, 'out_ch': 48, 'kernel': 5, 'stride': 2, 'padding': 2},
        ]
    })
    
    # Config 5: Deep but very narrow
    configs.append({
        'name': 'deep_very_narrow',
        'layers': [
            {'in_ch': 3, 'out_ch': 8, 'kernel': 3, 'stride': 2, 'padding': 1},
            {'in_ch': 8, 'out_ch': 12, 'kernel': 3, 'stride': 1, 'padding': 1},
            {'in_ch': 12, 'out_ch': 16, 'kernel': 3, 'stride': 2, 'padding': 1},
            {'in_ch': 16, 'out_ch': 24, 'kernel': 3, 'stride': 1, 'padding': 1},
            {'in_ch': 24, 'out_ch': 32, 'kernel': 3, 'stride': 1, 'padding': 1},
            {'in_ch': 32, 'out_ch': 64, 'kernel': 3, 'stride': 1, 'padding': 1},
        ]
    })
    
    # Config 6: Minimal channels
    configs.append({
        'name': 'minimal_channels',
        'layers': [
            {'in_ch': 3, 'out_ch': 16, 'kernel': 3, 'stride': 2, 'padding': 1},
            {'in_ch': 16, 'out_ch': 32, 'kernel': 3, 'stride': 2, 'padding': 1},
            {'in_ch': 32, 'out_ch': 64, 'kernel': 3, 'stride': 1, 'padding': 1},
        ]
    })
    
    return configs

def analyze_encoder_configurations():
    """Analyze different encoder configurations for optimal performance."""
    
    print("Vision Transformer Patch Encoder Configuration Analysis")
    print("Testing Multiple Kernel Sizes and Stride Patterns")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test parameters - optimized for lightweight models
    img_size = 224
    patch_size = 16
    embed_dim = 64  # Much smaller embedding for 2M parameter target
    batch_size = 4
    
    # Get configurations to test
    configs = get_encoder_configurations()
    
    # Create test input and move to device
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    results = []
    
    for config in configs:
        print(f"\n{'='*20} {config['name']} {'='*20}")
        
        try:
            # Create encoder and move to device
            encoder = ConvPatchEncoder(patch_size, embed_dim, config).to(device)
            encoder.eval()  # Set to eval mode for timing
            
            # Calculate receptive field
            receptive_field = calculate_receptive_field(config)
            
            # Count parameters
            num_params = sum(p.numel() for p in encoder.parameters())
            param_memory_mb = num_params * 4 / (1024 * 1024)  # 4 bytes per float32
            
            # Warmup runs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
            warmup_runs = 3
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = encoder(x)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
            
            # Timing runs
            num_timing_runs = 10
            times = []
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
            for _ in range(num_timing_runs):
                start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
                
                if device.type == 'cuda':
                    start_time.record()
                else:
                    start_time = time.time()
                
                with torch.no_grad():
                    patches, patch_grid = encoder(x)
                
                if device.type == 'cuda':
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                else:
                    elapsed_time = time.time() - start_time
                    
                times.append(elapsed_time)
            
            # Calculate timing statistics
            avg_forward_time = sum(times) / len(times)
            min_forward_time = min(times)
            
            # Memory measurements
            if device.type == 'cuda':
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                current_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                peak_memory_mb = 0
                current_memory_mb = 0
            
            # Output memory calculation
            output_memory_mb = (patches.numel() * 4) / (1024 * 1024)  # 4 bytes per float32
            
            # Calculate metrics
            num_patches = patches.shape[1]
            patch_grid_h, patch_grid_w = patch_grid
            rf_patch_ratio = receptive_field / patch_size
            
            # Training considerations
            param_efficiency = receptive_field / (num_params / 1000)  # RF per 1K params
            throughput = batch_size / avg_forward_time  # images/sec
            
            # Warnings for aggressive configurations
            warnings = []
            if encoder.output_stride > 32:
                warnings.append(f"Very aggressive stride ({encoder.output_stride}x)")
            if patch_grid_h * patch_grid_w < 16:
                warnings.append(f"Small patch grid ({patch_grid_h}x{patch_grid_w})")
            if num_params > 2000000:
                warnings.append("Exceeds 2M parameter target")
            
            result = {
                'config': config['name'],
                'num_layers': len(config['layers']),
                'num_patches': num_patches,
                'patch_grid': (patch_grid_h, patch_grid_w),
                'output_stride': encoder.output_stride,
                'receptive_field': receptive_field,
                'num_params': num_params,
                'param_memory_mb': param_memory_mb,
                'output_memory_mb': output_memory_mb,
                'peak_memory_mb': peak_memory_mb,
                'rf_patch_ratio': rf_patch_ratio,
                'param_efficiency': param_efficiency,
                'avg_forward_time': avg_forward_time,
                'min_forward_time': min_forward_time,
                'throughput': throughput,
                'warnings': warnings
            }
            
            results.append(result)
            
            print(f"Layers: {len(config['layers'])}")
            print(f"Output stride: {encoder.output_stride}x")
            print(f"Layer specs:")
            for i, layer in enumerate(config['layers']):
                print(f"  L{i+1}: {layer['in_ch']}->{layer['out_ch']}, K={layer['kernel']}, S={layer['stride']}")
            print(f"Patch grid: {patch_grid_h}x{patch_grid_w} = {num_patches} patches")
            print(f"Receptive Field: {receptive_field}px")
            print(f"Parameters: {num_params:,} ({param_memory_mb:.1f} MB)")
            print(f"Output memory: {output_memory_mb:.2f} MB")
            if device.type == 'cuda':
                print(f"Peak GPU memory: {peak_memory_mb:.1f} MB")
            print(f"RF/Patch Ratio: {rf_patch_ratio:.2f}")
            print(f"Forward Time: {avg_forward_time:.4f}s ± {(max(times) - min(times)):.4f}s")
            print(f"Throughput: {throughput:.1f} images/sec")
            print(f"Param Efficiency: {param_efficiency:.4f}")
            if warnings:
                print(f"⚠️  Warnings: {', '.join(warnings)}")
            
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            continue
    
    # Analysis and recommendations
    print(f"\n{'=' * 80}")
    print("CONFIGURATION COMPARISON:")
    print("=" * 80)
    
    print(f"{'Config':<20} {'Layers':<6} {'Stride':<6} {'RF':<4} {'Params':<8} {'Time(s)':<8} {'Grid':<8} {'Score':<5}")
    print("-" * 80)
    
    best_score = 0
    best_config = None
    
    for result in results:
        # Enhanced scoring system (higher is better) - optimized for lightweight models
        score = 0
        
        # RF/patch ratio scoring (optimal around 0.5-1.5)
        if 0.5 <= result['rf_patch_ratio'] <= 1.5:
            score += 3
        elif 0.3 <= result['rf_patch_ratio'] <= 2.0:
            score += 2
        else:
            score += 1
            
        # Parameter efficiency (favor lightweight models for laptops)
        if result['num_params'] < 1000000:  # <1M - excellent
            score += 4
        elif result['num_params'] < 2000000:  # <2M - very good
            score += 3
        elif result['num_params'] < 5000000:  # <5M - acceptable
            score += 2
        else:  # >5M - too heavy for laptops
            score += 0
            
        # Training speed (critical for laptop training)
        if result['throughput'] > 100:  # Very fast
            score += 3
        elif result['throughput'] > 50:  # Fast
            score += 2
        elif result['throughput'] > 30:  # Acceptable
            score += 1
        else:  # Too slow
            score += 0
            
        # Memory efficiency (important for laptop GPU)
        total_memory = result['param_memory_mb'] + result['output_memory_mb']
        if total_memory < 20:
            score += 2
        elif total_memory < 50:
            score += 1
            
        # Patch grid size (avoid too aggressive)
        grid_size = result['patch_grid'][0] * result['patch_grid'][1]
        if grid_size >= 16:
            score += 1
            
        # Penalty for warnings
        score -= len(result['warnings'])
        
        result['score'] = max(0, score)  # Ensure non-negative
        
        if result['score'] > best_score:
            best_score = result['score']
            best_config = result
        
        grid_str = f"{result['patch_grid'][0]}x{result['patch_grid'][1]}"
        print(f"{result['config']:<20} {result['num_layers']:<6} {result['output_stride']:<6}x "
              f"{result['receptive_field']:<4} {result['num_params']:<8,} "
              f"{result['avg_forward_time']:<8.4f} {grid_str:<8} {result['score']:<5}")
    
    # Recommendations
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    if best_config:
        print(f"🏆 BEST LIGHTWEIGHT CONFIGURATION: {best_config['config']}")
        print(f"   Score: {best_config['score']}/13")
        print(f"   Layers: {best_config['num_layers']}")
        print(f"   Output stride: {best_config['output_stride']}x")
        print(f"   Patch grid: {best_config['patch_grid'][0]}x{best_config['patch_grid'][1]}")
        print(f"   Receptive Field: {best_config['receptive_field']}px")
        print(f"   Parameters: {best_config['num_params']:,} (~{best_config['num_params']/1000000:.1f}M)")
        print(f"   Training Speed: {best_config['throughput']:.1f} img/sec")
        print(f"   Total Memory: {best_config['param_memory_mb'] + best_config['output_memory_mb']:.1f} MB")
        if best_config['warnings']:
            print(f"   ⚠️  Considerations: {', '.join(best_config['warnings'])}")
        
    print(f"\nKey Insights for Lightweight Laptop Training:")
    print(f"• Target <2M parameters for efficient laptop training")
    print(f"• Larger initial kernels reduce layer count and training time")
    print(f"• Channel counts should stay low (16, 32, 64, 128 max)")
    print(f"• Higher throughput (>50 img/sec) means faster epoch times")
    print(f"• Balance RF coverage (50-150% of patch) with parameter budget")
    print(f"• Maintain reasonable patch grid size (>4x4) for spatial resolution")
    print(f"• Watch for over-aggressive strides that lose too much spatial info")
    
    return results

if __name__ == "__main__":
    results = analyze_encoder_configurations()