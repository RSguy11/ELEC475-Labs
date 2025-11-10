"""
Test script for the Custom SMNet segmentation model
ELEC475 Lab 3 - Step 2.2 - Original Custom Architecture
"""

import torch
import torch.nn as nn
from model import SMNet

def test_custom_model():
    """Test the custom SMNet architecture"""
    print("=" * 60)
    print("Custom SMNet Segmentation Model Test")
    print("Original Design for Lab 3 Step 2.2")
    print("=" * 60)
    
    # Test different base dimension configurations (optimized for 2M budget)
    base_dims = [12, 16, 18, 20]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test input
    batch_size = 2
    input_size = 224
    test_input = torch.randn(batch_size, 3, input_size, input_size, device=device)
    
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Target output: segmentation mask [{batch_size}, 21, {input_size}, {input_size}]")
    
    for base_dim in base_dims:
        print(f"\n{'='*20} Base Dimension: {base_dim} {'='*20}")
        
        try:
            # Create custom model
            model = SMNet(num_classes=21, base_dim=base_dim).to(device)
            model.eval()
            
            # Get model info
            model_info = model.get_model_info()
            
            # Forward pass timing
            import time
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            forward_time = time.time() - start_time
            
            # Calculate memory usage
            param_memory_mb = model_info['total_parameters'] * 4 / (1024 * 1024)
            output_memory_mb = output.numel() * 4 / (1024 * 1024)
            
            print(f"Model: {model_info['model_name']}")
            print(f"Architecture: {model_info['architecture_type']}")
            print(f"Total parameters: {model_info['total_parameters']:,}")
            print(f"Parameter memory: {param_memory_mb:.1f} MB")
            print(f"Output shape: {output.shape}")
            print(f"Output memory: {output_memory_mb:.1f} MB")
            print(f"Forward time: {forward_time:.4f}s")
            print(f"Throughput: {batch_size/forward_time:.1f} images/sec")
            
            print(f"\nCustom Features:")
            for i, feature in enumerate(model_info['custom_features'], 1):
                print(f"  {i}. {feature}")
            
            # Check parameter budget
            param_target = 2000000  # 2M parameters
            if model_info['total_parameters'] <= param_target:
                print(f"✅ Within parameter budget ({model_info['total_parameters']:,} <= {param_target:,})")
            else:
                print(f"❌ Exceeds parameter budget ({model_info['total_parameters']:,} > {param_target:,})")
            
            # Verify output shape
            expected_shape = (batch_size, 21, input_size, input_size)
            if output.shape == expected_shape:
                print(f"✅ Correct output shape")
            else:
                print(f"❌ Wrong output shape: {output.shape} vs {expected_shape}")
                
        except Exception as e:
            print(f"❌ Model failed: {e}")
            continue
    
    print(f"\n{'=' * 60}")
    print("SIMPLIFIED CUSTOM SMNET ARCHITECTURE SUMMARY:")
    print("=" * 60)
    print("Structure: Image → Simple Encoder → Bottleneck → Simple Decoder → Segmentation Head")
    print("\nSimple Custom Features:")
    print("• GELU activation functions (instead of standard ReLU)")
    print("• Basic encoder-decoder with skip connections")  
    print("• Simple upsampling blocks")
    print("• Progressive dimension scaling: base_dim → 2x → 3x → 4x → 5x")
    print("• Residual connections in encoder stages")
    print("• Straightforward feature fusion")
    print("• Lightweight design for efficiency")
    print("\nSimplified and easy to understand - perfect for Lab 3 Step 2.2")

def analyze_custom_architecture():
    """Analyze the custom model architecture in detail"""
    print(f"\n{'=' * 60}")
    print("CUSTOM ARCHITECTURE ANALYSIS:")
    print("=" * 60)
    
    model = SMNet(num_classes=21, base_dim=24)
    model_info = model.get_model_info()
    
    print(f"Model Name: {model_info['model_name']}")
    print(f"Base Dimension: {model_info['base_dimension']}")
    print(f"Total Parameters: {model_info['total_parameters']:,}")
    print(f"Architecture Type: {model_info['architecture_type']}")
    
    print(f"\nSimple Component Breakdown:")
    print(f"1. CustomConvBlock: Simple conv block with GELU + 1x1 refinement")
    print(f"2. SimpleUpsampler: Basic upsampling with concatenation & fusion")
    print(f"3. Progressive Scaling: Straightforward dimension progression")
    print(f"4. Skip Connections: Basic feature fusion from encoder to decoder")
    
    print(f"\nKey Simplifications:")
    print(f"• Removed complex MultiScaleBlock (not needed for lab)")
    print(f"• Simple upsampling instead of complex feature alignment")
    print(f"• Basic concatenation for skip connections")
    print(f"• Lightweight design with much fewer parameters")
    print(f"• Easy to understand and implement")

if __name__ == "__main__":
    test_custom_model()
    analyze_custom_architecture()