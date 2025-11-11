"""
Debug script to test SMNet model and compare with standard training setup
"""

import torch
import torch.nn as nn
import sys
import os

# Add paths
sys.path.append('../2_2_Custom_SMNet')
sys.path.append('../2_1_Evaluate_Model')

from model import SMNet

def test_model_output():
    """Test if SMNet produces valid outputs"""
    print("Testing SMNet model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SMNet(num_classes=21, base_dim=16).to(device)
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Check output statistics
        print(f"Output min: {output.min().item():.4f}")
        print(f"Output max: {output.max().item():.4f}")
        print(f"Output mean: {output.mean().item():.4f}")
        print(f"Output std: {output.std().item():.4f}")
        
        # Check if output contains NaN or Inf
        if torch.isnan(output).any():
            print("❌ WARNING: Output contains NaN!")
        if torch.isinf(output).any():
            print("❌ WARNING: Output contains Inf!")
        
        # Check prediction distribution
        predictions = torch.argmax(output, dim=1)
        unique_preds, counts = torch.unique(predictions, return_counts=True)
        print(f"Predicted classes: {unique_preds.cpu().numpy()}")
        print(f"Prediction counts: {counts.cpu().numpy()}")
        
        # Check if all predictions are the same class
        if len(unique_preds) == 1:
            print(f"❌ WARNING: Model only predicts class {unique_preds[0].item()}")
        else:
            print(f"✅ Model predicts {len(unique_preds)} different classes")
            
        # Check class probabilities
        probs = torch.softmax(output, dim=1)
        print(f"Probability min: {probs.min().item():.4f}")
        print(f"Probability max: {probs.max().item():.4f}")
        
        return True

def compare_models():
    """Compare distillation model with standard training model if available"""
    standard_model_path = "../2_3_train_test_model/best_smnet_model_base16.pth"
    distill_model_path = "response_based_distillation/response_based_model.pth"
    
    if os.path.exists(standard_model_path):
        print(f"\nComparing with standard training model...")
        checkpoint = torch.load(standard_model_path, map_location='cpu')
        miou = checkpoint.get('miou', 'Unknown')
        epoch = checkpoint.get('epoch', 'Unknown')
        loss = checkpoint.get('loss', 'Unknown')
        print(f"Standard model mIoU: {miou}")
        print(f"Standard model epoch: {epoch}")
        print(f"Standard model loss: {loss}")
    
    if os.path.exists(distill_model_path):
        print(f"\nDistillation model exists")
        checkpoint = torch.load(distill_model_path, map_location='cpu')
        print(f"Distillation model mIoU: {checkpoint.get('miou', 'Unknown'):.4f}")
        print(f"Distillation model epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Distillation model loss: {checkpoint.get('loss', 'Unknown'):.4f}")

if __name__ == "__main__":
    print("="*50)
    print("SMNet Model Debugging")
    print("="*50)
    
    test_model_output()
    compare_models()
    
    print("\n" + "="*50)
    print("Debug complete!")