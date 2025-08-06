#!/usr/bin/env python
"""
Test script to verify MinutiaeNet weight loading and feature extraction.
"""

import torch
import numpy as np
from pathlib import Path
from model import get_architecture

def test_minutiaenet_weights():
    """Test MinutiaeNet weight loading and feature extraction"""
    
    print("Testing MinutiaeNet weight loading...")
    print("="*50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if weights file exists
    weights_path = "weights/minutiaenet_livdet.h5"
    if not Path(weights_path).exists():
        print(f"Error: Weights file not found at {weights_path}")
        return False
    
    print(f"Weights file found: {weights_path}")
    
    try:
        # Create model with pretrained weights
        print("Creating simplified MinutiaeNet model with pretrained weights...")
        model = get_architecture('minutiaenet_simple', device=device, pretrained_path=weights_path)
        
        # Test with dummy input
        print("Testing feature extraction with dummy input...")
        batch_size = 2
        dummy_input = torch.randn(batch_size, 1, 400, 400).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model.extract_features(dummy_input)
        
        print(f"Feature extraction successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output features shape: {features.shape}")
        print(f"Feature norm (should be ~1.0): {torch.norm(features, dim=1)}")
        
        # Test siamese forward pass
        print("\nTesting Siamese forward pass...")
        img1 = torch.randn(batch_size, 1, 400, 400).to(device)
        img2 = torch.randn(batch_size, 1, 400, 400).to(device)
        
        with torch.no_grad():
            similarity = model(img1, img2)
        
        print(f"Siamese forward pass successful!")
        print(f"Similarity scores shape: {similarity.shape}")
        print(f"Similarity scores range: {similarity.min().item():.4f} - {similarity.max().item():.4f}")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("\n✅ MinutiaeNet weight loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ MinutiaeNet weight loading test FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_weight_converter():
    """Test the weight converter utility"""
    
    print("\nTesting weight converter utility...")
    print("="*50)
    
    try:
        from utils.weight_converter import load_minutiaenet_weights
        from model.minutiaenet_simple import SimpleMinutiaeNetFeatureExtractor
        
        # Create model
        model = MinutiaeNetFeatureExtractor()
        
        # Test weight loading
        weights_path = "weights/minutiaenet_livdet.h5"
        updated_model = load_minutiaenet_weights(model, weights_path)
        
        print("✅ Weight converter test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Weight converter test FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("MinutiaeNet Weight Loading Test")
    print("="*60)
    
    # Test weight converter
    converter_success = test_weight_converter()
    
    # Test full model
    model_success = test_minutiaenet_weights()
    
    if converter_success and model_success:
        print("\n🎉 All tests PASSED! MinutiaeNet is ready to use.")
        print("\nYou can now run:")
        print("python train_minutiaenet.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.") 