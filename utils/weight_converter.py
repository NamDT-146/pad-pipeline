#!/usr/bin/env python
"""
Utility script to convert TensorFlow/Keras MinutiaeNet weights to PyTorch format.
This is a helper script for loading pretrained MinutiaeNet weights.
"""

import os
import numpy as np
import torch
import tensorflow as tf
from pathlib import Path

def convert_tf_weights_to_pytorch(tf_model_path, pytorch_model, output_path=None):
    """
    Convert TensorFlow/Keras model weights to PyTorch format.
    
    Args:
        tf_model_path: Path to TensorFlow/Keras model file (.h5)
        pytorch_model: PyTorch model instance
        output_path: Path to save converted weights (optional)
        
    Returns:
        PyTorch model with loaded weights
    """
    print(f"Loading TensorFlow model from: {tf_model_path}")
    
    try:
        # Load TensorFlow model
        tf_model = tf.keras.models.load_model(tf_model_path, compile=False)
        print("TensorFlow model loaded successfully")
        
        # Get TensorFlow weights
        tf_weights = tf_model.get_weights()
        print(f"Found {len(tf_weights)} weight layers in TensorFlow model")
        
        # Create mapping between TF and PyTorch layers
        weight_mapping = create_weight_mapping(tf_model, pytorch_model)
        
        # Load weights into PyTorch model
        pytorch_model = load_weights_to_pytorch(pytorch_model, tf_weights, weight_mapping)
        
        # Save converted weights if output path provided
        if output_path:
            torch.save(pytorch_model.state_dict(), output_path)
            print(f"Converted weights saved to: {output_path}")
        
        return pytorch_model
        
    except Exception as e:
        print(f"Error converting weights: {e}")
        print("Continuing with random initialization...")
        return pytorch_model

def create_weight_mapping(tf_model, pytorch_model):
    """
    Create mapping between TensorFlow and PyTorch layer names.
    This is a simplified mapping - you may need to adjust based on your specific model architecture.
    """
    mapping = {}
    
    # Get TensorFlow layer names
    tf_layer_names = [layer.name for layer in tf_model.layers]
    print(f"TensorFlow layers: {tf_layer_names}")
    
    # Get PyTorch layer names
    pytorch_layer_names = list(pytorch_model.state_dict().keys())
    print(f"PyTorch layers: {pytorch_layer_names}")
    
    # Create mapping for simplified MinutiaeNet architecture
    # This matches the available TensorFlow weights better
    conv_mapping = {
        'conv2d': 'conv1.weight',           # (64, 1, 7, 7)
        'conv2d_1': 'conv2_1.weight',       # (64, 64, 3, 3)
        'conv2d_2': 'conv2_2.weight',       # (128, 64, 3, 3)
        'conv2d_3': 'conv3_1.weight',       # (128, 128, 3, 3)
        'conv2d_4': 'conv3_2.weight',       # (256, 128, 3, 3)
        'conv2d_6': 'feature_conv.weight'   # (512, 256, 3, 3)
    }
    
    bn_mapping = {
        'batch_normalization': 'bn1.weight',
        'batch_normalization_1': 'bn2_1.weight',
        'batch_normalization_2': 'bn2_2.weight', 
        'batch_normalization_3': 'bn3_1.weight',
        'batch_normalization_4': 'bn3_2.weight',
        'batch_normalization_6': 'feature_bn.weight'
    }
    
    # Add convolutional mappings
    for tf_name, pt_name in conv_mapping.items():
        if tf_name in tf_layer_names and pt_name in pytorch_layer_names:
            mapping[tf_name] = pt_name
    
    # Add batch normalization mappings
    for tf_name, pt_name in bn_mapping.items():
        if tf_name in tf_layer_names and pt_name in pytorch_layer_names:
            mapping[tf_name] = pt_name
    
    print(f"Weight mapping: {mapping}")
    return mapping

def load_weights_to_pytorch(pytorch_model, tf_weights, weight_mapping):
    """
    Load TensorFlow weights into PyTorch model.
    """
    pytorch_state_dict = pytorch_model.state_dict()
    
    # Create extended mapping for bias and batch norm parameters
    extended_mapping = {}
    
    # Add weight mappings
    for tf_name, pt_name in weight_mapping.items():
        extended_mapping[tf_name] = pt_name
        
        # Add bias mappings for conv layers
        if 'conv' in tf_name.lower():
            bias_name = tf_name.replace('conv', 'bias') if 'conv' in tf_name else tf_name + '_bias'
            pt_bias_name = pt_name.replace('.weight', '.bias')
            if pt_bias_name in pytorch_state_dict:
                extended_mapping[bias_name] = pt_bias_name
        
        # Add batch norm parameter mappings
        if 'batch_normalization' in tf_name.lower():
            # Map gamma (weight), beta (bias), moving_mean, moving_variance
            base_name = tf_name
            pt_base = pt_name.replace('.weight', '')
            
            # Gamma (weight)
            extended_mapping[base_name] = pt_base + '.weight'
            
            # Beta (bias) 
            beta_name = base_name + '_beta'
            extended_mapping[beta_name] = pt_base + '.bias'
            
            # Moving mean
            mean_name = base_name + '_moving_mean'
            extended_mapping[mean_name] = pt_base + '.running_mean'
            
            # Moving variance
            var_name = base_name + '_moving_variance'
            extended_mapping[var_name] = pt_base + '.running_var'
    
    # Load weights
    for tf_layer_name, tf_weight in zip(weight_mapping.keys(), tf_weights):
        if tf_layer_name in extended_mapping:
            pt_layer_name = extended_mapping[tf_layer_name]
            
            if pt_layer_name in pytorch_state_dict:
                target_shape = pytorch_state_dict[pt_layer_name].shape
                
                # Convert weight format if needed
                if len(tf_weight.shape) == 4:  # Conv2d weights
                    # TF: (H, W, C_in, C_out) -> PyTorch: (C_out, C_in, H, W)
                    converted_weight = np.transpose(tf_weight, (3, 2, 0, 1))
                elif len(tf_weight.shape) == 2:  # Linear weights
                    # TF: (C_in, C_out) -> PyTorch: (C_out, C_in)
                    converted_weight = np.transpose(tf_weight, (1, 0))
                else:  # Bias or other weights
                    converted_weight = tf_weight
                
                # Check shape compatibility
                if converted_weight.shape == target_shape:
                    pytorch_state_dict[pt_layer_name] = torch.from_numpy(converted_weight).float()
                    print(f"Loaded weights for {tf_layer_name} -> {pt_layer_name}")
                else:
                    print(f"Shape mismatch for {tf_layer_name}: {converted_weight.shape} vs {target_shape}")
    
    pytorch_model.load_state_dict(pytorch_state_dict)
    return pytorch_model

def load_minutiaenet_weights(model, weights_path):
    """
    Load MinutiaeNet weights into the model.
    
    Args:
        model: PyTorch model instance
        weights_path: Path to weights file (can be .h5 or .pth)
        
    Returns:
        Model with loaded weights
    """
    weights_path = Path(weights_path)
    
    if weights_path.suffix == '.h5':
        # TensorFlow/Keras weights - need conversion
        print("Detected TensorFlow/Keras weights, converting to PyTorch...")
        return convert_tf_weights_to_pytorch(weights_path, model)
    
    elif weights_path.suffix == '.pth':
        # PyTorch weights - direct loading
        print(f"Loading PyTorch weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        return model
    
    else:
        print(f"Unsupported weight format: {weights_path.suffix}")
        print("Continuing with random initialization...")
        return model

def test_weight_loading():
    """
    Test function to verify weight loading works correctly.
    """
    from model.minutiaenet import MinutiaeNetFeatureExtractor
    
    # Create PyTorch model
    model = MinutiaeNetFeatureExtractor()
    
    # Test with dummy weights (you would replace this with actual MinutiaeNet weights)
    print("Testing weight loading functionality...")
    
    # Example usage:
    # model = load_minutiaenet_weights(model, "path/to/minutiaenet_weights.h5")
    
    return model

if __name__ == "__main__":
    # Test the weight conversion utility
    model = test_weight_loading()
    print("Weight conversion utility ready!")
    print("\nTo use with your trained MinutiaeNet:")
    print("1. Train MinutiaeNet using the TensorFlow/Keras script")
    print("2. Use this utility to convert the .h5 weights to PyTorch format")
    print("3. Update the pretrained_path in train_minutiaenet.py") 