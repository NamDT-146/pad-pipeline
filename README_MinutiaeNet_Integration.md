# MinutiaeNet Integration for Fingerprint Matching

This document describes the integration of MinutiaeNet as a feature extractor for fingerprint matching in your presentation attack detection (PAD) pipeline.

## Overview

MinutiaeNet has been integrated as an alternative feature extractor that can be used instead of the standard Siamese network or MobileNetV2. The integration includes:

1. **MinutiaeNet Feature Extractor**: PyTorch implementation of MinutiaeNet for feature extraction
2. **Siamese Network with MinutiaeNet**: Combines MinutiaeNet features with similarity learning
3. **Enhanced Metrics**: FMR, FNMR, EER, and ROC curve analysis
4. **Weight Conversion Utility**: Tools to convert TensorFlow/Keras weights to PyTorch

## Architecture

### MinutiaeNetFeatureExtractor
- **Input**: Single fingerprint image (400x400x1)
- **Output**: Feature embeddings (512-dimensional)
- **Architecture**: Simplified version of MinutiaeNet with 3 convolutional blocks
- **Features**: L2-normalized embeddings for similarity comparison

### MinutiaeNetSiamese
- **Input**: Two fingerprint images
- **Output**: Similarity score (0-1)
- **Architecture**: 
  - MinutiaeNet feature extractor for both images
  - Similarity network for comparison
  - Absolute difference + MLP for final score

## Usage

### 1. Training MinutiaeNet (TensorFlow/Keras)

First, train the original MinutiaeNet using the TensorFlow implementation:

```bash
cd MinutiaeNet
python simple_train_livdet.py
```

This will create trained weights in the output directory.

### 2. Converting Weights (Optional)

If you want to use pretrained MinutiaeNet weights:

```bash
python utils/weight_converter.py
```

This utility helps convert TensorFlow weights to PyTorch format.

### 3. Training with MinutiaeNet Feature Extractor

Use the new training script:

```bash
python train_minutiaenet.py
```

Or modify your existing training script:

```python
from model import get_architecture

# Use MinutiaeNet instead of other architectures
model = get_architecture('minutiaenet', device=device, pretrained_path=None)
```

## Enhanced Metrics

The integration includes comprehensive biometric evaluation metrics:

### Basic Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Verification Metrics
- **FMR (False Match Rate)**: Rate of impostor comparisons accepted as genuine
- **FNMR (False Non-Match Rate)**: Rate of genuine comparisons rejected as impostor
- **GAR (Genuine Acceptance Rate)**: Rate of genuine comparisons correctly accepted
- **FAR (False Acceptance Rate)**: Same as FMR
- **FRR (False Rejection Rate)**: Same as FNMR

### Presentation Attack Detection Metrics
- **APCER**: Attack Presentation Classification Error Rate
- **BPCER**: Bona fide Presentation Classification Error Rate
- **IAPMR**: Impostor Attack Presentation Match Rate
- **IMG_accuracy**: Image-level accuracy
- **SGAR**: Spoof/Genuine Accept Rate

### ROC Analysis
- **ROC AUC**: Area under the ROC curve
- **EER (Equal Error Rate)**: Point where FAR = FRR
- **ROC Curve**: Visual representation of TPR vs FPR

## File Structure

```
model/
├── minutiaenet.py          # MinutiaeNet PyTorch implementation
├── metrics.py              # Enhanced metrics (FMR, FNMR, EER, ROC)
├── siamesenetwork.py       # Original Siamese network
├── mobilenetv2.py          # MobileNetV2 implementation
└── __init__.py            # Model factory

utils/
├── weight_converter.py     # TensorFlow to PyTorch weight conversion
└── __init__.py            # Utils package

train_minutiaenet.py        # Training script with enhanced metrics
```

## Configuration

### Model Parameters

```python
# MinutiaeNet configuration
model_config = {
    'input_shape': (400, 400, 1),
    'embedding_size': 512,
    'pretrained_path': None  # Path to pretrained weights
}
```

### Training Parameters

```python
# Training configuration
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.001
OUTPUT_DIR = 'output/minutiaenet_training'
```

## Output Files

The training script generates:

1. **Model Files**:
   - `best_model_eer.pth`: Best model based on EER
   - `best_model_loss.pth`: Best model based on validation loss
   - `final_model.pth`: Final trained model
   - `feature_model.pth`: Feature extraction model only

2. **Metrics Files**:
   - `test_metrics.json`: Comprehensive test metrics
   - `training_history.png`: Training curves
   - `roc_curve.png`: ROC curve with EER point

3. **Logs**:
   - Training progress with all metrics
   - EER and ROC AUC tracking

## Comparison with Other Models

| Model | Feature Extractor | Matching Method | Use Case |
|-------|------------------|-----------------|----------|
| Siamese | Custom CNN | Similarity Network | PAD |
| MobileNetV2 | MobileNetV2 | Similarity Network | PAD |
| MinutiaeNet | MinutiaeNet | Similarity Network | PAD + Feature Enhancement |

## Advantages of MinutiaeNet Integration

1. **Domain-Specific Features**: MinutiaeNet is specifically designed for fingerprint analysis
2. **Enhanced Feature Quality**: Better representation of fingerprint characteristics
3. **Transfer Learning**: Can leverage pretrained MinutiaeNet weights
4. **Comprehensive Evaluation**: Full suite of biometric metrics

## Next Steps

1. **Train MinutiaeNet**: Complete the TensorFlow training
2. **Convert Weights**: Use the weight converter utility
3. **Train Siamese Network**: Use MinutiaeNet features for matching
4. **Evaluate Performance**: Compare with other architectures using the enhanced metrics

## Troubleshooting

### Common Issues

1. **Weight Loading Errors**: Ensure TensorFlow model is compatible
2. **Memory Issues**: Reduce batch size for large models
3. **Metric Calculation**: Check input format (0-1 range for predictions)

### Performance Tips

1. **Use GPU**: Enable CUDA for faster training
2. **Data Augmentation**: Increase training data variety
3. **Hyperparameter Tuning**: Experiment with learning rates and architectures

## References

- Original MinutiaeNet paper: [MinutiaeNet: Learning Representations of Fingerprint Minutiae](https://arxiv.org/abs/1812.00002)
- Biometric metrics: ISO/IEC 30107-1:2016
- ROC analysis: [Receiver Operating Characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) 