# Import fingerprint preprocessing functionality from transforms
from .transforms import (
    FingerprintNormalization,
    FingerprintHistogramEqualization,
    FingerprintPreprocessingPipeline,
    create_fingerprint_transforms,
    FingerprintRandomRotation,
    FingerprintRandomHorizontalFlip,
    FingerprintRandomVerticalFlip,
    FingerprintGaussianBlur,
    FingerprintElasticTransform,
    FingerprintRandomCrop
)

# Re-export for convenience
__all__ = [
    'FingerprintNormalization',
    'FingerprintHistogramEqualization', 
    'FingerprintPreprocessingPipeline',
    'create_fingerprint_transforms',
    'FingerprintRandomRotation',
    'FingerprintRandomHorizontalFlip',
    'FingerprintRandomVerticalFlip',
    'FingerprintGaussianBlur',
    'FingerprintElasticTransform',
    'FingerprintRandomCrop'
]

def get_default_args(mode='train'):
    """
    Create arguments object with default parameters, matching test_enhancement.py.
    
    Args:
        mode: 'train', 'val', or 'test' - affects augmentation settings
    
    Returns:
        Args object with all required parameters
    """
    class Args:
        def __init__(self):
            # Common parameters
            self.img_size = 1000  # Standard image size for training
            
            # Preprocessing parameters
            self.fingerprint_normalization = True
            self.histogram_equalization = True
            
            # Augmentation parameters (only applied in training mode)
            self.rotation_degrees = 30 if mode == 'train' else 0
            self.horizontal_flip_p = 0.5 if mode == 'train' else 0
            self.vertical_flip_p = 0.3 if mode == 'train' else 0
            self.blur_p = 0.7 if mode == 'train' else 0
            self.elastic_p = 0.3 if mode == 'train' else 0
            self.crop_scale = (0.9, 1.1) if mode == 'train' else (1.0, 1.0)
            self.crop_ratio = (0.9, 1.1) if mode == 'train' else (1.0, 1.0)
            
            # Enhancement parameters (from test_enhancement.py)
            self.apply_orientation = True
            self.apply_ogorman = True
            self.apply_binarization = True
            self.apply_thinning = True  # Thinning might remove too much information for verification
            self.orientation_block_size = 16
            self.orientation_smooth_sigma = 1.0
            self.ogorman_filter_size = 7
            self.ogorman_sigma_u = 2.0
            self.ogorman_sigma_v = 0.5
            self.binarization_window_size = 9
            self.thinning_iterations = 1
    
    return Args()