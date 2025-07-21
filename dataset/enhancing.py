import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Union, Tuple, Optional
import math
from scipy import ndimage
from skimage import morphology
import numba



class EnhancedGradientOrientationEstimation:
    """
    Enhanced gradient-based orientation estimation that converts angle range 
    from [-PI/4, PI/4] to [0, PI] to remove inconsistency.
    
    Based on the enhanced gradient-based approach mentioned in the paper.
    """
    
    def __init__(self, block_size=16, smooth_sigma=1.0):
        self.block_size = block_size
        self.smooth_sigma = smooth_sigma
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Compute enhanced orientation field for fingerprint image.
        
        Args:
            image: Input grayscale fingerprint image
            
        Returns:
            Orientation field with angles in range [0, PI]
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute gradients using Sobel operators
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute orientation using enhanced gradient method
        # Convert from [-PI/4, PI/4] to [0, PI] range
        orientation = np.zeros_like(image, dtype=np.float64)
        
        # Block-wise processing for stability
        h, w = image.shape
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                # Define block boundaries
                i_end = min(i + self.block_size, h)
                j_end = min(j + self.block_size, w)
                
                # Extract block gradients
                block_gx = grad_x[i:i_end, j:j_end]
                block_gy = grad_y[i:i_end, j:j_end]
                
                # Compute local orientation using double angle representation
                # This removes the 180-degree ambiguity
                gxx = np.sum(block_gx * block_gx)
                gyy = np.sum(block_gy * block_gy)
                gxy = np.sum(block_gx * block_gy)
                
                # Enhanced orientation calculation
                if gxx != gyy:
                    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
                else:
                    theta = 0.0
                
                # Convert to [0, PI] range (enhancement from paper)
                if theta < 0:
                    theta += np.pi
                
                # Assign to block
                orientation[i:i_end, j:j_end] = theta
        
        # Apply Gaussian smoothing to orientation field
        if self.smooth_sigma > 0:
            orientation = ndimage.gaussian_filter(orientation, sigma=self.smooth_sigma)
        
        return orientation

    def __repr__(self):
        return f'{self.__class__.__name__}(block_size={self.block_size}, smooth_sigma={self.smooth_sigma})'


class EnhancedOGormanFilter:
    """
    Enhanced O'Gorman filter for fingerprint image enhancement with linear interpolation
    to solve the rotation coordinate problem mentioned in the paper.
    
    Uses anisotropic smoothening kernel oriented parallel to ridges.
    """
    
    def __init__(self, filter_size=7, sigma_u=2.0, sigma_v=0.5):
        self.filter_size = filter_size
        self.sigma_u = sigma_u  # Along ridge direction
        self.sigma_v = sigma_v  # Perpendicular to ridge direction
        
        # Create base 0° direction filter (7x7 matrix)
        self.base_filter = self._create_base_filter()
    
    def _create_base_filter(self):
        """Create the base 7x7 O'Gorman filter for 0° direction."""
        size = self.filter_size
        center = size // 2
        filter_kernel = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                u = i - center
                v = j - center
                
                # O'Gorman filter formula
                filter_kernel[i, j] = np.exp(-(u**2 / (2 * self.sigma_u**2) + 
                                              v**2 / (2 * self.sigma_v**2)))
        
        # Normalize filter
        filter_kernel = filter_kernel / np.sum(filter_kernel)
        return filter_kernel
    
    def _rotate_filter(self, angle):
        """
        Rotate the base filter by given angle using linear interpolation
        to solve the coordinate problem mentioned in the paper.
        """
        size = self.filter_size
        center = size // 2
        rotated_filter = np.zeros((size, size))
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        for i in range(size):
            for j in range(size):
                # Original coordinates relative to center
                x = i - center
                y = j - center
                
                # Rotate coordinates
                x_rot = x * cos_angle - y * sin_angle + center
                y_rot = x * sin_angle + y * cos_angle + center
                
                # Linear interpolation to solve non-integer coordinate problem
                if 0 <= x_rot < size - 1 and 0 <= y_rot < size - 1:
                    x1, y1 = int(x_rot), int(y_rot)
                    x2, y2 = x1 + 1, y1 + 1
                    
                    # Bilinear interpolation weights
                    wx = x_rot - x1
                    wy = y_rot - y1
                    
                    # Interpolate using the base filter
                    value = (1 - wx) * (1 - wy) * self.base_filter[x1, y1] + \
                            wx * (1 - wy) * self.base_filter[x2, y1] + \
                            (1 - wx) * wy * self.base_filter[x1, y2] + \
                            wx * wy * self.base_filter[x2, y2]
                    
                    rotated_filter[i, j] = value
        
        return rotated_filter
    
    def __call__(self, image: np.ndarray, orientation_field: np.ndarray) -> np.ndarray:
        """
        Apply enhanced O'Gorman filter to fingerprint image.
        
        Args:
            image: Input fingerprint image
            orientation_field: Orientation field from gradient estimation
            
        Returns:
            Enhanced fingerprint image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        enhanced = np.zeros_like(image, dtype=np.float64)
        h, w = image.shape
        pad = self.filter_size // 2
        
        # Pad image to handle borders
        padded_image = np.pad(image, pad, mode='reflect')
        
        # Apply filter pixel by pixel with orientation-based rotation
        for i in range(h):
            for j in range(w):
                # Get local orientation
                theta = orientation_field[i, j]
                
                # Rotate filter according to local orientation
                rotated_filter = self._rotate_filter(theta)
                
                # Extract local region
                region = padded_image[i:i+self.filter_size, j:j+self.filter_size]
                
                # Apply rotated filter
                enhanced[i, j] = np.sum(region * rotated_filter)
        
        # Normalize to [0, 255] range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        return enhanced

    def __repr__(self):
        return f'{self.__class__.__name__}(filter_size={self.filter_size})'


# class AdaptiveBinarization:
#     """
#     Adaptive thresholding using 9x9 matrix as mentioned in the paper.
#     Uses threshold value as mean of all neighbor pixels in 9x9 window.
#     """
    
#     def __init__(self, window_size=9):
#         self.window_size = window_size
    
#     def __call__(self, image: np.ndarray) -> np.ndarray:
#         """
#         Apply adaptive binarization using 9x9 matrix.
        
#         Args:
#             image: Input grayscale image
            
#         Returns:
#             Binary image (0 and 1 values)
#         """
#         if len(image.shape) == 3:
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
#         # Convert to float for calculations
#         image_float = image.astype(np.float64)
#         binary = np.zeros_like(image, dtype=np.uint8)
        
#         pad = self.window_size // 2
#         padded = np.pad(image_float, pad, mode='reflect')
        
#         h, w = image.shape
        
#         # Apply adaptive thresholding
#         for i in range(h):
#             for j in range(w):
#                 # Extract 9x9 neighborhood
#                 neighborhood = padded[i:i+self.window_size, j:j+self.window_size]
                
#                 # Calculate threshold as mean of neighborhood
#                 threshold = np.mean(neighborhood)
                
#                 # Apply thresholding: if pixel > threshold, set to 1, else 0
#                 if image[i, j] > threshold:
#                     binary[i, j] = 1
#                 else:
#                     binary[i, j] = 0
        
#         return binary

#     def __repr__(self):
#         return f'{self.__class__.__name__}(window_size={self.window_size})'

class AdaptiveBinarization:
    """
    Adaptive thresholding using 9x9 matrix as mentioned in the paper.
    Uses threshold value as mean of all neighbor pixels in 9x9 window.
    Optimized implementation using OpenCV.
    """
    
    def __init__(self, window_size=9):
        self.window_size = window_size
        # Must be odd
        if self.window_size % 2 == 0:
            self.window_size += 1
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive binarization using OpenCV's adaptiveThreshold.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary image (0 and 1 values)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply OpenCV's adaptiveThreshold - much faster than manual implementation
        binary = cv2.adaptiveThreshold(
            image,
            1,  # Max value
            cv2.ADAPTIVE_THRESH_MEAN_C,  # Mean thresholding
            cv2.THRESH_BINARY,
            self.window_size,
            0  # Constant subtracted from mean
        )
        
        return binary

    def __repr__(self):
        return f'{self.__class__.__name__}(window_size={self.window_size})'

class EnhancedZhangSuenThinning:
    """
    Enhanced Zhang-Suen thinning algorithm with vectorized operations.
    Uses skimage.morphology for better performance.
    """
    
    def __init__(self, iterations=2):
        self.iterations = iterations
    
    def __call__(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Apply enhanced Zhang-Suen thinning with optimized implementation.
        
        Args:
            binary_image: Input binary image (0 and 1 values)
            
        Returns:
            Thinned binary image
        """
        # Ensure binary values are 0 and 1
        image = (binary_image > 0).astype(np.uint8)
        
        # Use skimage's optimized thinning implementation
        # This is a C-based implementation of Zhang-Suen algorithm
        thinned = morphology.thin(image).astype(np.uint8)
        
        # Apply fix ridge algorithm using morphological operations
        for _ in range(self.iterations):
            # Remove isolated pixels (Rule 1)
            kernel = np.ones((3, 3), np.uint8)
            kernel[1, 1] = 0  # Center pixel doesn't count
            isolated_mask = cv2.filter2D(thinned.astype(np.float32), -1, kernel) == 0
            thinned[isolated_mask] = 0
            
            # Add pixels where needed (Rule 2)
            neighbors_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
            neighbor_count = cv2.filter2D(thinned.astype(np.float32), -1, neighbors_kernel)
            thinned[(neighbor_count >= 3) & (thinned == 0)] = 1
        
        return thinned
    def __repr__(self):
        return f'{self.__class__.__name__}(iterations={self.iterations})'


class FingerprintEnhancementPipeline:
    """
    Complete fingerprint enhancement pipeline that integrates all enhancement
    techniques mentioned in the papers.
    """
    
    def __init__(self, 
                 apply_orientation=True,
                 apply_ogorman=True,
                 apply_binarization=True,
                 apply_thinning=True,
                 orientation_params=None,
                 ogorman_params=None,
                 binarization_params=None,
                 thinning_params=None):
        """
        Initialize enhancement pipeline.
        
        Args:
            apply_orientation: Whether to apply enhanced gradient orientation
            apply_ogorman: Whether to apply enhanced O'Gorman filter
            apply_binarization: Whether to apply adaptive binarization
            apply_thinning: Whether to apply enhanced Zhang-Suen thinning
            *_params: Parameter dictionaries for each enhancement step
        """
        self.apply_orientation = apply_orientation
        self.apply_ogorman = apply_ogorman
        self.apply_binarization = apply_binarization
        self.apply_thinning = apply_thinning
        
        # Initialize enhancement components
        orientation_params = orientation_params or {}
        self.orientation_estimator = EnhancedGradientOrientationEstimation(**orientation_params)
        
        ogorman_params = ogorman_params or {}
        self.ogorman_filter = EnhancedOGormanFilter(**ogorman_params)
        
        binarization_params = binarization_params or {}
        self.binarizer = AdaptiveBinarization(**binarization_params)
        
        thinning_params = thinning_params or {}
        self.thinner = EnhancedZhangSuenThinning(**thinning_params)
    
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> dict:
        """
        Apply complete enhancement pipeline.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Dictionary containing results from each enhancement step
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        results = {'original': image.copy()}

        print("starting orientation estimation...")
        
        # Step 1: Enhanced gradient-based orientation estimation
        if self.apply_orientation:
            orientation_field = self.orientation_estimator(image)
            results['orientation_field'] = orientation_field
        else:
            orientation_field = None

        print("starting O'Gorman filter...")
        
        # Step 2: Enhanced O'Gorman filter for image enhancement
        if self.apply_ogorman and orientation_field is not None:
            enhanced_image = self.ogorman_filter(image, orientation_field)
            results['enhanced'] = enhanced_image
            current_image = enhanced_image
        else:
            current_image = image

        print("starting adaptive binarization...")
        
        # Step 3: Adaptive binarization using 9x9 matrix
        if self.apply_binarization:
            binary_image = self.binarizer(current_image)
            results['binary'] = binary_image
            current_image = binary_image
        
        print("starting enhanced Zhang-Suen thinning...")
        # Step 4: Enhanced Zhang-Suen thinning with fix ridge algorithm
        if self.apply_thinning and self.apply_binarization:
            thinned_image = self.thinner(current_image)
            results['thinned'] = thinned_image

        print("enhancement pipeline completed!")
        
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(apply_orientation={self.apply_orientation}, ' \
               f'apply_ogorman={self.apply_ogorman}, apply_binarization={self.apply_binarization}, ' \
               f'apply_thinning={self.apply_thinning})'


def create_fingerprint_enhancement(args):
    """
    Factory function to create fingerprint enhancement pipeline similar to preprocessing.
    
    Args:
        args: Arguments object containing enhancement parameters
        
    Returns:
        FingerprintEnhancementPipeline instance
    """
    # Extract parameters from args
    apply_orientation = getattr(args, 'apply_orientation', True)
    apply_ogorman = getattr(args, 'apply_ogorman', True)
    apply_binarization = getattr(args, 'apply_binarization', True)
    apply_thinning = getattr(args, 'apply_thinning', True)
    
    # Orientation estimation parameters
    orientation_params = {
        'block_size': getattr(args, 'orientation_block_size', 16),
        'smooth_sigma': getattr(args, 'orientation_smooth_sigma', 1.0)
    }
    
    # O'Gorman filter parameters
    ogorman_params = {
        'filter_size': getattr(args, 'ogorman_filter_size', 7),
        'sigma_u': getattr(args, 'ogorman_sigma_u', 2.0),
        'sigma_v': getattr(args, 'ogorman_sigma_v', 0.5)
    }
    
    # Binarization parameters
    binarization_params = {
        'window_size': getattr(args, 'binarization_window_size', 9)
    }
    
    # Thinning parameters
    thinning_params = {
        'iterations': getattr(args, 'thinning_iterations', 2)
    }
    
    return FingerprintEnhancementPipeline(
        apply_orientation=apply_orientation,
        apply_ogorman=apply_ogorman,
        apply_binarization=apply_binarization,
        apply_thinning=apply_thinning,
        orientation_params=orientation_params,
        ogorman_params=ogorman_params,
        binarization_params=binarization_params,
        thinning_params=thinning_params
    )


def example_usage():
    """Example of how to use the enhancement pipeline."""
    
    # Mock args object
    class Args:
        def __init__(self):
            self.apply_orientation = False
            self.apply_ogorman = False
            self.apply_binarization = True
            self.apply_thinning = True
            self.orientation_block_size = 16
            self.ogorman_filter_size = 7
            self.binarization_window_size = 9
            self.thinning_iterations = 1
    
    args = Args()
    
    # Create enhancement pipeline
    enhancer = create_fingerprint_enhancement(args)
    
    # Example with dummy image
    dummy_image = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    results = enhancer(dummy_image)
    
    print("Enhancement pipeline completed!")
    print(f"Available results: {list(results.keys())}")
    
    return results


if __name__ == "__main__":
    example_usage()