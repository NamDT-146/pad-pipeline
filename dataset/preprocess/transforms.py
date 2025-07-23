# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Code in this file is adapted from timm

import torch
import torchvision.transforms.functional as F
try:
    from torchvision.transforms.functional import InterpolationMode
    has_interpolation_mode = True
except ImportError:
    has_interpolation_mode = False
from PIL import Image
import warnings
import math
import random
import numpy as np


class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype)

_pil_interpolation_to_str = {
    Image.NEAREST: 'nearest',
    Image.BILINEAR: 'bilinear',
    Image.BICUBIC: 'bicubic',
    Image.BOX: 'box',
    Image.HAMMING: 'hamming',
    Image.LANCZOS: 'lanczos',
}
_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}


if has_interpolation_mode:
    _torch_interpolation_to_str = {
        InterpolationMode.NEAREST: 'nearest',
        InterpolationMode.BILINEAR: 'bilinear',
        InterpolationMode.BICUBIC: 'bicubic',
        InterpolationMode.BOX: 'box',
        InterpolationMode.HAMMING: 'hamming',
        InterpolationMode.LANCZOS: 'lanczos',
    }
    _str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}
else:
    _pil_interpolation_to_torch = {}
    _torch_interpolation_to_str = {}


def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]


def str_to_interp_mode(mode_str):
    if has_interpolation_mode:
        return _str_to_torch_interpolation[mode_str]
    else:
        return _str_to_pil_interpolation[mode_str]


def interp_mode_to_str(mode):
    if has_interpolation_mode:
        return _torch_interpolation_to_str[mode]
    else:
        return _pil_interpolation_to_str[mode]


_RANDOM_INTERPOLATION = (str_to_interp_mode('bilinear'), str_to_interp_mode('bicubic'))


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([interp_mode_to_str(x) for x in self.interpolation])
        else:
            interpolate_str = interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class FingerprintNormalization:
    """
    Fingerprint image normalization using mean and variance as described in 
    Shalash and Abou-Chadi [4] method.
    
    This implements the normalization equation:
    G(i,j) = M0 + sqrt(VAR0*(I(i,j)-M)^2/VAR) if I(i,j) > M
           = M0 - sqrt(VAR0*(I(i,j)-M)^2/VAR) otherwise
    
    Where M0 = 100, VAR0 = 100 are desired mean and variance.
    """
    
    def __init__(self, desired_mean=100.0, desired_variance=100.0):
        self.M0 = desired_mean
        self.VAR0 = desired_variance
    
    def __call__(self, pil_img):
        """
        Apply normalization to fingerprint image.
        
        Args:
            pil_img: PIL Image (grayscale)
            
        Returns:
            PIL Image: Normalized fingerprint image
        """
        # Convert PIL to numpy array
        img = np.array(pil_img, dtype=np.float32)
        
        # Calculate current mean and variance
        M = np.mean(img)
        VAR = np.var(img)
        
        # Avoid division by zero
        if VAR == 0:
            VAR = 1e-8
            
        # Apply normalization formula
        normalized = np.zeros_like(img)
        
        # Create masks for the two conditions
        condition = img > M
        
        # Apply formula for pixels > mean
        factor = np.sqrt(self.VAR0 * ((img - M) ** 2) / VAR)
        normalized[condition] = self.M0 + factor[condition]
        normalized[~condition] = self.M0 - factor[~condition]
        
        # Clip values to valid range and convert back to uint8
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(normalized, mode='L')

    def __repr__(self):
        return self.__class__.__name__ + f'(desired_mean={self.M0}, desired_variance={self.VAR0})'


class FingerprintHistogramEqualization:
    """
    Histogram equalization specifically for fingerprint images.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, pil_img):
        """Apply histogram equalization to fingerprint image."""
        # Convert PIL to numpy array
        img = np.array(pil_img, dtype=np.uint8)
        
        # Apply histogram equalization
        try:
            import cv2
            equalized = cv2.equalizeHist(img)
        except ImportError:
            # Use manual implementation if OpenCV not available
            equalized = self._manual_hist_eq(img)
        
        # Convert back to PIL Image
        return Image.fromarray(equalized, mode='L')
    
    def _manual_hist_eq(self, img):
        """Manual histogram equalization without OpenCV."""
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # Use linear interpolation of cdf to find new pixel values
        equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized)
        return equalized.reshape(img.shape).astype(np.uint8)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class FingerprintGaussianBlur:
    """
    Gaussian blur for fingerprint preprocessing and augmentation.
    """
    
    def __init__(self, kernel_size=3, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, pil_img):
        """Apply Gaussian blur to fingerprint image."""
        try:
            import torchvision.transforms as transforms
            blur_transform = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)
            return blur_transform(pil_img)
        except ImportError:
            # Fallback: return original image if torchvision not available
            return pil_img

    def __repr__(self):
        return self.__class__.__name__ + f'(kernel_size={self.kernel_size}, sigma={self.sigma})'


class FingerprintRandomRotation:
    """
    Random rotation specifically designed for fingerprint images.
    Keeps rotations smaller to preserve fingerprint structure.
    """
    
    def __init__(self, degrees=15, interpolation='bilinear', expand=False, center=None, fill=0):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.interpolation = str_to_interp_mode(interpolation)
        self.expand = expand
        self.center = center
        self.fill = fill
    
    def __call__(self, pil_img):
        """Apply random rotation to fingerprint image."""
        angle = random.uniform(*self.degrees)
        return F.rotate(pil_img, angle, self.interpolation, self.expand, self.center, self.fill)

    def __repr__(self):
        return self.__class__.__name__ + f'(degrees={self.degrees})'


class FingerprintRandomCrop:
    """
    Random crop for fingerprint images with padding if necessary.
    """
    
    def __init__(self, size, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
    
    def __call__(self, pil_img):
        """Apply random crop to fingerprint image."""
        return F.crop(pil_img, *self._get_params(pil_img))
    
    def _get_params(self, img):
        """Get parameters for random crop."""
        w, h = img.size
        th, tw = self.size
        
        if self.pad_if_needed and w < tw:
            img = F.pad(img, (tw - w, 0), self.fill, self.padding_mode)
            w = tw
        if self.pad_if_needed and h < th:
            img = F.pad(img, (0, th - h), self.fill, self.padding_mode)
            h = th
            
        if w == tw and h == th:
            return 0, 0, h, w
            
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size})'


class FingerprintRandomHorizontalFlip:
    """
    Random horizontal flip for fingerprint augmentation.
    """
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, pil_img):
        """Apply random horizontal flip to fingerprint image."""
        if random.random() < self.p:
            return F.hflip(pil_img)
        return pil_img

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class FingerprintRandomVerticalFlip:
    """
    Random vertical flip for fingerprint augmentation.
    """
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, pil_img):
        """Apply random vertical flip to fingerprint image."""
        if random.random() < self.p:
            return F.vflip(pil_img)
        return pil_img

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class FingerprintElasticTransform:
    """
    Elastic transformation for fingerprint augmentation.
    Simulates natural deformations that can occur during fingerprint capture.
    """
    
    def __init__(self, alpha=1.0, sigma=0.1, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, pil_img):
        """Apply elastic transformation to fingerprint image."""
        if random.random() < self.p:
            return self._elastic_transform(pil_img)
        return pil_img
    
    def _elastic_transform(self, pil_img):
        """Apply elastic transformation using scipy if available."""
        try:
            from scipy.ndimage import gaussian_filter, map_coordinates
            
            img = np.array(pil_img)
            shape = img.shape
            
            # Generate random displacement fields
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
            
            # Apply transformation
            transformed = map_coordinates(img, indices, order=1, mode='reflect')
            transformed = transformed.reshape(shape).astype(np.uint8)
            
            return Image.fromarray(transformed, mode='L')
        except ImportError:
            # Fallback: return original image if scipy not available
            return pil_img

    def __repr__(self):
        return self.__class__.__name__ + f'(alpha={self.alpha}, sigma={self.sigma}, p={self.p})'

class ReshapeForEnhancer:
    """Ensures numpy array has the correct shape for the enhancer."""
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            # If shape is [1, H, W] (channel first), convert to [H, W]
            if img.ndim == 3 and img.shape[0] == 1:
                return np.squeeze(img, axis=0)
            # If shape is [H, W, 1] (channel last), convert to [H, W]
            elif img.ndim == 3 and img.shape[2] == 1:
                return np.squeeze(img, axis=2)
        return img

class FingerprintPreprocessingPipeline:
    """
    Complete preprocessing pipeline for fingerprint images that integrates
    with PyTorch training pipelines.
    """
    
    def __init__(self, 
                 img_size=224,
                 apply_normalization=True,
                 apply_hist_eq=False,
                 training=True,
                 augmentation_params=None):
        """
        Initialize fingerprint preprocessing pipeline.
        
        Args:
            img_size: Target image size
            apply_normalization: Whether to apply paper's normalization
            apply_hist_eq: Whether to apply histogram equalization
            training: Whether in training mode (enables augmentations)
            augmentation_params: Dictionary of augmentation parameters
        """
        self.img_size = img_size
        self.apply_normalization = apply_normalization
        self.apply_hist_eq = apply_hist_eq
        self.training = training
        
        # Default augmentation parameters
        default_aug_params = {
            'rotation_degrees': 15,
            'horizontal_flip_p': 0.5,
            'vertical_flip_p': 0.3,
            'blur_p': 0.2,
            'elastic_p': 0.3,
            'crop_scale': (0.8, 1.0),
            'crop_ratio': (0.9, 1.1)
        }
        
        if augmentation_params:
            default_aug_params.update(augmentation_params)
        self.aug_params = default_aug_params
        
        # Build transformation pipeline
        self.transforms = self._build_transforms()
    
    def _build_transforms(self):
        """Build the transformation pipeline."""
        transform_list = []
        
        # Convert to grayscale if needed (fingerprints are typically grayscale)
        transform_list.append(lambda x: x.convert('L') if x.mode != 'L' else x)
        
        # Apply fingerprint-specific normalization
        if self.apply_normalization:
            transform_list.append(FingerprintNormalization())
        
        # Apply histogram equalization
        if self.apply_hist_eq:
            transform_list.append(FingerprintHistogramEqualization())
        
        # Training augmentations
        if self.training:
            # Random rotation
            transform_list.append(FingerprintRandomRotation(
                degrees=self.aug_params['rotation_degrees']
            ))
            
            # Random resized crop
            transform_list.append(RandomResizedCropAndInterpolation(
                size=self.img_size,
                scale=self.aug_params['crop_scale'],
                ratio=self.aug_params['crop_ratio']
            ))
            
            # Random flips
            transform_list.append(FingerprintRandomHorizontalFlip(
                p=self.aug_params['horizontal_flip_p']
            ))
            transform_list.append(FingerprintRandomVerticalFlip(
                p=self.aug_params['vertical_flip_p']
            ))
            
            # Gaussian blur
            if random.random() < self.aug_params['blur_p']:
                transform_list.append(FingerprintGaussianBlur(kernel_size=3, sigma=1.0))
            
            # Elastic transformation
            transform_list.append(FingerprintElasticTransform(
                alpha=1.0, sigma=0.1, p=self.aug_params['elastic_p']
            ))
        else:
            # Validation/test: just resize
            transform_list.append(lambda x: x.resize((self.img_size, self.img_size), Image.BILINEAR))
        
        # Convert to numpy
        transform_list.append(ToNumpy())

        transform_list.append(ReshapeForEnhancer())
        
        # Normalize to [0, 1] range
        # transform_list.append(lambda x: x.astype(np.float32) / 255.0)        
        return Compose(transform_list)
    
    def __call__(self, pil_img):
        """Apply the complete preprocessing pipeline."""
        return self.transforms(pil_img)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'img_size={self.img_size}, '
                f'apply_normalization={self.apply_normalization}, '
                f'apply_hist_eq={self.apply_hist_eq}, '
                f'training={self.training})')


class Compose:
    """
    Compose several transforms together.
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


def create_fingerprint_transforms(args, mode='train'):
    """
    Factory function to create fingerprint preprocessing transforms.
    
    Args:
        args: Arguments object containing preprocessing parameters
        mode: 'train', 'val', or 'test'
        
    Returns:
        FingerprintPreprocessingPipeline instance
    """
    img_size = getattr(args, 'img_size', 224)
    apply_normalization = getattr(args, 'fingerprint_normalization', True)
    apply_hist_eq = getattr(args, 'histogram_equalization', False)
    
    # Augmentation parameters from args
    augmentation_params = {
        'rotation_degrees': getattr(args, 'rotation_degrees', 15),
        'horizontal_flip_p': getattr(args, 'horizontal_flip_p', 0.5),
        'vertical_flip_p': getattr(args, 'vertical_flip_p', 0.3),
        'blur_p': getattr(args, 'blur_p', 0.2),
        'elastic_p': getattr(args, 'elastic_p', 0.3),
        'crop_scale': getattr(args, 'crop_scale', (0.8, 1.0)),
        'crop_ratio': getattr(args, 'crop_ratio', (0.9, 1.1))
    }
    
    training = mode == 'train'
    
    return FingerprintPreprocessingPipeline(
        img_size=img_size,
        apply_normalization=apply_normalization,
        apply_hist_eq=apply_hist_eq,
        training=training,
        augmentation_params=augmentation_params
    )

