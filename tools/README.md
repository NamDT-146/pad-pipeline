# Fingerprint Processing Pipeline

This module provides QThread-based classes for processing fingerprint images in a multi-threaded pipeline.

## Architecture

```
[Input Buffer] → [Preprocessor Thread] → [Enhancement Queue] → [Enhancer Thread] → [Extractor Queue]
```

## Classes

### 1. FingerprintPreprocessor (`tools/preprocessor.py`)

**Purpose**: Takes raw fingerprint images and applies preprocessing transformations.

**Input**: Raw images from `input_buffer` queue
**Output**: Preprocessed images to `enhance_queue`

**Features**:
- Loads images from file paths or processes PIL/numpy images
- Applies normalization, augmentation, and resizing
- Supports different modes ('train', 'val', 'test')
- Thread-safe queue management
- Progress monitoring and error handling

### 2. FingerprintEnhancer (`tools/enhancer.py`)

**Purpose**: Takes preprocessed images and applies enhancement algorithms.

**Input**: Preprocessed images from `enhance_queue`
**Output**: Enhanced results to `extractor_queue`

**Features**:
- Enhanced gradient-based orientation estimation
- O'Gorman filter with linear interpolation
- Adaptive binarization using 9×9 matrix
- Enhanced Zhang-Suen thinning with fix ridge algorithm
- Thread-safe processing with progress signals

## Dependencies

```bash
pip install PyQt5 numpy pillow scipy scikit-image opencv-python torch torchvision
```

## Usage Example

```python
import queue
from tools.preprocessor import FingerprintPreprocessor
from tools.enhancer import FingerprintEnhancer

# Create args object with parameters
class Args:
    def __init__(self):
        # Preprocessing parameters
        self.img_size = 224
        self.fingerprint_normalization = True
        self.histogram_equalization = False
        
        # Enhancement parameters
        self.apply_orientation = True
        self.apply_ogorman = True
        self.apply_binarization = True
        self.apply_thinning = True

# Create queues
input_buffer = queue.Queue(maxsize=100)
enhance_queue = queue.Queue(maxsize=50)
extractor_queue = queue.Queue(maxsize=50)

args = Args()

# Create and start threads
preprocessor = FingerprintPreprocessor(input_buffer, enhance_queue, args)
enhancer = FingerprintEnhancer(enhance_queue, extractor_queue, args)

preprocessor.start()
enhancer.start()

# Add images for processing
preprocessor.add_image("/path/to/fingerprint.bmp", "fp_001")

# Monitor results
while not extractor_queue.empty():
    result = extractor_queue.get()
    print(f"Processed {result['id']}")

# Stop threads
preprocessor.stop()
enhancer.stop()
```

## Signals and Monitoring

Both classes emit PyQt signals for monitoring:

### Preprocessor Signals
- `preprocessing_progress(int)`: Progress percentage
- `preprocessing_completed(str, object)`: Image ID and result
- `preprocessing_error(str, str)`: Image ID and error message
- `processing_stats(dict)`: Processing statistics

### Enhancer Signals
- `enhancement_progress(int)`: Progress percentage
- `enhancement_completed(str, dict)`: Image ID and enhancement results
- `enhancement_error(str, str)`: Image ID and error message
- `processing_stats(dict)`: Processing statistics

## Thread Control

Both classes support:
- `start()`: Start processing
- `stop()`: Stop processing gracefully
- `pause()`: Pause processing
- `resume()`: Resume processing
- `get_statistics()`: Get current statistics
- `get_queue_sizes()`: Get queue sizes

## Configuration Parameters

### Preprocessing Parameters (args object):
- `img_size`: Target image size (default: 224)
- `fingerprint_normalization`: Apply normalization (default: True)
- `histogram_equalization`: Apply histogram equalization (default: False)
- `rotation_degrees`: Random rotation range (default: 15)
- `horizontal_flip_p`: Horizontal flip probability (default: 0.5)
- `vertical_flip_p`: Vertical flip probability (default: 0.3)
- `blur_p`: Gaussian blur probability (default: 0.2)
- `elastic_p`: Elastic transform probability (default: 0.3)
- `crop_scale`: Random crop scale range (default: (0.8, 1.0))
- `crop_ratio`: Random crop ratio range (default: (0.9, 1.1))

### Enhancement Parameters (args object):
- `apply_orientation`: Apply orientation estimation (default: True)
- `apply_ogorman`: Apply O'Gorman filter (default: True)
- `apply_binarization`: Apply adaptive binarization (default: True)
- `apply_thinning`: Apply enhanced thinning (default: True)
- `orientation_block_size`: Block size for orientation (default: 16)
- `orientation_smooth_sigma`: Smoothing sigma (default: 1.0)
- `ogorman_filter_size`: O'Gorman filter size (default: 7)
- `ogorman_sigma_u`: Sigma along ridge direction (default: 2.0)
- `ogorman_sigma_v`: Sigma perpendicular to ridge (default: 0.5)
- `binarization_window_size`: Binarization window size (default: 9)
- `thinning_iterations`: Fix ridge iterations (default: 2)

## Output Data Structure

### Preprocessor Output (to enhance_queue):
```python
{
    'id': 'image_identifier',
    'original_image': PIL.Image or numpy.ndarray,
    'preprocessed_image': processed_image,
    'metadata': {
        'preprocessing_mode': 'train/val/test',
        'processing_time': timestamp,
        # ... additional metadata
    }
}
```

### Enhancer Output (to extractor_queue):
```python
{
    'id': 'image_identifier',
    'original_image': original_image,
    'preprocessed_image': preprocessed_image,
    'enhancement_results': {
        'original': original_array,
        'orientation_field': orientation_array,
        'enhanced': enhanced_array,
        'binary': binary_array,
        'thinned': thinned_array
    },
    'metadata': metadata_dict,
    'processing_time': timestamp
}
```

## Error Handling

Both classes handle errors gracefully:
- Invalid image formats are skipped with error signals
- Queue timeouts are handled to prevent deadlocks
- Processing errors are logged and signaled
- Threads can be safely stopped even during processing

## Performance Considerations

- Use appropriate queue sizes to balance memory usage and throughput
- Batch processing can improve performance for large datasets
- Monitor queue sizes to identify bottlenecks
- Consider using multiple worker threads for CPU-intensive operations

## Example Integration with PyTorch

```python
import torch
from torch.utils.data import Dataset, DataLoader

class FingerprintDataset(Dataset):
    def __init__(self, data_paths, labels, args, mode='train'):
        self.data_paths = data_paths
        self.labels = labels
        self.preprocessor = create_fingerprint_transforms(args, mode=mode)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.data_paths[idx])
        processed = self.preprocessor(image)
        return processed, self.labels[idx]
    
    def __len__(self):
        return len(self.data_paths)

# Create dataset and dataloader
dataset = FingerprintDataset(image_paths, labels, args, mode='train')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Testing

Run the example pipeline:
```bash
python tools/pipeline_example.py
```

This will create synthetic fingerprint images and process them through the complete pipeline.
