"""
Example showing how to use FingerprintPreprocessor and FingerprintEnhancer together.

This creates a complete pipeline:
Input Buffer -> Preprocessor -> Enhancement Queue -> Enhancer -> Extractor Queue
"""

import sys
import queue
import time
import numpy as np
from PIL import Image

# Import the processing classes
try:
    from PyQt5.QtWidgets import QApplication
    from tools.preprocessor import FingerprintPreprocessor
    from tools.enhancer import FingerprintEnhancer
    PYQT_AVAILABLE = True
except ImportError:
    print("PyQt5 not available. Install with: pip install PyQt5")
    PYQT_AVAILABLE = False


def create_pipeline_example():
    """Create and run a complete fingerprint processing pipeline."""
    
    if not PYQT_AVAILABLE:
        print("Cannot run example without PyQt5")
        return
    
    # Mock args object with all necessary parameters
    class Args:
        def __init__(self):
            # Preprocessing parameters
            self.img_size = 224
            self.fingerprint_normalization = True
            self.histogram_equalization = False
            self.rotation_degrees = 15
            self.horizontal_flip_p = 0.5
            self.vertical_flip_p = 0.3
            self.blur_p = 0.2
            self.elastic_p = 0.3
            self.crop_scale = (0.8, 1.0)
            self.crop_ratio = (0.9, 1.1)
            
            # Enhancement parameters
            self.apply_orientation = True
            self.apply_ogorman = True
            self.apply_binarization = True
            self.apply_thinning = True
            self.orientation_block_size = 16
            self.orientation_smooth_sigma = 1.0
            self.ogorman_filter_size = 7
            self.ogorman_sigma_u = 2.0
            self.ogorman_sigma_v = 0.5
            self.binarization_window_size = 9
            self.thinning_iterations = 2
    
    # Create queues for the pipeline
    input_buffer = queue.Queue(maxsize=100)
    enhance_queue = queue.Queue(maxsize=50)
    extractor_queue = queue.Queue(maxsize=50)
    
    args = Args()
    
    # Create processor instances
    preprocessor = FingerprintPreprocessor(
        input_buffer=input_buffer,
        enhance_queue=enhance_queue,
        args=args,
        mode='train',
        batch_size=2
    )
    
    enhancer = FingerprintEnhancer(
        enhance_queue=enhance_queue,
        extractor_queue=extractor_queue,
        args=args,
        batch_size=2
    )
    
    # Connect signals for monitoring
    def on_preprocessing_completed(img_id, result):
        print(f"✓ Preprocessed: {img_id}")
    
    def on_preprocessing_error(img_id, error):
        print(f"✗ Preprocessing error for {img_id}: {error}")
    
    def on_enhancement_completed(img_id, results):
        print(f"✓ Enhanced: {img_id} -> {list(results.keys())}")
    
    def on_enhancement_error(img_id, error):
        print(f"✗ Enhancement error for {img_id}: {error}")
    
    def on_processing_stats(stats):
        print(f"Stats: {stats}")
    
    # Connect signals
    preprocessor.preprocessing_completed.connect(on_preprocessing_completed)
    preprocessor.preprocessing_error.connect(on_preprocessing_error)
    preprocessor.processing_stats.connect(on_processing_stats)
    
    enhancer.enhancement_completed.connect(on_enhancement_completed)
    enhancer.enhancement_error.connect(on_enhancement_error)
    enhancer.processing_stats.connect(on_processing_stats)
    
    print("Starting fingerprint processing pipeline...")
    
    # Start both threads
    preprocessor.start()
    enhancer.start()
    
    # Generate and add test fingerprint images
    print("Adding test images to pipeline...")
    
    for i in range(5):
        # Create synthetic fingerprint-like image
        dummy_image = np.random.randint(50, 200, (300, 300), dtype=np.uint8)
        
        # Add some ridge-like patterns
        x, y = np.meshgrid(np.linspace(0, 10, 300), np.linspace(0, 10, 300))
        pattern = np.sin(x * 2) * np.cos(y * 2) * 50 + 128
        dummy_image = np.clip(dummy_image + pattern.astype(np.uint8), 0, 255)
        
        image_id = f"synthetic_{i:03d}"
        metadata = {
            "source": "synthetic",
            "index": i,
            "created_at": time.time()
        }
        
        preprocessor.add_image(dummy_image, image_id, metadata)
        time.sleep(0.1)  # Small delay between additions
    
    print("All test images added. Processing...")
    
    # Monitor processing for a while
    start_time = time.time()
    timeout = 30  # 30 seconds timeout
    
    while time.time() - start_time < timeout:
        # Check queue sizes
        preproc_stats = preprocessor.get_statistics()
        enhance_stats = enhancer.get_statistics()
        
        print(f"\rProgress - Preprocessed: {preproc_stats['processed_count']}, "
              f"Enhanced: {enhance_stats['processed_count']}, "
              f"Queues: {preproc_stats['queue_sizes']}", end="")
        
        # Check if extractor queue has results
        try:
            while not extractor_queue.empty():
                result = extractor_queue.get_nowait()
                print(f"\n→ Final result for {result['id']}: "
                      f"{list(result['enhancement_results'].keys())}")
        except queue.Empty:
            pass
        
        # Break if all processing is complete
        if (extractor_queue.qsize() > 0 and 
            input_buffer.empty() and 
            enhance_queue.empty()):
            print("\nProcessing appears complete!")
            break
        
        time.sleep(0.5)
    
    print(f"\nFinal statistics:")
    print(f"Preprocessor: {preprocessor.get_statistics()}")
    print(f"Enhancer: {enhancer.get_statistics()}")
    print(f"Extractor queue size: {extractor_queue.qsize()}")
    
    # Stop threads
    print("Stopping processing threads...")
    preprocessor.stop()
    enhancer.stop()
    
    print("Pipeline example completed!")


def simple_single_image_example():
    """Simple example processing a single image through the pipeline."""
    
    if not PYQT_AVAILABLE:
        print("Cannot run example without PyQt5")
        return
    
    # Create a simple synthetic fingerprint image
    print("Creating synthetic fingerprint image...")
    
    # Generate ridge-like pattern
    size = 300
    x, y = np.meshgrid(np.linspace(0, 20, size), np.linspace(0, 20, size))
    
    # Create sinusoidal ridges with some rotation
    angle = np.pi / 6  # 30 degrees
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    
    ridges = np.sin(x_rot * 2) * 127 + 128
    ridges = np.clip(ridges, 0, 255).astype(np.uint8)
    
    # Add some noise
    noise = np.random.normal(0, 15, (size, size))
    fingerprint = np.clip(ridges + noise, 0, 255).astype(np.uint8)
    
    # Save as example
    Image.fromarray(fingerprint, mode='L').save('/tmp/synthetic_fingerprint.png')
    print("Synthetic fingerprint saved to /tmp/synthetic_fingerprint.png")
    
    return fingerprint


if __name__ == "__main__":
    if PYQT_AVAILABLE:
        # Initialize Qt application
        app = QApplication(sys.argv)
        
        # Run the pipeline example
        create_pipeline_example()
        
        # Create synthetic fingerprint
        simple_single_image_example()
        
    else:
        # Show what the pipeline structure looks like
        print("""
Fingerprint Processing Pipeline Structure:

[Input Buffer] -> [Preprocessor Thread] -> [Enhancement Queue] -> [Enhancer Thread] -> [Extractor Queue]

1. FingerprintPreprocessor:
   - Takes raw images from input_buffer
   - Applies normalization, augmentation, resizing
   - Outputs preprocessed images to enhance_queue

2. FingerprintEnhancer:
   - Takes preprocessed images from enhance_queue
   - Applies orientation estimation, O'Gorman filter, binarization, thinning
   - Outputs enhanced results to extractor_queue

Both threads run independently and communicate via thread-safe queues.
Each thread can be paused, resumed, and stopped safely.
Progress and error signals are emitted for monitoring.

To run this pipeline, install PyQt5:
pip install PyQt5
        """)
