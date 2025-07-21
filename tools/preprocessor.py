import sys
import queue
import threading
import time
from typing import Optional, Dict, Any, Union
import numpy as np
from PIL import Image
import os

# PyQt5 imports with fallback
try:
    from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
    from PyQt5.QtWidgets import QApplication
    PYQT_AVAILABLE = True
except ImportError:
    print("PyQt5 not available. Install with: pip install PyQt5")
    # Create dummy classes for development
    class QThread:
        def __init__(self): pass
        def start(self): pass
        def stop(self): pass
        def wait(self, timeout): return True
        def terminate(self): pass
        def isRunning(self): return False
        def msleep(self, ms): time.sleep(ms/1000)
    
    class pyqtSignal:
        def __init__(self, *args): pass
        def emit(self, *args): pass
        def connect(self, func): pass
    
    class QMutex:
        def lock(self): pass
        def unlock(self): pass
    
    class QWaitCondition:
        def wait(self, mutex): pass
        def wakeAll(self): pass
    
    class QApplication:
        def __init__(self, args): pass
    
    PYQT_AVAILABLE = False

# Import fingerprint processing functions from dataset module
try:
    from dataset.preprocessing import create_fingerprint_transforms
except ImportError as e:
    print(f"Error importing dataset modules: {e}")
    print("Make sure the dataset module is in your Python path")


class FingerprintPreprocessor(QThread):
    """
    QThread class for fingerprint preprocessing.
    
    Takes raw images from input buffer, applies preprocessing transforms,
    and puts results into enhance_queue for enhancement processing.
    """
    
    # PyQt signals for communication
    preprocessing_progress = pyqtSignal(int)  # Progress percentage
    preprocessing_completed = pyqtSignal(str, object)  # Image ID and preprocessed image
    preprocessing_error = pyqtSignal(str, str)  # Image ID and error message
    processing_stats = pyqtSignal(dict)  # Processing statistics
    
    def __init__(self, 
                 input_buffer: queue.Queue,
                 enhance_queue: queue.Queue,
                 args: Any,
                 mode: str = 'train',
                 max_workers: int = 1,
                 batch_size: int = 1):
        """
        Initialize the fingerprint preprocessor thread.
        
        Args:
            input_buffer: Input queue containing raw images
            enhance_queue: Output queue for preprocessed images
            args: Arguments object containing preprocessing parameters
            mode: Processing mode ('train', 'val', 'test')
            max_workers: Maximum number of worker threads
            batch_size: Number of images to process in batch
        """
        super().__init__()
        
        # Queue management
        self.input_buffer = input_buffer
        self.enhance_queue = enhance_queue
        
        # Processing parameters
        self.args = args
        self.mode = mode
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Thread control
        self._stop_flag = False
        self._pause_flag = False
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()
        
        # Statistics tracking
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
        # Create preprocessing pipeline
        try:
            self.preprocessor = create_fingerprint_transforms(args, mode=mode)
            print(f"Fingerprint preprocessing pipeline initialized successfully for mode: {mode}")
        except Exception as e:
            print(f"Error initializing preprocessing pipeline: {e}")
            self.preprocessor = None
    
    def run(self):
        """Main thread execution loop."""
        print("Fingerprint preprocessor thread started")
        self.start_time = time.time()
        
        if self.preprocessor is None:
            self.preprocessing_error.emit("INIT_ERROR", "Preprocessing pipeline not initialized")
            return
        
        while not self._stop_flag:
            try:
                # Handle pause state
                self._mutex.lock()
                while self._pause_flag and not self._stop_flag:
                    self._pause_condition.wait(self._mutex)
                self._mutex.unlock()
                
                if self._stop_flag:
                    break
                
                # Get batch of images from input_buffer
                batch_data = self._get_batch()
                
                if not batch_data:
                    # No data available, sleep briefly
                    self.msleep(100)
                    continue
                
                # Process batch
                self._process_batch(batch_data)
                
            except Exception as e:
                print(f"Error in preprocessor thread: {e}")
                self.preprocessing_error.emit("THREAD_ERROR", str(e))
                self.msleep(1000)  # Wait before retrying
        
        self._emit_final_stats()
        print("Fingerprint preprocessor thread stopped")
    
    def _get_batch(self) -> list:
        """Get a batch of images from the input_buffer."""
        batch = []
        
        try:
            # Try to get up to batch_size items from queue
            for _ in range(self.batch_size):
                try:
                    # Non-blocking get with short timeout
                    item = self.input_buffer.get(timeout=0.1)
                    batch.append(item)
                except queue.Empty:
                    break
            
        except Exception as e:
            print(f"Error getting batch from input_buffer: {e}")
        
        return batch
    
    def _process_batch(self, batch_data: list):
        """Process a batch of raw images."""
        for item in batch_data:
            if self._stop_flag:
                break
            
            try:
                # Extract data from queue item
                image_id = item.get('id', 'unknown')
                image_data = item.get('image')
                image_path = item.get('path')
                metadata = item.get('metadata', {})
                
                # Load image if path is provided
                if image_data is None and image_path:
                    image_data = self._load_image(image_path)
                
                if image_data is None:
                    self.preprocessing_error.emit(image_id, "No image data or invalid path")
                    continue
                
                # Apply preprocessing
                preprocessed_image = self._preprocess_image(image_data, image_id)
                
                if preprocessed_image is not None:
                    # Prepare output data
                    output_data = {
                        'id': image_id,
                        'original_image': image_data,
                        'preprocessed_image': preprocessed_image,
                        'metadata': {
                            **metadata,
                            'preprocessing_mode': self.mode,
                            'processing_time': time.time()
                        }
                    }
                    
                    # Put preprocessed data into enhance_queue
                    try:
                        self.enhance_queue.put(output_data, timeout=5.0)
                        self.processed_count += 1
                        
                        # Emit completion signal
                        self.preprocessing_completed.emit(image_id, preprocessed_image)
                        
                    except queue.Full:
                        print(f"Enhancement queue full, skipping image {image_id}")
                        self.error_count += 1
                
                # Mark task as done in input_buffer
                self.input_buffer.task_done()
                
                # Emit progress
                self._emit_progress()
                
            except Exception as e:
                print(f"Error processing image {item.get('id', 'unknown')}: {e}")
                self.preprocessing_error.emit(item.get('id', 'unknown'), str(e))
                self.error_count += 1
                
                # Still mark task as done
                try:
                    self.input_buffer.task_done()
                except:
                    pass
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load image from file path."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image using PIL
            image = Image.open(image_path)
            
            # Convert to grayscale if needed (fingerprints are typically grayscale)
            if image.mode != 'L':
                image = image.convert('L')
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def _preprocess_image(self, image_data: Union[Image.Image, np.ndarray], image_id: str) -> Optional[object]:
        """Apply preprocessing transforms to image."""
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image_data, np.ndarray):
                if image_data.dtype != np.uint8:
                    image_data = (image_data * 255).astype(np.uint8)
                
                # Handle different array shapes
                if len(image_data.shape) == 3:
                    if image_data.shape[2] == 3:  # RGB
                        image_data = Image.fromarray(image_data, mode='RGB').convert('L')
                    elif image_data.shape[2] == 1:  # Grayscale with channel dim
                        image_data = Image.fromarray(image_data.squeeze(), mode='L')
                elif len(image_data.shape) == 2:  # Grayscale
                    image_data = Image.fromarray(image_data, mode='L')
                else:
                    raise ValueError(f"Unsupported image shape: {image_data.shape}")
            
            # Ensure PIL Image is in grayscale mode
            if isinstance(image_data, Image.Image) and image_data.mode != 'L':
                image_data = image_data.convert('L')
            
            # Apply preprocessing pipeline
            preprocessed = self.preprocessor(image_data)
            
            return preprocessed
            
        except Exception as e:
            print(f"Error preprocessing image {image_id}: {e}")
            self.preprocessing_error.emit(image_id, f"Preprocessing failed: {str(e)}")
            return None
    
    def _emit_progress(self):
        """Emit progress signal with current statistics."""
        if self.processed_count % 10 == 0:  # Emit every 10 processed images
            progress_percentage = min(100, (self.processed_count * 100) // max(1, self.input_buffer.qsize() + self.processed_count))
            self.preprocessing_progress.emit(progress_percentage)
    
    def _emit_final_stats(self):
        """Emit final processing statistics."""
        if self.start_time:
            total_time = time.time() - self.start_time
            stats = {
                'processed_count': self.processed_count,
                'error_count': self.error_count,
                'total_time': total_time,
                'processing_rate': self.processed_count / max(1, total_time),
                'success_rate': self.processed_count / max(1, self.processed_count + self.error_count),
                'mode': self.mode
            }
            self.processing_stats.emit(stats)
    
    def add_image(self, image_data: Union[str, Image.Image, np.ndarray], 
                  image_id: str = None, metadata: Dict = None):
        """
        Add a single image to the input buffer for processing.
        
        Args:
            image_data: Image data (file path, PIL Image, or numpy array)
            image_id: Unique identifier for the image
            metadata: Additional metadata for the image
        """
        if image_id is None:
            image_id = f"img_{int(time.time() * 1000)}"
        
        if metadata is None:
            metadata = {}
        
        item = {
            'id': image_id,
            'metadata': metadata
        }
        
        if isinstance(image_data, str):
            item['path'] = image_data
        else:
            item['image'] = image_data
        
        try:
            self.input_buffer.put(item, timeout=1.0)
        except queue.Full:
            print(f"Input buffer full, cannot add image {image_id}")
    
    def add_images_batch(self, images_data: list):
        """
        Add multiple images to the input buffer for processing.
        
        Args:
            images_data: List of tuples (image_data, image_id, metadata)
        """
        for image_info in images_data:
            if len(image_info) >= 2:
                image_data, image_id = image_info[:2]
                metadata = image_info[2] if len(image_info) > 2 else {}
                self.add_image(image_data, image_id, metadata)
    
    def stop(self):
        """Stop the preprocessor thread."""
        print("Stopping fingerprint preprocessor thread...")
        self._stop_flag = True
        
        # Resume if paused to allow thread to exit
        if self._pause_flag:
            self.resume()
        
        # Wait for thread to finish
        if not self.wait(5000):  # 5 second timeout
            print("Force terminating preprocessor thread")
            self.terminate()
    
    def pause(self):
        """Pause the preprocessor thread."""
        self._mutex.lock()
        self._pause_flag = True
        self._mutex.unlock()
        print("Fingerprint preprocessor paused")
    
    def resume(self):
        """Resume the preprocessor thread."""
        self._mutex.lock()
        self._pause_flag = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        print("Fingerprint preprocessor resumed")
    
    def is_paused(self) -> bool:
        """Check if the preprocessor is paused."""
        return self._pause_flag
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes for monitoring."""
        return {
            'input_buffer_size': self.input_buffer.qsize(),
            'enhance_queue_size': self.enhance_queue.qsize()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'elapsed_time': elapsed_time,
            'processing_rate': self.processed_count / max(1, elapsed_time),
            'queue_sizes': self.get_queue_sizes(),
            'is_paused': self.is_paused(),
            'is_running': self.isRunning(),
            'mode': self.mode
        }
    
    def set_mode(self, mode: str):
        """
        Change processing mode and reinitialize preprocessor.
        
        Args:
            mode: New processing mode ('train', 'val', 'test')
        """
        if mode != self.mode:
            self.mode = mode
            try:
                self.preprocessor = create_fingerprint_transforms(self.args, mode=mode)
                print(f"Preprocessing mode changed to: {mode}")
            except Exception as e:
                print(f"Error changing preprocessing mode: {e}")


# Example usage and testing
def example_usage():
    """Example of how to use the FingerprintPreprocessor."""
    
    # Mock args object for preprocessing
    class Args:
        def __init__(self):
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
    
    # Create queues
    input_buffer = queue.Queue(maxsize=100)
    enhance_queue = queue.Queue(maxsize=100)
    
    # Create and start preprocessor
    args = Args()
    preprocessor = FingerprintPreprocessor(input_buffer, enhance_queue, args, mode='train')
    
    # Connect signals (in real application)
    preprocessor.preprocessing_completed.connect(lambda img_id, result: 
                                               print(f"Preprocessed image {img_id}: {type(result)}"))
    preprocessor.preprocessing_error.connect(lambda img_id, error: 
                                           print(f"Error preprocessing {img_id}: {error}"))
    
    # Start processing
    preprocessor.start()
    
    # Add test images
    dummy_image = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    preprocessor.add_image(dummy_image, "test_001", {"source": "synthetic"})
    
    # Add batch of images
    batch_images = [
        (np.random.randint(0, 255, (280, 280), dtype=np.uint8), "test_002", {"source": "batch"}),
        (np.random.randint(0, 255, (320, 320), dtype=np.uint8), "test_003", {"source": "batch"})
    ]
    preprocessor.add_images_batch(batch_images)
    
    # Let it process for a bit
    time.sleep(3)
    
    # Check statistics
    stats = preprocessor.get_statistics()
    print(f"Processing stats: {stats}")
    
    # Stop preprocessor
    preprocessor.stop()
    
    print("Example completed")


if __name__ == "__main__":
    # Initialize Qt application for testing
    app = QApplication(sys.argv)
    example_usage()
    sys.exit()
