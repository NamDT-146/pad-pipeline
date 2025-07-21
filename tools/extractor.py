import sys
import queue
import threading
import time
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
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

# Import the SiameseNetwork model
try:
    from model.siamesenetwork import SiameseNetwork
except ImportError as e:
    print(f"Error importing model: {e}")
    print("Make sure the model module is in your Python path")


class FingerprintExtractor(QThread):
    """
    QThread class for fingerprint feature extraction.
    
    Takes enhanced fingerprint images from the enhancer_queue,
    extracts feature vectors using the SiameseNetwork model,
    and puts results into matcher_queue for further processing.
    """
    
    # PyQt signals for communication
    extraction_progress = pyqtSignal(int)  # Progress percentage
    extraction_completed = pyqtSignal(str, torch.Tensor)  # Image ID and feature vector
    extraction_error = pyqtSignal(str, str)  # Image ID and error message
    processing_stats = pyqtSignal(dict)  # Processing statistics
    
    def __init__(self, 
                 enhancer_queue: queue.Queue,
                 matcher_queue: queue.Queue,
                 model_path: str,
                 device: Optional[torch.device] = None,
                 batch_size: int = 1):
        """
        Initialize the fingerprint extractor thread.
        
        Args:
            enhancer_queue: Input queue containing enhanced images
            matcher_queue: Output queue for feature vectors
            model_path: Path to the saved model weights
            device: Torch device (CPU/CUDA)
            batch_size: Number of images to process in batch
        """
        super().__init__()
        
        # Queue management
        self.enhancer_queue = enhancer_queue
        self.matcher_queue = matcher_queue
        
        # Processing parameters
        self.model_path = model_path
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        
        # Load model
        self.model = self._load_model()
        print(f"Feature extraction model loaded on {self.device}")
    
    def _load_model(self) -> Optional[SiameseNetwork]:
        """Load the SiameseNetwork model from the saved weights."""
        try:
            model = SiameseNetwork().to(self.device)
            
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()  # Set to evaluation mode
                return model
            else:
                print(f"Model file not found: {self.model_path}")
                return None
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def run(self):
        """Main thread execution loop."""
        print("Fingerprint extractor thread started")
        self.start_time = time.time()
        
        if self.model is None:
            self.extraction_error.emit("INIT_ERROR", "Model could not be loaded")
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
                
                # Get batch of images from enhancer_queue
                batch_data = self._get_batch()
                
                if not batch_data:
                    # No data available, sleep briefly
                    self.msleep(100)
                    continue
                
                # Process batch
                self._process_batch(batch_data)
                
            except Exception as e:
                print(f"Error in extractor thread: {e}")
                self.extraction_error.emit("THREAD_ERROR", str(e))
                self.msleep(1000)  # Wait before retrying
        
        self._emit_final_stats()
        print("Fingerprint extractor thread stopped")
    
    def _get_batch(self) -> list:
        """Get a batch of enhanced images from the enhancer_queue."""
        batch = []
        
        try:
            # Try to get up to batch_size items from queue
            for _ in range(self.batch_size):
                try:
                    # Non-blocking get with short timeout
                    item = self.enhancer_queue.get(timeout=0.1)
                    batch.append(item)
                except queue.Empty:
                    break
            
        except Exception as e:
            print(f"Error getting batch from enhancer_queue: {e}")
        
        return batch
    
    def _process_batch(self, batch_data: list):
        """Process a batch of enhanced images."""
        for item in batch_data:
            if self._stop_flag:
                break
            
            try:
                # Extract data from queue item
                image_id = item.get('id', 'unknown')
                enhancement_results = item.get('enhancement_results', {})
                metadata = item.get('metadata', {})
                
                # Get the enhanced image (prefer enhanced, fallback to binary)
                if 'enhanced' in enhancement_results:
                    fingerprint_image = enhancement_results['enhanced']
                elif 'binary' in enhancement_results:
                    fingerprint_image = enhancement_results['binary']
                else:
                    # Fallback to preprocessed image
                    fingerprint_image = item.get('preprocessed_image')
                    if isinstance(fingerprint_image, Image.Image):
                        fingerprint_image = np.array(fingerprint_image)
                
                if fingerprint_image is None:
                    self.extraction_error.emit(image_id, "No valid image found for feature extraction")
                    continue
                
                # Extract features
                feature_vector = self._extract_features(fingerprint_image, image_id)
                
                if feature_vector is not None:
                    # Prepare output data
                    output_data = {
                        'id': image_id,
                        'feature_vector': feature_vector,
                        'metadata': metadata,
                        'processing_time': time.time()
                    }
                    
                    # Put feature vector into matcher_queue
                    try:
                        self.matcher_queue.put(output_data, timeout=5.0)
                        self.processed_count += 1
                        
                        # Emit completion signal
                        self.extraction_completed.emit(image_id, feature_vector)
                        
                    except queue.Full:
                        print(f"Matcher queue full, skipping image {image_id}")
                        self.error_count += 1
                
                # Mark task as done in enhancer_queue
                self.enhancer_queue.task_done()
                
                # Emit progress
                self._emit_progress()
                
            except Exception as e:
                print(f"Error processing image {item.get('id', 'unknown')}: {e}")
                self.extraction_error.emit(item.get('id', 'unknown'), str(e))
                self.error_count += 1
                
                # Still mark task as done
                try:
                    self.enhancer_queue.task_done()
                except:
                    pass
    
    def _extract_features(self, image, image_id: str) -> Optional[torch.Tensor]:
        """Extract feature vector from an enhanced fingerprint image."""
        try:
            with torch.no_grad():
                # Convert to numpy array if PIL Image
                if isinstance(image, Image.Image):
                    image = np.array(image)
                
                # Normalize if needed
                if image.dtype != np.float32:
                    image = image.astype(np.float32) / 255.0
                
                # Create tensor
                if len(image.shape) == 2:  # Add channel dimension if grayscale
                    image = image[np.newaxis, ...]
                
                # Add batch dimension if not present
                if len(image.shape) == 3:
                    image = image[np.newaxis, ...]
                
                # Convert to PyTorch tensor and move to device
                image_tensor = torch.tensor(image, device=self.device)
                
                # Extract features
                feature_vector = self.model.extract_features(image_tensor)
                
                return feature_vector
                
        except Exception as e:
            print(f"Error extracting features from image {image_id}: {e}")
            self.extraction_error.emit(image_id, f"Feature extraction failed: {str(e)}")
            return None
    
    def _emit_progress(self):
        """Emit progress signal with current statistics."""
        if self.processed_count % 10 == 0:  # Emit every 10 processed images
            progress_percentage = min(100, (self.processed_count * 100) // max(1, self.enhancer_queue.qsize() + self.processed_count))
            self.extraction_progress.emit(progress_percentage)
    
    def _emit_final_stats(self):
        """Emit final processing statistics."""
        if self.start_time:
            total_time = time.time() - self.start_time
            stats = {
                'processed_count': self.processed_count,
                'error_count': self.error_count,
                'total_time': total_time,
                'processing_rate': self.processed_count / max(1, total_time),
                'success_rate': self.processed_count / max(1, self.processed_count + self.error_count)
            }
            self.processing_stats.emit(stats)
    
    def stop(self):
        """Stop the extractor thread."""
        print("Stopping fingerprint extractor thread...")
        self._stop_flag = True
        
        # Resume if paused to allow thread to exit
        if self._pause_flag:
            self.resume()
        
        # Wait for thread to finish
        if not self.wait(5000):  # 5 second timeout
            print("Force terminating extractor thread")
            self.terminate()
    
    def pause(self):
        """Pause the extractor thread."""
        self._mutex.lock()
        self._pause_flag = True
        self._mutex.unlock()
        print("Fingerprint extractor paused")
    
    def resume(self):
        """Resume the extractor thread."""
        self._mutex.lock()
        self._pause_flag = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        print("Fingerprint extractor resumed")
    
    def is_paused(self) -> bool:
        """Check if the extractor is paused."""
        return self._pause_flag
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes for monitoring."""
        return {
            'enhancer_queue_size': self.enhancer_queue.qsize(),
            'matcher_queue_size': self.matcher_queue.qsize()
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
            'is_running': self.isRunning()
        }


# Example usage and testing
def example_usage():
    """Example of how to use the FingerprintExtractor."""
    
    # Create queues
    enhancer_queue = queue.Queue(maxsize=100)
    matcher_queue = queue.Queue(maxsize=100)
    
    # Create mock enhanced data
    dummy_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    enhanced_data = {
        'id': 'test_001',
        'original_image': dummy_image,
        'preprocessed_image': dummy_image,
        'enhancement_results': {'enhanced': dummy_image},
        'metadata': {'source': 'test'}
    }
    
    # Put test data in queue
    enhancer_queue.put(enhanced_data)
    
    # Create and start extractor
    model_path = 'feature_model.pth'  # Path to saved model
    extractor = FingerprintExtractor(enhancer_queue, matcher_queue, model_path)
    
    # Connect signals (in real application)
    extractor.extraction_completed.connect(lambda img_id, features: 
                                         print(f"Extracted features from {img_id}: shape={features.shape}"))
    extractor.extraction_error.connect(lambda img_id, error: 
                                     print(f"Error extracting {img_id}: {error}"))
    
    # Start processing
    extractor.start()
    
    # Let it process for a bit
    time.sleep(2)
    
    # Check results
    if not matcher_queue.empty():
        result = matcher_queue.get()
        print(f"Feature vector shape: {result['feature_vector'].shape}")
    
    # Stop extractor
    extractor.stop()
    
    print("Example completed")


if __name__ == "__main__":
    # Initialize Qt application for testing
    app = QApplication(sys.argv)
    example_usage()
    sys.exit()