import sys
import queue
import threading
import time
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image

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
    from dataset.preprocess.enhancing import create_fingerprint_enhancement
    from dataset.preprocess.preprocessing import create_fingerprint_transforms
except ImportError as e:
    print(f"Error importing dataset modules: {e}")
    print("Make sure the dataset module is in your Python path")


class FingerprintEnhancer(QThread):
    """
    QThread class for fingerprint enhancement processing.
    
    Takes preprocessed images from enhance_queue, applies enhancement algorithms,
    and puts results into extractor_queue for further processing.
    """
    
    # PyQt signals for communication
    enhancement_progress = pyqtSignal(int)  # Progress percentage
    enhancement_completed = pyqtSignal(str, dict)  # Image ID and results
    enhancement_error = pyqtSignal(str, str)  # Image ID and error message
    processing_stats = pyqtSignal(dict)  # Processing statistics
    
    def __init__(self, 
                 enhance_queue: queue.Queue,
                 extractor_queue: queue.Queue,
                 args: Any,
                 max_workers: int = 1,
                 batch_size: int = 1):
        """
        Initialize the fingerprint enhancer thread.
        
        Args:
            enhance_queue: Input queue containing preprocessed images
            extractor_queue: Output queue for enhanced images
            args: Arguments object containing enhancement parameters
            max_workers: Maximum number of worker threads
            batch_size: Number of images to process in batch
        """
        super().__init__()
        
        # Queue management
        self.enhance_queue = enhance_queue
        self.extractor_queue = extractor_queue
        
        # Processing parameters
        self.args = args
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
        
        # Create enhancement pipeline
        try:
            self.enhancer = create_fingerprint_enhancement(args)
            print("Fingerprint enhancement pipeline initialized successfully")
        except Exception as e:
            print(f"Error initializing enhancement pipeline: {e}")
            self.enhancer = None
    
    def run(self):
        """Main thread execution loop."""
        print("Fingerprint enhancer thread started")
        self.start_time = time.time()
        
        if self.enhancer is None:
            self.enhancement_error.emit("INIT_ERROR", "Enhancement pipeline not initialized")
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
                
                # Get batch of images from enhance_queue
                batch_data = self._get_batch()
                
                if not batch_data:
                    # No data available, sleep briefly
                    self.msleep(100)
                    continue
                
                # Process batch
                self._process_batch(batch_data)
                
            except Exception as e:
                print(f"Error in enhancer thread: {e}")
                self.enhancement_error.emit("THREAD_ERROR", str(e))
                self.msleep(1000)  # Wait before retrying
        
        self._emit_final_stats()
        print("Fingerprint enhancer thread stopped")
    
    def _get_batch(self) -> list:
        """Get a batch of images from the enhance_queue."""
        batch = []
        
        try:
            # Try to get up to batch_size items from queue
            for _ in range(self.batch_size):
                try:
                    # Non-blocking get with short timeout
                    item = self.enhance_queue.get(timeout=0.1)
                    batch.append(item)
                except queue.Empty:
                    break
            
        except Exception as e:
            print(f"Error getting batch from enhance_queue: {e}")
        
        return batch
    
    def _process_batch(self, batch_data: list):
        """Process a batch of preprocessed images."""
        for item in batch_data:
            if self._stop_flag:
                break
            
            try:
                # Extract data from queue item
                image_id = item.get('id', 'unknown')
                preprocessed_image = item.get('preprocessed_image')
                metadata = item.get('metadata', {})
                
                if preprocessed_image is None:
                    self.enhancement_error.emit(image_id, "No preprocessed image found")
                    continue
                
                # Apply enhancement pipeline
                enhancement_results = self._enhance_image(preprocessed_image, image_id)
                
                if enhancement_results is not None:
                    # Prepare output data
                    output_data = {
                        'id': image_id,
                        'original_image': item.get('original_image'),
                        'preprocessed_image': preprocessed_image,
                        'enhancement_results': enhancement_results,
                        'metadata': metadata,
                        'processing_time': time.time()
                    }
                    
                    # Put enhanced data into extractor_queue
                    try:
                        self.extractor_queue.put(output_data, timeout=5.0)
                        self.processed_count += 1
                        
                        # Emit completion signal
                        self.enhancement_completed.emit(image_id, enhancement_results)
                        
                    except queue.Full:
                        print(f"Extractor queue full, skipping image {image_id}")
                        self.error_count += 1
                
                # Mark task as done in enhance_queue
                self.enhance_queue.task_done()
                
                # Emit progress
                self._emit_progress()
                
            except Exception as e:
                print(f"Error processing image {item.get('id', 'unknown')}: {e}")
                self.enhancement_error.emit(item.get('id', 'unknown'), str(e))
                self.error_count += 1
                
                # Still mark task as done
                try:
                    self.enhance_queue.task_done()
                except:
                    pass
    
    def _enhance_image(self, preprocessed_image, image_id: str) -> Optional[Dict]:
        """Apply enhancement algorithms to preprocessed image."""
        try:
            # Convert to PIL Image if needed
            if isinstance(preprocessed_image, np.ndarray):
                if preprocessed_image.dtype != np.uint8:
                    preprocessed_image = (preprocessed_image * 255).astype(np.uint8)
                preprocessed_image = Image.fromarray(preprocessed_image, mode='L')
            
            # Apply enhancement pipeline
            enhancement_results = self.enhancer(preprocessed_image)
            
            return enhancement_results
            
        except Exception as e:
            print(f"Error enhancing image {image_id}: {e}")
            self.enhancement_error.emit(image_id, f"Enhancement failed: {str(e)}")
            return None
    
    def _emit_progress(self):
        """Emit progress signal with current statistics."""
        if self.processed_count % 10 == 0:  # Emit every 10 processed images
            progress_percentage = min(100, (self.processed_count * 100) // max(1, self.enhance_queue.qsize() + self.processed_count))
            self.enhancement_progress.emit(progress_percentage)
    
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
        """Stop the enhancer thread."""
        print("Stopping fingerprint enhancer thread...")
        self._stop_flag = True
        
        # Resume if paused to allow thread to exit
        if self._pause_flag:
            self.resume()
        
        # Wait for thread to finish
        if not self.wait(5000):  # 5 second timeout
            print("Force terminating enhancer thread")
            self.terminate()
    
    def pause(self):
        """Pause the enhancer thread."""
        self._mutex.lock()
        self._pause_flag = True
        self._mutex.unlock()
        print("Fingerprint enhancer paused")
    
    def resume(self):
        """Resume the enhancer thread."""
        self._mutex.lock()
        self._pause_flag = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        print("Fingerprint enhancer resumed")
    
    def is_paused(self) -> bool:
        """Check if the enhancer is paused."""
        return self._pause_flag
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes for monitoring."""
        return {
            'enhance_queue_size': self.enhance_queue.qsize(),
            'extractor_queue_size': self.extractor_queue.qsize()
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
    """Example of how to use the FingerprintEnhancer."""
    
    # Mock args object for enhancement
    class Args:
        def __init__(self):
            self.apply_orientation = False
            self.apply_ogorman = True
            self.apply_binarization = True
            self.apply_thinning = True
            self.orientation_block_size = 16
            self.ogorman_filter_size = 7
            self.binarization_window_size = 9
            self.thinning_iterations = 1
    
    # Create queues
    enhance_queue = queue.Queue(maxsize=100)
    extractor_queue = queue.Queue(maxsize=100)
    
    # Create mock preprocessed data
    dummy_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    preprocessed_data = {
        'id': 'test_001',
        'original_image': dummy_image,
        'preprocessed_image': Image.fromarray(dummy_image, mode='L'),
        'metadata': {'source': 'test'}
    }
    
    # Put test data in queue
    enhance_queue.put(preprocessed_data)
    
    # Create and start enhancer
    args = Args()
    enhancer = FingerprintEnhancer(enhance_queue, extractor_queue, args)
    
    # Connect signals (in real application)
    enhancer.enhancement_completed.connect(lambda img_id, results: 
                                         print(f"Enhanced image {img_id}: {list(results.keys())}"))
    enhancer.enhancement_error.connect(lambda img_id, error: 
                                     print(f"Error enhancing {img_id}: {error}"))
    
    # Start processing
    enhancer.start()
    
    # Let it process for a bit
    time.sleep(2)
    
    # Stop enhancer
    enhancer.stop()
    
    print("Example completed")


if __name__ == "__main__":
    # Initialize Qt application for testing
    app = QApplication(sys.argv)
    example_usage()
    sys.exit()