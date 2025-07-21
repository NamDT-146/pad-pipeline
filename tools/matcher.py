import sys
import os
import queue
import time
import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn.functional as F

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


class FingerprintMatcher(QThread):
    """
    QThread class for fingerprint matching.
    
    Takes feature vectors from the extractor_queue, 
    matches them against a database of known fingerprints,
    and determines if the fingerprint belongs to a known person or is unrecognized.
    """
    
    # PyQt signals for communication
    matching_progress = pyqtSignal(int)  # Progress percentage
    matching_result = pyqtSignal(str, str, float)  # Image ID, matched ID or "unrecognized", and similarity score
    matching_error = pyqtSignal(str, str)  # Image ID and error message
    processing_stats = pyqtSignal(dict)  # Processing statistics
    
    def __init__(self, 
                 extractor_queue: queue.Queue,
                 result_queue: Optional[queue.Queue] = None,
                 model_path: str = None,
                 database_path: str = None,
                 threshold: float = 0.75,
                 device: Optional[torch.device] = None,
                 batch_size: int = 1):
        """
        Initialize the fingerprint matcher thread.
        
        Args:
            extractor_queue: Input queue containing feature vectors
            result_queue: Output queue for matching results (optional)
            model_path: Path to the saved model weights (for full model)
            database_path: Path to the fingerprint database
            threshold: Similarity threshold for positive identification
            device: Torch device (CPU/CUDA)
            batch_size: Number of items to process in batch
        """
        super().__init__()
        
        # Queue management
        self.extractor_queue = extractor_queue
        self.result_queue = result_queue
        
        # Processing parameters
        self.model_path = model_path
        self.database_path = database_path
        self.threshold = threshold
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Thread control
        self._stop_flag = False
        self._pause_flag = False
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()
        
        # Statistics tracking
        self.processed_count = 0
        self.matched_count = 0
        self.unrecognized_count = 0
        self.error_count = 0
        self.start_time = None
        
        # Initialize fingerprint database
        self.database = {}
        self.database_embeddings = None
        self.database_ids = []
        
        # Load model and database
        self.model = self._load_model() if self.model_path else None
        self._load_database()
        
        print(f"Fingerprint matcher initialized with {len(self.database)} registered fingerprints")
        print(f"Matcher running on {self.device} with threshold {self.threshold}")
    
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
    
    def _load_database(self):
        """Load the fingerprint database."""
        try:
            if self.database_path and os.path.exists(self.database_path):
                # Check if it's a directory or a file
                if os.path.isdir(self.database_path):
                    # Load all database files in the directory
                    self._load_database_from_directory()
                else:
                    # Load from a single file
                    extension = os.path.splitext(self.database_path)[1].lower()
                    if extension == '.pt' or extension == '.pth':
                        self._load_database_from_torch()
                    elif extension == '.json':
                        self._load_database_from_json()
                    elif extension == '.npz':
                        self._load_database_from_numpy()
                    else:
                        print(f"Unsupported database file format: {extension}")
                
                # Prepare tensor for efficient batch matching
                self._prepare_database_tensor()
            else:
                print("No database path provided or database not found. Starting with empty database.")
                
        except Exception as e:
            print(f"Error loading database: {e}")
    
    def _load_database_from_directory(self):
        """Load database from a directory of fingerprint embedding files."""
        file_count = 0
        for filename in os.listdir(self.database_path):
            filepath = os.path.join(self.database_path, filename)
            
            # Skip directories
            if os.path.isdir(filepath):
                continue
                
            # Process each file based on extension
            extension = os.path.splitext(filename)[1].lower()
            
            try:
                if extension == '.pt' or extension == '.pth':
                    # Parse ID from filename
                    identity = os.path.splitext(filename)[0]
                    embedding = torch.load(filepath, map_location=self.device)
                    self.database[identity] = embedding
                    file_count += 1
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        print(f"Loaded {file_count} fingerprint embeddings from directory")
    
    def _load_database_from_torch(self):
        """Load database from a PyTorch file."""
        try:
            db = torch.load(self.database_path, map_location=self.device)
            if isinstance(db, dict):
                self.database = db
                print(f"Loaded {len(self.database)} fingerprint embeddings from PyTorch file")
            else:
                print(f"Invalid database format in {self.database_path}")
        except Exception as e:
            print(f"Error loading PyTorch database: {e}")
    
    def _load_database_from_json(self):
        """Load database from a JSON file."""
        try:
            with open(self.database_path, 'r') as f:
                db_dict = json.load(f)
            
            # Convert to tensors
            for identity, embedding_list in db_dict.items():
                self.database[identity] = torch.tensor(embedding_list, 
                                                      device=self.device, 
                                                      dtype=torch.float32)
            
            print(f"Loaded {len(self.database)} fingerprint embeddings from JSON file")
        except Exception as e:
            print(f"Error loading JSON database: {e}")
    
    def _load_database_from_numpy(self):
        """Load database from a NumPy file."""
        try:
            db = np.load(self.database_path, allow_pickle=True)
            
            # Convert to dict of tensors
            if 'embeddings' in db and 'identities' in db:
                embeddings = db['embeddings']
                identities = db['identities']
                
                for i, identity in enumerate(identities):
                    self.database[str(identity)] = torch.tensor(embeddings[i], 
                                                              device=self.device,
                                                              dtype=torch.float32)
            
            print(f"Loaded {len(self.database)} fingerprint embeddings from NumPy file")
        except Exception as e:
            print(f"Error loading NumPy database: {e}")
    
    def _prepare_database_tensor(self):
        """Prepare a tensor containing all database embeddings for efficient matching."""
        if not self.database:
            self.database_embeddings = None
            self.database_ids = []
            return
            
        # Create a list of all embeddings
        embeddings = []
        ids = []
        
        for identity, embedding in self.database.items():
            if isinstance(embedding, list):
                # Handle multiple embeddings per identity
                for emb in embedding:
                    embeddings.append(emb)
                    ids.append(identity)
            else:
                # Single embedding per identity
                embeddings.append(embedding)
                ids.append(identity)
        
        # Stack all embeddings into a single tensor
        if embeddings:
            self.database_embeddings = torch.stack(embeddings) if isinstance(embeddings[0], torch.Tensor) else torch.tensor(embeddings, device=self.device)
            self.database_ids = ids
            print(f"Prepared database tensor with shape {self.database_embeddings.shape}")
        else:
            self.database_embeddings = None
            self.database_ids = []
    
    def add_fingerprint(self, identity: str, embedding: torch.Tensor, save_to_disk: bool = True):
        """
        Add a new fingerprint to the database.
        
        Args:
            identity: ID or name of the person
            embedding: Feature vector of the fingerprint
            save_to_disk: Whether to save the updated database to disk
        """
        # Add to database
        self.database[identity] = embedding
        
        # Update the database tensor
        self._prepare_database_tensor()
        
        # Save to disk if requested
        if save_to_disk and self.database_path:
            self._save_database()
        
        print(f"Added fingerprint for {identity} to database")
    
    def remove_fingerprint(self, identity: str, save_to_disk: bool = True):
        """
        Remove a fingerprint from the database.
        
        Args:
            identity: ID or name of the person to remove
            save_to_disk: Whether to save the updated database to disk
        """
        if identity in self.database:
            del self.database[identity]
            
            # Update the database tensor
            self._prepare_database_tensor()
            
            # Save to disk if requested
            if save_to_disk and self.database_path:
                self._save_database()
            
            print(f"Removed fingerprint for {identity} from database")
        else:
            print(f"Identity {identity} not found in database")
    
    def _save_database(self):
        """Save the fingerprint database to disk."""
        try:
            if not self.database_path:
                print("No database path specified, cannot save database")
                return
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.database_path)), exist_ok=True)
            
            # Save based on file extension
            extension = os.path.splitext(self.database_path)[1].lower()
            
            if extension == '.pt' or extension == '.pth':
                torch.save(self.database, self.database_path)
            elif extension == '.json':
                # Convert tensors to lists
                db_dict = {}
                for identity, embedding in self.database.items():
                    if isinstance(embedding, torch.Tensor):
                        db_dict[identity] = embedding.cpu().tolist()
                    else:
                        db_dict[identity] = embedding
                
                with open(self.database_path, 'w') as f:
                    json.dump(db_dict, f)
            elif extension == '.npz':
                # Convert to numpy arrays
                identities = list(self.database.keys())
                embeddings = [self.database[identity].cpu().numpy() if isinstance(self.database[identity], torch.Tensor)
                             else self.database[identity] for identity in identities]
                
                np.savez(self.database_path, embeddings=embeddings, identities=identities)
            else:
                print(f"Unsupported database file format: {extension}")
                return
                
            print(f"Saved database with {len(self.database)} fingerprints to {self.database_path}")
            
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def run(self):
        """Main thread execution loop."""
        print("Fingerprint matcher thread started")
        self.start_time = time.time()
        
        while not self._stop_flag:
            try:
                # Handle pause state
                self._mutex.lock()
                while self._pause_flag and not self._stop_flag:
                    self._pause_condition.wait(self._mutex)
                self._mutex.unlock()
                
                if self._stop_flag:
                    break
                
                # Get batch of feature vectors from extractor_queue
                batch_data = self._get_batch()
                
                if not batch_data:
                    # No data available, sleep briefly
                    self.msleep(100)
                    continue
                
                # Process batch
                self._process_batch(batch_data)
                
            except Exception as e:
                print(f"Error in matcher thread: {e}")
                self.matching_error.emit("THREAD_ERROR", str(e))
                self.msleep(1000)  # Wait before retrying
        
        self._emit_final_stats()
        print("Fingerprint matcher thread stopped")
    
    def _get_batch(self) -> list:
        """Get a batch of feature vectors from the extractor_queue."""
        batch = []
        
        try:
            # Try to get up to batch_size items from queue
            for _ in range(self.batch_size):
                try:
                    # Non-blocking get with short timeout
                    item = self.extractor_queue.get(timeout=0.1)
                    batch.append(item)
                except queue.Empty:
                    break
            
        except Exception as e:
            print(f"Error getting batch from extractor_queue: {e}")
        
        return batch
    
    def _process_batch(self, batch_data: list):
        """Process a batch of feature vectors."""
        for item in batch_data:
            if self._stop_flag:
                break
            
            try:
                # Extract data from queue item
                image_id = item.get('id', 'unknown')
                feature_vector = item.get('feature_vector')
                metadata = item.get('metadata', {})
                
                if feature_vector is None:
                    self.matching_error.emit(image_id, "No feature vector found for matching")
                    continue
                
                # Match against database
                matched_id, similarity_score = self._match_fingerprint(feature_vector)
                
                # Determine if the fingerprint is recognized
                if similarity_score >= self.threshold:
                    result = matched_id
                    self.matched_count += 1
                else:
                    result = "unrecognized"
                    self.unrecognized_count += 1
                
                # Prepare output data
                output_data = {
                    'id': image_id,
                    'result': result,
                    'similarity_score': similarity_score,
                    'metadata': metadata,
                    'processing_time': time.time()
                }
                
                # Put result into result_queue if provided
                if self.result_queue is not None:
                    try:
                        self.result_queue.put(output_data, timeout=5.0)
                    except queue.Full:
                        print(f"Result queue full, skipping result for {image_id}")
                
                # Emit result signal
                self.matching_result.emit(image_id, result, float(similarity_score))
                
                # Update statistics
                self.processed_count += 1
                
                # Mark task as done in extractor_queue
                self.extractor_queue.task_done()
                
                # Emit progress
                self._emit_progress()
                
            except Exception as e:
                print(f"Error processing feature vector {item.get('id', 'unknown')}: {e}")
                self.matching_error.emit(item.get('id', 'unknown'), str(e))
                self.error_count += 1
                
                # Still mark task as done
                try:
                    self.extractor_queue.task_done()
                except:
                    pass
    
    def _match_fingerprint(self, feature_vector: torch.Tensor) -> Tuple[str, float]:
        """
        Match a feature vector against the database.
        
        Args:
            feature_vector: The feature vector to match
            
        Returns:
            Tuple of (matched_identity, similarity_score)
        """
        if self.database_embeddings is None or len(self.database_embeddings) == 0:
            return "no_database", 0.0
        
        # Ensure feature vector is on the correct device
        if feature_vector.device != self.device:
            feature_vector = feature_vector.to(self.device)
        
        # Ensure feature vector is normalized
        feature_vector = F.normalize(feature_vector, p=2, dim=1)
        
        # Calculate similarity with all database embeddings
        with torch.no_grad():
            # Calculate element-wise squared difference
            diff = self.database_embeddings - feature_vector
            diff_squared = diff * diff
            
            if self.model and hasattr(self.model, 'similarity_net'):
                # Use model's similarity network if available
                similarity_scores = self.model.similarity_net(diff_squared).squeeze(-1)
            else:
                # Fallback to euclidean distance
                distances = torch.sum(diff_squared, dim=1)
                similarity_scores = 1.0 - torch.sqrt(distances) / 2.0  # Normalize to [0,1]
        
        # Get the best match
        best_score, best_idx = torch.max(similarity_scores, dim=0)
        best_id = self.database_ids[best_idx.item()]
        
        return best_id, best_score.item()
    
    def _emit_progress(self):
        """Emit progress signal with current statistics."""
        if self.processed_count % 10 == 0:  # Emit every 10 processed items
            progress_percentage = min(100, (self.processed_count * 100) // max(1, self.extractor_queue.qsize() + self.processed_count))
            self.matching_progress.emit(progress_percentage)
    
    def _emit_final_stats(self):
        """Emit final processing statistics."""
        if self.start_time:
            total_time = time.time() - self.start_time
            stats = {
                'processed_count': self.processed_count,
                'matched_count': self.matched_count,
                'unrecognized_count': self.unrecognized_count,
                'error_count': self.error_count,
                'total_time': total_time,
                'processing_rate': self.processed_count / max(1, total_time),
                'success_rate': (self.matched_count + self.unrecognized_count) / max(1, self.processed_count)
            }
            self.processing_stats.emit(stats)
    
    def stop(self):
        """Stop the matcher thread."""
        print("Stopping fingerprint matcher thread...")
        self._stop_flag = True
        
        # Resume if paused to allow thread to exit
        if self._pause_flag:
            self.resume()
        
        # Wait for thread to finish
        if not self.wait(5000):  # 5 second timeout
            print("Force terminating matcher thread")
            self.terminate()
    
    def pause(self):
        """Pause the matcher thread."""
        self._mutex.lock()
        self._pause_flag = True
        self._mutex.unlock()
        print("Fingerprint matcher paused")
    
    def resume(self):
        """Resume the matcher thread."""
        self._mutex.lock()
        self._pause_flag = False
        self._pause_condition.wakeAll()
        self._mutex.unlock()
        print("Fingerprint matcher resumed")
    
    def is_paused(self) -> bool:
        """Check if the matcher is paused."""
        return self._pause_flag
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes for monitoring."""
        return {
            'extractor_queue_size': self.extractor_queue.qsize(),
            'result_queue_size': self.result_queue.qsize() if self.result_queue else 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        return {
            'processed_count': self.processed_count,
            'matched_count': self.matched_count,
            'unrecognized_count': self.unrecognized_count,
            'error_count': self.error_count,
            'database_size': len(self.database),
            'elapsed_time': elapsed_time,
            'processing_rate': self.processed_count / max(1, elapsed_time),
            'queue_sizes': self.get_queue_sizes(),
            'is_paused': self.is_paused(),
            'is_running': self.isRunning()
        }


# Example usage and testing
def example_usage():
    """Example of how to use the FingerprintMatcher."""
    
    # Create queues
    extractor_queue = queue.Queue(maxsize=100)
    result_queue = queue.Queue(maxsize=100)
    
    # Create mock data
    feature_dim = 512
    mock_feature = torch.randn(1, feature_dim)  # Random feature vector
    
    # Create a small mock database
    db_path = "mock_db.pt"
    mock_db = {
        "person1": torch.randn(1, feature_dim),
        "person2": torch.randn(1, feature_dim),
        "person3": torch.randn(1, feature_dim)
    }
    torch.save(mock_db, db_path)
    
    # Put test data in queue
    extractor_queue.put({
        'id': 'test_001',
        'feature_vector': mock_feature,
        'metadata': {'source': 'test'}
    })
    
    # Create and start matcher
    matcher = FingerprintMatcher(
        extractor_queue=extractor_queue,
        result_queue=result_queue,
        database_path=db_path,
        threshold=0.75
    )
    
    # Connect signals (in real application)
    matcher.matching_result.connect(lambda img_id, result, score: 
                                  print(f"Match result for {img_id}: {result} (score: {score:.4f})"))
    matcher.matching_error.connect(lambda img_id, error: 
                                 print(f"Error matching {img_id}: {error}"))
    
    # Start processing
    matcher.start()
    
    # Let it process for a bit
    time.sleep(2)
    
    # Check results
    if not result_queue.empty():
        result = result_queue.get()
        print(f"Match result: {result['result']} with score {result['similarity_score']:.4f}")
    
    # Stop matcher
    matcher.stop()
    
    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print("Example completed")


if __name__ == "__main__":
    # Initialize Qt application for testing
    app = QApplication(sys.argv)
    example_usage()
    sys.exit()