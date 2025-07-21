import os
import sys
import time
import queue
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from PyQt5.QtCore import QCoreApplication, QTimer, QObject, pyqtSlot
from PyQt5.QtWidgets import QApplication

from tools.preprocessor import FingerprintPreprocessor
from tools.enhancer import FingerprintEnhancer
from tools.extractor import FingerprintExtractor
from tools.matcher import FingerprintMatcher
from dataset.siamesepair import create_default_args


class FingerprintPipeline(QObject):
    """Main class for fingerprint processing pipeline."""
    
    def __init__(self, args):
        super().__init__()
        
        # Parse arguments
        self.args = args
        self.reference_path = Path(args.reference_path)
        self.test_path = Path(args.test_path)
        self.model_path = args.model_path
        self.database_path = args.database_path
        self.threshold = args.threshold
        self.device = torch.device(args.device)
        
        # Create queues
        self.input_buffer = queue.Queue(maxsize=100)
        self.enhance_queue = queue.Queue(maxsize=100)
        self.extractor_queue = queue.Queue(maxsize=100)
        self.matcher_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        
        # Create processing arguments
        self.process_args = create_default_args(mode='test')
        
        # Initialize threads
        self.preprocessor = FingerprintPreprocessor(
            self.input_buffer, 
            self.enhance_queue, 
            self.process_args, 
            mode='test'
        )
        
        self.enhancer = FingerprintEnhancer(
            self.enhance_queue, 
            self.extractor_queue, 
            self.process_args
        )
        
        self.extractor = FingerprintExtractor(
            self.extractor_queue, 
            self.matcher_queue, 
            self.model_path, 
            device=self.device
        )
        
        self.matcher = FingerprintMatcher(
            self.matcher_queue, 
            self.result_queue, 
            model_path=self.model_path,
            database_path=self.database_path,
            threshold=self.threshold,
            device=self.device
        )
        
        # Connect signals
        self.preprocessor.preprocessing_error.connect(self.handle_preprocessing_error)
        self.enhancer.enhancement_error.connect(self.handle_enhancement_error)
        self.extractor.extraction_error.connect(self.handle_extraction_error)
        self.matcher.matching_result.connect(self.handle_match_result)
        self.matcher.matching_error.connect(self.handle_matching_error)
        
        # Statistics
        self.total_processed = 0
        self.total_matched = 0
        self.total_unmatched = 0
        self.start_time = None
        self.results = {}
        
        # Processing flags
        self.registration_complete = False
        self.processing_complete = False
        
        print("Fingerprint pipeline initialized")
    
    def start(self):
        """Start the fingerprint processing pipeline."""
        print("Starting fingerprint processing pipeline...")
        self.start_time = time.time()
        
        # Start all threads
        self.preprocessor.start()
        self.enhancer.start()
        self.extractor.start()
        self.matcher.start()
        
        # Register fingerprints if path is provided
        if self.reference_path.exists():
            self.register_fingerprints()
        else:
            print(f"Reference path not found: {self.reference_path}")
            self.registration_complete = True
        
        # Process test fingerprints if path is provided
        if self.test_path.exists():
            self.process_test_fingerprints()
        else:
            print(f"Test path not found: {self.test_path}")
            self.processing_complete = True
        
        # Set up periodic status checks
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_status)
        self.status_timer.start(5000)  # Check every 5 seconds
    
    def register_fingerprints(self):
        """Register fingerprints from the reference folder."""
        print(f"Registering fingerprints from {self.reference_path}...")
        
        # Get list of subdirectories (one per subject)
        subdirs = [d for d in self.reference_path.iterdir() if d.is_dir()]
        
        if not subdirs:
            # If no subdirectories, assume flat structure with subject_id in filename
            self._register_flat_directory(self.reference_path)
        else:
            # Register each subject's fingerprints
            for subject_dir in subdirs:
                subject_id = subject_dir.name
                self._register_subject(subject_id, subject_dir)
        
        self.registration_complete = True
        print(f"Registration complete. {len(self.matcher.database)} fingerprints registered.")
    
    def _register_flat_directory(self, directory):
        """Register fingerprints from a flat directory structure."""
        image_files = self._get_image_files(directory)
        
        for image_path in image_files:
            # Extract subject ID from filename (assuming format "subject_id_*.ext")
            filename = image_path.stem
            parts = filename.split('_')
            
            if len(parts) >= 1:
                subject_id = parts[0]
                self._process_enrollment_image(subject_id, image_path)
            else:
                print(f"Cannot parse subject ID from filename: {filename}")
    
    def _register_subject(self, subject_id, subject_dir):
        """Register all fingerprints for a single subject."""
        image_files = self._get_image_files(subject_dir)
        
        for image_path in image_files:
            self._process_enrollment_image(subject_id, image_path)
    
    def _process_enrollment_image(self, subject_id, image_path):
        """Process a single enrollment image and add it to the database."""
        try:
            # Load image
            image = Image.open(image_path).convert('L')
            
            # Process the image through the pipeline
            # For enrollment, we'll do synchronous processing
            preprocessed = self.preprocessor._preprocess_image(image, f"enroll_{subject_id}")
            if preprocessed is None:
                print(f"Failed to preprocess enrollment image: {image_path}")
                return
                
            enhanced_results = self.enhancer._enhance_image(preprocessed, f"enroll_{subject_id}")
            if enhanced_results is None:
                print(f"Failed to enhance enrollment image: {image_path}")
                return
            
            # Get the enhanced image
            if 'enhanced' in enhanced_results:
                fingerprint_image = enhanced_results['enhanced']
            elif 'binary' in enhanced_results:
                fingerprint_image = enhanced_results['binary']
            else:
                fingerprint_image = np.array(preprocessed)
            
            # Extract features
            with torch.no_grad():
                # Convert to numpy array if PIL Image
                if isinstance(fingerprint_image, Image.Image):
                    fingerprint_image = np.array(fingerprint_image)
                
                # Normalize if needed
                if fingerprint_image.dtype != np.float32:
                    fingerprint_image = fingerprint_image.astype(np.float32) / 255.0
                
                # Create tensor
                if len(fingerprint_image.shape) == 2:  # Add channel dimension if grayscale
                    fingerprint_image = fingerprint_image[np.newaxis, ...]
                
                # Add batch dimension if not present
                if len(fingerprint_image.shape) == 3:
                    fingerprint_image = fingerprint_image[np.newaxis, ...]
                
                # Convert to PyTorch tensor and move to device
                image_tensor = torch.tensor(fingerprint_image, device=self.device)
                
                # Extract features
                feature_vector = self.extractor.model.extract_features(image_tensor)
            
            # Add to database
            self.matcher.add_fingerprint(subject_id, feature_vector, save_to_disk=False)
            print(f"Enrolled fingerprint for subject {subject_id} from {image_path.name}")
            
        except Exception as e:
            print(f"Error enrolling fingerprint {image_path}: {str(e)}")
    
    def process_test_fingerprints(self):
        """Process test fingerprints from the test folder."""
        print(f"Processing test fingerprints from {self.test_path}...")
        
        # Get list of image files
        image_files = self._get_image_files(self.test_path)
        print(f"Found {len(image_files)} test images")
        
        # Add all images to input buffer
        for i, image_path in enumerate(image_files):
            image_id = f"test_{i:04d}"
            self.preprocessor.add_image(str(image_path), image_id, {
                'path': str(image_path),
                'filename': image_path.name
            })
            print(f"Added test image to queue: {image_path.name}")
    
    def _get_image_files(self, directory):
        """Get all image files in a directory."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.BMP']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(directory.glob(f"*{ext}")))
        
        return image_files
    
    @pyqtSlot()
    def check_status(self):
        """Check processing status and stop when complete."""
        if self.processing_complete and self.registration_complete:
            # Check if all queues are empty
            if (self.input_buffer.empty() and 
                self.enhance_queue.empty() and 
                self.extractor_queue.empty() and 
                self.matcher_queue.empty()):
                
                # Process any remaining results
                while not self.result_queue.empty():
                    self._process_result()
                
                # Stop periodic checks
                self.status_timer.stop()
                
                # Wait a bit to ensure all signals are processed
                QTimer.singleShot(1000, self.stop)
        else:
            # Process any available results
            while not self.result_queue.empty():
                self._process_result()
            
            # Print status
            print(f"Status update:")
            print(f"  Input queue: {self.input_buffer.qsize()}")
            print(f"  Enhance queue: {self.enhance_queue.qsize()}")
            print(f"  Extractor queue: {self.extractor_queue.qsize()}")
            print(f"  Matcher queue: {self.matcher_queue.qsize()}")
            print(f"  Result queue: {self.result_queue.qsize()}")
            print(f"  Processed: {self.total_processed}")
            print(f"  Matched: {self.total_matched}")
            print(f"  Unmatched: {self.total_unmatched}")
    
    def _process_result(self):
        """Process a single result from the result queue."""
        try:
            result = self.result_queue.get(block=False)
            self.results[result['id']] = result
            self.total_processed += 1
            
            if result['result'] != "unrecognized":
                self.total_matched += 1
            else:
                self.total_unmatched += 1
                
            self.result_queue.task_done()
        except queue.Empty:
            pass
    
    def stop(self):
        """Stop the fingerprint processing pipeline."""
        print("Stopping fingerprint processing pipeline...")
        
        # Stop all threads
        self.preprocessor.stop()
        self.enhancer.stop()
        self.extractor.stop()
        self.matcher.stop()
        
        # Save database if needed
        if self.args.save_database:
            self.matcher._save_database()
        
        # Print final results
        self.print_results()
        
        # Exit application if requested
        if self.args.quit_when_done:
            QCoreApplication.quit()
    
    def print_results(self):
        """Print final processing results."""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "="*50)
        print("Fingerprint Processing Results")
        print("="*50)
        print(f"Total images processed: {self.total_processed}")
        print(f"Total matched: {self.total_matched}")
        print(f"Total unmatched: {self.total_unmatched}")
        print(f"Match rate: {self.total_matched/max(1, self.total_processed)*100:.2f}%")
        print(f"Total processing time: {elapsed_time:.2f} seconds")
        print(f"Average time per image: {elapsed_time/max(1, self.total_processed):.4f} seconds")
        print("="*50)
        
        # Print detailed results if requested
        if self.args.verbose:
            print("\nDetailed Results:")
            for result_id, result in sorted(self.results.items()):
                print(f"Image {result_id}: {result['result']} (score: {result['similarity_score']:.4f})")
    
    @pyqtSlot(str, object)
    def handle_preprocessing_error(self, image_id, error):
        print(f"Preprocessing error for {image_id}: {error}")
    
    @pyqtSlot(str, str)
    def handle_enhancement_error(self, image_id, error):
        print(f"Enhancement error for {image_id}: {error}")
    
    @pyqtSlot(str, str)
    def handle_extraction_error(self, image_id, error):
        print(f"Extraction error for {image_id}: {error}")
    
    @pyqtSlot(str, str, float)
    def handle_match_result(self, image_id, result, score):
        print(f"Match result for {image_id}: {result} (score: {score:.4f})")
    
    @pyqtSlot(str, str)
    def handle_matching_error(self, image_id, error):
        print(f"Matching error for {image_id}: {error}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fingerprint processing pipeline")
    
    parser.add_argument("--reference-path", type=str, required=True,
                        help="Path to reference fingerprint images for enrollment")
    
    parser.add_argument("--test-path", type=str, required=True,
                        help="Path to test fingerprint images for matching")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to saved model weights")
    
    parser.add_argument("--database-path", type=str, default="fingerprint_database.pt",
                        help="Path to fingerprint database file")
    
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Similarity threshold for matching (0-1)")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")
    
    parser.add_argument("--save-database", action="store_true",
                        help="Save updated database after processing")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results")
    
    parser.add_argument("--quit-when-done", action="store_true",
                        help="Quit application when processing is complete")
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create pipeline
    pipeline = FingerprintPipeline(args)
    
    # Start pipeline
    pipeline.start()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()