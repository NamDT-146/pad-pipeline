import os
import sys
import time
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Import the core functionality directly
from model.siamesenetwork import SiameseNetwork
from dataset.siamesepair import create_default_args
from dataset.preprocessing import create_fingerprint_transforms
from dataset.enhancing import create_fingerprint_enhancement


def preprocess_image(image, args):
    """Preprocess a fingerprint image."""
    preprocessor = create_fingerprint_transforms(args)
    return preprocessor(image)


def enhance_image(image, args):
    """Enhance a preprocessed fingerprint image."""
    enhancer = create_fingerprint_enhancement(args)
    
    # Apply enhancement
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        enhanced_results = enhancer(image)
        return enhanced_results['thinned']
    
    # Fallback to original image
    if isinstance(image, Image.Image):
        return np.array(image)
    return image


def extract_features(image, model, device):
    """Extract feature vector from a fingerprint image."""
    
    with torch.no_grad():
        feature_vector = model.extract_features(image.unsqueeze(0).to(device))
    
    return feature_vector


def match_features(feature_vector, database, threshold=0.75):
    """Match a feature vector against the fingerprint database."""
    if not database:
        return "no_database", 0.0
    
    best_match = None
    best_score = 0.0
    
    for subject_id, reference_features in database.items():
        # Calculate squared difference
        diff = reference_features - feature_vector
        diff_squared = diff * diff
        
        # Calculate similarity score (simplified)
        sim_score = 1.0 - torch.sqrt(torch.sum(diff_squared)).item() / 2.0
        
        if sim_score > best_score:
            best_score = sim_score
            best_match = subject_id
    
    # Return best match if score is above threshold
    if best_score >= threshold:
        return best_match, best_score
    else:
        return "unrecognized", best_score


def create_database(reference_path, model_path, database_path, device="cuda"):
    """Create a fingerprint database from reference images."""
    print(f"Creating fingerprint database from {reference_path}")
    start_time = time.time()
    
    # Load model
    model = SiameseNetwork().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create preprocessing arguments
    args = create_default_args(mode='test')
    
    # Initialize database
    database = {}
    reference_path = Path(reference_path)
    
    # Get list of subdirectories (one per subject)
    subdirs = [d for d in reference_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        # If no subdirectories, assume flat structure with subject_id in filename
        image_files = get_image_files(reference_path)
        
        for image_path in image_files:
            # Extract subject ID from filename (assuming format "subject_id_*.ext")
            filename = image_path.stem
            parts = filename.split('_')
            
            if len(parts) >= 1:
                subject_id = parts[0]
                print(f"Processing {image_path.name} for subject {subject_id}")
                
                # Process image
                try:
                    # Load and process image
                    image = Image.open(image_path).convert('L')
                    preprocessed = preprocess_image(image, args)
                    enhanced = enhance_image(preprocessed, args)
                    features = extract_features(enhanced, model, device)
                    
                    # Add to database
                    database[subject_id] = features
                    print(f"  Added fingerprint for subject {subject_id}")
                    
                except Exception as e:
                    print(f"  Error processing {image_path}: {e}")
            else:
                print(f"Cannot parse subject ID from filename: {filename}")
    else:
        # Process each subject directory
        for subject_dir in subdirs:
            subject_id = subject_dir.name
            print(f"Processing subject {subject_id}")
            
            image_files = get_image_files(subject_dir)
            
            if not image_files:
                print(f"  No images found for subject {subject_id}")
                continue
            
            # Process first image for each subject
            try:
                image_path = image_files[0]
                print(f"  Processing {image_path.name}")
                
                # Load and process image
                image = Image.open(image_path).convert('L')
                preprocessed = preprocess_image(image, args)
                enhanced = enhance_image(preprocessed, args)
                features = extract_features(enhanced, model, device)
                
                # Add to database
                database[subject_id] = features
                print(f"  Added fingerprint for subject {subject_id}")
                
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
    
    # Save database
    torch.save(database, database_path)
    
    elapsed_time = time.time() - start_time
    print(f"Database creation completed in {elapsed_time:.2f} seconds")
    print(f"Added {len(database)} fingerprints to database")
    print(f"Database saved to {database_path}")
    
    return database


def append_to_database(reference_path, database_path, model_path, device="cuda"):
    """Append new fingerprints to an existing database."""
    print(f"Appending fingerprints from {reference_path} to database {database_path}")
    start_time = time.time()
    
    # Load model
    model = SiameseNetwork().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create preprocessing arguments
    args = create_default_args(mode='test')
    
    # Load existing database
    try:
        database = torch.load(database_path, map_location=device)
        print(f"Loaded existing database with {len(database)} fingerprints")
    except FileNotFoundError:
        print(f"Database file {database_path} not found. Creating new database.")
        database = {}
    except Exception as e:
        print(f"Error loading database: {e}. Creating new database.")
        database = {}
    
    # Initialize counters
    added_count = 0
    skipped_count = 0
    error_count = 0
    
    reference_path = Path(reference_path)
    
    # Get list of subdirectories (one per subject)
    subdirs = [d for d in reference_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        # If no subdirectories, assume flat structure with subject_id in filename
        image_files = get_image_files(reference_path)
        
        for image_path in image_files:
            # Extract subject ID from filename (assuming format "subject_id_*.ext")
            filename = image_path.stem
            parts = filename.split('_')
            
            if len(parts) >= 1:
                subject_id = parts[0]
                print(f"Processing {image_path.name} for subject {subject_id}")
                
                # Check if subject already exists
                if subject_id in database:
                    print(f"  Subject {subject_id} already exists in database, skipping")
                    skipped_count += 1
                    continue
                
                # Process image
                try:
                    # Load and process image
                    image = Image.open(image_path).convert('L')
                    preprocessed = preprocess_image(image, args)
                    enhanced = enhance_image(preprocessed, args)
                    features = extract_features(enhanced, model, device)
                    
                    # Add to database
                    database[subject_id] = features
                    print(f"  Added fingerprint for subject {subject_id}")
                    added_count += 1
                    
                except Exception as e:
                    print(f"  Error processing {image_path}: {e}")
                    error_count += 1
            else:
                print(f"Cannot parse subject ID from filename: {filename}")
    else:
        # Process each subject directory
        for subject_dir in subdirs:
            subject_id = subject_dir.name
            print(f"Processing subject {subject_id}")
            
            # Check if subject already exists
            if subject_id in database:
                print(f"  Subject {subject_id} already exists in database, skipping")
                skipped_count += 1
                continue
            
            image_files = get_image_files(subject_dir)
            
            if not image_files:
                print(f"  No images found for subject {subject_id}")
                continue
            
            # Process first image for each subject
            try:
                image_path = image_files[0]
                print(f"  Processing {image_path.name}")
                
                # Load and process image
                image = Image.open(image_path).convert('L')
                preprocessed = preprocess_image(image, args)
                enhanced = enhance_image(preprocessed, args)
                features = extract_features(enhanced, model, device)
                
                # Add to database
                database[subject_id] = features
                print(f"  Added fingerprint for subject {subject_id}")
                added_count += 1
                
            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                error_count += 1
    
    # Save database
    torch.save(database, database_path)
    
    elapsed_time = time.time() - start_time
    print(f"Database update completed in {elapsed_time:.2f} seconds")
    print(f"Added {added_count} new fingerprints to database")
    print(f"Skipped {skipped_count} existing fingerprints")
    print(f"Encountered {error_count} errors")
    print(f"Database now contains {len(database)} fingerprints")
    print(f"Updated database saved to {database_path}")
    
    return database


def verify_fingerprint(image_path, database_path, model_path, threshold=0.75, device="cuda"):
    """Verify a single fingerprint against the database."""
    print(f"Verifying fingerprint: {image_path}")
    start_time = time.time()
    
    # Load model
    model = SiameseNetwork().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create preprocessing arguments
    args = create_default_args(mode='test')
    
    # Load database
    database = torch.load(database_path, map_location=device)
    print(f"Loaded database with {len(database)} fingerprints")
    
    try:
        # Load and process image
        image = Image.open(image_path).convert('L')
        print(f"Image loaded: {image.size}")
        print(type(image))
        
        # Process image
        print("Preprocessing image...")
        preprocessed = preprocess_image(image, args)
        print(type(preprocessed))
        print(preprocessed.shape)

        print("Enhancing image...")
        enhanced = enhance_image(preprocessed, args)
        print(type(enhanced))
        print(enhanced.shape)

        print("Extracting features...")
        features = extract_features(enhanced, model, device)
        print(type(features))
        print(features.shape)

        # Match against database
        print("Matching against database...")
        result, score = match_features(features, database, threshold)
        
        elapsed_time = time.time() - start_time
        print(f"Verification completed in {elapsed_time:.2f} seconds")
        print(f"Result: {result}")
        print(f"Score: {score:.4f}")
        
        return result, score
        
    except Exception as e:
        print(f"Error verifying fingerprint: {e}")
        return "error", 0.0


def get_image_files(directory):
    """Get all image files in a directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.BMP']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(directory.glob(f"*{ext}")))
    
    return image_files


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fingerprint Pipeline Tester")
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Create database mode
    create_parser = subparsers.add_parser("create", help="Create fingerprint database")
    create_parser.add_argument("--reference-path", required=True, 
                             help="Path to reference fingerprint images")
    create_parser.add_argument("--model-path", required=True,
                             help="Path to the model weights")
    create_parser.add_argument("--database-path", default="fingerprint_database.pt",
                             help="Path to save the fingerprint database")
    create_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                             help="Device to run on (cuda or cpu)")
    
    # Append to database mode
    append_parser = subparsers.add_parser("append", help="Append fingerprints to existing database")
    append_parser.add_argument("--reference-path", required=True, 
                             help="Path to reference fingerprint images to append")
    append_parser.add_argument("--database-path", required=True,
                             help="Path to existing fingerprint database")
    append_parser.add_argument("--model-path", required=True,
                             help="Path to the model weights")
    append_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                             help="Device to run on (cuda or cpu)")
    
    # Verify fingerprint mode
    verify_parser = subparsers.add_parser("verify", help="Verify a single fingerprint")
    verify_parser.add_argument("--image-path", required=True,
                             help="Path to the fingerprint image to verify")
    verify_parser.add_argument("--database-path", required=True,
                             help="Path to the fingerprint database")
    verify_parser.add_argument("--model-path", required=True,
                             help="Path to the model weights")
    verify_parser.add_argument("--threshold", type=float, default=0.75,
                             help="Similarity threshold (0-1)")
    verify_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                             help="Device to run on (cuda or cpu)")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.mode == "create":
        create_database(
            args.reference_path,
            args.model_path,
            args.database_path,
            args.device
        )
    elif args.mode == "append":
        append_to_database(
            args.reference_path,
            args.database_path,
            args.model_path,
            args.device
        )
    elif args.mode == "verify":
        verify_fingerprint(
            args.image_path,
            args.database_path,
            args.model_path,
            args.threshold,
            args.device
        )
    else:
        print("Please specify a mode: create, append, or verify")
        sys.exit(1)


if __name__ == "__main__":
    main()