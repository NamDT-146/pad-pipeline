import os
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from fingerprint import FingerprintMatcher
import torch

app = Flask(__name__)
app.secret_key = 'demo_secret'  # for flash messages

# Configuration
MODEL_PATH = '#'  # Path to model weights
DATABASE_PATH = '#'  # Path to fingerprint database

# Create required directories
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

# Initialize fingerprint matcher
matcher = FingerprintMatcher(
    database_path=DATABASE_PATH,
    model_path=MODEL_PATH,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    mode = request.form.get('mode')             # "register" or "validate"
    file = request.files.get('fingerprint')     # .bmp file
    subject_id = request.form.get('subject_id', None)
    
    if not file or mode not in ('register', 'validate'):
        flash('Please upload a .bmp and select an action.')
        return redirect(url_for('home'))

    # Get file data directly from memory without saving
    fingerprint_data = file.read()

    if mode == 'register':
        if not subject_id:
            flash('Please provide a subject ID for registration.')
            return redirect(url_for('home'))
            
        # Register fingerprint directly from memory
        success, message = matcher.register_fingerprint(fingerprint_data, subject_id)
        
        if success:
            flash(f"Fingerprint registered: {message}")
        else:
            flash(f"Registration failed: {message}")
            
        return redirect(url_for('home'))

    # Validation mode
    threshold = float(request.form.get('threshold', 0.75))
    identity, score = matcher.verify_fingerprint(fingerprint_data, threshold)
    
    if identity not in ["unrecognized", "no_database", "error"]:
        flash(f"Matched: {identity} (confidence: {score:.2f})")
    else:
        if identity == "no_database":
            flash("No fingerprints in database yet. Please register first.")
        elif identity == "error":
            flash(f"Error during verification. Score: {score:.2f}")
        else:
            flash(f"No match found. Best score: {score:.2f}")
            
    return redirect(url_for('home'))

@app.route('/subjects', methods=['GET'])
def list_subjects():
    """Get all registered subjects"""
    subjects = matcher.get_all_subjects()
    return jsonify({"subjects": subjects, "count": len(subjects)})

@app.route('/subjects/<subject_id>', methods=['DELETE'])
def remove_subject(subject_id):
    """Remove a subject from the database"""
    success, message = matcher.remove_subject(subject_id)
    return jsonify({"success": success, "message": message})

if __name__ == '__main__':
    app.run(debug=True)