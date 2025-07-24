import os
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from fingerprint import FingerprintMatcher
import torch

app = Flask(__name__)
app.secret_key = 'demo_secret'  # for flash messages

# Configuration
MODEL_PATH = "weights/bad_model.pth"
DATABASE_PATH = "database/fingerprint_database.pt"

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
    architecture = request.form.get('architecture', 'siamese')
    
    if not file or mode not in ('register', 'validate'):
        flash('Please upload a .bmp and select an action.')
        return redirect(url_for('home'))

    fingerprint_data = file.read()

    # Use selected architecture for this request
    matcher = FingerprintMatcher(
        database_path=DATABASE_PATH,
        model_path=MODEL_PATH,
        model=architecture,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    if mode == 'register':
        if not subject_id:
            flash('Please provide a subject ID for registration.')
            return redirect(url_for('home'))
        success, message = matcher.register_fingerprint(fingerprint_data, subject_id)
        if success:
            flash(f"Fingerprint registered: {message}")
        else:
            flash(f"Registration failed: {message}")
        return redirect(url_for('home'))

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

# ===================== SIMPLE FINGERPRINT COMPARISON =====================
@app.route('/compare', methods=['POST'])
def compare_fingerprints():
    """Simple endpoint to compare two fingerprint images and say if they are of the same person or not."""
    file1 = request.files.get('fingerprint1')
    file2 = request.files.get('fingerprint2')
    architecture = request.form.get('architecture', 'siamese')
    if not file1 or not file2:
        flash('Please upload both fingerprint images.')
        return redirect(url_for('home'))
    data1 = file1.read()
    data2 = file2.read()
    matcher = FingerprintMatcher(
        database_path=DATABASE_PATH,
        model_path=MODEL_PATH,
        model=architecture,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    try:
        same_person, score = matcher.compare_fingerprints(data1, data2)
        if same_person:
            flash(f"Result: Same person! (score: {score:.2f})")
        else:
            flash(f"Result: Different person. (score: {score:.2f})")
    except Exception as e:
        flash(f"Error comparing fingerprints: {str(e)}")
    return redirect(url_for('home'))
# ===================== END SIMPLE FINGERPRINT COMPARISON =====================

if __name__ == '__main__':
    app.run(debug=True)