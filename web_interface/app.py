import os
import sys
import cv2
import time
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

# --- 1. SETUP PATHS ---
# Get the root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# path to 'Models' folder
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

# path to 'Ensemble' folder inside Models
ENSEMBLE_DIR = os.path.join(MODELS_DIR, 'Ensemble')

# Add BOTH to system path so Python finds everything
sys.path.append(MODELS_DIR)
sys.path.append(ENSEMBLE_DIR)



app = Flask(__name__)

app = Flask(__name__)

# --- 2. CONFIGURATION ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PREDICT_FOLDER = os.path.join('static', 'predictions')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICT_FOLDER'] = PREDICT_FOLDER

# Ensure folders exist
os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)
os.makedirs(os.path.join(app.root_path, PREDICT_FOLDER), exist_ok=True)

# --- 3. IMPORT AI ENGINE ---
# We wrap this in a try-except block so the server doesn't crash if paths are wrong
try:
    from Ensemble.single_inference import load_ensemble_models, predict_single
    AI_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import Ensemble engine. Error: {e}")
    AI_ENGINE_AVAILABLE = False

# Global variable to hold loaded models
loaded_models = []

# --- 4. ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inference')
def inference():
    return render_template('inference.html', ai_ready=AI_ENGINE_AVAILABLE)

@app.route('/analysis')
def analysis():
    """
    Renders the Analysis page.
    """
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not AI_ENGINE_AVAILABLE:
        return jsonify({'error': 'AI Engine not available'}), 500
    if not loaded_models:
        return jsonify({'error': 'Models loading...'}), 503

    # 1. Handle Input Image
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    timestamp = int(time.time())
    filename = f"{timestamp}_{file.filename}"
    upload_path = os.path.join(app.root_path, UPLOAD_FOLDER, filename)
    file.save(upload_path)

    # 2. Handle Ground Truth (Optional)
    true_mask_path = None
    if 'mask' in request.files:
        mask_file = request.files['mask']
        if mask_file.filename != '':
            mask_filename = f"truth_{filename}.png" # Ensure png
            true_mask_path = os.path.join(app.root_path, UPLOAD_FOLDER, mask_filename)
            mask_file.save(true_mask_path)

    try:
        # 3. Run Inference
        # Returns: path_to_pred, score (float or None), path_to_color_gt (or None)
        pred_abs, score, gt_abs = predict_single(
            loaded_models, 
            upload_path, 
            os.path.join(app.root_path, PREDICT_FOLDER),
            true_mask_path
        )
        
        # 4. Prepare Response
        response = {
            'original_url': f"/{UPLOAD_FOLDER}/{filename}",
            'mask_url': f"/{PREDICT_FOLDER}/{os.path.basename(pred_abs)}",
            'message': 'Segmentation Successful'
        }
        
        if score is not None:
            response['score'] = f"{score:.4f}" # Format to 4 decimals
            response['gt_url'] = f"/{PREDICT_FOLDER}/{os.path.basename(gt_abs)}"

        return jsonify(response)

    except Exception as e:
        print(f"❌ Inference Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- Add this to app.py (anywhere after the analysis route) ---

@app.route('/get_chart/<phase>/<filename>')
def get_chart(phase, filename):
    """
    Serves images (confusion matrix, metrics) from the Result folder of each model.
    """
    # Map phases to their specific result folders
    # This matches the folder structure used in get_metrics
    phase_map = {
        'phase1': 'Models/phase1/Result',
        'phase1 Remastered': 'Models/phase1/Result', 
        'phase3': 'Models/phase3/Result',
        'phase4': 'Models/phase4/Result',
        'phase5': 'Models/phase5/Result',
        'phase6': 'Models/phase6/Result',
        'ensemble': 'Models/Ensemble/Result'
    }
    
    # 1. Check if phase exists in our map
    if phase not in phase_map:
        return f"Phase '{phase}' not found in map", 404
        
    # 2. Construct the full path
    # We combine BASE_DIR (root) + The Model Folder + Result
    relative_path = phase_map[phase]
    directory = os.path.join(BASE_DIR, relative_path)
    
    # 3. Serve the file
    try:
        return send_from_directory(directory, filename)
    except FileNotFoundError:
        return f"File {filename} not found in {directory}", 404


@app.route('/get_metrics/<phase>')
def get_metrics(phase):
    """
    Reads the evaluation_metrics.txt file for a specific phase 
    and returns the scores as JSON for the Model Analysis page.
    """
    # Map phases to their specific result folders (matching your get_chart map)
    phase_map = {
        'phase1': 'Models/phase1/Result',
        'phase3': 'Models/phase3/Result',
        'phase4': 'Models/phase4/Result',
        'phase5': 'Models/phase5/Result',
        'phase6': 'Models/phase6/Result',
        'ensemble': 'Models/Ensemble/Result'
    }

    if phase not in phase_map:
        return jsonify({"error": "Phase not found"}), 404

    # Construct path to the text file
    file_path = os.path.join(BASE_DIR, phase_map[phase], "evaluation_metrics.txt")

    # If file doesn't exist, return a placeholder
    if not os.path.exists(file_path):
        return jsonify({"metrics": ["Data Pending"]})

    metrics = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Look for lines containing keywords like SCORE or mIoU
            for line in lines:
                if any(key in line for key in ["SCORE", "mIoU", "Score", "Pixel Acc", "IoU"]):
                    # Clean up and format the string for the UI badge
                    text = line.strip()
                    text = text.split()
                    text=f"IoU: {text[-1]}"
                    metrics.append(text)
                    # We usually only want the top 1-2 metrics for the badges
                    if len(metrics) >= 2: break 
            
            if not metrics:
                metrics.append("File found (No specific score detected)")
    except Exception as e:
        print(f"Error reading metrics: {e}")
        return jsonify({"metrics": ["Error reading data"]})

    return jsonify({"metrics": metrics})

# --- 5. SERVER STARTUP ---
if __name__ == '__main__':
    print("------------------------------------------------")
    print("🚀 STARTING SEMANTIC SEGMENTATION SERVER (HF)")
    print("------------------------------------------------")
    
    # 1. Safety Check: Only try to load models if the engine imported correctly
    if AI_ENGINE_AVAILABLE:
        try:
            print("⏳ Initializing Ensemble Models...")
            loaded_models = load_ensemble_models()
            print("✅ Models Loaded Successfully.")
        except Exception as e:
            print(f"❌ Critical Error loading models: {e}")
            # We don't exit; we let the server start so we can debug via logs
    else:
        print("⚠️ AI ENGINE UNAVAILABLE: Check imports and paths.")

    # 2. Production Config for Hugging Face
    # host='0.0.0.0' makes it accessible outside the container
    # port=7860 is the standard HF Spaces port
    app.run(host='0.0.0.0', port=7860, debug=False)