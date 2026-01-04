"""
Main Flask Application for PCB Defect Detection
Connects the model, routes, and web interface
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import json
from datetime import datetime

# Import model predictor
from src.model.predictor import PCBDefectPredictor
from src.utils.file_handler import allowed_file, save_uploaded_file
from src.config.settings import Config

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize model predictor (singleton pattern)
predictor = None

def get_predictor():
    """Get or initialize the predictor (lazy loading)"""
    global predictor
    if predictor is None:
        predictor = PCBDefectPredictor(
            model_path=app.config['MODEL_PATH'],
            conf_threshold=app.config['CONF_THRESHOLD'],
            iou_threshold=app.config['IOU_THRESHOLD']
        )
    return predictor


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and run prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify({'error': 'Invalid file type. Use JPG, JPEG, or PNG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Get predictor and run inference
        model = get_predictor()
        results = model.predict(
            image_path=filepath,
            save_output=True,
            output_dir=app.config['RESULTS_FOLDER']
        )
        
        # Get annotated image path
        annotated_filename = f"annotated_{unique_filename}"
        
        # Prepare response
        response = {
            'success': True,
            'filename': unique_filename,
            'annotated_filename': annotated_filename,
            'verdict': results['verdict'],
            'num_detections': results['num_detections'],
            'defect_summary': results['defect_summary'],
            'detections': results['detections'],
            'timestamp': results['timestamp']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction (JSON input)"""
    try:
        data = request.get_json()
        
        if 'image_path' not in data:
            return jsonify({'error': 'image_path required'}), 400
        
        image_path = data['image_path']
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        # Get predictor and run inference
        model = get_predictor()
        results = model.predict(
            image_path=image_path,
            save_output=data.get('save_output', False),
            output_dir=app.config['RESULTS_FOLDER']
        )
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch prediction"""
    try:
        data = request.get_json()
        
        if 'image_paths' not in data:
            return jsonify({'error': 'image_paths required'}), 400
        
        image_paths = data['image_paths']
        
        # Get predictor and run batch inference
        model = get_predictor()
        results = model.predict_batch(
            image_paths=image_paths,
            save_outputs=data.get('save_outputs', False),
            output_dir=app.config['RESULTS_FOLDER']
        )
        
        return jsonify({'results': results, 'count': len(results)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/config', methods=['GET', 'POST'])
def config():
    """Get or update configuration"""
    global predictor
    
    if request.method == 'GET':
        return jsonify({
            'conf_threshold': app.config['CONF_THRESHOLD'],
            'iou_threshold': app.config['IOU_THRESHOLD'],
            'model_path': app.config['MODEL_PATH']
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            # Update config
            if 'conf_threshold' in data:
                app.config['CONF_THRESHOLD'] = float(data['conf_threshold'])
            if 'iou_threshold' in data:
                app.config['IOU_THRESHOLD'] = float(data['iou_threshold'])
            
            # Reinitialize predictor with new config
            predictor = None
            
            return jsonify({'success': True, 'message': 'Configuration updated'})
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500


@app.route('/model-info')
def model_info():
    """Get model information"""
    try:
        model = get_predictor()
        info = model.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        model = get_predictor()
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("PCB Defect Detection System - Starting Server")
    print("=" * 60)
    print(f"Model: {app.config['MODEL_PATH']}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Results folder: {app.config['RESULTS_FOLDER']}")
    print(f"Confidence threshold: {app.config['CONF_THRESHOLD']}")
    print(f"IoU threshold: {app.config['IOU_THRESHOLD']}")
    print("=" * 60)
    print("\nServer running at: http://localhost:5000")
    print("API Documentation: http://localhost:5000/api/docs")
    print("\nPress CTRL+C to quit")
    print("=" * 60)
    
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )