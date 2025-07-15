from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
import uuid
from functools import wraps
import json
from pathlib import Path
import traceback
from model_utils import analyze_csv, predict_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = 'uploads'
    MODEL_FOLDER = 'models'
    ALLOWED_EXTENSIONS = {'csv'}

app.config.from_object(Config)

# Create necessary directories
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.MODEL_FOLDER, exist_ok=True)

# Store for model metadata
MODEL_METADATA = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def handle_errors(f):
    """Decorator for error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            return jsonify({'error': 'File not found'}), 404
        except ValueError as e:
            logger.error(f"Value error: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Internal server error'}), 500
    return decorated_function

def validate_input(data, required_fields):
    """Validate input data"""
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    return True

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/upload', methods=['POST'])
@handle_errors
def upload_csv():
    """Upload CSV and train model"""
    # Validate file upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
    
    # Get parameters
    target_column = request.form.get('target_column')
    model_type = request.form.get('model_type', 'random_forest')
    
    if not target_column:
        return jsonify({'error': 'Target column is required'}), 400
    
    # Validate model type
    valid_models = ['random_forest', 'linear_regression', 'decision_tree', 'svm', 'xgboost']
    if model_type not in valid_models:
        return jsonify({'error': f'Invalid model type. Must be one of: {valid_models}'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    filepath = os.path.join(Config.UPLOAD_FOLDER, f"{file_id}_{filename}")
    file.save(filepath)
    
    try:
        # Read and validate CSV
        df = pd.read_csv(filepath)
        logger.info(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in CSV'}), 400
        
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400
        
        # Analyze CSV and train model
        results, model = analyze_csv(df, target_column, model_type)
        
        # Save model with unique ID
        model_id = str(uuid.uuid4())
        model_path = os.path.join(Config.MODEL_FOLDER, f"{model_id}.pkl")
        joblib.dump(model, model_path)
        
        # Store model metadata
        MODEL_METADATA[model_id] = {
            'created_at': datetime.now().isoformat(),
            'model_type': model_type,
            'target_column': target_column,
            'features': list(df.columns.drop(target_column)),
            'data_shape': df.shape,
            'filename': filename
        }
        
        # Save metadata to file
        metadata_path = os.path.join(Config.MODEL_FOLDER, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(MODEL_METADATA, f, indent=2)
        
        logger.info(f"Model trained successfully: {model_id}")
        
        # Add model_id to results
        results['model_id'] = model_id
        results['model_metadata'] = MODEL_METADATA[model_id]
        
        return jsonify(results)
        
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/predict', methods=['POST'])
@handle_errors
def predict():
    """Make predictions using trained model"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    model_id = data.get('model_id')
    if not model_id:
        return jsonify({'error': 'Model ID is required'}), 400
    
    if model_id not in MODEL_METADATA:
        return jsonify({'error': 'Model not found'}), 404
    
    model_path = os.path.join(Config.MODEL_FOLDER, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404
    
    # Load model
    model = joblib.load(model_path)
    
    # Validate input features
    metadata = MODEL_METADATA[model_id]
    expected_features = metadata['features']
    
    input_data = data.get('input_data', {})
    missing_features = set(expected_features) - set(input_data.keys())
    if missing_features:
        return jsonify({'error': f'Missing features: {list(missing_features)}'}), 400
    
    # Make prediction
    prediction = predict_input(model, input_data)
    
    logger.info(f"Prediction made for model {model_id}")
    
    return jsonify({
        'prediction': prediction,
        'model_id': model_id,
        'model_type': metadata['model_type'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/models', methods=['GET'])
@handle_errors
def list_models():
    """List all available models"""
    # Load metadata from file if it exists
    metadata_path = os.path.join(Config.MODEL_FOLDER, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            MODEL_METADATA.update(json.load(f))
    
    return jsonify({
        'models': MODEL_METADATA,
        'count': len(MODEL_METADATA)
    })

@app.route('/api/models/<model_id>', methods=['GET'])
@handle_errors
def get_model_info(model_id):
    """Get information about a specific model"""
    if model_id not in MODEL_METADATA:
        return jsonify({'error': 'Model not found'}), 404
    
    return jsonify({
        'model_id': model_id,
        'metadata': MODEL_METADATA[model_id]
    })

@app.route('/api/models/<model_id>', methods=['DELETE'])
@handle_errors
def delete_model(model_id):
    """Delete a specific model"""
    if model_id not in MODEL_METADATA:
        return jsonify({'error': 'Model not found'}), 404
    
    model_path = os.path.join(Config.MODEL_FOLDER, f"{model_id}.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)
    
    del MODEL_METADATA[model_id]
    
    # Update metadata file
    metadata_path = os.path.join(Config.MODEL_FOLDER, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(MODEL_METADATA, f, indent=2)
    
    logger.info(f"Model deleted: {model_id}")
    
    return jsonify({'message': 'Model deleted successfully'})

@app.route('/api/download-model/<model_id>', methods=['GET'])
@handle_errors
def download_model(model_id):
    """Download a specific model"""
    if model_id not in MODEL_METADATA:
        return jsonify({'error': 'Model not found'}), 404
    
    model_path = os.path.join(Config.MODEL_FOLDER, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404
    
    return send_file(
        model_path,
        as_attachment=True,
        download_name=f"model_{model_id}.pkl"
    )

@app.route('/api/batch-predict', methods=['POST'])
@handle_errors
def batch_predict():
    """Make batch predictions"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    model_id = data.get('model_id')
    if not model_id:
        return jsonify({'error': 'Model ID is required'}), 400
    
    if model_id not in MODEL_METADATA:
        return jsonify({'error': 'Model not found'}), 404
    
    batch_data = data.get('batch_data', [])
    if not batch_data:
        return jsonify({'error': 'Batch data is required'}), 400
    
    model_path = os.path.join(Config.MODEL_FOLDER, f"{model_id}.pkl")
    model = joblib.load(model_path)
    
    predictions = []
    for item in batch_data:
        try:
            prediction = predict_input(model, item)
            predictions.append({
                'input': item,
                'prediction': prediction,
                'status': 'success'
            })
        except Exception as e:
            predictions.append({
                'input': item,
                'error': str(e),
                'status': 'error'
            })
    
    return jsonify({
        'predictions': predictions,
        'total': len(predictions),
        'model_id': model_id,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

# Load existing metadata on startup
def load_existing_metadata():
    """Load existing model metadata on startup"""
    metadata_path = os.path.join(Config.MODEL_FOLDER, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                MODEL_METADATA.update(json.load(f))
            logger.info(f"Loaded {len(MODEL_METADATA)} existing models")
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")

if __name__ == '__main__':
    load_existing_metadata()
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
