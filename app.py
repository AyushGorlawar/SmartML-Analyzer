from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import os
from model_utils import analyze_csv, predict_input

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model.pkl'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    file = request.files['file']
    target_column = request.form.get('target_column')
    model_type = request.form.get('model_type', 'random_forest')
    df = pd.read_csv(file.stream)
    results, model = analyze_csv(df, target_column, model_type)
    joblib.dump(model, MODEL_PATH)
    return jsonify(results)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    model = joblib.load(MODEL_PATH)
    prediction = predict_input(model, data)
    return jsonify({'prediction': prediction})

@app.route('/api/download-model', methods=['GET'])
def download_model():
    return send_file(MODEL_PATH, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 