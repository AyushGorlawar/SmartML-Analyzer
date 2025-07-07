# SmartML Analyzer

SmartML Analyzer is a professional, user-friendly web app for instant machine learning analysis. Upload your CSV dataset, select a model, and get performance metrics, visualizations, and model explainability—all in a beautiful dashboard. Test new predictions and download your trained model.

## Features
- **Upload Dataset**: Drag-and-drop or browse to upload CSV files.
- **Model Selection**: Choose from Random Forest, Logistic Regression, Decision Tree, or SVM.
- **Performance Metrics**: View accuracy, precision, recall, F1, ROC AUC, and confusion matrix.
- **Visualizations**: Confusion matrix, ROC curve, and feature importance plots.
- **Model Explainability**: SHAP summary plots for feature contributions.
- **Test & Predict**: Enter new feature values and get predictions from the trained model.
- **Download Model**: Export the trained model as a .pkl file.

## Setup Instructions

### Backend (Python/Flask)
1. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Run the backend server:
   ```bash
   python app.py
   ```
   The backend will be available at `http://localhost:5000/`.

### Frontend
- The frontend is served by Flask at the root URL. Open `http://localhost:5000/` in your browser.

## Usage
1. Upload your CSV dataset.
2. Select the target column and ML model.
3. View metrics, visualizations, and explainability.
4. Test new predictions in the "Test & Predict" section.
5. Download your trained model if desired.

## Example Dataset
You can use the included `iris_sample.csv` or your own CSV files.


