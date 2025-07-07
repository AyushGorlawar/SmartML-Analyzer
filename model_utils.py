import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import shap

PLOTS_DIR = os.path.join('static', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def analyze_csv(df, target_column=None, model_type='random_forest'):
    if target_column is None:
        target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Auto-encode categorical features
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model selection
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    elif model_type == 'svm':
        model = SVC(probability=True)
    else:
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        try:
            roc_auc = float(roc_auc_score(y_test, y_proba, multi_class='ovr'))
        except Exception:
            roc_auc = None
    else:
        roc_auc = None

    # --- Visualizations ---
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    # ROC Curve (only for binary or multiclass with OvR)
    roc_path = None
    y_unique = np.unique(np.array(y))
    if y_proba is not None and isinstance(y_proba, np.ndarray) and (len(y_unique) == 2 or (len(y_unique) > 2 and y_proba.shape[1] > 1)):
        plt.figure(figsize=(6, 5))
        if len(y_unique) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            plt.plot(fpr, tpr, label='ROC curve')
        else:
            for i in range(y_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
                plt.plot(fpr, tpr, label=f'Class {i}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        roc_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()

    # Feature Importance (for tree-based models)
    feat_imp_path = None
    if hasattr(model, 'feature_importances_') and isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(7, 5))
        plt.title('Feature Importance')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        feat_imp_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
        plt.savefig(feat_imp_path, bbox_inches='tight')
        plt.close()

    # --- SHAP Explainability ---
    shap_summary_path = None
    try:
        explainer = None
        if model_type in ['random_forest', 'decision_tree']:
            explainer = shap.TreeExplainer(model)
        elif model_type == 'logistic_regression':
            explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
        elif model_type == 'svm':
            explainer = shap.KernelExplainer(model.predict_proba, np.array(X_train))
        if explainer is not None:
            X_test_sample = np.array(X_test[:100])
            if hasattr(explainer, 'shap_values'):
                shap_values = explainer.shap_values(X_test_sample)
            else:
                shap_values = explainer.shap_values(X_test_sample)
            plt.figure(figsize=(8, 5))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], X_test[:100], show=False)
            else:
                shap.summary_plot(shap_values, X_test[:100], show=False)
            shap_summary_path = os.path.join(PLOTS_DIR, 'shap_summary.png')
            plt.savefig(shap_summary_path, bbox_inches='tight')
            plt.close()
    except Exception as e:
        shap_summary_path = None

    results = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1': float(f1_score(y_test, y_pred, average='weighted')),
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc,
        'confusion_matrix_img': '/static/plots/confusion_matrix.png',
        'roc_curve_img': '/static/plots/roc_curve.png' if roc_path else None,
        'feature_importance_img': '/static/plots/feature_importance.png' if feat_imp_path else None,
        'shap_summary_img': '/static/plots/shap_summary.png' if shap_summary_path else None,
    }
    return results, model

def predict_input(model, data):
    # data: dict of feature_name: value
    X_new = pd.DataFrame([data])
    return model.predict(X_new)[0] 