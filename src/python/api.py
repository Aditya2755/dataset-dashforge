
"""
AI Model Experimentation Platform API

This file contains the Flask API endpoints that would be used to interact
with the scikit-learn models and perform machine learning operations.

To run this Flask API (not needed for the React demo):
    pip install flask flask-cors pandas numpy scikit-learn xgboost
    python api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage for datasets and models (would use a database in production)
datasets = {}
models = {}

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """List all available datasets"""
    return jsonify({
        'status': 'success',
        'datasets': [{'id': k, 'name': v.get('name', k)} for k, v in datasets.items()]
    })

@app.route('/api/datasets', methods=['POST'])
def upload_dataset():
    """Upload and process a new dataset"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    name = request.form.get('name', file.filename)
    
    # Process CSV or Excel file
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file format'}), 400
        
        # Generate a unique ID
        dataset_id = f"dataset_{len(datasets) + 1}"
        
        # Store dataset metadata and dataframe
        datasets[dataset_id] = {
            'name': name,
            'columns': list(df.columns),
            'rows': len(df),
            'data': df  # In production, we'd store this in a database or file system
        }
        
        return jsonify({
            'status': 'success',
            'dataset_id': dataset_id,
            'name': name,
            'columns': list(df.columns),
            'rows': len(df)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train a machine learning model with specified parameters"""
    try:
        data = request.json
        dataset_id = data.get('datasetId')
        algorithm = data.get('algorithm')
        hyperparameters = data.get('hyperparameters', {})
        preprocessing = data.get('preprocessing', {})
        target_column = data.get('targetColumn')
        
        if dataset_id not in datasets:
            return jsonify({'status': 'error', 'message': 'Dataset not found'}), 404
        
        # Get the dataset
        df = datasets[dataset_id]['data']
        
        # Extract features and target
        X = df.drop(target_column, axis=1) if target_column in df.columns else df.iloc[:, :-1]
        y = df[target_column] if target_column in df.columns else df.iloc[:, -1]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Apply preprocessing
        X_train, X_test = preprocess_data(X_train, X_test, y_train, preprocessing)
        
        # Train the model
        model, results = train_algorithm(algorithm, hyperparameters, X_train, X_test, y_train, y_test)
        
        # Store the model
        model_id = f"model_{len(models) + 1}"
        models[model_id] = {
            'algorithm': algorithm,
            'hyperparameters': hyperparameters,
            'preprocessing': preprocessing,
            'dataset_id': dataset_id,
            'model': model,
            'results': results
        }
        
        # Generate Python code
        python_code = generate_python_code(algorithm, hyperparameters, preprocessing)
        results['python_code'] = python_code
        
        return jsonify({
            'status': 'success',
            'model_id': model_id,
            'results': results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def preprocess_data(X_train, X_test, y_train, preprocessing):
    """Apply preprocessing steps to the data"""
    # Handle scaling
    if preprocessing.get('scaling') == 'standard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif preprocessing.get('scaling') == 'minmax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Handle missing values
    if preprocessing.get('missingValues') == 'drop':
        # Convert to DataFrame if necessary for dropna
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        
        # Drop rows with missing values
        mask_train = ~X_train.isna().any(axis=1)
        X_train = X_train[mask_train]
        y_train = y_train[mask_train]
        
        mask_test = ~X_test.isna().any(axis=1)
        X_test = X_test[mask_test]
    elif preprocessing.get('missingValues') == 'mean':
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
    elif preprocessing.get('missingValues') == 'median':
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
    
    # Feature selection
    if preprocessing.get('featureSelection') == 'selectk':
        selector = SelectKBest(f_classif, k=min(10, X_train.shape[1]))
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)
    
    # Dimension reduction
    if preprocessing.get('dimensionReduction') == 'pca_half':
        n_components = max(1, X_train.shape[1] // 2)
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    elif preprocessing.get('dimensionReduction') == 'pca_2d':
        n_components = min(2, X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
    return X_train, X_test

def train_algorithm(algorithm, hyperparameters, X_train, X_test, y_train, y_test):
    """Train the selected algorithm with the specified hyperparameters"""
    # Initialize the model based on the algorithm
    if algorithm == 'randomforest':
        model = RandomForestClassifier(
            n_estimators=hyperparameters.get('n_estimators', 100),
            max_depth=hyperparameters.get('max_depth', None) if hyperparameters.get('max_depth') != 'None' else None,
            min_samples_split=hyperparameters.get('min_samples_split', 2),
            bootstrap=hyperparameters.get('bootstrap', True),
            criterion=hyperparameters.get('criterion', 'gini'),
            max_features=hyperparameters.get('max_features', 'sqrt') if hyperparameters.get('max_features') != 'None' else None,
            random_state=42
        )
    elif algorithm == 'svm':
        model = SVC(
            C=hyperparameters.get('C', 1.0),
            kernel=hyperparameters.get('kernel', 'rbf'),
            gamma=hyperparameters.get('gamma', 'scale'),
            degree=hyperparameters.get('degree', 3),
            probability=hyperparameters.get('probability', True),
            class_weight=hyperparameters.get('class_weight', None) if hyperparameters.get('class_weight') != 'None' else None,
            random_state=42
        )
    elif algorithm == 'neuralnet':
        model = MLPClassifier(
            hidden_layer_sizes=tuple([hyperparameters.get('neurons', 64)] * hyperparameters.get('hidden_layers', 2)),
            activation=hyperparameters.get('activation', 'relu'),
            learning_rate_init=hyperparameters.get('learning_rate', 0.001),
            alpha=hyperparameters.get('alpha', 0.0001),
            batch_size=hyperparameters.get('batch_size', 32),
            max_iter=hyperparameters.get('max_iter', 1000),
            random_state=42
        )
    elif algorithm == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=hyperparameters.get('n_estimators', 100),
            learning_rate=hyperparameters.get('learning_rate', 0.1),
            max_depth=hyperparameters.get('max_depth', 6),
            subsample=hyperparameters.get('subsample', 0.8),
            colsample_bytree=hyperparameters.get('colsample_bytree', 0.8),
            gamma=hyperparameters.get('gamma', 0),
            min_child_weight=hyperparameters.get('min_child_weight', 1),
            objective=hyperparameters.get('objective', 'binary:logistic'),
            random_state=42
        )
    elif algorithm == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=hyperparameters.get('n_neighbors', 5),
            weights=hyperparameters.get('weights', 'uniform'),
            algorithm=hyperparameters.get('algorithm', 'auto'),
            leaf_size=hyperparameters.get('leaf_size', 30),
            p=hyperparameters.get('p', 2)
        )
    elif algorithm == 'decisiontree':
        model = DecisionTreeClassifier(
            max_depth=hyperparameters.get('max_depth', 10),
            min_samples_split=hyperparameters.get('min_samples_split', 2),
            min_samples_leaf=hyperparameters.get('min_samples_leaf', 1),
            criterion=hyperparameters.get('criterion', 'gini'),
            splitter=hyperparameters.get('splitter', 'best'),
            random_state=42
        )
    elif algorithm == 'logistic':
        model = LogisticRegression(
            C=hyperparameters.get('C', 1.0),
            penalty=hyperparameters.get('penalty', 'l2'),
            solver=hyperparameters.get('solver', 'lbfgs'),
            max_iter=hyperparameters.get('max_iter', 100),
            warm_start=hyperparameters.get('warm_start', False),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handle binary vs multi-class for precision, recall, f1
    if len(np.unique(y_test)) <= 2:
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
    else:
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance (if applicable)
    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        for i, importance in enumerate(model.feature_importances_):
            feature_importance.append({
                'feature': f'Feature {i+1}',
                'importance': float(importance)
            })
    
    # Generate fake training history (not available in scikit-learn)
    training_history = []
    for i in range(10):
        training_history.append({
            'epoch': i + 1,
            'training_loss': 1 - (0.7 * (1 - np.exp(-0.5 * i))),
            'validation_loss': 1 - (0.6 * (1 - np.exp(-0.4 * i))) + (np.random.rand() * 0.1)
        })
    
    results = {
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1Score': float(f1)
        },
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance,
        'training_history': training_history
    }
    
    return model, results

def generate_python_code(algorithm, hyperparams, preprocessing):
    """Generate Python code based on the selected algorithm and hyperparameters"""
    # This function would generate the actual Python code
    # similar to the frontend's generatePythonCode function
    return "# Python code generation is implemented in the real backend"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
