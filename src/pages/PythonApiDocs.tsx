import React from 'react';
import Navbar from '@/components/Navbar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileDown } from "lucide-react";
import { Link } from 'react-router-dom';

const PythonApiDocs = () => {
  const handleDownloadAPI = () => {
    const element = document.createElement('a');
    fetch('/src/python/api.py')
      .then(response => response.text())
      .then(text => {
        const file = new Blob([text], {type: 'text/plain'});
        element.href = URL.createObjectURL(file);
        element.download = 'api.py';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
      });
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navbar />
      
      <div className="container mx-auto p-6 flex-1">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold">Python API Documentation</h1>
          <div className="flex gap-4">
            <Button onClick={handleDownloadAPI} className="gap-2">
              <FileDown className="w-4 h-4" />
              Download API File
            </Button>
            <Link to="/">
              <Button variant="outline">Back to Dashboard</Button>
            </Link>
          </div>
        </div>
        
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Overview</CardTitle>
            <CardDescription>
              Python backend API for the AI Model Experimentation Platform using scikit-learn
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p>
              This documentation outlines the Python API that powers the backend of our AI Model Experimentation Platform.
              The API is built with Flask and integrates scikit-learn models for machine learning tasks.
            </p>

            <h3 className="text-lg font-semibold mt-4 mb-2">Requirements</h3>
            <pre className="bg-muted p-3 rounded-md text-sm mb-4">
              pip install flask flask-cors pandas numpy scikit-learn xgboost
            </pre>

            <h3 className="text-lg font-semibold mt-4 mb-2">Running the API</h3>
            <pre className="bg-muted p-3 rounded-md text-sm">
              python api.py
            </pre>
            <p className="text-sm text-muted-foreground mt-2">
              This will start the Flask server on http://localhost:5000
            </p>
          </CardContent>
        </Card>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Endpoints</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-2">GET /api/datasets</h3>
              <p className="mb-2">List all available datasets</p>
              <div className="bg-muted p-3 rounded-md text-sm">
                <p className="font-semibold">Response:</p>
                <pre>{`{
  "status": "success",
  "datasets": [
    { "id": "dataset_1", "name": "Iris Dataset" },
    { "id": "dataset_2", "name": "Wine Quality" }
  ]
}`}</pre>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-2">POST /api/datasets</h3>
              <p className="mb-2">Upload a new dataset (CSV or Excel file)</p>
              <div className="bg-muted p-3 rounded-md text-sm mb-3">
                <p className="font-semibold">Request:</p>
                <pre>{`// Form data
{
  "file": [binary file data],
  "name": "My Dataset"
}`}</pre>
              </div>
              <div className="bg-muted p-3 rounded-md text-sm">
                <p className="font-semibold">Response:</p>
                <pre>{`{
  "status": "success",
  "dataset_id": "dataset_3",
  "name": "My Dataset",
  "columns": ["feature1", "feature2", "target"],
  "rows": 150
}`}</pre>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-2">POST /api/train</h3>
              <p className="mb-2">Train a model with specified algorithm and hyperparameters</p>
              <div className="bg-muted p-3 rounded-md text-sm mb-3">
                <p className="font-semibold">Request:</p>
                <pre>{`{
  "datasetId": "dataset_1",
  "algorithm": "randomforest",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10
    // other parameters...
  },
  "preprocessing": {
    "scaling": "standard",
    "missingValues": "mean",
    "featureSelection": "none",
    "dimensionReduction": "none"
  },
  "targetColumn": "species"
}`}</pre>
              </div>
              <div className="bg-muted p-3 rounded-md text-sm">
                <p className="font-semibold">Response:</p>
                <pre>{`{
  "status": "success",
  "model_id": "model_1",
  "results": {
    "metrics": {
      "accuracy": 0.95,
      "precision": 0.94,
      "recall": 0.95,
      "f1Score": 0.945
    },
    "confusion_matrix": [[30, 2], [1, 27]],
    "feature_importance": [
      {"feature": "Feature 1", "importance": 0.4},
      {"feature": "Feature 2", "importance": 0.3}
    ],
    "training_history": [
      {"epoch": 1, "training_loss": 0.8, "validation_loss": 0.85}
    ],
    "python_code": "# Python code for model training"
  }
}`}</pre>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Supported Algorithms</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="border rounded-md p-4">
                <h3 className="font-semibold mb-2">Random Forest</h3>
                <p className="text-sm text-muted-foreground mb-2">Ensemble learning method using multiple decision trees</p>
                <p className="text-xs">Key hyperparameters: n_estimators, max_depth, min_samples_split</p>
              </div>
              
              <div className="border rounded-md p-4">
                <h3 className="font-semibold mb-2">Support Vector Machine</h3>
                <p className="text-sm text-muted-foreground mb-2">Effective in high dimensional spaces</p>
                <p className="text-xs">Key hyperparameters: C, kernel, gamma</p>
              </div>
              
              <div className="border rounded-md p-4">
                <h3 className="font-semibold mb-2">Neural Network</h3>
                <p className="text-sm text-muted-foreground mb-2">Deep learning with multiple layers for complex patterns</p>
                <p className="text-xs">Key hyperparameters: hidden_layers, neurons, activation, learning_rate</p>
              </div>
              
              <div className="border rounded-md p-4">
                <h3 className="font-semibold mb-2">XGBoost</h3>
                <p className="text-sm text-muted-foreground mb-2">Gradient boosting optimized for speed and performance</p>
                <p className="text-xs">Key hyperparameters: n_estimators, learning_rate, max_depth</p>
              </div>
              
              <div className="border rounded-md p-4">
                <h3 className="font-semibold mb-2">K-Nearest Neighbors</h3>
                <p className="text-sm text-muted-foreground mb-2">Simple and effective instance-based learning algorithm</p>
                <p className="text-xs">Key hyperparameters: n_neighbors, weights, algorithm</p>
              </div>
              
              <div className="border rounded-md p-4">
                <h3 className="font-semibold mb-2">Decision Tree</h3>
                <p className="text-sm text-muted-foreground mb-2">Simple tree-based model with good interpretability</p>
                <p className="text-xs">Key hyperparameters: max_depth, min_samples_split, criterion</p>
              </div>
              
              <div className="border rounded-md p-4">
                <h3 className="font-semibold mb-2">Logistic Regression</h3>
                <p className="text-sm text-muted-foreground mb-2">Simple and interpretable linear model for classification</p>
                <p className="text-xs">Key hyperparameters: C, penalty, solver</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      <footer className="border-t py-6 bg-card mt-8">
        <div className="container mx-auto px-6 text-center text-sm text-muted-foreground">
          <p>© 2023 DatasetDashforge. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default PythonApiDocs;
