
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChartBar, PlayCircle, Sliders, Settings, Activity, Database } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { DataPreprocessor } from "@/components/DataPreprocessor";

interface ModelTrainerProps {
  datasetId: string;
  onTrainComplete: (results: any) => void;
}

interface Algorithm {
  id: string;
  name: string;
  type: string;
  description: string;
  hyperparameters: Hyperparameter[];
}

interface Hyperparameter {
  id: string;
  name: string;
  type: "range" | "select" | "switch";
  min?: number;
  max?: number;
  step?: number;
  default: any;
  options?: string[];
}

const ALGORITHMS: Algorithm[] = [
  {
    id: "randomforest",
    name: "Random Forest",
    type: "Classification/Regression",
    description: "Ensemble learning method using multiple decision trees",
    hyperparameters: [
      { id: "n_estimators", name: "Number of Trees", type: "range", min: 10, max: 500, step: 10, default: 100 },
      { id: "max_depth", name: "Max Depth", type: "range", min: 1, max: 30, step: 1, default: 10 },
      { id: "min_samples_split", name: "Min Samples to Split", type: "range", min: 2, max: 20, step: 1, default: 2 },
      { id: "bootstrap", name: "Bootstrap Samples", type: "switch", default: true },
      { id: "criterion", name: "Split Criterion", type: "select", default: "gini", options: ["gini", "entropy"] },
      { id: "max_features", name: "Max Features", type: "select", default: "sqrt", options: ["sqrt", "log2", "None"] },
    ]
  },
  {
    id: "svm",
    name: "Support Vector Machine",
    type: "Classification/Regression",
    description: "Effective in high dimensional spaces",
    hyperparameters: [
      { id: "C", name: "Regularization Parameter", type: "range", min: 0.1, max: 10, step: 0.1, default: 1.0 },
      { id: "kernel", name: "Kernel", type: "select", default: "rbf", options: ["linear", "poly", "rbf", "sigmoid"] },
      { id: "gamma", name: "Kernel Coefficient", type: "range", min: 0.001, max: 1, step: 0.001, default: 0.1 },
      { id: "degree", name: "Polynomial Degree", type: "range", min: 1, max: 10, step: 1, default: 3 },
      { id: "probability", name: "Enable Probability", type: "switch", default: true },
      { id: "class_weight", name: "Class Weight", type: "select", default: "None", options: ["None", "balanced"] },
    ]
  },
  {
    id: "neuralnet",
    name: "Neural Network",
    type: "Classification/Regression",
    description: "Deep learning with multiple layers for complex patterns",
    hyperparameters: [
      { id: "hidden_layers", name: "Hidden Layers", type: "range", min: 1, max: 5, step: 1, default: 2 },
      { id: "neurons", name: "Neurons per Layer", type: "range", min: 8, max: 256, step: 8, default: 64 },
      { id: "activation", name: "Activation Function", type: "select", default: "relu", options: ["relu", "tanh", "sigmoid"] },
      { id: "learning_rate", name: "Learning Rate", type: "range", min: 0.0001, max: 0.1, step: 0.0001, default: 0.001 },
      { id: "dropout", name: "Dropout Rate", type: "range", min: 0, max: 0.5, step: 0.1, default: 0.2 },
      { id: "batch_size", name: "Batch Size", type: "range", min: 8, max: 256, step: 8, default: 32 },
      { id: "alpha", name: "Regularization Alpha", type: "range", min: 0.0001, max: 0.1, step: 0.0001, default: 0.0001 },
      { id: "max_iter", name: "Max Iterations", type: "range", min: 100, max: 2000, step: 100, default: 1000 },
    ]
  },
  {
    id: "xgboost",
    name: "XGBoost",
    type: "Classification/Regression",
    description: "Gradient boosting optimized for speed and performance",
    hyperparameters: [
      { id: "n_estimators", name: "Number of Estimators", type: "range", min: 50, max: 1000, step: 50, default: 100 },
      { id: "learning_rate", name: "Learning Rate", type: "range", min: 0.01, max: 0.3, step: 0.01, default: 0.1 },
      { id: "max_depth", name: "Max Depth", type: "range", min: 3, max: 10, step: 1, default: 6 },
      { id: "subsample", name: "Subsample Ratio", type: "range", min: 0.5, max: 1, step: 0.1, default: 0.8 },
      { id: "colsample_bytree", name: "Column Sample by Tree", type: "range", min: 0.5, max: 1, step: 0.1, default: 0.8 },
      { id: "gamma", name: "Minimum Loss Reduction", type: "range", min: 0, max: 1, step: 0.1, default: 0 },
      { id: "min_child_weight", name: "Min Child Weight", type: "range", min: 1, max: 10, step: 1, default: 1 },
      { id: "objective", name: "Objective", type: "select", default: "binary:logistic", options: ["binary:logistic", "multi:softmax", "reg:squarederror"] },
    ]
  },
  {
    id: "knn",
    name: "K-Nearest Neighbors",
    type: "Classification/Regression",
    description: "Simple and effective instance-based learning algorithm",
    hyperparameters: [
      { id: "n_neighbors", name: "Number of Neighbors", type: "range", min: 1, max: 30, step: 1, default: 5 },
      { id: "weights", name: "Weight Function", type: "select", default: "uniform", options: ["uniform", "distance"] },
      { id: "algorithm", name: "Algorithm", type: "select", default: "auto", options: ["auto", "ball_tree", "kd_tree", "brute"] },
      { id: "leaf_size", name: "Leaf Size", type: "range", min: 10, max: 100, step: 10, default: 30 },
      { id: "p", name: "Power Parameter", type: "range", min: 1, max: 5, step: 1, default: 2 },
    ]
  },
  {
    id: "decisiontree",
    name: "Decision Tree",
    type: "Classification/Regression",
    description: "Simple tree-based model with good interpretability",
    hyperparameters: [
      { id: "max_depth", name: "Max Depth", type: "range", min: 1, max: 30, step: 1, default: 10 },
      { id: "min_samples_split", name: "Min Samples to Split", type: "range", min: 2, max: 20, step: 1, default: 2 },
      { id: "min_samples_leaf", name: "Min Samples per Leaf", type: "range", min: 1, max: 20, step: 1, default: 1 },
      { id: "criterion", name: "Split Criterion", type: "select", default: "gini", options: ["gini", "entropy"] },
      { id: "splitter", name: "Splitter", type: "select", default: "best", options: ["best", "random"] },
    ]
  },
  {
    id: "logistic",
    name: "Logistic Regression",
    type: "Classification",
    description: "Simple and interpretable linear model for classification",
    hyperparameters: [
      { id: "C", name: "Regularization Strength", type: "range", min: 0.1, max: 10, step: 0.1, default: 1.0 },
      { id: "penalty", name: "Penalty", type: "select", default: "l2", options: ["l1", "l2", "elasticnet", "none"] },
      { id: "solver", name: "Solver", type: "select", default: "lbfgs", options: ["newton-cg", "lbfgs", "liblinear", "sag", "saga"] },
      { id: "max_iter", name: "Max Iterations", type: "range", min: 100, max: 1000, step: 100, default: 100 },
      { id: "warm_start", name: "Warm Start", type: "switch", default: false },
    ]
  },
];

const ModelTrainer: React.FC<ModelTrainerProps> = ({ datasetId, onTrainComplete }) => {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string | null>(null);
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeTab, setActiveTab] = useState<'algorithm' | 'hyperparameters' | 'preprocessing'>('algorithm');
  const [preprocessingOptions, setPreprocessingOptions] = useState({
    scaling: 'none',
    missingValues: 'drop',
    featureSelection: 'none',
    dimensionReduction: 'none'
  });
  const { toast } = useToast();

  const handleAlgorithmSelect = (algorithmId: string) => {
    setSelectedAlgorithm(algorithmId);
    
    // Initialize hyperparameters with default values
    const algorithm = ALGORITHMS.find(a => a.id === algorithmId);
    if (algorithm) {
      const defaultParams = algorithm.hyperparameters.reduce((acc, param) => {
        acc[param.id] = param.default;
        return acc;
      }, {} as Record<string, any>);
      
      setHyperparameters(defaultParams);
    }
  };
  
  const handleHyperparameterChange = (parameterId: string, value: any) => {
    setHyperparameters(prev => ({
      ...prev,
      [parameterId]: value
    }));
  };

  const handlePreprocessingOptionChange = (option: string, value: string) => {
    setPreprocessingOptions(prev => ({
      ...prev,
      [option]: value
    }));
  };
  
  const handleTrainStart = () => {
    if (!selectedAlgorithm) return;
    
    setIsTraining(true);
    setProgress(0);
    
    toast({
      title: "Training Started",
      description: `Training model using ${ALGORITHMS.find(a => a.id === selectedAlgorithm)?.name}`,
    });

    // Prepare data for API call
    const trainingData = {
      datasetId,
      algorithm: selectedAlgorithm,
      hyperparameters,
      preprocessing: preprocessingOptions,
    };

    // Mock API call to Python backend
    console.log('Sending training request to Python API:', trainingData);
    
    // Simulate training progress
    const interval = setInterval(() => {
      setProgress(prev => {
        const newProgress = prev + (Math.random() * 10);
        if (newProgress >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            setIsTraining(false);
            generateFakeResults();
          }, 500);
          return 100;
        }
        return newProgress;
      });
    }, 300);
  };
  
  const generateFakeResults = () => {
    // Generate more realistic fake results based on the algorithm and hyperparameters
    
    // Base accuracy varies by algorithm
    let baseAccuracy = 0.75;
    if (selectedAlgorithm === 'randomforest' || selectedAlgorithm === 'xgboost') {
      baseAccuracy = 0.85; // Random Forest and XGBoost typically perform well
    } else if (selectedAlgorithm === 'neuralnet') {
      baseAccuracy = 0.82; // Neural networks can be very good
    } else if (selectedAlgorithm === 'svm') {
      baseAccuracy = 0.80; // SVM often performs well
    } else if (selectedAlgorithm === 'knn') {
      baseAccuracy = 0.75; // KNN might be less accurate
    } else if (selectedAlgorithm === 'decisiontree') {
      baseAccuracy = 0.70; // Single decision trees are usually less accurate
    }
    
    // Hyperparameter influence on results
    let hyperparameterModifier = 0;
    
    if (selectedAlgorithm === 'randomforest' || selectedAlgorithm === 'xgboost') {
      // More trees generally improves performance to a point
      if (hyperparameters.n_estimators > 100) {
        hyperparameterModifier += 0.03;
      }
      // Too deep trees can overfit
      if (hyperparameters.max_depth > 15) {
        hyperparameterModifier -= 0.02;
      }
    } else if (selectedAlgorithm === 'neuralnet') {
      // More layers can help but may lead to overfitting
      if (hyperparameters.hidden_layers > 3) {
        hyperparameterModifier += 0.02;
      }
      // Better learning rate
      if (hyperparameters.learning_rate < 0.01) {
        hyperparameterModifier += 0.03;
      }
    }
    
    // Preprocessing influence on results
    if (preprocessingOptions.scaling !== 'none') {
      hyperparameterModifier += 0.02; // Scaling usually helps
    }
    if (preprocessingOptions.dimensionReduction !== 'none') {
      hyperparameterModifier += 0.01; // Dimension reduction can help
    }
    
    // Add some randomness
    const randomness = (Math.random() * 0.06) - 0.03;
    
    // Calculate final accuracy and related metrics
    let accuracy = Math.min(0.98, Math.max(0.5, baseAccuracy + hyperparameterModifier + randomness));
    let precision = accuracy - (Math.random() * 0.05);
    let recall = accuracy - (Math.random() * 0.07);
    let f1Score = 2 * (precision * recall) / (precision + recall);
    
    // Confusion matrix based on accuracy
    const totalSamples = 100;
    const truePositives = Math.floor((accuracy * totalSamples) / 2);
    const trueNegatives = Math.floor((accuracy * totalSamples) / 2);
    const falsePositives = Math.floor(((1 - accuracy) * totalSamples) / 2);
    const falseNegatives = totalSamples - truePositives - trueNegatives - falsePositives;
    
    // Generate training history (learning curve)
    const trainingHistory = Array.from({ length: 10 }, (_, i) => {
      // Create decreasing loss values that plateau
      const epoch = i + 1;
      const baseLoss = 1.0 - (0.8 * (1 - Math.exp(-0.4 * epoch)));
      const trainingLoss = baseLoss * (1 - (hyperparameterModifier / 2));
      // Validation loss is usually higher and more variable
      const validationLoss = trainingLoss * (1 + (Math.random() * 0.2)) + (Math.random() * 0.05);
      
      return {
        epoch,
        training_loss: parseFloat(trainingLoss.toFixed(4)),
        validation_loss: parseFloat(validationLoss.toFixed(4))
      };
    });
    
    // Feature importance based on algorithm
    const featureImportance = [];
    const featureCount = 8;
    
    // Create feature names based on dataset
    const featureNames = Array.from({ length: featureCount }, (_, i) => 
      datasetId.includes('iris') 
        ? ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'][i % 4] 
        : `Feature ${i+1}`
    );
    
    // Generate feature importance values
    let totalImportance = 0;
    const rawImportances = Array.from({ length: featureCount }, () => Math.random());
    const sumImportances = rawImportances.reduce((sum, val) => sum + val, 0);
    
    for (let i = 0; i < featureCount; i++) {
      const importance = rawImportances[i] / sumImportances;
      featureImportance.push({
        feature: featureNames[i],
        importance: parseFloat(importance.toFixed(4))
      });
    }
    
    featureImportance.sort((a, b) => b.importance - a.importance);
    
    const results = {
      metrics: {
        accuracy: parseFloat(accuracy.toFixed(4)),
        precision: parseFloat(precision.toFixed(4)),
        recall: parseFloat(recall.toFixed(4)),
        f1Score: parseFloat(f1Score.toFixed(4)),
      },
      confusion_matrix: [
        [trueNegatives, falsePositives],
        [falseNegatives, truePositives]
      ],
      feature_importance: featureImportance,
      training_history: trainingHistory,
      python_code: generatePythonCode(selectedAlgorithm, hyperparameters, preprocessingOptions)
    };
    
    toast({
      title: "Training Complete",
      description: `Model training finished with ${(accuracy * 100).toFixed(2)}% accuracy`,
    });
    
    onTrainComplete(results);
  };

  const generatePythonCode = (algorithmId: string | null, params: Record<string, any>, preprocessing: any) => {
    if (!algorithmId) return '';
    
    const algorithm = ALGORITHMS.find(a => a.id === algorithmId);
    if (!algorithm) return '';
    
    // Generate Python code based on the selected algorithm and hyperparameters
    let code = `
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
`;

    // Add preprocessing imports
    code += `
# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
`;

    // Add model-specific imports
    switch (algorithmId) {
      case 'randomforest':
        code += 'from sklearn.ensemble import RandomForestClassifier\n';
        break;
      case 'svm':
        code += 'from sklearn.svm import SVC\n';
        break;
      case 'neuralnet':
        code += 'from sklearn.neural_network import MLPClassifier\n';
        break;
      case 'xgboost':
        code += 'import xgboost as xgb\n';
        break;
      case 'knn':
        code += 'from sklearn.neighbors import KNeighborsClassifier\n';
        break;
      case 'decisiontree':
        code += 'from sklearn.tree import DecisionTreeClassifier\n';
        break;
      case 'logistic':
        code += 'from sklearn.linear_model import LogisticRegression\n';
        break;
    }
    
    // Add data loading
    code += `
# Load dataset
# Replace with your data loading code
data = pd.read_csv('your_dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
`;

    // Add preprocessing steps
    code += `
# Preprocessing steps
`;
    
    if (preprocessing.scaling !== 'none') {
      if (preprocessing.scaling === 'standard') {
        code += `
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)`;
      } else if (preprocessing.scaling === 'minmax') {
        code += `
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)`;
      }
    }
    
    if (preprocessing.missingValues !== 'none') {
      if (preprocessing.missingValues === 'drop') {
        code += `
# Drop rows with missing values
X_train = pd.DataFrame(X_train).dropna()
y_train = y_train[X_train.index]
X_test = pd.DataFrame(X_test).dropna()
y_test = y_test[X_test.index]`;
      } else if (preprocessing.missingValues === 'mean') {
        code += `
# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)`;
      } else if (preprocessing.missingValues === 'median') {
        code += `
# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)`;
      }
    }
    
    if (preprocessing.featureSelection !== 'none') {
      code += `
# Feature selection
selector = SelectKBest(f_classif, k=10)  # Select top 10 features
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)`;
    }
    
    if (preprocessing.dimensionReduction !== 'none') {
      const n_components = preprocessing.dimensionReduction === 'pca_half' ? 'X_train.shape[1] // 2' : '2';
      code += `
# Dimension reduction with PCA
pca = PCA(n_components=${n_components})
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)`;
    }
    
    // Add model creation and hyperparameters
    code += `
# Model definition
`;
    
    switch (algorithmId) {
      case 'randomforest':
        code += `model = RandomForestClassifier(
    n_estimators=${params.n_estimators},
    max_depth=${params.max_depth !== 'None' ? params.max_depth : 'None'},
    min_samples_split=${params.min_samples_split},
    bootstrap=${params.bootstrap},
    criterion='${params.criterion}',
    max_features='${params.max_features === 'None' ? 'None' : params.max_features}',
    random_state=42
)`;
        break;
      case 'svm':
        code += `model = SVC(
    C=${params.C},
    kernel='${params.kernel}',
    gamma=${params.gamma},
    degree=${params.degree},
    probability=${params.probability},
    class_weight=${params.class_weight === 'None' ? 'None' : "'" + params.class_weight + "'"},
    random_state=42
)`;
        break;
      case 'neuralnet':
        code += `model = MLPClassifier(
    hidden_layer_sizes=(${Array(params.hidden_layers).fill(params.neurons).join(', ')}),
    activation='${params.activation}',
    learning_rate_init=${params.learning_rate},
    alpha=${params.alpha},
    batch_size=${params.batch_size},
    max_iter=${params.max_iter},
    random_state=42
)`;
        break;
      case 'xgboost':
        code += `model = xgb.XGBClassifier(
    n_estimators=${params.n_estimators},
    learning_rate=${params.learning_rate},
    max_depth=${params.max_depth},
    subsample=${params.subsample},
    colsample_bytree=${params.colsample_bytree},
    gamma=${params.gamma},
    min_child_weight=${params.min_child_weight},
    objective='${params.objective}',
    random_state=42
)`;
        break;
      case 'knn':
        code += `model = KNeighborsClassifier(
    n_neighbors=${params.n_neighbors},
    weights='${params.weights}',
    algorithm='${params.algorithm}',
    leaf_size=${params.leaf_size},
    p=${params.p}
)`;
        break;
      case 'decisiontree':
        code += `model = DecisionTreeClassifier(
    max_depth=${params.max_depth},
    min_samples_split=${params.min_samples_split},
    min_samples_leaf=${params.min_samples_leaf},
    criterion='${params.criterion}',
    splitter='${params.splitter}',
    random_state=42
)`;
        break;
      case 'logistic':
        code += `model = LogisticRegression(
    C=${params.C},
    penalty='${params.penalty}',
    solver='${params.solver}',
    max_iter=${params.max_iter},
    warm_start=${params.warm_start},
    random_state=42
)`;
        break;
    }
    
    // Add training, evaluation, and visualization
    code += `

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Feature importance (if applicable)
`;

    if (['randomforest', 'xgboost', 'decisiontree'].includes(algorithmId)) {
      code += `
# Plot feature importance
import matplotlib.pyplot as plt

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    features = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])]
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
`;
    }
    
    return code;
  };
  
  const selectedAlgorithmData = ALGORITHMS.find(a => a.id === selectedAlgorithm);
  
  return (
    <Card className="w-full animate-fade-in">
      <CardHeader>
        <CardTitle className="text-xl font-bold">Train Model</CardTitle>
        <CardDescription>Select an algorithm, configure preprocessing and hyperparameters</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as any)}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="algorithm">
              <ChartBar className="w-4 h-4 mr-2" />
              Algorithm
            </TabsTrigger>
            <TabsTrigger value="preprocessing" disabled={!selectedAlgorithm}>
              <Database className="w-4 h-4 mr-2" />
              Preprocessing
            </TabsTrigger>
            <TabsTrigger value="hyperparameters" disabled={!selectedAlgorithm}>
              <Sliders className="w-4 h-4 mr-2" />
              Hyperparameters
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="algorithm" className="space-y-4 mt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {ALGORITHMS.map((algorithm) => (
                <Card 
                  key={algorithm.id}
                  className={`dataset-card cursor-pointer h-full ${selectedAlgorithm === algorithm.id ? 'ring-2 ring-primary' : ''}`}
                  onClick={() => handleAlgorithmSelect(algorithm.id)}
                >
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg font-medium">{algorithm.name}</CardTitle>
                    <CardDescription className="text-xs">{algorithm.type}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{algorithm.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="preprocessing" className="space-y-6 mt-4">
            <DataPreprocessor 
              options={preprocessingOptions}
              onOptionChange={handlePreprocessingOptionChange}
            />
          </TabsContent>
          
          <TabsContent value="hyperparameters" className="space-y-6 mt-4">
            {selectedAlgorithmData?.hyperparameters.map((param) => (
              <div key={param.id} className="space-y-2">
                <div className="flex justify-between">
                  <Label htmlFor={param.id} className="text-sm font-medium">
                    {param.name}
                  </Label>
                  {param.type !== "switch" && (
                    <span className="text-sm text-muted-foreground">
                      {hyperparameters[param.id]}
                    </span>
                  )}
                </div>
                
                {param.type === "range" && (
                  <Slider
                    id={param.id}
                    min={param.min}
                    max={param.max}
                    step={param.step}
                    value={[hyperparameters[param.id]]}
                    onValueChange={(value) => handleHyperparameterChange(param.id, value[0])}
                    className="parameter-slider"
                  />
                )}
                
                {param.type === "select" && (
                  <Select
                    value={hyperparameters[param.id]}
                    onValueChange={(value) => handleHyperparameterChange(param.id, value)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select option" />
                    </SelectTrigger>
                    <SelectContent>
                      {param.options?.map((option) => (
                        <SelectItem key={option} value={option}>
                          {option}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
                
                {param.type === "switch" && (
                  <div className="flex items-center space-x-2">
                    <Switch
                      id={param.id}
                      checked={hyperparameters[param.id]}
                      onCheckedChange={(checked) => handleHyperparameterChange(param.id, checked)}
                    />
                    <Label htmlFor={param.id}>
                      {hyperparameters[param.id] ? "Enabled" : "Disabled"}
                    </Label>
                  </div>
                )}
              </div>
            ))}
          </TabsContent>
        </Tabs>
        
        {isTraining && (
          <div className="mt-8 space-y-2">
            <div className="flex justify-between">
              <Label className="text-sm">Training Progress</Label>
              <span className="text-sm text-muted-foreground">{Math.floor(progress)}%</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2.5">
              <div 
                className="bg-primary h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <p className="text-sm text-muted-foreground animate-pulse-subtle">
              Training model on {datasetId} dataset...
            </p>
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-end">
        <Button
          onClick={handleTrainStart}
          disabled={!selectedAlgorithm || isTraining}
          className="gap-2"
        >
          {isTraining ? (
            <Activity className="h-4 w-4 animate-pulse" />
          ) : (
            <PlayCircle className="h-4 w-4" />
          )}
          {isTraining ? "Training..." : "Train Model"}
        </Button>
      </CardFooter>
    </Card>
  );
};

export default ModelTrainer;
