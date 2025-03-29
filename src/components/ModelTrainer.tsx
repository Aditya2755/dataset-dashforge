
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChartBar, PlayCircle, Sliders, Settings, Activity } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

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
    ]
  }
];

const ModelTrainer: React.FC<ModelTrainerProps> = ({ datasetId, onTrainComplete }) => {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string | null>(null);
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
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
  
  const handleTrainStart = () => {
    if (!selectedAlgorithm) return;
    
    setIsTraining(true);
    setProgress(0);
    
    toast({
      title: "Training Started",
      description: `Training model using ${ALGORITHMS.find(a => a.id === selectedAlgorithm)?.name}`,
    });
    
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
    // Generate fake results based on the algorithm and hyperparameters
    const accuracy = 0.7 + (Math.random() * 0.25);
    const precision = 0.65 + (Math.random() * 0.3);
    const recall = 0.6 + (Math.random() * 0.35);
    const f1Score = 0.68 + (Math.random() * 0.28);
    
    const results = {
      metrics: {
        accuracy,
        precision,
        recall,
        f1Score,
      },
      confusion_matrix: [
        [Math.floor(Math.random() * 50) + 30, Math.floor(Math.random() * 15) + 5],
        [Math.floor(Math.random() * 15) + 5, Math.floor(Math.random() * 50) + 30]
      ],
      feature_importance: Array.from({ length: 6 }, () => Math.random()).map((v, i) => ({
        feature: `Feature ${i+1}`,
        importance: v
      })).sort((a, b) => b.importance - a.importance),
      training_history: Array.from({ length: 10 }, (_, i) => ({
        epoch: i + 1,
        training_loss: 1 - (0.7 * (1 - Math.exp(-0.5 * i))),
        validation_loss: 1 - (0.6 * (1 - Math.exp(-0.4 * i))) + (Math.random() * 0.1)
      }))
    };
    
    toast({
      title: "Training Complete",
      description: `Model training finished with ${(accuracy * 100).toFixed(2)}% accuracy`,
    });
    
    onTrainComplete(results);
  };
  
  const selectedAlgorithmData = ALGORITHMS.find(a => a.id === selectedAlgorithm);
  
  return (
    <Card className="w-full animate-fade-in">
      <CardHeader>
        <CardTitle className="text-xl font-bold">Train Model</CardTitle>
        <CardDescription>Select an algorithm and configure hyperparameters</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="algorithm">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="algorithm">
              <ChartBar className="w-4 h-4 mr-2" />
              Algorithm
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
