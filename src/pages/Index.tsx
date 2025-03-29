
import React, { useState } from 'react';
import Navbar from '@/components/Navbar';
import DatasetSelector from '@/components/DatasetSelector';
import ModelTrainer from '@/components/ModelTrainer';
import ResultsDashboard from '@/components/ResultsDashboard';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const Index = () => {
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [modelResults, setModelResults] = useState<any | null>(null);
  const [step, setStep] = useState<'dataset' | 'train' | 'results'>('dataset');

  const handleDatasetSelect = (datasetId: string) => {
    setSelectedDataset(datasetId);
    setStep('train');
  };

  const handleTrainComplete = (results: any) => {
    setModelResults(results);
    setStep('results');
  };

  const renderStepContent = () => {
    switch (step) {
      case 'dataset':
        return <DatasetSelector onSelectDataset={handleDatasetSelect} />;
      case 'train':
        return selectedDataset ? 
          <ModelTrainer datasetId={selectedDataset} onTrainComplete={handleTrainComplete} /> : 
          null;
      case 'results':
        return modelResults ? 
          <ResultsDashboard results={modelResults} /> : 
          null;
      default:
        return <DatasetSelector onSelectDataset={handleDatasetSelect} />;
    }
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navbar />
      
      <div className="flex-1 container mx-auto p-6">
        <header className="mb-8 text-center mt-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            AI Model Experimentation Platform
          </h1>
          <p className="text-muted-foreground mt-2 max-w-2xl mx-auto">
            Upload datasets, train models with various algorithms, and analyze results with interactive visualizations
          </p>
        </header>
        
        <Card className="mb-6">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex space-x-4">
                <div className={`flex items-center ${step === 'dataset' ? 'text-primary' : 'text-muted-foreground'}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center mr-2 ${step === 'dataset' ? 'bg-primary text-white' : 'bg-muted'}`}>1</div>
                  <span>Select Dataset</span>
                </div>
                <div className="w-8 h-px bg-border self-center"></div>
                <div className={`flex items-center ${step === 'train' ? 'text-primary' : 'text-muted-foreground'}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center mr-2 ${step === 'train' ? 'bg-primary text-white' : 'bg-muted'}`}>2</div>
                  <span>Configure & Train</span>
                </div>
                <div className="w-8 h-px bg-border self-center"></div>
                <div className={`flex items-center ${step === 'results' ? 'text-primary' : 'text-muted-foreground'}`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center mr-2 ${step === 'results' ? 'bg-primary text-white' : 'bg-muted'}`}>3</div>
                  <span>Analyze Results</span>
                </div>
              </div>
              
              {step !== 'dataset' && (
                <button 
                  onClick={() => setStep('dataset')}
                  className="text-sm text-muted-foreground hover:text-primary transition-colors"
                >
                  Start Over
                </button>
              )}
            </div>
          </CardContent>
        </Card>
        
        <div className="mb-8">
          {renderStepContent()}
        </div>
      </div>
      
      <footer className="border-t py-6 bg-card">
        <div className="container mx-auto px-6 text-center text-sm text-muted-foreground">
          <p>© 2023 DatasetDashforge. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
