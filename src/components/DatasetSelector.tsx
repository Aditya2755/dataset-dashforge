
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Upload, Database, ChartBar, Check } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

interface DatasetSelectorProps {
  onSelectDataset: (dataset: string) => void;
}

const BUILT_IN_DATASETS = [
  { id: "iris", name: "Iris Flower", size: "150 samples", type: "Classification", description: "Classic dataset for classification tasks" },
  { id: "boston", name: "Boston Housing", size: "506 samples", type: "Regression", description: "Housing price prediction dataset" },
  { id: "mnist", name: "MNIST Digits", size: "70,000 samples", type: "Image Classification", description: "Handwritten digit recognition" },
  { id: "wine", name: "Wine Quality", size: "1,599 samples", type: "Classification", description: "Wine quality based on chemical properties" },
];

const DatasetSelector: React.FC<DatasetSelectorProps> = ({ onSelectDataset }) => {
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const { toast } = useToast();

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setSelectedDataset('custom');
      toast({
        title: "Dataset Uploaded",
        description: `Successfully uploaded ${file.name}`,
      });
    }
  };

  const handleBuiltInSelect = (datasetId: string) => {
    setSelectedDataset(datasetId);
    onSelectDataset(datasetId);
    toast({
      title: "Dataset Selected",
      description: `Selected ${BUILT_IN_DATASETS.find(d => d.id === datasetId)?.name} dataset`,
    });
  };

  const handleConfirmSelection = () => {
    if (selectedDataset === 'custom' && uploadedFile) {
      onSelectDataset('custom');
    } else if (selectedDataset) {
      onSelectDataset(selectedDataset);
    }
  };

  return (
    <Card className="w-full animate-fade-in">
      <CardHeader>
        <CardTitle className="text-xl font-bold">Select Dataset</CardTitle>
        <CardDescription>Upload your own dataset or choose from our pre-built collection</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="built-in">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="built-in">Built-in Datasets</TabsTrigger>
            <TabsTrigger value="upload">Upload Dataset</TabsTrigger>
          </TabsList>
          
          <TabsContent value="built-in" className="mt-4 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {BUILT_IN_DATASETS.map((dataset) => (
                <Card 
                  key={dataset.id}
                  className={`dataset-card cursor-pointer ${selectedDataset === dataset.id ? 'ring-2 ring-primary' : ''}`}
                  onClick={() => handleBuiltInSelect(dataset.id)}
                >
                  <CardHeader className="pb-2">
                    <div className="flex justify-between items-start">
                      <CardTitle className="text-lg font-medium">{dataset.name}</CardTitle>
                      {selectedDataset === dataset.id && (
                        <div className="h-5 w-5 rounded-full bg-primary flex items-center justify-center">
                          <Check className="h-3 w-3 text-white" />
                        </div>
                      )}
                    </div>
                    <CardDescription className="text-xs">{dataset.type} • {dataset.size}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">{dataset.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="upload" className="mt-4">
            <div className="border-2 border-dashed rounded-lg p-8 text-center">
              <div className="flex flex-col items-center">
                <Upload className="h-10 w-10 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">Upload Dataset</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  CSV, JSON or Excel files supported
                </p>
                
                <div className="w-full max-w-sm">
                  <Label htmlFor="dataset-file" className="sr-only">
                    Choose file
                  </Label>
                  <input
                    id="dataset-file"
                    type="file"
                    className="hidden"
                    accept=".csv,.json,.xlsx,.xls"
                    onChange={handleFileUpload}
                  />
                  <div className="flex flex-col gap-2">
                    <Button asChild variant="outline" className="w-full">
                      <label htmlFor="dataset-file">Choose file</label>
                    </Button>
                    {uploadedFile && (
                      <p className="text-sm text-center text-muted-foreground">
                        Selected: {uploadedFile.name}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="flex justify-end">
        <Button 
          onClick={handleConfirmSelection}
          disabled={!selectedDataset}
          className="gap-2"
        >
          <Database className="h-4 w-4" />
          Use Selected Dataset
        </Button>
      </CardFooter>
    </Card>
  );
};

export default DatasetSelector;
