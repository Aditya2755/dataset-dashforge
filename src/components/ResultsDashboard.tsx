
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import { ChartBar, LineChart as LineChartIcon, FileText, Download } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

interface ResultsDashboardProps {
  results: {
    metrics: {
      accuracy: number;
      precision: number;
      recall: number;
      f1Score: number;
    };
    confusion_matrix: number[][];
    feature_importance: { feature: string; importance: number }[];
    training_history: { epoch: number; training_loss: number; validation_loss: number }[];
    python_code?: string;
  };
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ results }) => {
  const [activeTab, setActiveTab] = useState<string>("metrics");
  const { toast } = useToast();

  const metricsData = [
    { name: 'Accuracy', value: results.metrics.accuracy * 100 },
    { name: 'Precision', value: results.metrics.precision * 100 },
    { name: 'Recall', value: results.metrics.recall * 100 },
    { name: 'F1 Score', value: results.metrics.f1Score * 100 }
  ];

  const confusionMatrix = results.confusion_matrix;
  
  const handleDownloadCode = () => {
    const element = document.createElement('a');
    const file = new Blob([results.python_code || ''], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = 'model_training_code.py';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
    
    toast({
      title: "Code Downloaded",
      description: "Python code has been downloaded successfully",
    });
  };

  return (
    <Card className="w-full animate-fade-in">
      <CardHeader>
        <CardTitle className="text-xl font-bold">Model Training Results</CardTitle>
        <CardDescription>Analyze performance metrics and visualizations</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="metrics">
              <ChartBar className="w-4 h-4 mr-2" />
              Metrics
            </TabsTrigger>
            <TabsTrigger value="history">
              <LineChartIcon className="w-4 h-4 mr-2" />
              Training
            </TabsTrigger>
            <TabsTrigger value="features">
              <ChartBar className="w-4 h-4 mr-2" />
              Features
            </TabsTrigger>
            <TabsTrigger value="code">
              <FileText className="w-4 h-4 mr-2" />
              Python Code
            </TabsTrigger>
          </TabsList>

          <TabsContent value="metrics" className="space-y-4 mt-4">
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Performance Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={metricsData}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                        <Bar dataKey="value" fill="#8884d8" name="Value (%)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Confusion Matrix</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-center h-64">
                    <table className="border-collapse">
                      <thead>
                        <tr>
                          <th className="p-2 border"></th>
                          <th className="p-2 border bg-muted">Predicted 0</th>
                          <th className="p-2 border bg-muted">Predicted 1</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <th className="p-2 border bg-muted">Actual 0</th>
                          <td className="p-2 border text-center bg-green-100">{confusionMatrix[0][0]}</td>
                          <td className="p-2 border text-center bg-red-100">{confusionMatrix[0][1]}</td>
                        </tr>
                        <tr>
                          <th className="p-2 border bg-muted">Actual 1</th>
                          <td className="p-2 border text-center bg-red-100">{confusionMatrix[1][0]}</td>
                          <td className="p-2 border text-center bg-green-100">{confusionMatrix[1][1]}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="history" className="space-y-4 mt-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Training History</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={results.training_history}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="training_loss" stroke="#8884d8" name="Training Loss" />
                      <Line type="monotone" dataKey="validation_loss" stroke="#82ca9d" name="Validation Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="features" className="space-y-4 mt-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Feature Importance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={results.feature_importance.slice(0, 10)}
                      layout="vertical"
                      margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="feature" width={80} />
                      <Tooltip />
                      <Bar dataKey="importance" fill="#82ca9d" name="Importance" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="code" className="space-y-4 mt-4">
            <Card>
              <CardHeader className="pb-2 flex flex-row items-center justify-between">
                <CardTitle className="text-lg">Python Code</CardTitle>
                <Button size="sm" onClick={handleDownloadCode} className="gap-2">
                  <Download className="w-4 h-4" />
                  Download
                </Button>
              </CardHeader>
              <CardContent>
                <div className="max-h-[600px] overflow-auto bg-muted rounded-md p-4">
                  <pre className="text-xs whitespace-pre-wrap font-mono">
                    {results.python_code || "# No Python code available for this model"}
                  </pre>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default ResultsDashboard;
