
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend, PieChart, Pie, Cell } from 'recharts';
import { Button } from "@/components/ui/button";
import { Download, FileDown, Share2 } from "lucide-react";

interface ResultsDashboardProps {
  results: {
    metrics: {
      accuracy: number;
      precision: number;
      recall: number;
      f1Score: number;
    };
    confusion_matrix: number[][];
    feature_importance: Array<{
      feature: string;
      importance: number;
    }>;
    training_history: Array<{
      epoch: number;
      training_loss: number;
      validation_loss: number;
    }>;
  };
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ results }) => {
  const [activeTab, setActiveTab] = useState('overview');

  const formattedMetrics = [
    { name: 'Accuracy', value: results.metrics.accuracy },
    { name: 'Precision', value: results.metrics.precision },
    { name: 'Recall', value: results.metrics.recall },
    { name: 'F1 Score', value: results.metrics.f1Score },
  ];

  const COLORS = ['#4f46e5', '#06b6d4', '#10b981', '#f59e0b'];

  const confusionMatrix = [
    { name: 'True Negative', value: results.confusion_matrix[0][0] },
    { name: 'False Positive', value: results.confusion_matrix[0][1] },
    { name: 'False Negative', value: results.confusion_matrix[1][0] },
    { name: 'True Positive', value: results.confusion_matrix[1][1] },
  ];

  return (
    <Card className="w-full animate-fade-in">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-xl font-bold">Model Performance Results</CardTitle>
          <CardDescription>Detailed analysis of your trained model</CardDescription>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="gap-1">
            <FileDown className="h-4 w-4" />
            Export
          </Button>
          <Button variant="outline" size="sm" className="gap-1">
            <Share2 className="h-4 w-4" />
            Share
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="overview" className="w-full" onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-4 mb-6">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="metrics">Detailed Metrics</TabsTrigger>
            <TabsTrigger value="features">Feature Importance</TabsTrigger>
            <TabsTrigger value="training">Training History</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Performance Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={formattedMetrics}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                        <Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(2)}%`} />
                        <Bar dataKey="value" fill="#4f46e5" />
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
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={confusionMatrix}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {confusionMatrix.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => value} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="grid grid-cols-2 gap-2 mt-4 text-xs">
                    <div className="flex items-center">
                      <div className="w-3 h-3 mr-1" style={{ backgroundColor: COLORS[0] }}></div>
                      <span>True Negative: {results.confusion_matrix[0][0]}</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 mr-1" style={{ backgroundColor: COLORS[1] }}></div>
                      <span>False Positive: {results.confusion_matrix[0][1]}</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 mr-1" style={{ backgroundColor: COLORS[2] }}></div>
                      <span>False Negative: {results.confusion_matrix[1][0]}</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 mr-1" style={{ backgroundColor: COLORS[3] }}></div>
                      <span>True Positive: {results.confusion_matrix[1][1]}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Model Training History</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={results.training_history}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="training_loss" stroke="#4f46e5" name="Training Loss" />
                      <Line type="monotone" dataKey="validation_loss" stroke="#f59e0b" name="Validation Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="metrics" className="space-y-4">
            {/* Detailed metrics content */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {formattedMetrics.map((metric) => (
                <Card key={metric.name}>
                  <CardContent className="pt-6">
                    <div className="text-center">
                      <p className="text-sm font-medium text-muted-foreground mb-1">{metric.name}</p>
                      <h3 className="text-3xl font-bold">{(metric.value * 100).toFixed(2)}%</h3>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
            
            <Card>
              <CardHeader>
                <CardTitle>Confusion Matrix Details</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 border rounded overflow-hidden">
                  <div className="border bg-primary/10 p-4 text-center font-medium">
                    <p>True Negative</p>
                    <p className="text-2xl mt-2">{results.confusion_matrix[0][0]}</p>
                  </div>
                  <div className="border bg-destructive/10 p-4 text-center font-medium">
                    <p>False Positive</p>
                    <p className="text-2xl mt-2">{results.confusion_matrix[0][1]}</p>
                  </div>
                  <div className="border bg-destructive/10 p-4 text-center font-medium">
                    <p>False Negative</p>
                    <p className="text-2xl mt-2">{results.confusion_matrix[1][0]}</p>
                  </div>
                  <div className="border bg-primary/10 p-4 text-center font-medium">
                    <p>True Positive</p>
                    <p className="text-2xl mt-2">{results.confusion_matrix[1][1]}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="features" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Feature Importance</CardTitle>
                <CardDescription>The most influential features for model predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={results.feature_importance}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="feature" type="category" width={100} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="importance" fill="#10b981" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="training" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Training History</CardTitle>
                <CardDescription>Loss values throughout the training process</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={results.training_history}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="training_loss" stroke="#4f46e5" name="Training Loss" activeDot={{ r: 8 }} />
                      <Line type="monotone" dataKey="validation_loss" stroke="#f59e0b" name="Validation Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter className="flex justify-between">
        <p className="text-sm text-muted-foreground">
          Results generated {new Date().toLocaleDateString()}
        </p>
        <Button>
          <Download className="mr-2 h-4 w-4" />
          Download Full Report
        </Button>
      </CardFooter>
    </Card>
  );
};

export default ResultsDashboard;
