
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ChartPie, BarChart, LineChart, List, CheckSquare } from "lucide-react";
import { 
  BarChart as ReBarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  LineChart as ReLineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from 'recharts';

interface ResultsDashboardProps {
  results: any;
}

const COLORS = ['#8b5cf6', '#6366f1', '#3b82f6', '#0ea5e9'];
const CONFUSION_MATRIX_COLORS = ['#8B5CF6', '#C7D2FE'];

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ results }) => {
  if (!results) return null;
  
  const { metrics, confusion_matrix, feature_importance, training_history } = results;
  
  // Format confusion matrix for display
  const confusionMatrixData = [
    { name: 'True Negative', value: confusion_matrix[0][0] },
    { name: 'False Positive', value: confusion_matrix[0][1] },
    { name: 'False Negative', value: confusion_matrix[1][0] },
    { name: 'True Positive', value: confusion_matrix[1][1] },
  ];
  
  // Format metrics for display
  const metricsData = [
    { name: 'Accuracy', value: metrics.accuracy },
    { name: 'Precision', value: metrics.precision },
    { name: 'Recall', value: metrics.recall },
    { name: 'F1 Score', value: metrics.f1Score },
  ];
  
  return (
    <Card className="w-full animate-fade-in">
      <CardHeader>
        <CardTitle className="text-xl font-bold">Model Performance</CardTitle>
        <CardDescription>Detailed analysis of your trained model</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="overview">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">
              <ChartPie className="w-4 h-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="features">
              <BarChart className="w-4 h-4 mr-2" />
              Features
            </TabsTrigger>
            <TabsTrigger value="history">
              <LineChart className="w-4 h-4 mr-2" />
              History
            </TabsTrigger>
            <TabsTrigger value="predictions">
              <CheckSquare className="w-4 h-4 mr-2" />
              Predictions
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview" className="mt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="dashboard-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-md">Performance Metrics</CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="grid grid-cols-2 gap-4">
                    {metricsData.map((metric) => (
                      <div key={metric.name} className="space-y-1">
                        <p className="text-sm text-muted-foreground">{metric.name}</p>
                        <p className="metric-value">{(metric.value * 100).toFixed(1)}%</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
              
              <Card className="dashboard-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-md">Confusion Matrix</CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={confusionMatrixData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {confusionMatrixData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="features" className="mt-4">
            <Card className="dashboard-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-md">Feature Importance</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <ResponsiveContainer width="100%" height={300}>
                  <ReBarChart
                    data={feature_importance}
                    layout="vertical"
                    margin={{ top: 20, right: 30, left: 40, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 1]} />
                    <YAxis dataKey="feature" type="category" scale="band" width={80} />
                    <Tooltip formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Importance']} />
                    <Bar dataKey="importance" fill="#8b5cf6" barSize={20} radius={[0, 4, 4, 0]} />
                  </ReBarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="history" className="mt-4">
            <Card className="dashboard-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-md">Training History</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <ResponsiveContainer width="100%" height={300}>
                  <ReLineChart
                    data={training_history}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="training_loss" stroke="#8b5cf6" strokeWidth={2} activeDot={{ r: 8 }} />
                    <Line type="monotone" dataKey="validation_loss" stroke="#a855f7" strokeWidth={2} />
                  </ReLineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="predictions" className="mt-4">
            <Card className="dashboard-card">
              <CardHeader className="pb-2">
                <CardTitle className="text-md">Sample Predictions</CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="relative overflow-x-auto rounded-md border">
                  <table className="w-full text-sm text-left">
                    <thead className="text-xs uppercase bg-muted">
                      <tr>
                        <th scope="col" className="px-6 py-3">Input</th>
                        <th scope="col" className="px-6 py-3">Predicted</th>
                        <th scope="col" className="px-6 py-3">Actual</th>
                        <th scope="col" className="px-6 py-3">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Array.from({ length: 5 }).map((_, index) => {
                        // Generate some fake prediction data
                        const correct = Math.random() > 0.3;
                        const confidence = 0.7 + (Math.random() * 0.3);
                        return (
                          <tr key={index} className="bg-card border-b">
                            <td className="px-6 py-4">Sample {index + 1}</td>
                            <td className="px-6 py-4">Class {correct ? "A" : "B"}</td>
                            <td className="px-6 py-4">Class {correct ? "A" : "B"}</td>
                            <td className="px-6 py-4">
                              <div className="flex items-center">
                                <div className="w-full bg-muted rounded-full h-2.5 mr-2">
                                  <div className={`h-2.5 rounded-full ${correct ? 'bg-primary' : 'bg-orange-500'}`} style={{ width: `${confidence * 100}%` }}></div>
                                </div>
                                <span>{(confidence * 100).toFixed(0)}%</span>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
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
