
import React from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface DataPreprocessorProps {
  options: {
    scaling: string;
    missingValues: string;
    featureSelection: string;
    dimensionReduction: string;
  };
  onOptionChange: (option: string, value: string) => void;
}

export const DataPreprocessor: React.FC<DataPreprocessorProps> = ({ 
  options, 
  onOptionChange 
}) => {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="scaling">Data Scaling</Label>
            <Select
              value={options.scaling}
              onValueChange={(value) => onOptionChange('scaling', value)}
            >
              <SelectTrigger id="scaling">
                <SelectValue placeholder="Select scaling method" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="standard">Standard Scaling (Z-score)</SelectItem>
                <SelectItem value="minmax">Min-Max Scaling (0-1)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-1">
              {options.scaling === 'standard' 
                ? 'Standardizes features by removing the mean and scaling to unit variance'
                : options.scaling === 'minmax'
                ? 'Scales features to a fixed range between 0 and 1'
                : 'No scaling applied to the data'}
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="missingValues">Missing Values Handling</Label>
            <Select
              value={options.missingValues}
              onValueChange={(value) => onOptionChange('missingValues', value)}
            >
              <SelectTrigger id="missingValues">
                <SelectValue placeholder="Select missing values strategy" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="drop">Drop rows with missing values</SelectItem>
                <SelectItem value="mean">Impute with mean</SelectItem>
                <SelectItem value="median">Impute with median</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-1">
              {options.missingValues === 'drop' 
                ? 'Removes rows containing missing values'
                : options.missingValues === 'mean'
                ? 'Replaces missing values with the mean of the column'
                : options.missingValues === 'median'
                ? 'Replaces missing values with the median of the column'
                : 'No handling of missing values'}
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="featureSelection">Feature Selection</Label>
            <Select
              value={options.featureSelection}
              onValueChange={(value) => onOptionChange('featureSelection', value)}
            >
              <SelectTrigger id="featureSelection">
                <SelectValue placeholder="Select feature selection method" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="selectk">SelectKBest (ANOVA F-value)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-1">
              {options.featureSelection === 'selectk' 
                ? 'Selects top features using ANOVA F-value between label/features'
                : 'No feature selection applied'}
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="dimensionReduction">Dimension Reduction</Label>
            <Select
              value={options.dimensionReduction}
              onValueChange={(value) => onOptionChange('dimensionReduction', value)}
            >
              <SelectTrigger id="dimensionReduction">
                <SelectValue placeholder="Select dimension reduction method" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                <SelectItem value="pca_half">PCA (50% components)</SelectItem>
                <SelectItem value="pca_2d">PCA (2 components)</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-1">
              {options.dimensionReduction === 'pca_half' 
                ? 'Reduces dimensions using PCA to 50% of original features'
                : options.dimensionReduction === 'pca_2d'
                ? 'Reduces dimensions using PCA to 2 components (for visualization)'
                : 'No dimension reduction applied'}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
