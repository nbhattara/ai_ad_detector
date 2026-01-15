"""
app/ml/feature_extractor.py
Feature engineering and selection using Pearson correlation
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import List, Dict
from app.core.config import CORRELATION_THRESHOLD


class FeatureSelector:
    """Feature selection using Pearson correlation"""
    
    def __init__(self, correlation_threshold: float = CORRELATION_THRESHOLD):
        self.correlation_threshold = correlation_threshold
        self.selected_features = []
        self.feature_correlations = {}
    
    def calculate_correlations(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Calculate Pearson correlation for each feature"""
        correlations = {}
        
        for column in X.columns:
            corr, p_value = pearsonr(X[column], y)
            correlations[column] = {
                'correlation': corr,
                'p_value': p_value,
                'abs_correlation': abs(corr)
            }
        
        return correlations
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features based on correlation threshold"""
        self.feature_correlations = self.calculate_correlations(X, y)
        
        # Select features above threshold
        self.selected_features = [
            feature for feature, stats in self.feature_correlations.items()
            if stats['abs_correlation'] >= self.correlation_threshold
        ]
        
        # Sort by absolute correlation
        self.selected_features.sort(
            key=lambda x: self.feature_correlations[x]['abs_correlation'],
            reverse=True
        )
        
        return self.selected_features
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame to include only selected features"""
        if not self.selected_features:
            raise ValueError("No features selected. Run select_features() first.")
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select features and transform in one step"""
        self.select_features(X, y)
        return self.transform(X)


def extract_features_from_dict(data: dict) -> pd.DataFrame:
    """
    Convert feature dictionary to DataFrame
    
    Args:
        data: Dictionary of features
        
    Returns:
        DataFrame with single row
    """
    return pd.DataFrame([data])