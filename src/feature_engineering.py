"""
Feature engineering module for sentiment analysis.

This module handles TF-IDF vectorization and feature extraction
for machine learning models.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class SentimentFeatureExtractor:
    """
    Feature extractor for sentiment analysis using TF-IDF vectorization.
    
    This class handles the creation and transformation of TF-IDF features
    from preprocessed text data.
    """
    
    def __init__(self, max_features: int = 20000, ngram_range: tuple = (1, 3),
                 min_df: int = 3, max_df: float = 0.9):
        """
        Initialize the feature extractor with TF-IDF parameters.
        
        Args:
            max_features (int): Maximum number of features to extract. Default: 20000
            ngram_range (tuple): Range of n-grams to use (min, max). Default: (1, 3)
            min_df (int): Minimum document frequency. Default: 3
            max_df (float): Maximum document frequency (as proportion). Default: 0.9
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.feature_names = None
    
    def fit_transform(self, texts: list) -> object:
        """
        Fit the TF-IDF vectorizer and transform text data.
        
        Args:
            texts (list): List of preprocessed text strings to fit on
            
        Returns:
            sparse matrix: TF-IDF feature matrix
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True
        )
        
        X_tfidf = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        return X_tfidf
    
    def transform(self, texts: list) -> object:
        """
        Transform text data using the fitted vectorizer.
        
        Args:
            texts (list): List of preprocessed text strings
            
        Returns:
            sparse matrix: TF-IDF feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self) -> list:
        """
        Get the list of feature names (terms) from the vectorizer.
        
        Returns:
            list: Array of feature names
        """
        if self.feature_names is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        return self.feature_names


def prepare_data(X_features, y_labels, test_size: float = 0.1,
                random_state: int = 42):
    """
    Split data into training and validation sets with stratification.
    
    Args:
        X_features: Feature matrix (sparse or dense)
        y_labels: Target labels
        test_size (float): Proportion of data to use for validation. Default: 0.1
        random_state (int): Random seed for reproducibility. Default: 42
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_features, y_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labels
    )
    
    return X_train, X_val, y_train, y_val