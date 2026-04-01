"""
Model training and evaluation module for sentiment analysis.

This module handles training, hyperparameter optimization, and evaluation
of sentiment classification models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SentimentClassifier:
    """
    Sentiment classifier using Logistic Regression.
    
    This class handles model training, hyperparameter optimization,
    and feature importance extraction.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the sentiment classifier.
        
        Args:
            random_state (int): Random seed for reproducibility. Default: 42
        """
        self.random_state = random_state
        self.model = None
        self.best_c = None
        self.best_score = None
    
    def find_best_hyperparameters(self, X_train, y_train, X_val, y_val,
                                  c_values: list = [0.1, 1.0, 10.0, 100.0]) -> dict:
        """
        Find the best regularization parameter (C) through validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            c_values (list): List of C values to test. Default: [0.1, 1.0, 10.0, 100.0]
            
        Returns:
            dict: Dictionary with best_c, best_score, and results for all C values
        """
        results = {}
        best_score = 0
        best_c = None
        best_model = None
        
        print("Testing Logistic Regression with different regularization parameters...")
        
        for c in c_values:
            model = LogisticRegression(
                C=c,
                max_iter=1000,
                random_state=self.random_state,
                solver='liblinear'
            )
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            results[c] = score
            
            print(f"  C={c}, Validation Accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_c = c
                best_model = model
        
        self.best_c = best_c
        self.best_score = best_score
        self.model = best_model
        
        return {
            'best_c': best_c,
            'best_score': best_score,
            'results': results
        }
    
    def train_final_model(self, X_train, y_train) -> None:
        """
        Train the final model on all training data using the best hyperparameters.
        
        Args:
            X_train: Full training features
            y_train: Full training labels
        """
        if self.best_c is None:
            raise ValueError("Best hyperparameters not found. Run find_best_hyperparameters first.")
        
        print(f"\nTraining final model with C={self.best_c}...")
        self.model = LogisticRegression(
            C=self.best_c,
            max_iter=1000,
            random_state=self.random_state,
            solver='liblinear'
        )
        self.model.fit(X_train, y_train)
        print("Model training completed.")
    
    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        Perform k-fold cross-validation on the model.
        
        Args:
            X: Features
            y: Labels
            cv (int): Number of folds. Default: 5
            
        Returns:
            dict: Cross-validation results including mean and std
        """
        if self.model is None:
            raise ValueError("Model not trained. Train model first.")
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"\n{cv}-fold Cross-Validation Results:")
        print(f"  Mean Accuracy: {cv_scores.mean():.4f}")
        print(f"  Std Deviation: {cv_scores.std():.4f}")
        
        return {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
    
    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> dict:
        """
        Extract and return the most important features from the model.
        
        Args:
            feature_names (list): List of feature names from vectorizer
            top_n (int): Number of top features to return. Default: 20
            
        Returns:
            dict: Dictionary with positive and negative features and their coefficients
        """
        if self.model is None or not hasattr(self.model, 'coef_'):
            raise ValueError("Model not trained or does not have coefficients.")
        
        coefs = self.model.coef_[0]
        feature_names = np.array(feature_names)
        
        # Get indices of top positive and negative coefficients
        top_positive_idx = coefs.argsort()[-top_n:][::-1]
        top_negative_idx = coefs.argsort()[:top_n]
        
        positive_features = {
            feature_names[i]: coefs[i] 
            for i in top_positive_idx
        }
        
        negative_features = {
            feature_names[i]: coefs[i] 
            for i in top_negative_idx
        }
        
        return {
            'positive_features': positive_features,
            'negative_features': negative_features
        }
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        return self.model.predict(X)
    
    def evaluate(self, X, y, label_names: list = None) -> dict:
        """
        Evaluate model performance on a dataset.
        
        Args:
            X: Features
            y: True labels
            label_names (list): Names for the labels (e.g., ['negative', 'positive'])
            
        Returns:
            dict: Evaluation metrics including accuracy, classification report, confusion matrix
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        report = classification_report(
            y, y_pred,
            target_names=label_names,
            output_dict=True
        )
        
        cm = confusion_matrix(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }