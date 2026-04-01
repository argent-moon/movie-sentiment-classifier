"""
Text preprocessing module for sentiment analysis.

This module provides functions for cleaning and normalizing text data
including contraction expansion, special character handling, and more.
"""

import re
import pandas as pd


def advanced_preprocess_text(text: str) -> str:
    """
    Apply text preprocessing to clean review text.
    
    This function performs the following operations:
    - Converts text to lowercase
    - Removes HTML tags
    - Expands common contractions
    - Removes special characters while preserving sentiment indicators (!, ?)
    - Handles multiple punctuation marks
    - Removes extra whitespace
    
    Args:
        text (str): Raw text to be preprocessed
        
    Returns:
        str: Cleaned and normalized text
        
    Example:
        >>> text = "This is AMAZING!!!! I loved it!!!"
        >>> advanced_preprocess_text(text)
        'this is amazing ! i loved it ! '
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Replace common contractions
    contractions = {
        "n't": " not",
        "'ve": " have",
        "'ll": " will",
        "'re": " are",
        "'m": " am",
        "'d": " would",
        "'s": " is"
    }
    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)
    
    # Remove special characters but keep important punctuation that might signal sentiment
    text = re.sub(r'[^a-zA-Z\s!?.]', ' ', text)
    
    # Replace multiple exclamation or question marks with single ones (preserving sentiment signals)
    text = re.sub(r'!+', ' ! ', text)
    text = re.sub(r'\?+', ' ? ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_dataframe(df: pd.DataFrame, text_column: str) -> list:
    """
    Preprocess all text entries in a DataFrame column.
    
    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_column (str): Name of the column containing text to preprocess
        
    Returns:
        list: List of preprocessed texts
    """
    return [advanced_preprocess_text(text) for text in df[text_column]]


def extract_text_features(texts: list) -> pd.DataFrame:
    """
    Extract numerical features from preprocessed text.
    
    Extracts the following features:
    - text_length: Total number of characters
    - word_count: Number of words
    - exclamation_count: Number of exclamation marks
    - question_count: Number of question marks
    
    Args:
        texts (list): List of preprocessed text strings
        
    Returns:
        pd.DataFrame: DataFrame containing extracted features
    """
    features = pd.DataFrame()
    features['text_length'] = [len(text) for text in texts]
    features['word_count'] = [
        len(text.split()) if isinstance(text, str) else 0 
        for text in texts
    ]
    features['exclamation_count'] = [
        text.count('!') if isinstance(text, str) else 0 
        for text in texts
    ]
    features['question_count'] = [
        text.count('?') if isinstance(text, str) else 0 
        for text in texts
    ]
    return features
