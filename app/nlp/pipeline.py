"""
app/nlp/pipeline.py
NLP pipeline for text-based ad detection using TF-IDF
"""
import re
from typing import Tuple
import numpy as np


class TextPreprocessor:
    """Text preprocessing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def extract_features(text: str) -> dict:
        """Extract text features for ad detection"""
        words = text.split()
        
        # Ad-related keywords
        special_offer_keywords = ['sale', 'discount', 'offer', 'deal', 'save', 'free', 'promo']
        urgency_keywords = ['now', 'today', 'hurry', 'limited', 'expires', 'last chance']
        
        text_lower = text.lower()
        
        features = {
            'text_length': len(text),
            'word_count': len(words),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'exclamation_count': text.count('!'),
            'special_offer': int(any(kw in text_lower for kw in special_offer_keywords)),
            'urgency_words': int(any(kw in text_lower for kw in urgency_keywords)),
            'price_mentioned': int(any(char in text for char in ['$', '€', '£', '%']))
        }
        
        return features


class NLPPredictor:
    """NLP prediction wrapper"""
    
    def __init__(self, vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model
        self.preprocessor = TextPreprocessor()
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict if text is an ad
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Get confidence
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(probabilities[prediction])
        else:
            confidence = 1.0  # Default for models without probability
        
        result = 'ad' if prediction == 1 else 'not_ad'
        
        return result, confidence