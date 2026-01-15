"""
app/routers/detection.py
Detection routes for ML, NLP, and LSTM predictions
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import joblib
import pandas as pd
import torch
from pathlib import Path
from app.models.db_models import get_db
from app.schemas.user import MLDetectionRequest, NLPDetectionRequest, DetectionResponse
from app.crud.log import create_ad_log
from app.dependencies import get_optional_user_id
from app.core.config import ML_MODEL_DIR, NLP_MODEL_DIR, DL_MODEL_DIR
from app.ml.feature_extractor import extract_features_from_dict
from app.nlp.pipeline import NLPPredictor
from app.dl.lstm import LSTMPredictor, LSTMAdDetector

router = APIRouter(prefix="/detect", tags=["Detection"])

# Global model cache
_ml_model = None
_ml_scaler = None
_ml_selector = None
_nlp_predictor = None
_lstm_predictor = None


def load_ml_models():
    """Load ML models into cache"""
    global _ml_model, _ml_scaler, _ml_selector
    
    if _ml_model is None:
        model_path = ML_MODEL_DIR / "ml_model.pkl"
        scaler_path = ML_MODEL_DIR / "scaler.pkl"
        selector_path = ML_MODEL_DIR / "selector.pkl"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not trained. Run: python app/ml/train_ml.py"
            )
        
        _ml_model = joblib.load(model_path)
        _ml_scaler = joblib.load(scaler_path)
        _ml_selector = joblib.load(selector_path)
    
    return _ml_model, _ml_scaler, _ml_selector


def load_nlp_model():
    """Load NLP model into cache"""
    global _nlp_predictor
    
    if _nlp_predictor is None:
        vectorizer_path = NLP_MODEL_DIR / "tfidf_vectorizer.pkl"
        model_path = NLP_MODEL_DIR / "nlp_model.pkl"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NLP model not trained. Run: python app/nlp/train_nlp.py"
            )
        
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        _nlp_predictor = NLPPredictor(vectorizer, model)
    
    return _nlp_predictor


def load_lstm_model():
    """Load LSTM model into cache"""
    global _lstm_predictor
    
    if _lstm_predictor is None:
        model_path = DL_MODEL_DIR / "lstm_model.pth"
        tokenizer_path = DL_MODEL_DIR / "tokenizer.pkl"
        
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LSTM model not trained. Run: python app/dl/train_dl.py"
            )
        
        tokenizer = joblib.load(tokenizer_path)
        
        model = LSTMAdDetector(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.3
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        _lstm_predictor = LSTMPredictor(model, tokenizer, device='cpu')
    
    return _lstm_predictor


@router.post("/ml", response_model=DetectionResponse)
def detect_with_ml(
    request: MLDetectionRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_optional_user_id)
):
    """
    Feature-based ML detection
    
    Detects advertisements using engineered features with ML algorithms:
    - Logistic Regression
    - Naive Bayes
    - SVM
    - Random Forest
    - KNN
    
    Uses Pearson correlation for feature selection.
    """
    # Load models
    model, scaler, selector = load_ml_models()
    
    # Convert request to DataFrame
    features_dict = request.dict()
    df = extract_features_from_dict(features_dict)
    
    # Apply feature selection
    df_selected = selector.transform(df)
    
    # Scale features
    df_scaled = scaler.transform(df_selected)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    
    # Get confidence
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(df_scaled)[0]
        confidence = float(probabilities[prediction])
    else:
        confidence = 1.0
    
    result = 'ad' if prediction == 1 else 'not_ad'
    
    # Log prediction
    log = create_ad_log(
        db=db,
        user_id=user_id,
        detection_method='ml',
        input_data=features_dict,
        prediction=result,
        confidence=confidence,
        num_features=len(features_dict)
    )
    
    return {
        "prediction": result,
        "confidence": confidence,
        "method": "ml",
        "log_id": log.id
    }


@router.post("/text", response_model=DetectionResponse)
def detect_with_nlp(
    request: NLPDetectionRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_optional_user_id)
):
    """
    Text-based NLP detection
    
    Detects advertisements using TF-IDF (unigram + bigram) with NLP classification.
    """
    # Load model
    predictor = load_nlp_model()
    
    # Predict
    prediction, confidence = predictor.predict(request.text)
    
    # Log prediction
    log = create_ad_log(
        db=db,
        user_id=user_id,
        detection_method='nlp',
        input_data={'text': request.text[:500]},  # Store first 500 chars
        prediction=prediction,
        confidence=confidence,
        text_length=len(request.text),
        word_count=len(request.text.split())
    )
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "method": "nlp",
        "log_id": log.id
    }


@router.post("/lstm", response_model=DetectionResponse)
def detect_with_lstm(
    request: NLPDetectionRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_optional_user_id)
):
    """
    Deep Learning LSTM detection
    
    Detects advertisements using PyTorch LSTM model for sequential text analysis.
    """
    # Load model
    predictor = load_lstm_model()
    
    # Predict
    prediction, confidence = predictor.predict(request.text)
    
    # Log prediction
    log = create_ad_log(
        db=db,
        user_id=user_id,
        detection_method='lstm',
        input_data={'text': request.text[:500]},
        prediction=prediction,
        confidence=confidence,
        text_length=len(request.text),
        word_count=len(request.text.split())
    )
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "method": "lstm",
        "log_id": log.id
    }