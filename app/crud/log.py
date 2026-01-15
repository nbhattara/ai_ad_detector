"""
app/crud/log.py
CRUD operations for AdLog model
"""
from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.db_models import AdLog
import json


def create_ad_log(
    db: Session,
    user_id: Optional[int],
    detection_method: str,
    input_data: dict,
    prediction: str,
    confidence: float,
    num_features: Optional[int] = None,
    text_length: Optional[int] = None,
    word_count: Optional[int] = None
) -> AdLog:
    """
    Create a new advertisement detection log
    
    Args:
        db: Database session
        user_id: User ID (can be None for anonymous)
        detection_method: 'ml', 'nlp', or 'lstm'
        input_data: Dictionary of input data
        prediction: 'ad' or 'not_ad'
        confidence: Prediction confidence score
        num_features: Number of features (for ML)
        text_length: Text length (for NLP/LSTM)
        word_count: Word count (for NLP/LSTM)
        
    Returns:
        Created AdLog instance
    """
    db_log = AdLog(
        user_id=user_id,
        detection_method=detection_method,
        input_data=json.dumps(input_data),
        prediction=prediction,
        confidence=confidence,
        num_features=num_features,
        text_length=text_length,
        word_count=word_count
    )
    
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    
    return db_log


def get_recent_logs(db: Session, limit: int = 10) -> List[AdLog]:
    """
    Get recent detection logs
    
    Args:
        db: Database session
        limit: Maximum number of logs to return
        
    Returns:
        List of AdLog instances
    """
    return db.query(AdLog).order_by(AdLog.created_at.desc()).limit(limit).all()


def get_total_detections(db: Session) -> int:
    """Get total number of detections"""
    return db.query(AdLog).count()


def get_total_ads_detected(db: Session) -> int:
    """Get total number of ads detected"""
    return db.query(AdLog).filter(AdLog.prediction == 'ad').count()


def get_logs_by_user(db: Session, user_id: int, limit: int = 20) -> List[AdLog]:
    """
    Get logs for a specific user
    
    Args:
        db: Database session
        user_id: User ID
        limit: Maximum number of logs to return
        
    Returns:
        List of AdLog instances
    """
    return db.query(AdLog).filter(AdLog.user_id == user_id).order_by(
        AdLog.created_at.desc()
    ).limit(limit).all()


def get_logs_by_method(db: Session, method: str, limit: int = 20) -> List[AdLog]:
    """
    Get logs by detection method
    
    Args:
        db: Database session
        method: Detection method ('ml', 'nlp', or 'lstm')
        limit: Maximum number of logs to return
        
    Returns:
        List of AdLog instances
    """
    return db.query(AdLog).filter(AdLog.detection_method == method).order_by(
        AdLog.created_at.desc()
    ).limit(limit).all()