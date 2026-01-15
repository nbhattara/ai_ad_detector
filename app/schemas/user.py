"""
app/schemas/user.py
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ==================== USER SCHEMAS ====================

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=6)


class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# ==================== AUTH SCHEMAS ====================

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ==================== DETECTION SCHEMAS ====================

class MLDetectionRequest(BaseModel):
    """Feature-based ML detection request"""
    click_through_rate: float = Field(..., ge=0, le=100)
    engagement_rate: float = Field(..., ge=0, le=100)
    time_on_page: float = Field(..., ge=0)
    image_count: int = Field(..., ge=0)
    link_count: int = Field(..., ge=0)
    caps_ratio: float = Field(..., ge=0, le=1)
    exclamation_count: int = Field(..., ge=0)
    special_offer: int = Field(..., ge=0, le=1)
    urgency_words: int = Field(..., ge=0, le=1)
    price_mentioned: int = Field(..., ge=0, le=1)


class NLPDetectionRequest(BaseModel):
    """Text-based NLP detection request"""
    text: str = Field(..., min_length=10, max_length=5000)


class DetectionResponse(BaseModel):
    """Detection result response"""
    prediction: str  # 'ad' or 'not_ad'
    confidence: float
    method: str
    log_id: int


# ==================== ADMIN SCHEMAS ====================

class DashboardStats(BaseModel):
    """Dashboard statistics"""
    total_users: int
    total_detections: int
    total_ads_detected: int
    recent_logs: List[dict]


class RecentLog(BaseModel):
    """Recent detection log"""
    id: int
    detection_method: str
    prediction: str
    confidence: float
    created_at: datetime
    
    class Config:
        from_attributes = True

      