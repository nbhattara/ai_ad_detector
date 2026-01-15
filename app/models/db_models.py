"""
app/models/db_models.py
Database models using SQLAlchemy ORM
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.core.config import DATABASE_URL

Base = declarative_base()


class User(Base):
    """User model for regular users"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Admin(Base):
    """Admin model for administrative users"""
    __tablename__ = "admins"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AdLog(Base):
    """Log model for advertisement detection results"""
    __tablename__ = "ad_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    
    # Detection method: 'ml', 'nlp', or 'lstm'
    detection_method = Column(String(20), nullable=False, index=True)
    
    # Input data stored as JSON string
    input_data = Column(Text, nullable=True)
    
    # Feature statistics
    num_features = Column(Integer, nullable=True)
    
    # Text statistics
    text_length = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    
    # Prediction results
    prediction = Column(String(20), nullable=False)  # 'ad' or 'not_ad'
    confidence = Column(Float, nullable=False)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# Database engine and session
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database and create all tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()