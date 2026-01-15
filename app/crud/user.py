
from sqlalchemy.orm import Session
from typing import Optional
from app.models.db_models import User, Admin
from app.core.security import hash_password, verify_password


# ==================== USER CRUD ====================

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, username: str, email: str, password: str) -> User:
    """Create a new user"""
    hashed_password = hash_password(password)
    
    db_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password"""
    user = get_user_by_username(db, username)
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    return user


def get_total_users(db: Session) -> int:
    """Get total number of users"""
    return db.query(User).count()


# ==================== ADMIN CRUD ====================

def get_admin_by_username(db: Session, username: str) -> Optional[Admin]:
    """Get admin by username"""
    return db.query(Admin).filter(Admin.username == username).first()


def create_admin(db: Session, username: str, email: str, password: str) -> Admin:
    """Create a new admin"""
    hashed_password = hash_password(password)
    
    db_admin = Admin(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    
    db.add(db_admin)
    db.commit()
    db.refresh(db_admin)
    
    return db_admin


def authenticate_admin(db: Session, username: str, password: str) -> Optional[Admin]:
    """Authenticate admin with username and password"""
    admin = get_admin_by_username(db, username)
    
    if not admin:
        return None
    
    if not verify_password(password, admin.hashed_password):
        return None
    
    return admin