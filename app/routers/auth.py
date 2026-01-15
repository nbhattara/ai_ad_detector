"""
app/routers/auth.py
Authentication routes: register and login
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.db_models import get_db
from app.schemas.user import UserCreate, UserResponse, LoginRequest, TokenResponse
from app.crud.user import (
    get_user_by_username,
    get_user_by_email,
    create_user,
    authenticate_user,
    authenticate_admin
)
from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user
    
    - **username**: Unique username (3-50 characters)
    - **email**: Valid email address
    - **password**: Password (minimum 6 characters)
    """
    # Check if username exists
    db_user = get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    db_user = get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    new_user = create_user(
        db=db,
        username=user.username,
        email=user.email,
        password=user.password
    )
    
    return new_user


@router.post("/login", response_model=TokenResponse)
def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    """
    Login for regular users
    
    - **username**: Username
    - **password**: Password
    
    Returns JWT access token
    """
    user = authenticate_user(db, username=credentials.username, password=credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": user.username, "user_id": user.id})
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/admin/login", response_model=TokenResponse)
def admin_login(credentials: LoginRequest, db: Session = Depends(get_db)):
    """
    Login for admin users
    
    - **username**: Admin username
    - **password**: Admin password
    
    Returns JWT access token for admin dashboard
    """
    admin = authenticate_admin(db, username=credentials.username, password=credentials.password)
    
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect admin credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token with admin role
    access_token = create_access_token(data={"sub": admin.username, "admin_id": admin.id, "role": "admin"})
    
    return {"access_token": access_token, "token_type": "bearer"}