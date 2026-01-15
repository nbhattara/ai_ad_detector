"""
app/routers/admin.py
Admin routes for dashboard and analytics
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.models.db_models import get_db
from app.dependencies import get_current_admin
from app.crud.user import get_total_users
from app.crud.log import get_total_detections, get_total_ads_detected, get_recent_logs
from app.schemas.user import DashboardStats
from pathlib import Path

router = APIRouter(prefix="/admin", tags=["Admin"])

# Templates directory
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/dashboard", response_class=HTMLResponse)
def admin_dashboard(
    request: Request,
    db: Session = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """
    Admin dashboard HTML page
    
    Requires admin authentication (JWT token in Authorization header).
    
    Displays:
    - Total users
    - Total detections
    - Total ads detected
    - Recent detection logs with confidence scores
    """
    # Get statistics
    total_users = get_total_users(db)
    total_detections = get_total_detections(db)
    total_ads = get_total_ads_detected(db)
    recent_logs = get_recent_logs(db, limit=20)
    
    # Format logs for template
    formatted_logs = []
    for log in recent_logs:
        formatted_logs.append({
            'id': log.id,
            'method': log.detection_method.upper(),
            'prediction': log.prediction.replace('_', ' ').title(),
            'confidence': f"{log.confidence * 100:.2f}%",
            'timestamp': log.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "admin_username": admin.username,
            "total_users": total_users,
            "total_detections": total_detections,
            "total_ads": total_ads,
            "recent_logs": formatted_logs
        }
    )


@router.get("/stats", response_model=DashboardStats)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    admin = Depends(get_current_admin)
):
    """
    Get dashboard statistics as JSON
    
    Requires admin authentication.
    
    Returns:
    - total_users: Number of registered users
    - total_detections: Total detection requests
    - total_ads_detected: Number of ads detected
    - recent_logs: List of recent detection logs
    """
    # Get statistics
    total_users = get_total_users(db)
    total_detections = get_total_detections(db)
    total_ads = get_total_ads_detected(db)
    recent_logs = get_recent_logs(db, limit=10)
    
    # Format logs
    formatted_logs = [
        {
            'id': log.id,
            'detection_method': log.detection_method,
            'prediction': log.prediction,
            'confidence': log.confidence,
            'created_at': log.created_at.isoformat()
        }
        for log in recent_logs
    ]
    
    return {
        "total_users": total_users,
        "total_detections": total_detections,
        "total_ads_detected": total_ads,
        "recent_logs": formatted_logs
    }