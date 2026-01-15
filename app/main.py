"""
app/main.py
Main FastAPI application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.db_models import init_db
from app.routers import auth, detection, admin

# Initialize database
init_db()

# Create FastAPI app
app = FastAPI(
    title="AI Advertisement Detection Platform",
    description="""
    Production-ready AI platform for detecting advertisements using:
    
    - **Machine Learning**: Feature-based detection with multiple algorithms
    - **NLP**: Text-based detection using TF-IDF (unigram + bigram)
    - **Deep Learning**: LSTM neural network for sequential text analysis
    
    ## Features:
    - JWT Authentication
    - Multiple detection methods
    - Admin dashboard with analytics
    - Comprehensive logging
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(detection.router)
app.include_router(admin.router)


@app.get("/")
def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "AI Advertisement Detection Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "authentication": {
                "register": "POST /auth/register",
                "login": "POST /auth/login",
                "admin_login": "POST /auth/admin/login"
            },
            "detection": {
                "ml_detection": "POST /detect/ml",
                "nlp_detection": "POST /detect/text",
                "lstm_detection": "POST /detect/lstm"
            },
            "admin": {
                "dashboard": "GET /admin/dashboard",
                "stats": "GET /admin/stats"
            }
        }
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "ai-ad-detection-platform"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)