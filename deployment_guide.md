# 🎯 Complete Setup & Deployment Guide

## 📋 Table of Contents
1. [Quick Setup](#quick-setup)
2. [Manual Setup](#manual-setup)
3. [Testing the Platform](#testing)
4. [Deployment Options](#deployment)
5. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# 1. Navigate to project
cd ai_ad_detection_platform

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Run setup (does everything!)
python setup.py
```

**This will:**
- ✅ Install all dependencies
- ✅ Initialize database
- ✅ Train ML models
- ✅ Train NLP models
- ✅ Train LSTM models
- ✅ Create admin user (admin/admin123)

### Option 2: One-Line Docker Setup (Coming Soon)

```bash
docker-compose up --build
```

---

## 🔧 Manual Setup

If you prefer step-by-step control:

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Initialize Database
```python
python -c "from app.models.db_models import init_db; init_db()"
```

### Step 3: Train Models

**Train ML models (5-10 minutes):**
```bash
python app/ml/train_ml.py
```

**Train NLP models (3-5 minutes):**
```bash
python app/nlp/train_nlp.py
```

**Train LSTM models (5-8 minutes):**
```bash
python app/dl/train_dl.py
```

### Step 4: Create Admin User
```python
python -c "
from app.models.db_models import SessionLocal
from app.crud.user import create_admin

db = SessionLocal()
admin = create_admin(db, 'admin', 'admin@example.com', 'admin123')
print(f'Admin created: {admin.username}')
db.close()
"
```

### Step 5: Start Server
```bash
python app/main.py
```

Server will run at: **http://localhost:8000**

---

## 🧪 Testing the Platform

### 1. Check Health
```bash
curl http://localhost:8000/health
```

### 2. View API Documentation
Open browser: http://localhost:8000/docs

### 3. Test Registration
```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123"
  }'
```

### 4. Test Login
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }'
```

### 5. Test ML Detection
```bash
curl -X POST "http://localhost:8000/detect/ml" \
  -H "Content-Type: application/json" \
  -d '{
    "click_through_rate": 7.5,
    "engagement_rate": 1.2,
    "time_on_page": 10.5,
    "image_count": 7,
    "link_count": 13,
    "caps_ratio": 0.35,
    "exclamation_count": 6,
    "special_offer": 1,
    "urgency_words": 1,
    "price_mentioned": 1
  }'
```

### 6. Test NLP Detection
```bash
curl -X POST "http://localhost:8000/detect/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "HUGE SALE! 80% OFF everything! Shop now! Limited time only!"
  }'
```

### 7. Test Admin Login
```bash
curl -X POST "http://localhost:8000/auth/admin/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

### 8. Access Admin Dashboard
1. Get admin token from step 7
2. Open: http://localhost:8000/admin/dashboard
3. Add Authorization header: `Bearer <your_token>`

### 9. Run Python Test Suite
```bash
python api_examples.py
```

---

## 🚀 Deployment Options

### Option 1: Local Development
```bash
# Development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Production with Gunicorn
```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Option 3: Docker Deployment

**Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models during build
RUN python app/ml/train_ml.py && \
    python app/nlp/train_nlp.py && \
    python app/dl/train_dl.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**
```bash
docker build -t ai-ad-detection .
docker run -p 8000:8000 ai-ad-detection
```

### Option 4: Cloud Deployment

#### Heroku
```bash
# Install Heroku CLI
heroku login
heroku create your-app-name
git push heroku main
```

#### AWS EC2
```bash
# SSH into EC2 instance
ssh -i key.pem ubuntu@your-ec2-ip

# Clone repo
git clone <your-repo>
cd ai_ad_detection_platform

# Install dependencies
pip install -r requirements.txt

# Train models
python setup.py

# Run with nohup
nohup python app/main.py &
```

#### Google Cloud Run
```bash
gcloud run deploy ai-ad-detection \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 5: Nginx Reverse Proxy

**nginx.conf:**
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🔐 Production Security Checklist

Before deploying to production:

- [ ] Change `SECRET_KEY` in `app/core/config.py`
- [ ] Use strong passwords for admin
- [ ] Enable HTTPS/SSL
- [ ] Set up proper CORS origins
- [ ] Use PostgreSQL instead of SQLite
- [ ] Enable rate limiting
- [ ] Set up monitoring (Sentry, DataDog)
- [ ] Configure logging
- [ ] Use environment variables
- [ ] Regular security audits
- [ ] Set up backups

**Update config:**
```python
# app/core/config.py
import os
from secrets import token_urlsafe

SECRET_KEY = os.getenv("SECRET_KEY", token_urlsafe(32))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://...")
```

---

## 🐛 Troubleshooting

### Error: "Module not found"
```bash
# Make sure you're in virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Error: "Model not found"
```bash
# Retrain all models
python app/ml/train_ml.py
python app/nlp/train_nlp.py
python app/dl/train_dl.py
```

### Error: "Database locked"
```bash
# SQLite issue - restart server or use PostgreSQL
rm app/database.db
python -c "from app.models.db_models import init_db; init_db()"
```

### Error: "Port already in use"
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn app.main:app --port 8001
```

### Error: "CUDA not available" (for LSTM)
```python
# LSTM will automatically use CPU
# No action needed - CPU training works fine
```

### Error: "Admin already exists"
```python
# This is normal - admin was created during setup
# Use existing credentials: admin/admin123
```

### Performance Issues
```bash
# Increase workers
gunicorn app.main:app -w 8 -k uvicorn.workers.UvicornWorker

# Or enable caching
pip install redis
# Add Redis caching layer
```

---

## 📊 Monitoring & Logs

### Check Logs
```bash
# View application logs
tail -f app.log

# Check database size
ls -lh app/database.db

# Monitor API requests
# Add to app/main.py:
@app.middleware("http")
async def log_requests(request, call_next):
    print(f"{request.method} {request.url}")
    return await call_next(request)
```

### Performance Metrics
```python
# Add to your monitoring system
- Request latency
- Model inference time
- Detection accuracy
- Database query time
- Memory usage
- CPU usage
```

---

## 🔄 Updates & Maintenance

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Retrain Models (with new data)
```bash
# Add new data to data/dataset.csv
# Retrain
python app/ml/train_ml.py
```

### Database Migrations
```bash
# If you modify database schema
# Use Alembic for migrations
pip install alembic
alembic init migrations
alembic revision --autogenerate -m "description"
alembic upgrade head
```

### Backup Database
```bash
# Backup SQLite
cp app/database.db backups/database_$(date +%Y%m%d).db

# Backup PostgreSQL
pg_dump dbname > backup.sql
```

---

## 🎓 Next Steps

### For Learning
1. Read through code documentation
2. Experiment with different models
3. Add new features
4. Implement unit tests
5. Try different datasets

### For Production
1. Set up CI/CD pipeline
2. Add monitoring (Prometheus/Grafana)
3. Implement caching (Redis)
4. Add rate limiting
5. Set up load balancing
6. Configure auto-scaling

### For Portfolio
1. Deploy to cloud (Heroku/AWS)
2. Create demo video
3. Write blog post
4. Share on GitHub
5. Add to resume/CV

---

## 📞 Support

### Resources
- **API Docs**: http://localhost:8000/docs
- **Project Summary**: PROJECT_SUMMARY.md
- **Quick Start**: QUICKSTART.md
- **Full README**: README.md

### Getting Help
- Check documentation first
- Search GitHub issues
- Stack Overflow
- Create issue on GitHub

---



