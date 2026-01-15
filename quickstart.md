# 🚀 Quick Start Guide

## Installation (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Setup Script (One Command!)
```bash
python setup.py
```

This will:
- ✅ Initialize database
- ✅ Train all ML models
- ✅ Train NLP models  
- ✅ Train LSTM models
- ✅ Create admin user

### 3. Start the Server
```bash
python app/main.py
```

Server runs at: **http://localhost:8000**

---

## 🎯 Testing the API

### Test ML Detection
```bash
curl -X POST "http://localhost:8000/detect/ml" \
  -H "Content-Type: application/json" \
  -d '{
    "click_through_rate": 6.5,
    "engagement_rate": 1.5,
    "time_on_page": 8.0,
    "image_count": 5,
    "link_count": 10,
    "caps_ratio": 0.3,
    "exclamation_count": 4,
    "special_offer": 1,
    "urgency_words": 1,
    "price_mentioned": 1
  }'
```

### Test NLP Detection
```bash
curl -X POST "http://localhost:8000/detect/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "SALE! 50% OFF everything! Limited time only!"
  }'
```

---

## 🔑 Admin Access

### Default Credentials
- **Username**: `admin`
- **Password**: `admin123`

### Login to Get Token
```bash
curl -X POST "http://localhost:8000/auth/admin/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

### Access Dashboard
1. Login at: `POST /auth/admin/login`
2. Copy the `access_token`
3. Visit: `http://localhost:8000/admin/dashboard`
4. Add token in Authorization header: `Bearer <token>`

---

## 📚 API Documentation

Interactive docs: **http://localhost:8000/docs**

---

## 🎓 For Students & Developers

This project is perfect for:
- ✅ Final year projects
- ✅ Internship portfolios
- ✅ Resume projects
- ✅ Learning ML/NLP/Deep Learning
- ✅ Building production APIs

---

## 🐛 Troubleshooting

### Error: Module not found
```bash
pip install -r requirements.txt
```

### Error: Models not found
```bash
python app/ml/train_ml.py
python app/nlp/train_nlp.py
python app/dl/train_dl.py
```

### Error: Database issues
```bash
# Delete database and reinitialize
rm app/database.db
python -c "from app.models.db_models import init_db; init_db()"
```

---

## 📞 Need Help?

Check the full README.md for detailed documentation!