"""
app/core/config.py
Core configuration settings for the platform
"""
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Security
SECRET_KEY = "your-secret-key-change-in-production-min-32-chars-long"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Database
DATABASE_URL = f"sqlite:///{BASE_DIR}/app/database.db"

# Model paths
ML_MODEL_DIR = BASE_DIR / "app" / "ml" / "models"
NLP_MODEL_DIR = BASE_DIR / "app" / "nlp" / "models"
DL_MODEL_DIR = BASE_DIR / "app" / "dl" / "models"

# Ensure directories exist
for directory in [ML_MODEL_DIR, NLP_MODEL_DIR, DL_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ML settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CORRELATION_THRESHOLD = 0.1

# NLP settings
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# LSTM settings
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 32
MAX_LENGTH = 200