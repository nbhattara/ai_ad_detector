"""
app/ml/train_ml.py
Train ML models: Logistic Regression, Naive Bayes, SVM, Random Forest, KNN
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path
from app.core.config import ML_MODEL_DIR, RANDOM_STATE, TEST_SIZE
from app.ml.feature_extractor import FeatureSelector


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic training data"""
    np.random.seed(RANDOM_STATE)
    
    # Generate ad samples
    n_ads = n_samples // 2
    ads_data = {
        'click_through_rate': np.random.uniform(2.5, 8.0, n_ads),
        'engagement_rate': np.random.uniform(1.0, 5.0, n_ads),
        'time_on_page': np.random.uniform(5, 30, n_ads),
        'image_count': np.random.randint(2, 10, n_ads),
        'link_count': np.random.randint(3, 15, n_ads),
        'caps_ratio': np.random.uniform(0.1, 0.4, n_ads),
        'exclamation_count': np.random.randint(1, 8, n_ads),
        'special_offer': np.random.choice([0, 1], n_ads, p=[0.2, 0.8]),
        'urgency_words': np.random.choice([0, 1], n_ads, p=[0.3, 0.7]),
        'price_mentioned': np.random.choice([0, 1], n_ads, p=[0.2, 0.8]),
        'is_ad': [1] * n_ads
    }
    
    # Generate non-ad samples
    n_non_ads = n_samples - n_ads
    non_ads_data = {
        'click_through_rate': np.random.uniform(0.1, 2.0, n_non_ads),
        'engagement_rate': np.random.uniform(3.0, 10.0, n_non_ads),
        'time_on_page': np.random.uniform(30, 180, n_non_ads),
        'image_count': np.random.randint(0, 4, n_non_ads),
        'link_count': np.random.randint(1, 5, n_non_ads),
        'caps_ratio': np.random.uniform(0.0, 0.15, n_non_ads),
        'exclamation_count': np.random.randint(0, 2, n_non_ads),
        'special_offer': np.random.choice([0, 1], n_non_ads, p=[0.9, 0.1]),
        'urgency_words': np.random.choice([0, 1], n_non_ads, p=[0.85, 0.15]),
        'price_mentioned': np.random.choice([0, 1], n_non_ads, p=[0.8, 0.2]),
        'is_ad': [0] * n_non_ads
    }
    
    ads_df = pd.DataFrame(ads_data)
    non_ads_df = pd.DataFrame(non_ads_data)
    
    df = pd.concat([ads_df, non_ads_df], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    return df


def train_ml_models():
    """Train all ML models and save the best one"""
    print("\n" + "="*60)
    print("TRAINING ML MODELS")
    print("="*60)
    
    # Generate data
    print("\nGenerating training data...")
    df = generate_sample_data(n_samples=1000)
    print(f"✓ Generated {len(df)} samples")
    
    # Split features and target
    X = df.drop('is_ad', axis=1)
    y = df['is_ad']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Feature selection
    print("\nApplying Pearson correlation feature selection...")
    selector = FeatureSelector()
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    print(f"✓ Selected features: {selector.selected_features}")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Train and evaluate
    results = {}
    best_model = None
    best_score = 0
    best_name = None
    
    print("\nTraining models...")
    for name, model in models.items():
        print(f"\n{name}:")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}
        
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    df_results = pd.DataFrame(results).T.sort_values('f1_score', ascending=False)
    print(df_results.to_string())
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name} (F1-Score: {best_score:.4f})")
    print("="*60)
    
    # Save models
    print(f"\nSaving models to {ML_MODEL_DIR}...")
    joblib.dump(best_model, ML_MODEL_DIR / "ml_model.pkl")
    joblib.dump(scaler, ML_MODEL_DIR / "scaler.pkl")
    joblib.dump(selector, ML_MODEL_DIR / "selector.pkl")
    
    print("✓ ML model saved")
    print("✓ Scaler saved")
    print("✓ Feature selector saved")
    print("\n✓ ML training complete!")


if __name__ == "__main__":
    train_ml_models()