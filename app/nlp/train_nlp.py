"""
app/nlp/train_nlp.py
Train NLP model using TF-IDF (unigram + bigram)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from app.core.config import NLP_MODEL_DIR, RANDOM_STATE, TEST_SIZE, MAX_FEATURES, NGRAM_RANGE
from app.nlp.pipeline import TextPreprocessor


def generate_text_samples(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic text data"""
    np.random.seed(RANDOM_STATE)
    
    ad_templates = [
        "LIMITED TIME OFFER! Get {discount}% OFF on all products! Shop now and save big!",
        "SALE ALERT! Buy now and get FREE shipping! Don't miss out on this amazing deal!",
        "Special promotion! Order today and receive a FREE gift! Hurry, while stocks last!",
        "EXCLUSIVE DEAL: Save ${amount} on your purchase! Click here to claim your discount NOW!",
        "Flash Sale! Up to {discount}% off everything! Shop today and get instant savings!",
        "Amazing offer! Buy one get one FREE! Limited time only! Order now!",
        "HUGE DISCOUNTS! Everything must go! Save up to ${amount} today!",
        "Special offer for you! Use code SAVE{discount} at checkout! Expires soon!",
        "Don't miss out! {discount}% OFF all items! Shop now before it's too late!",
        "Best deal of the year! Save ${amount} instantly! Click to buy now!"
    ]
    
    non_ad_templates = [
        "The research paper discusses the methodology and findings of the study conducted over three years.",
        "According to the latest report, climate change continues to affect global weather patterns.",
        "The conference will feature keynote speakers from various universities and research institutions.",
        "This article explores the historical context and cultural significance of ancient civilizations.",
        "The documentary examines the social and economic factors that shaped modern society.",
        "Scientists have discovered new evidence that supports the existing theoretical framework.",
        "The book provides a comprehensive analysis of political movements in the 20th century.",
        "Experts discuss the technological innovations that have transformed the industry.",
        "The study reveals important insights into human behavior and cognitive processes.",
        "This essay evaluates different philosophical perspectives on ethics and morality."
    ]
    
    texts = []
    labels = []
    
    # Generate ads
    for _ in range(n_samples // 2):
        template = np.random.choice(ad_templates)
        text = template.format(
            discount=np.random.choice([20, 30, 40, 50, 60, 70]),
            amount=np.random.choice([10, 20, 50, 100, 200])
        )
        texts.append(text)
        labels.append(1)
    
    # Generate non-ads
    for _ in range(n_samples // 2):
        text = np.random.choice(non_ad_templates)
        texts.append(text)
        labels.append(0)
    
    df = pd.DataFrame({'text': texts, 'is_ad': labels})
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def train_nlp_model():
    """Train NLP model with TF-IDF"""
    print("\n" + "="*60)
    print("TRAINING NLP MODEL")
    print("="*60)
    
    # Generate data
    print("\nGenerating text data...")
    df = generate_text_samples(n_samples=1000)
    print(f"✓ Generated {len(df)} text samples")
    
    # Preprocess
    preprocessor = TextPreprocessor()
    df['text'] = df['text'].apply(preprocessor.clean_text)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['is_ad'], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['is_ad']
    )
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # TF-IDF Vectorization
    print(f"\nVectorizing with TF-IDF (ngram_range={NGRAM_RANGE}, max_features={MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words='english',
        lowercase=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1)
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_name = None
    
    print("\nTraining NLP models...")
    for name, model in models.items():
        print(f"\n{name}:")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
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
    
    print(f"\n{'='*60}")
    print(f"BEST NLP MODEL: {best_name} (F1-Score: {best_score:.4f})")
    print("="*60)
    
    # Save
    print(f"\nSaving models to {NLP_MODEL_DIR}...")
    joblib.dump(vectorizer, NLP_MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(best_model, NLP_MODEL_DIR / "nlp_model.pkl")
    
    print("✓ TF-IDF vectorizer saved")
    print("✓ NLP model saved")
    print("\n✓ NLP training complete!")


if __name__ == "__main__":
    train_nlp_model()