"""
app/dl/train_dl.py
Train LSTM deep learning model
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from app.core.config import (
    DL_MODEL_DIR, RANDOM_STATE, TEST_SIZE, EMBEDDING_DIM, HIDDEN_DIM,
    NUM_LAYERS, DROPOUT, LEARNING_RATE, EPOCHS, BATCH_SIZE, MAX_LENGTH
)
from app.dl.lstm import TextTokenizer, LSTMAdDetector


class AdDataset(Dataset):
    """PyTorch Dataset"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def generate_text_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate text data"""
    np.random.seed(RANDOM_STATE)
    
    ad_templates = [
        "LIMITED TIME! {discount}% OFF! Shop now and save!",
        "SALE! FREE shipping! Amazing deal!",
        "Special offer! Order today! FREE gift!",
        "EXCLUSIVE! Save ${amount}! Click NOW!",
        "Flash Sale! {discount}% off! Shop today!",
    ]
    
    non_ad_templates = [
        "The research discusses methodology and findings.",
        "The report examines climate change patterns.",
        "The conference features university speakers.",
        "The article explores historical context.",
        "The study reveals behavioral insights.",
    ]
    
    texts, labels = [], []
    
    for _ in range(n_samples // 2):
        template = np.random.choice(ad_templates)
        text = template.format(
            discount=np.random.choice([20, 30, 50]),
            amount=np.random.choice([10, 50, 100])
        )
        texts.append(text)
        labels.append(1)
    
    for _ in range(n_samples // 2):
        texts.append(np.random.choice(non_ad_templates))
        labels.append(0)
    
    df = pd.DataFrame({'text': texts, 'is_ad': labels})
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def train_lstm_model():
    """Train LSTM model"""
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Generate data
    print("\nGenerating data...")
    df = generate_text_data(n_samples=1000)
    print(f"✓ Generated {len(df)} samples")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['is_ad'], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['is_ad']
    )
    
    # Build tokenizer
    print("\nBuilding tokenizer...")
    tokenizer = TextTokenizer(max_length=MAX_LENGTH)
    tokenizer.build_vocab(X_train.tolist())
    print(f"✓ Vocabulary size: {tokenizer.vocab_size}")
    
    # Convert to sequences
    X_train_seq = np.array([tokenizer.text_to_sequence(text) for text in X_train])
    X_test_seq = np.array([tokenizer.text_to_sequence(text) for text in X_test])
    
    # Create datasets
    train_dataset = AdDataset(X_train_seq, y_train.values)
    test_dataset = AdDataset(X_test_seq, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("\nInitializing LSTM model...")
    model = LSTMAdDetector(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    print("\nEvaluating...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            preds = (outputs >= 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().astype(int))
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"\n{'='*60}")
    print("LSTM MODEL PERFORMANCE")
    print("="*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Save
    print(f"\nSaving models to {DL_MODEL_DIR}...")
    torch.save(model.state_dict(), DL_MODEL_DIR / "lstm_model.pth")
    joblib.dump(tokenizer, DL_MODEL_DIR / "tokenizer.pkl")
    
    print("✓ LSTM model saved")
    print("✓ Tokenizer saved")
    print("\n✓ LSTM training complete!")


if __name__ == "__main__":
    train_lstm_model()