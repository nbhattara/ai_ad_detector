"""
app/dl/lstm.py
LSTM Deep Learning model for ad detection
"""
import torch
import torch.nn as nn
import numpy as np
import re
from typing import List, Tuple, Dict


class TextTokenizer:
    """Tokenizer for LSTM input"""
    
    def __init__(self, max_length: int = 200):
        self.max_length = max_length
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_size = 2
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        for text in texts:
            words = self.tokenize(text)
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size += 1
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text.split()
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence"""
        words = self.tokenize(text)
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Pad or truncate
        if len(sequence) < self.max_length:
            sequence += [self.vocab['<PAD>']] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return sequence


class LSTMAdDetector(nn.Module):
    """LSTM model for advertisement detection"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(LSTMAdDetector, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out.squeeze()


class LSTMPredictor:
    """LSTM prediction wrapper"""
    
    def __init__(self, model: LSTMAdDetector, tokenizer: TextTokenizer, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict if text is an ad
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Tokenize
        sequence = self.tokenizer.text_to_sequence(text)
        
        # Convert to tensor
        x = torch.LongTensor([sequence]).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(x)
            confidence = output.item()
        
        prediction = 'ad' if confidence >= 0.5 else 'not_ad'
        
        return prediction, confidence