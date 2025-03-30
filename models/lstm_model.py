"""
LSTM model for intent classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LEN, N_CLASS, DEVICE

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
voc_size = len(tokenizer.vocab)

class LSTMModel(nn.Module):
    """
    LSTM model for intent classification.
    
    Uses a Long Short-Term Memory network to capture
    sequential dependencies in the text.
    """
    def __init__(self):
        super().__init__()
        emb_dim = 100
        
        # Embedding layer
        self.embd = nn.Embedding(voc_size, emb_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(emb_dim, 50, batch_first=True)
        
        # Flatten and linear layers
        self.flatten = nn.Flatten()
        self.lin = nn.Linear(1250, 100)  # 25 * 50 = 1250, then reduce to 100
        self.out = nn.Linear(100, N_CLASS)

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Logits for each class of shape (batch_size, n_class)
        """
        # x shape: (batch_size, seq_len=25)
        x = self.embd(x)  # (batch_size, 25, 100)
        
        # Process through LSTM
        x, _ = self.lstm(x)  # (batch_size, 25, 50)
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x)  # (batch_size, 25*50=1250)
        x = self.lin(x)      # (batch_size, 100)
        x = F.relu(x)
        x = self.out(x)      # (batch_size, 11)
        
        return x