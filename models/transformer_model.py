"""
Transformer model for intent classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LEN, N_CLASS, DEVICE

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
voc_size = len(tokenizer.vocab)

class TransformerModel(nn.Module):
    """
    Transformer model for intent classification.
    
    Uses a lightweight transformer encoder to capture
    contextual relationships in the text.
    """
    def __init__(self):
        super().__init__()
        emb_dim = 100
        
        # Embedding layer
        self.embd = nn.Embedding(voc_size, emb_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=2, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Output layers
        self.flatten = nn.Flatten()
        self.out = nn.Linear(2500, N_CLASS)  # 25 * 100 = 2500

    def forward(self, x):
        """
        Forward pass through the Transformer model.
        
        Args:
            x (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Logits for each class of shape (batch_size, n_class)
        """
        # x shape: (batch_size, seq_len=25)
        x = self.embd(x)  # (batch_size, 25, 100)
        
        # Process through transformer encoder
        x = self.transformer_encoder(x)  # still (batch_size, 25, 100)
        
        # Flatten and project to output classes
        x = self.flatten(x)  # (batch_size, 2500)
        x = self.out(x)      # (batch_size, 11)
        
        return x