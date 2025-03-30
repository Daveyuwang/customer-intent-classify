"""
TextCNN model for intent classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LEN, N_CLASS, DEVICE

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
voc_size = len(tokenizer.vocab)

class TextCNN(nn.Module):
    """
    TextCNN model for intent classification.
    
    Uses convolutional layers with different kernel sizes
    to capture different n-gram features from the text.
    """
    def __init__(self):
        super().__init__()
        emb_dim = 100
        kernels = [3, 4, 5]
        kernel_number = [150, 150, 150]
        
        # Embedding layer with vocab_size same as BERT tokenizer
        self.embd = nn.Embedding(voc_size, emb_dim)
        
        # Convolution layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(MAX_LEN, kn, k, padding=k) 
            for (k, kn) in zip(kernels, kernel_number)
        ])
        
        self.dropout = nn.Dropout(0.1)
        self.lin = nn.Linear(sum(kernel_number), 50)
        self.out = nn.Linear(50, N_CLASS)

    def forward(self, x):
        """
        Forward pass through the TextCNN model.
        
        Args:
            x (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Logits for each class of shape (batch_size, n_class)
        """
        # x shape: (batch_size, seq_len=25)
        # embedded shape: (batch_size, seq_len=25, emb_dim=100)
        x = self.embd(x)
        
        # Apply convolutions with different kernel sizes
        x = [F.relu(conv(x)) for conv in self.convs]
        
        # Global max pooling over each convolutional output
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
        # Concatenate results from each kernel
        x = torch.cat(x, 1)
        
        x = self.dropout(x)
        x = F.relu(x)
        x = self.lin(x)  # (batch_size, 50)
        x = self.out(x)  # (batch_size, n_class=11)
        
        return x