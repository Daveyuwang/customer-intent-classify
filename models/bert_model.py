"""
BERT model for intent classification.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from config import MODEL_NAME, N_CLASS, DEVICE

class BertModel(nn.Module):
    """
    BERT model for intent classification.
    
    Uses a pre-trained BERT model with a classification head
    to predict intent categories.
    """
    def __init__(self):
        super().__init__()
        # Create a standard Transformers classification model with n_class outputs
        self.bert_block = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=N_CLASS
        ).to(DEVICE)

    def forward(self, text_b):
        """
        Forward pass through the BERT model.
        
        Args:
            text_b (torch.Tensor): Input token IDs
            
        Returns:
            torch.Tensor: Logits for each class
        """
        x = self.bert_block(text_b)
        return x.logits  # (batch_size, n_class)