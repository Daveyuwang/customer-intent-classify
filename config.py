"""
Configuration parameters for the customer intent recognition system.
"""

import torch

# Model parameters
MAX_LEN = 25
N_CLASS = 11
MODEL_NAME = 'bert-base-uncased'

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Intent classes
INTENT_CLASSES = [
    'ACCOUNT', 'CANCELLATION_FEE', 'CONTACT', 'DELIVERY', 'FEEDBACK',
    'INVOICE', 'NEWSLETTER', 'ORDER', 'PAYMENT', 'REFUND',
    'SHIPPING_ADDRESS'
]

# Training parameters
BATCH_SIZE = 8
LEARNING_RATE_BERT = 2e-5
LEARNING_RATE_CNN_LSTM = 1e-3
LEARNING_RATE_TRANSFORMER = 2e-5
EPOCHS = 5

# LLM API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com"