"""
Data processing utilities for customer intent recognition.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LEN, BATCH_SIZE

def get_text_length(item):
    """
    Get the number of words in a text.
    
    Args:
        item (str): Input text
        
    Returns:
        int: Number of words
    """
    return len(item.split(' '))

def prepare_data(training_csv, testing_csv=None):
    """
    Prepare data for training and evaluation.
    
    Args:
        training_csv (str): Path to training CSV file
        testing_csv (str, optional): Path to testing CSV file
        
    Returns:
        tuple: Processed training and testing dataframes, and label encoder
    """
    # Load and shuffle the training data
    train_df = pd.read_csv(training_csv).sample(frac=1).reset_index(drop=True)
    
    # Calculate text lengths
    train_df['len'] = train_df['utterance'].astype('str').map(get_text_length)
    
    # Encode labels
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['category'])
    
    # Load and process test data if provided
    if testing_csv:
        test_df = pd.read_csv(testing_csv)
        test_df['label'] = le.transform(test_df['category'])
    else:
        # If no test file, split the training data
        train_size = int(0.8 * len(train_df))
        test_df = train_df[train_size:].reset_index(drop=True)
        train_df = train_df[:train_size].reset_index(drop=True)
    
    return train_df, test_df, le

class CustomDataset(Dataset):
    """
    Custom dataset for loading and tokenizing text data.
    """
    def __init__(self, pd_data, tokenizer):
        """
        Initialize the dataset.
        
        Args:
            pd_data (pandas.DataFrame): DataFrame with 'utterance' and 'label' columns
            tokenizer: Tokenizer for processing text
        """
        self.pd_data = pd_data
        self.text_list = list(pd_data['utterance'].astype('str'))
        self.label_list = list(pd_data['label'])
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.pd_data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: Tokenized text and label
        """
        one_text = self.text_list[idx]
        one_result = self.tokenizer(
            one_text,
            padding='max_length', 
            max_length=MAX_LEN, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Extract token IDs
        one_ids = one_result.input_ids[0]
        
        # Get corresponding label
        one_label = self.label_list[idx]
        one_label = torch.tensor(one_label).long()
        
        return one_ids, one_label

def create_data_loaders(train_df, test_df):
    """
    Create data loaders for training and testing.
    
    Args:
        train_df (pandas.DataFrame): Training data
        test_df (pandas.DataFrame): Testing data
        
    Returns:
        tuple: Training and testing data loaders
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_ds = CustomDataset(train_df, tokenizer)
    test_ds = CustomDataset(test_df, tokenizer)
    
    # Create data loaders
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    return train_dl, test_dl, tokenizer