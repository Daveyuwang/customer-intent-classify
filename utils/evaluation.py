"""
Evaluation utilities for customer intent recognition models.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from config import DEVICE

def get_predict_score(model, test_dl):
    """
    Get prediction scores for a test dataset.
    
    Args:
        model (nn.Module): Model to evaluate
        test_dl (DataLoader): Test data loader
        
    Returns:
        tuple: Probability arrays, predicted labels, and true labels
    """
    predict_list = []
    label_list = []
    predict_pro_list = []
    m_softmax = nn.Softmax(dim=1)
    
    model.eval()
    with torch.no_grad():
        for (b_text, y) in test_dl:
            b_text, y = b_text.to(DEVICE), y.to(DEVICE)

            predict_score = model(b_text)
            predict_pro = m_softmax(predict_score)
            predict_label = np.argmax(predict_score.detach().cpu().numpy(), axis=1)

            predict_pro_list.append(predict_pro.detach().cpu().numpy())
            predict_list.append(predict_label)
            label_list.append(y.detach().cpu().numpy())

    # Convert lists to arrays
    predict_pro_array = np.vstack(predict_pro_list)
    predict_array = np.hstack(predict_list)
    label_array = np.hstack(label_list)
    
    return predict_pro_array, predict_array, label_array

def evaluate_model(model, test_dl, class_names=None):
    """
    Evaluate a model on a test dataset with detailed metrics.
    
    Args:
        model (nn.Module): Model to evaluate
        test_dl (DataLoader): Test data loader
        class_names (list, optional): List of class names
        
    Returns:
        dict: Evaluation metrics
    """
    # Get predictions
    predict_pro_array, predict_array, label_array = get_predict_score(model, test_dl)
    
    # Calculate classification report
    if class_names:
        report = classification_report(
            label_array, 
            predict_array, 
            target_names=class_names, 
            digits=3,
            output_dict=True
        )
    else:
        report = classification_report(
            label_array, 
            predict_array, 
            digits=3,
            output_dict=True
        )
    
    # Calculate confusion matrix
    cm = confusion_matrix(label_array, predict_array)
    
    # Calculate normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Calculate accuracy
    accuracy = report['accuracy']
    
    # Prepare results
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'predict_array': predict_array,
        'label_array': label_array,
        'predict_pro_array': predict_pro_array
    }
    
    return results