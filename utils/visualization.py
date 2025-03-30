"""
Visualization utilities for customer intent recognition.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_acc_loss(train_acc_list, test_acc_list, train_loss_list, test_loss_list, epochs=None):
    """
    Plot accuracy and loss curves for training and validation.
    
    Args:
        train_acc_list (list): Training accuracy values
        test_acc_list (list): Validation accuracy values
        train_loss_list (list): Training loss values
        test_loss_list (list): Validation loss values
        epochs (int, optional): Number of epochs
    """
    if epochs is None:
        epochs = len(train_acc_list)
    
    epochs_range = range(epochs)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_list, label='Training Accuracy')
    plt.plot(epochs_range, test_acc_list, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_list, label='Training Loss')
    plt.plot(epochs_range, test_loss_list, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def plot_confusion_matrix(cm, class_names=None, normalize=False, figsize=(10, 10), cmap='Blues'):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list, optional): List of class names
        normalize (bool): Whether to normalize the confusion matrix
        figsize (tuple): Figure size
        cmap (str): Colormap name
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.3f'
    else:
        title = 'Confusion Matrix'
        fmt = 'g'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap=cmap, 
        cbar=True,
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto"
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(labels, class_names=None, figsize=(12, 6)):
    """
    Plot class distribution.
    
    Args:
        labels (list or array): Label values
        class_names (list, optional): List of class names
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    if class_names:
        # Count occurrences of each class
        counts = np.bincount(labels)
        # Use class names for x-axis
        plt.bar(class_names, counts)
        plt.xticks(rotation=45, ha='right')
    else:
        # Plot histogram
        plt.hist(labels, bins=np.arange(0, max(labels) + 1.5) - 0.5)
        plt.xticks(np.arange(0, max(labels) + 1))
    
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_text_length_distribution(text_lengths, bins=20, figsize=(10, 6)):
    """
    Plot distribution of text lengths.
    
    Args:
        text_lengths (list or array): Text lengths
        bins (int): Number of bins for histogram
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.hist(text_lengths, bins=bins)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.axvline(x=np.mean(text_lengths), color='r', linestyle='--', 
                label=f'Mean: {np.mean(text_lengths):.2f}')
    plt.axvline(x=np.median(text_lengths), color='g', linestyle='--', 
                label=f'Median: {np.median(text_lengths):.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confidence_distribution(predictions, true_labels, figsize=(10, 6)):
    """
    Plot distribution of model confidence for correct and incorrect predictions.
    
    Args:
        predictions (numpy.ndarray): Predicted probabilities for each class
        true_labels (list or array): True label values
        figsize (tuple): Figure size
    """
    # Get the confidence scores (max probability)
    confidence_scores = np.max(predictions, axis=1)
    
    # Get predicted labels
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Determine correct and incorrect predictions
    correct_mask = predicted_labels == true_labels
    
    # Separate confidence scores for correct and incorrect predictions
    correct_confidence = confidence_scores[correct_mask]
    incorrect_confidence = confidence_scores[~correct_mask]
    
    plt.figure(figsize=figsize)
    
    # Plot histograms
    plt.hist(correct_confidence, alpha=0.7, bins=20, label='Correct Predictions')
    plt.hist(incorrect_confidence, alpha=0.7, bins=20, label='Incorrect Predictions')
    
    plt.title('Distribution of Model Confidence')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_model_comparison(model_names, accuracy_scores, figsize=(12, 6)):
    """
    Plot comparison of model accuracy scores.
    
    Args:
        model_names (list): List of model names
        accuracy_scores (list or array): Corresponding accuracy scores
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create bar chart
    bars = plt.bar(model_names, accuracy_scores, width=0.6)
    
    # Add accuracy values on top of bars
    for bar, score in zip(bars, accuracy_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{score:.4f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, max(accuracy_scores) + 0.1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()