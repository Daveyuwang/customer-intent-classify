"""
Training script for customer intent recognition models.
"""

import argparse
import torch
import torch.nn as nn
from config import (
    DEVICE, N_CLASS, MAX_LEN, MODEL_NAME, 
    LEARNING_RATE_BERT, LEARNING_RATE_CNN_LSTM, 
    LEARNING_RATE_TRANSFORMER, EPOCHS, INTENT_CLASSES
)
from utils.data_processing import prepare_data, create_data_loaders
from utils.training import train_test_epoch, save_trained_models
from utils.evaluation import evaluate_model
from utils.visualization import plot_acc_loss, plot_confusion_matrix, plot_model_comparison

from models.bert_model import BertModel
from models.cnn_model import TextCNN
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.combined_model import BERT_CNN_LSTM

def train_model(model, train_dl, test_dl, learning_rate, epochs):
    """
    Train a model and return performance metrics.
    
    Args:
        model (nn.Module): Model to train
        train_dl (DataLoader): Training data loader
        test_dl (DataLoader): Test data loader
        learning_rate (float): Learning rate
        epochs (int): Number of epochs
        
    Returns:
        tuple: Training history and evaluation metrics
    """
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print(f"\nTraining {model.__class__.__name__}...")
    history = train_test_epoch(train_dl, test_dl, model, loss_fn, optimizer, epochs)
    
    # Evaluate the model
    print(f"\nEvaluating {model.__class__.__name__}...")
    results = evaluate_model(model, test_dl, INTENT_CLASSES)
    
    return history, results

def main(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments
    """
    print(f"Using device: {DEVICE}")
    
    # Prepare data
    train_df, test_df, label_encoder = prepare_data(args.train_file, args.test_file)
    train_dl, test_dl, tokenizer = create_data_loaders(train_df, test_df)
    
    # Initialize models
    models = {
        "bert": BertModel().to(DEVICE),
        "cnn": TextCNN().to(DEVICE),
        "lstm": LSTMModel().to(DEVICE),
        "transformer": TransformerModel().to(DEVICE),
        "combined": BERT_CNN_LSTM().to(DEVICE)
    }
    
    # Learning rates for different models
    learning_rates = {
        "bert": LEARNING_RATE_BERT,
        "cnn": LEARNING_RATE_CNN_LSTM,
        "lstm": LEARNING_RATE_CNN_LSTM,
        "transformer": LEARNING_RATE_TRANSFORMER,
        "combined": LEARNING_RATE_BERT
    }
    
    # Train and evaluate each model
    histories = {}
    results = {}
    accuracies = []
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()} model")
        print(f"{'='*50}")
        
        history, result = train_model(
            model, 
            train_dl, 
            test_dl, 
            learning_rates[model_name], 
            args.epochs or EPOCHS
        )
        
        histories[model_name] = history
        results[model_name] = result
        accuracies.append(result['accuracy'])
        
        # Plot training history
        if args.plot:
            plot_acc_loss(
                history[1],  # train_acc_list
                history[3],  # test_acc_list
                history[0],  # train_loss_list
                history[2]   # test_loss_list
            )
            
            # Plot confusion matrix
            plot_confusion_matrix(
                result['confusion_matrix'],
                class_names=INTENT_CLASSES,
                normalize=True
            )
    
    # Save trained models
    save_trained_models(models, args.output_prefix)
    
    # Plot model comparison
    if args.plot:
        plot_model_comparison(
            list(models.keys()),
            accuracies
        )
    
    # Print summary of results
    print("\nModel Accuracy Comparison:")
    print("-" * 40)
    for model_name, result in results.items():
        print(f"{model_name.upper()}: {result['accuracy']:.4f}")
    
    # Identify best model
    best_model_name = list(results.keys())[accuracies.index(max(accuracies))]
    print(f"\nBest model: {best_model_name.upper()} with accuracy {max(accuracies):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train intent classification models")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--test_file", type=str, help="Path to test data CSV (optional)")
    parser.add_argument("--epochs", type=int, help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--output_prefix", type=str, default="", help="Prefix for saved model files")
    parser.add_argument("--plot", action="store_true", help="Generate plots during training")
    
    args = parser.parse_args()
    main(args)