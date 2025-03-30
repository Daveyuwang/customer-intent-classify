"""
Training utilities for customer intent recognition models.
"""

import torch
from config import DEVICE

def train(dataloader, model, loss_fn, optimizer):
    """
    Train the model for one epoch.
    
    Args:
        dataloader (DataLoader): Training data loader
        model (nn.Module): Model to train
        loss_fn: Loss function
        optimizer: Optimizer
    """
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    """
    Test the model on a dataset.
    
    Args:
        dataloader (DataLoader): Test data loader
        model (nn.Module): Model to evaluate
        loss_fn: Loss function
        
    Returns:
        tuple: Average loss and accuracy
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for (X, y) in dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, correct

def train_test_epoch(train_dl, test_dl, model, loss_fn, optimizer, epochs):
    """
    Train and evaluate a model for multiple epochs.
    
    Args:
        train_dl (DataLoader): Training data loader
        test_dl (DataLoader): Test data loader
        model (nn.Module): Model to train
        loss_fn: Loss function
        optimizer: Optimizer
        epochs (int): Number of epochs to train
        
    Returns:
        tuple: Lists of training and testing loss and accuracy
    """
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dl, model, loss_fn, optimizer)

        train_loss, train_correct = test(train_dl, model, loss_fn)
        test_loss, test_correct = test(test_dl, model, loss_fn)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_correct)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_correct)

    print("Training complete!")
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list

def save_trained_models(model_dict, filename_prefix=""):
    """
    Save trained models to disk.
    
    Args:
        model_dict (dict): Dictionary of models to save
        filename_prefix (str, optional): Prefix for filenames
    """
    print("Saving models...")
    
    for model_name, model in model_dict.items():
        filename = f"{filename_prefix}{model_name}_intent_model.pth"
        torch.save(model.state_dict(), filename)
        print(f"{model_name.capitalize()} model saved as '{filename}'")
    
    print("All models saved successfully!")