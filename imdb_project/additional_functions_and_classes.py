
"""
A series of functions for IMDB Review project.

"""
import os
import torch
import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.notebook import tqdm
from typing import Dict, List, Tuple
from nltk.tokenize import word_tokenize

class Vocabulary:
    """
    Create vocabulary and its inverse from a given dataset.

    Args:
        data (list of str): List of strings representing the text data.
        vocab_size (int): Size of the vocabulary to be created.

    Returns:
        tuple: A tuple containing two dictionaries - the vocabulary and its inverse.
            The vocabulary maps words to unique integer IDs.
            The inverse vocabulary maps integer IDs to their corresponding words.

    Example usage:
    >>> data = ["Hello world!", "How are you?", "Hello again!"]
    >>> vocab_size = 5
    >>> vocab, inverse_vocab = self.create_vocabs(data, vocab_size)
    >>> print(vocab)
    {'Hello': 2, 'world': 3, 'How': 4, 'are': 5, 'you': 6}
    >>> print(inverse_vocab)
    {2: 'Hello', 3: 'world', 4: 'How', 5: 'are', 6: 'you'}
    """
    def __init__(self, data, vocab_size=30000):
        self.vocab, self.inverse_vocab = self.create_vocabs(data, vocab_size)

    def create_vocabs(self, data, vocab_size):
        list_tokenized_train = [self.tokenize(text) for text in data]

        list_of_all_words = []
        for i in list_tokenized_train:
            list_of_all_words.extend(i)

        dictionary_of_all_words = Counter(list_of_all_words) 

        top_words = dictionary_of_all_words.most_common(vocab_size)
        new_dictionary = dict(top_words)

        vocab = {}
        for id, (key, _) in enumerate(new_dictionary.items()):
            vocab[key] = id + 2

        inverse_vocab = {}
        for id, (key, _) in enumerate(new_dictionary.items()):
            inverse_vocab[id + 2] = key

        return vocab, inverse_vocab

    def tokenize(self, text):
        return word_tokenize(text)

class Tokenizer:
    """
    Tokenizer class for converting text into tokenized format using a provided vocabulary.

    Args:
        vocab (Vocabulary): An instance of the Vocabulary class containing the vocabulary
                           and its inverse for tokenization.

    Attributes:
        vocab (dict): A dictionary mapping words to unique integer IDs.
        inverse_vocab (dict): A dictionary mapping integer IDs to their corresponding words.

    Methods:
        tokenize(tokens, inverse=False, max_length=128):
            Tokenizes the input text or tensor using the provided vocabulary.

    This class initializes with a Vocabulary instance, which contains the necessary mappings
    for tokenization. The 'tokenize' method allows for both forward and inverse tokenization.
    Forward tokenization converts a string of text into a sequence of integer IDs using the
    provided vocabulary. Inverse tokenization converts a tensor of integer IDs back into
    the original text using the inverse vocabulary.

    Example usage:
    >>> vocab = Vocabulary(data, vocab_size=30000)
    >>> tokenizer = Tokenizer(vocab)
    >>> tokens = tokenizer.tokenize("Hello world!")
    >>> print(tokens)
    tensor([2, 3, 0, 0, ...])  # Example output
    >>> inverse_tokens = tokenizer.tokenize(tokens, inverse=True)
    >>> print(inverse_tokens)
    "Hello world!"  # Example output
    """
    def __init__(self, vocab):
        self.vocab = vocab.vocab
        self.inverse_vocab = vocab.inverse_vocab

    def tokenize(self, tokens, inverse=False, max_length=128):
        
        if inverse == False:
            if not isinstance(tokens, str):
                raise ValueError("tokenization requires a string as input.")
            tokenized_text = []
            for word in tokens.split():
                if word in self.vocab:
                    tokenized_text.append(self.vocab[word])
                else:
                    tokenized_text.append(1)

            tokenized_text = tokenized_text + [0] * (max_length - len(tokenized_text))
            tokenized_text = tokenized_text[:max_length]

            return torch.LongTensor(tokenized_text)

        if inverse == True:
            if not isinstance(tokens, torch.Tensor):
                raise ValueError("Inverse tokenization requires a torch.Tensor as input.")
            retokenized_text = []
            for id in list(tokens.numpy()):
                if id in self.inverse_vocab:
                    retokenized_text.append(self.inverse_vocab[id])
                elif id == 0:
                    pass
                elif id == 1:
                    retokenized_text.append("<oov>")

            retokenized_text = " ".join(retokenized_text)

            return retokenized_text
    
def train_step(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device):
    """
    Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """    
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_train_pred = model(X)
        
        loss = loss_fn(y_train_pred, y)
        
        train_loss += loss.item() 
        
        optimizer.zero_grad()
        
        loss.backward() 
        
        optimizer.step()
        
        y_pred_class = torch.argmax(torch.softmax(y_train_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_train_pred)
        
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    return train_loss, train_acc

def test_step(
    model, 
    dataloader, 
    loss_fn,
    device):
    """
    Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    model.eval() 
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            test_pred_logits = model(X)
            
            loss = loss_fn(test_pred_logits, y)
            
            test_loss += loss.item()
            
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc

class ModelCheckpoint:
    """
    Saves the best-performing model based on validation loss during training.

    Args:
        save_path (str): The directory path where the model checkpoints will be saved.
        model_name (str): The name to be used when saving the model checkpoint file.
        save_best_only (bool): If True, only saves the model if it achieves a lower 
        validation loss than the previous best. If False, saves the model every time 
        the validation loss is checked. Default is True.
        verbose (int): Controls the verbosity of the output. A higher value will print 
        more information about the saving process.

    Attributes:
        save_path (str): The directory path where the model checkpoints will be saved.
        model_name (str): The name to be used when saving the model checkpoint file.
        save_best_only (bool): If True, only saves the model if it achieves a lower 
        validation loss than the previous best. If False, saves the model every time the validation loss is checked.
        verbose (int): Controls the verbosity of the output.
        best_loss (float): Tracks the best validation loss seen so far, initialized as positive infinity.
        counter (int): Keeps track of the number of times the validation loss has not improved.

    Methods:
        __init__(self, save_path, model_name, save_best_only, verbose): Initializes 
        the ModelCheckpoint object with the specified parameters.
        __call__(self, val_loss, model): Called after each epoch with the current 
        validation loss and model as arguments. Saves the model if it achieves a lower 
        validation loss based on the specified conditions.

        Example usage:
        checkpoint = ModelCheckpoint(save_path='models/', 
                                     model_name='best_model.pth', 
                                     save_best_only=True, 
                                     verbose=1)
                                     
        checkpoint(val_loss=0.1234, model=my_model)
    """

    def __init__(self, save_path, model_name, save_best_only, verbose):
        self.save_path = save_path
        self.model_name = model_name
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0

        os.makedirs(save_path, exist_ok=True)
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(self.save_path, self.model_name))
            if self.verbose > 0:
                print(f"Saved model with validation loss: {val_loss:.4f}.")  
        else:
            if not self.save_best_only:
                torch.save(model.state_dict(), os.path.join(self.save_path, self.model_name))
                if self.verbose > 0:
                    print(f"Saved model with validation loss: {val_loss:.4f}.")  
                    
            if self.verbose > 0:
                print(f"Model's validation loss: {self.best_loss:.4f} didn't improve.")
                
class EarlyStopping:
    """
    Monitors the validation loss during training and stops the training 
    process if the validation loss does not improve over a certain number of epochs.

    Args:
        patience (int): The number of consecutive epochs with no improvement 
        in validation loss before early stopping is triggered. Default is 5.
        min_delta (float): The minimum change in validation loss that is 
        considered an improvement. Default is 0, meaning any decrease in 
        validation loss is considered an improvement.

    Attributes:
    - counter (int): Keeps track of the number of consecutive epochs 
      with no improvement in validation loss.
    - best_loss (float): Tracks the best validation loss seen so far, 
      initialized as positive infinity.
    - early_stop (bool): Flag indicating whether early stopping should 
      be triggered.

    Methods:
    - __init__(self, patience=5, min_delta=0): Initializes the EarlyStopping 
      object with the specified patience and minimum delta values.
    - __call__(self, val_loss): Called after each epoch with the current 
      validation loss as an argument. Updates the internal state and returns 
      a boolean indicating whether early stopping should be triggered.
    """

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def train(
    model, 
    train_dataloader, 
    test_dataloader, 
    optimizer,
    loss_fn,
    epochs,
    device,
    model_checkpoint=None,
    early_stopping=None):
    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        model_checkpoint: A ModelCheckpoint instance for saving the model's results.
        early_stopping: An EarlyStopping instance for breaking out of the loop if the model doesn't improve.
        
    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
        train_acc: [...],
        test_loss: [...],
        test_acc: [...]} 
        For example if training for epochs=2: 
        {train_loss: [2.0616, 1.0537],
        train_acc: [0.3945, 0.3945],
        test_loss: [1.2641, 1.5706],
        test_acc: [0.3400, 0.2973]} 
    """
    results = {"train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
    }

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device)

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
            )

        if model_checkpoint is not None:
            model_checkpoint(test_loss, model)
            
        if early_stopping is not None:
            if early_stopping(val_loss=test_loss):
                print("Early stopping triggered")
                break

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

def plot_loss_curves(results):
    """
    Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
