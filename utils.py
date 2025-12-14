"""
Utility functions for training, evaluation, NTXentLoss, and Early Stopping.
Updated: Added get_all_predictions for detailed analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
import config

class NTXentLoss(nn.Module):
    """
    Implementation of the Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        # Concatenate the two views
        z = torch.cat((z_i, z_j), dim=0)
        
        # Calculate cosine similarity matrix
        sim_matrix = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
        
        # Create positive-pair mask
        n = 2 * self.batch_size
        mask = torch.eye(n, dtype=torch.bool, device=z.device)
        
        # Mask out self-similarity (diagonal)
        sim_matrix = sim_matrix[~mask].view(n, n - 1)
        
        # Positive pairs are (i, i+N) and (i+N, i)
        pos_pairs = torch.cat((z_j, z_i), dim=0)
        
        pos_sim = self.similarity_f(z.unsqueeze(1), pos_pairs.unsqueeze(1))
        
        # Concatenate positive similarity with negative similarities
        logits = torch.cat((pos_sim, sim_matrix), dim=1)
        logits /= self.temperature
        
        labels = torch.zeros(n, dtype=torch.long, device=z.device)
        
        loss = self.criterion(logits, labels)
        return loss / n

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    if len(all_labels) > 0:
        accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
        f1 = f1_score(all_labels, all_preds, average='macro')
    else:
        accuracy, f1 = 0.0, 0.0
    
    return avg_loss, accuracy, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    if len(all_labels) > 0:
        accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
        f1 = f1_score(all_labels, all_preds, average='macro')
    else:
        accuracy, f1 = 0.0, 0.0
    
    return avg_loss, accuracy, f1

def get_all_predictions(model, dataloader, device):
    """
    Returns all true labels and predicted labels for confusion matrix generation.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)