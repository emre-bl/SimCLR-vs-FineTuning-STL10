"""
Runs Experiment 1: Training from Scratch (Baseline).
Overfitting fix: Added Weight Decay (L2 Regularization).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import config
import data_setup
import models
import utils
import os

def main():
    print(f"--- Experiment 1: Training from Scratch (Baseline) ---")
    
    # Paths
    save_path = os.path.join(config.RESULTS_DIR, 'baseline_model.pth')

    # Data & Model
    # Now uses the stronger augmentations defined in data_setup.py
    train_loader = data_setup.get_stl10_loaders('train', data_setup.get_baseline_transforms(train=True), config.BATCH_SIZE_BASELINE)
    test_loader = data_setup.get_stl10_loaders('test', data_setup.get_baseline_transforms(train=False), config.BATCH_SIZE_BASELINE, shuffle=False)
    
    model = models.get_baseline_model().to(config.DEVICE)

    # Training Setup
    criterion = nn.CrossEntropyLoss()
    
    # ADDED: weight_decay=1e-4 (L2 Regularization)
    # This penalizes large weights, preventing the model from becoming too complex
    optimizer = optim.Adam(model.parameters(), lr=config.LR_BASELINE, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS_BASELINE)
    early_stopping = utils.EarlyStopping(patience=10, verbose=True, path=save_path)

    # Loop
    print(f"Starting training for {config.EPOCHS_BASELINE} epochs with regularization...")
    for epoch in range(config.EPOCHS_BASELINE):
        train_loss, train_acc, _ = utils.train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc, _ = utils.evaluate(model, test_loader, criterion, config.DEVICE)
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

if __name__ == "__main__":
    main()