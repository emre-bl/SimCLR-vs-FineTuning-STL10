"""
Runs  Fine-Tuning with Early Stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import config
import data_setup
import models
import utils

def main():
    print(f"--- Experiment 2: Fine-Tuning ---")
    print(f"Using device: {config.DEVICE}")

    transform = data_setup.get_finetune_transforms()
    
    train_loader = data_setup.get_stl10_loaders('train', transform, config.BATCH_SIZE_FINETUNE)
    val_loader = data_setup.get_stl10_loaders('test', transform, config.BATCH_SIZE_FINETUNE, shuffle=False)

    model = models.get_finetune_model().to(config.DEVICE)

    #Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    params_to_update = [
        {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": config.LR_FINETUNE_BACKBONE},
        {"params": model.fc.parameters(), "lr": config.LR_FINETUNE_HEAD}
    ]
    optimizer = optim.Adam(params_to_update, lr=config.LR_FINETUNE_HEAD)

    # Early Stopping
    early_stopping = utils.EarlyStopping(patience=5, verbose=True, path='finetune_model.pth')

    # Training Loop
    print(f"Starting training for {config.EPOCHS_FINETUNE} epochs...")
    
    for epoch in range(config.EPOCHS_FINETUNE):
        # Train
        train_loss, train_acc, train_f1 = utils.train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_f1 = utils.evaluate(
            model, val_loader, criterion, config.DEVICE
        )
        
        print(f"Epoch {epoch+1}/{config.EPOCHS_FINETUNE}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # Early Stopping Check
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print("Fine-tuning finished.")

if __name__ == "__main__":
    main()