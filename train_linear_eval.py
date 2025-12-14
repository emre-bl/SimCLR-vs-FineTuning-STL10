# train_linear_eval.py
"""
Runs Experiment 3 (Stage 2): Linear Evaluation with Early Stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models as torchvision_models
import config
import data_setup
import utils

def main():
    print(f"--- Experiment 3 (Stage 2): Linear Evaluation ---")
    print(f"Using device: {config.DEVICE}")

    # 1. Dataloaders
    train_transform = data_setup.get_baseline_transforms(train=True)
    test_transform = data_setup.get_baseline_transforms(train=False)
    
    train_loader = data_setup.get_stl10_loaders('train', train_transform, config.BATCH_SIZE_LINEAR)
    val_loader = data_setup.get_stl10_loaders('test', test_transform, config.BATCH_SIZE_LINEAR, shuffle=False)

    # 2. Model
    backbone = torchvision_models.resnet50(weights=None)
    num_ftrs = backbone.fc.in_features
    backbone.fc = nn.Identity()
    
    try:
        backbone.load_state_dict(torch.load("simclr_backbone.pth"))
        print("Loaded SSL weights from simclr_backbone.pth")
    except FileNotFoundError:
        print("Error: simclr_backbone.pth not found. Run train_simclr.py first.")
        return

    # FREEZE the backbone
    for param in backbone.parameters():
        param.requires_grad = False
        
    model = nn.Sequential(
        backbone,
        nn.Linear(num_ftrs, config.NUM_CLASSES)
    ).to(config.DEVICE)

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # model[1] is the Linear layer
    optimizer = optim.Adam(model[1].parameters(), lr=config.LR_LINEAR)
    
    # 4. Early Stopping
    early_stopping = utils.EarlyStopping(patience=10, verbose=True, path='linear_eval_model.pth')

    # 5. Training Loop
    print(f"Starting linear evaluation for {config.EPOCHS_LINEAR} epochs...")
    
    for epoch in range(config.EPOCHS_LINEAR):
        # Train
        train_loss, train_acc, train_f1 = utils.train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_f1 = utils.evaluate(
            model, val_loader, criterion, config.DEVICE
        )
        
        print(f"Epoch {epoch+1}/{config.EPOCHS_LINEAR}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # Early Stopping Check
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
            
    print("Linear evaluation finished.")

if __name__ == "__main__":
    main()