"""
Runs Experiment 3 (Stage 1): Self-Supervised Pre-training (SimCLR) with Early Stopping.
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import config
import data_setup
import models
import utils

def validate_simclr(model, loader, criterion, device):
    """
    Calculates contrastive loss on a validation set (without backprop).
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (views, _) in loader:
            # views is [tensor_view1, tensor_view2]
            images_v1, images_v2 = views
            images_v1 = images_v1.to(device)
            images_v2 = images_v2.to(device)
            
            z_i = model(images_v1)
            z_j = model(images_v2)
            
            loss = criterion(z_i, z_j)
            total_loss += loss.item() * images_v1.size(0)
            
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def main():
    print(f"--- Experiment 3 (Stage 1): SimCLR Pre-training ---")
    print(f"Using device: {config.DEVICE}")

    train_loader = data_setup.get_simclr_loader(config.BATCH_SIZE_SIMCLR)
    # Validation loader uses Test set but with Contrastive Transforms
    val_loader = data_setup.get_simclr_validation_loader(config.BATCH_SIZE_SIMCLR)

    model = models.SimCLRModel().to(config.DEVICE)

    #Loss and Optimizer
    criterion = utils.NTXentLoss(batch_size=config.BATCH_SIZE_SIMCLR).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR_SIMCLR, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS_SIMCLR)

    # Early Stopping (Monitoring Contrastive Loss)
    # We save the WHOLE SimCLR model first, then extract backbone later
    early_stopping = utils.EarlyStopping(patience=2, verbose=True, path='simclr_full_model.pth')

    # Training Loop
    print(f"Starting SimCLR training for {config.EPOCHS_SIMCLR} epochs...")
    
    for epoch in range(config.EPOCHS_SIMCLR):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS_SIMCLR}")
        
        for (views, labels) in pbar:
            images_v1, images_v2 = views
            images_v1 = images_v1.to(config.DEVICE)
            images_v2 = images_v2.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            z_i = model(images_v1)
            z_j = model(images_v2)
            
            loss = criterion(z_i, z_j)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images_v1.size(0)
            pbar.set_postfix({"loss": loss.item()})

        scheduler.step()
        
        # Validation Step
        val_loss = validate_simclr(model, val_loader, criterion, config.DEVICE)
        train_avg_loss = total_loss / len(train_loader.dataset)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early Stopping Check
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # 6. Load Best Model and Save ONLY Backbone
    print("Loading best model to save backbone...")
    model.load_state_dict(torch.load('simclr_full_model.pth'))
    torch.save(model.backbone.state_dict(), "simclr_backbone.pth")
    print("Saved best simclr_backbone.pth")

if __name__ == "__main__":
    main()