"""
Runs Linear Evaluation with Early Stopping.
Includes robust weight loading to handle different key formats.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models as torchvision_models
import config
import data_setup
import utils
import os

def load_backbone_weights(backbone, path):
    """
    Robustly loads backbone weights, handling common key mismatches.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")
        
    print(f"Loading weights from {path}...")
    state_dict = torch.load(path)
    
    # Check if it's a full checkpoint dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    new_state_dict = {}
    
    # 1. Detect if keys are from nn.Sequential (0.weight, 1.weight...)
    # This happens if backbone was defined as nn.Sequential(*list(resnet.children())[:-1])
    is_sequential_keys = any(k.startswith('0.') or k.startswith('1.') for k in state_dict.keys())
    
    if is_sequential_keys:
        print("Detected nn.Sequential keys (0., 1., ...). Mapping to ResNet names...")
        # Map indices to ResNet50 named layers
        # ResNet50 children order: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        layer_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
        
        for key, value in state_dict.items():
            # key format: "0.weight", "4.1.conv1.weight"
            parts = key.split('.')
            idx = int(parts[0])
            
            if idx < len(layer_names):
                layer_name = layer_names[idx]
                # Reconstruct key: "layer1" + ".1.conv1.weight"
                new_key = layer_name + '.' + '.'.join(parts[1:])
                # Clean up double dots if any (e.g. from empty suffixes)
                new_key = new_key.rstrip('.')
                new_state_dict[new_key] = value
                
    else:
        # 2. Standard or 'backbone.' prefix handling
        for key, value in state_dict.items():
            new_key = key
            
            # Remove 'backbone.' prefix if present (from full SimCLR model)
            if new_key.startswith('backbone.'):
                new_key = new_key.replace('backbone.', '')
                
            # Skip projection head weights
            if 'projection_head' in new_key or 'projector' in new_key:
                continue
                
            # Skip 'fc' weights (since we are replacing it anyway)
            if 'fc' in new_key:
                continue
                
            new_state_dict[new_key] = value

    # Load into the backbone
    # strict=False is important because we might be missing 'fc' or have mismatched avgpool keys
    msg = backbone.load_state_dict(new_state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}")
    # Verify important keys are loaded
    if 'conv1.weight' in msg.missing_keys and not '0.weight' in state_dict:
         print("WARNING: 'conv1.weight' was not loaded! Check weight file format.")

def main():
    print(f"--- Experiment 3 (Stage 2): Linear Evaluation ---")
    print(f"Using device: {config.DEVICE}")

    # 1. Dataloaders
    train_transform = data_setup.get_baseline_transforms(train=True)
    test_transform = data_setup.get_baseline_transforms(train=False)
    
    train_loader = data_setup.get_stl10_loaders('train', train_transform, config.BATCH_SIZE_LINEAR)
    val_loader = data_setup.get_stl10_loaders('test', test_transform, config.BATCH_SIZE_LINEAR, shuffle=False)

    # 2. Model
    # Load standard ResNet50
    backbone = torchvision_models.resnet50(weights=None)
    num_ftrs = backbone.fc.in_features
    backbone.fc = nn.Identity()
    
    # Try loading from available files
    try:
        if os.path.exists("simclr_backbone.pth"):
            load_backbone_weights(backbone, "simclr_backbone.pth")
        elif os.path.exists("simclr_full_model.pth"):
            print("Backbone file not found, trying full model...")
            load_backbone_weights(backbone, "simclr_full_model.pth")
        elif os.path.exists("simclr_full_model_final.pth"):
             print("Backbone file not found, trying final full model...")
             load_backbone_weights(backbone, "simclr_full_model_final.pth")
        else:
            print("Error: No SimCLR weight files found. Run train_simclr.py first.")
            return
    except Exception as e:
        print(f"Critical Error loading weights: {e}")
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
    # Optimize *only* the new linear layer
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