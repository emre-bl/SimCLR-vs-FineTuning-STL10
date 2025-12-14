"""
Defines the model architectures.
Overfitting fix: Added Dropout to the Baseline model.
"""

import torch
import torch.nn as nn
from torchvision import models
import config

def get_baseline_model():
    """Ex 1: ResNet-50 from scratch with Dropout."""
    model = models.resnet50(weights=None)
    
    # Replace final layer with Dropout + Linear
    # Dropout significantly reduces overfitting by randomly zeroing neurons
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, config.NUM_CLASSES)
    )
    return model

def get_finetune_model():
    """Ex 2: ResNet-50 pre-trained on ImageNet."""
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
    return model

class SimCLRModel(nn.Module):
    """Ex 3: Backbone + Projection Head."""
    def __init__(self, projection_dim=config.PROJECTION_DIM):
        super().__init__()
        
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.n_features = resnet.fc.in_features
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.n_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.size(0), -1) 
        z = self.projection_head(h)
        return z