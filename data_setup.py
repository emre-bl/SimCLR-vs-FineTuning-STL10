"""
Defines all data transformations and DataLoader functions.
Overfitting fix: Added stronger augmentations for baseline training.
"""

import torch
from torchvision import transforms, datasets
from torchvision.models import ResNet50_Weights
import config

class ContrastiveTransformations:
    """
    Applies the same set of aggressive augmentations twice to create two views.
    """
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]

def get_simclr_transforms():
    """
    Aggressive augmentations for SimCLR.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def get_baseline_transforms(train=True):
    """
    Updated: Stronger augmentations to prevent overfitting.
    """
    if train:
        return transforms.Compose([
            # Crop and resize forces the model to learn parts of the object
            transforms.RandomResizedCrop(96, scale=(0.8, 1.0)), 
            transforms.RandomHorizontalFlip(),
            # Color jitter makes model invariant to lighting changes
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # Test transforms (No augmentation)
        return transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def get_finetune_transforms():
    """
    ImageNet transforms for pre-trained models.
    """
    weights = ResNet50_Weights.DEFAULT
    return weights.transforms()

def get_stl10_loaders(split, transform, batch_size, shuffle=True):
    dataset = datasets.STL10(
        root=config.DATA_DIR,
        split=split,
        download=True,
        transform=transform
    )
    # num_workers=0 ensures stability on Windows
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0, 
        pin_memory=True
    )
    return loader

def get_simclr_loader(batch_size):
    transform = ContrastiveTransformations(
        base_transforms=get_simclr_transforms(),
        n_views=2
    )
    dataset = datasets.STL10(
        root=config.DATA_DIR,
        split='unlabeled', 
        download=True,
        transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    return loader