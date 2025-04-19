import os
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch

def prepare_data(data_dir, batch_size, augment):
    """
    Prepare training, validation, and test data loaders.
    The original dataset is assumed to have a 'train' and 'test' folder.
    A validation set is created by a stratified 80/20 split of the train folder.
    
    Args:
        data_dir (str): Path to dataset folder.
        batch_size (int): Batch size.
        augment (bool): Whether to apply data augmentation.
    
    Returns:
        train_loader, val_loader, test_loader, num_classes, class_weights
    """
    # Enhanced augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]) if augment else transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Validation uses a fixed transform without augmentation.
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Test transform same as validation.
    test_transform = val_transform

    # Load full training dataset
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    num_classes = len(full_dataset.classes)
    targets = [s[1] for s in full_dataset.samples]

    # Stratified split (80% train, 20% val)
    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, stratify=targets, random_state=42)
    train_dataset = Subset(full_dataset, train_idx)
    # For validation, use non-augmented transform:
    full_dataset_no_aug = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=val_transform)
    val_dataset = Subset(full_dataset_no_aug, val_idx)

    # Compute class weights for balanced training
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=[targets[i] for i in train_idx])
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Load test dataset (assumed available in 'test' folder)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes, class_weights
