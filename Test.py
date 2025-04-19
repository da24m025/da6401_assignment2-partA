import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import wandb

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#############################################
# Data Preparation
#############################################
def prepare_data(data_dir, batch_size, augment):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if augment else transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    num_classes = len(full_dataset.classes)
    targets = [s[1] for s in full_dataset.samples]

    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))), test_size=0.2,
        stratify=targets, random_state=42
    )
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(
        datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=val_transform),
        val_idx
    )

    class_weights = compute_class_weight(
        'balanced', classes=np.unique(targets),
        y=[targets[i] for i in train_idx]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes, class_weights

#############################################
# Model and Helper Functions
#############################################
class ResidualBlock(nn.Module):
    def __init__(self, conv, bn, activation, use_residual):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.activation = activation
        self.use_residual = use_residual

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        if self.use_residual and out.shape == x.shape:
            out = out + x
        return out

class CustomCNN(nn.Module):
    def __init__(self, num_filters, filter_size, activation, dense_neurons,
                 batch_norm=False, dropout=0.0, num_classes=10, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        layers = []
        in_ch = 3
        for i, nf in enumerate(num_filters):
            padding = (filter_size - 1) // 2
            conv = nn.Conv2d(in_ch, nf, kernel_size=filter_size, padding=padding)
            bn = nn.BatchNorm2d(nf) if batch_norm else nn.Identity()
            act = activation()
            layers.append(nn.Sequential(ResidualBlock(conv, bn, act, use_residual), nn.MaxPool2d(2)))
            in_ch = nf
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters[-1] * 7 * 7, dense_neurons),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dense_neurons, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_activation(name):
    return dict(ReLU=nn.ReLU, GELU=nn.GELU, SiLU=nn.SiLU, Mish=nn.Mish)[name]

def generate_filters(base_filters, organization):
    if organization == 'same': return [base_filters]*5
    if organization == 'double': return [base_filters*(2**i) for i in range(5)]
    if organization == 'half':   return [base_filters//(2**i) for i in reversed(range(5))]
    return [base_filters]*5

#############################################
# Training Function
#############################################
def final_train_and_save(config):
    # config: plain dict
    run = wandb.init(project="inaturalist_final", config=config)
    # use the original dict to access hyperparams
    data_dir = config['data_dir']
    batch_size = config['batch_size']
    augment = config['data_augmentation']

    train_loader, val_loader, test_loader, num_classes, class_weights = prepare_data(
        data_dir, batch_size, augment
    )

    model = CustomCNN(
        num_filters=generate_filters(config['num_filters'], config['filter_organization']),
        filter_size=config['filter_size'],
        activation=get_activation(config['activation']),
        dense_neurons=config['dense_neurons'],
        batch_norm=config['batch_norm'],
        dropout=config['dropout'],
        num_classes=num_classes,
        use_residual=config['use_residual']
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    scaler = GradScaler()

    best_acc = 0.0
    no_improve = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(): loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()
        avg_train = total_loss/len(train_loader)

        model.eval(); val_loss=0.0; preds=[]; labels=[]
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast(): loss = criterion(model(x), y)
                val_loss += loss.item()
                p = model(x).argmax(1).cpu().numpy(); preds.extend(p)
                labels.extend(y.cpu().numpy())
        avg_val = val_loss/len(val_loader)
        val_acc = 100*np.mean(np.array(preds)==np.array(labels))

        wandb.log({
            'epoch': epoch+1,
            'train_loss': avg_train,
            'val_loss': avg_val,
            'val_accuracy': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        print(f"Epoch {epoch+1}: train {avg_train:.4f}, val {avg_val:.4f}, acc {val_acc:.2f}%")

        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc; no_improve=0
            ckpt = os.path.join(run.dir, 'best_model.pth')
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
            if no_improve >= config['early_stop_patience']:
                print("Early stopping."); break

    run.finish()
    return ckpt, test_loader, num_classes

#############################################
# Evaluation & Visualization
#############################################
def inverse_normalize(t):
    inv = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                               std=[1/0.229, 1/0.224, 1/0.225])
    t = inv(t); t = torch.clamp(t,0,1)
    return transforms.ToPILImage()(t)

def evaluate_model(loader, model):
    model.eval(); all_p, all_l = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            all_p.extend(preds); all_l.extend(y.cpu().numpy())
    return (100*np.mean(np.array(all_p)==np.array(all_l)),
            precision_score(all_l, all_p, average='weighted', zero_division=0),
            recall_score(all_l, all_p, average='weighted', zero_division=0),
            f1_score(all_l, all_p, average='weighted', zero_division=0),
            confusion_matrix(all_l, all_p))

def plot_test_predictions(model, dataset, classes, num=10):
    idxs = random.sample(range(len(dataset)), num)
    fig, ax = plt.subplots(num,3, figsize=(12, num*3))
    for i, idx in enumerate(idxs):
        img, lbl = dataset[idx]
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
        pil = inverse_normalize(img)
        for j, title in enumerate(["Orig","Pred","True"]):
            ax[i,j].imshow(pil); ax[i,j].axis('off')
            ax[i,j].set_title(f"{title}: {classes[pred if j==1 else lbl]}")
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
    best_config = {
        'num_filters': 128,
        'filter_size': 3,
        'activation': 'GELU',
        'filter_organization': 'same',
        'data_augmentation': True,
        'batch_norm': False,
        'dropout': 0.2,
        'dense_neurons': 512,
        'batch_size': 32,
        'lr': 0.026992,
        'epochs': 50,
        'use_residual': False,
        'early_stop_patience': 5,
        'data_dir': '/kaggle/input/inaturalist/inaturalist_12K'
    }

    # Run training
    best_ckpt, test_loader, num_classes = final_train_and_save(best_config)

    # Evaluation run
    eval_run = wandb.init(project='inaturalist_final_evaluation2', reinit=True)
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    test_dataset = datasets.ImageFolder(os.path.join(best_config['data_dir'], 'test'), transform=transform)

    # Load model
    model = CustomCNN(
        num_filters=generate_filters(best_config['num_filters'], best_config['filter_organization']),
        filter_size=best_config['filter_size'],
        activation=get_activation(best_config['activation']),
        dense_neurons=best_config['dense_neurons'],
        batch_norm=best_config['batch_norm'],
        dropout=best_config['dropout'],
        num_classes=num_classes,
        use_residual=best_config['use_residual']
    ).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    acc, prec, rec, f1, cm = evaluate_model(test_loader, model)
    wandb.log({
        'test_accuracy': acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
        'confusion_matrix': wandb.plot.confusion_matrix(
            probs=None, y_true=test_dataset.targets, preds=[0]*len(test_dataset.targets), class_names=test_dataset.classes
        )
    })
    print(f"Test Acc: {acc:.2f}%, Prec: {prec:.2f}, Rec: {rec:.2f}, F1: {f1:.2f}")
    print("Confusion Matrix:\n", cm)

    plot_test_predictions(model, test_dataset, test_dataset.classes, num=10)
    eval_run.finish()
