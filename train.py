import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# … your prepare_data, CustomCNN, ResidualBlock, get_activation, generate_filters …
os.environ["WANDB_API_KEY"] = "e095fbd374bc0fa234acb179a6ec7620b57abf28"
def get_activation(name):
    return {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'SiLU': nn.SiLU,
        'Mish': nn.Mish
    }[name]

def generate_filters(base_filters, organization):
    if organization == 'same':
        return [base_filters] * 5
    elif organization == 'double':
        return [base_filters * (2 ** i) for i in range(5)]
    elif organization == 'half':
        return [base_filters // (2 ** i) for i in range(5)][::-1]  # Reverse to start small
    return [base_filters] * 5
def train():
    # this will seed everything for us
    wandb.init()
    config = wandb.config

    train_loader, val_loader, test_loader, num_classes, class_weights = prepare_data(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        augment=config.data_augmentation
    )

    model = CustomCNN(
        num_filters      = generate_filters(config.num_filters, config.filter_organization),
        filter_size      = config.filter_size,
        activation       = get_activation(config.activation),
        dense_neurons    = config.dense_neurons,
        batch_norm       = config.batch_norm,
        dropout          = config.dropout,
        num_classes      = num_classes,
        use_residual     = config.use_residual
    ).to(wandb.config.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.device))
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    scaler    = GradScaler()

    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(config.epochs):
        # TRAIN
        model.train()
        train_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # VALIDATE
        model.eval()
        val_loss = 0.0
        preds, targs = [], []
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(config.device), y.to(config.device)
                with autocast():
                    logits = model(x)
                    loss   = criterion(logits, y)
                val_loss += loss.item()
                preds .extend(logits.argmax(1).cpu().numpy())
                targs .extend(y.cpu().numpy())
        val_loss    /= len(val_loader)
        val_accuracy = 100 * np.mean(np.array(preds)==np.array(targs))

        # LOG EVERYTHING
        wandb.log({
            # losses & metrics
            "epoch":         epoch,
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "val_accuracy":  val_accuracy,
            # optimizer state
            "lr":            optimizer.param_groups[0]['lr'],
            # *** and all your hyperparameters again so they show up in the run table ***
            "hp/num_filters":        config.num_filters,
            "hp/filter_size":        config.filter_size,
            "hp/activation":         config.activation,
            "hp/filter_organization":config.filter_organization,
            "hp/data_augmentation":  config.data_augmentation,
            "hp/batch_norm":         config.batch_norm,
            "hp/dropout":            config.dropout,
            "hp/dense_neurons":      config.dense_neurons,
            "hp/batch_size":         config.batch_size,
            "hp/lr":                 config.lr,
            "hp/use_residual":       config.use_residual,
        })

        scheduler.step(val_accuracy)

        # early‑stop & checkpoint
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            no_improve   = 0
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best.pth"))
        else:
            no_improve += 1
            if no_improve >= config.early_stop_patience:
                break


if __name__ == "__main__":
    sweep_config = {
      'method': 'bayes',
      'metric': { 'name': 'val_accuracy', 'goal': 'maximize' },
      'parameters': {
        'num_filters':        {'values': [32,64,128]},
        'filter_size':        {'values': [3,5]},
        'activation':         {'values': ['ReLU','GELU','SiLU','Mish']},
        'filter_organization':{'values': ['same','double','half']},
        'data_augmentation':  {'values': [True,False]},
        'batch_norm':         {'values': [True,False]},
        'dropout':            {'values': [0.0,0.2,0.3]},
        'dense_neurons':      {'values': [256,512]},
        'batch_size':         {'values': [32,64]},
        'lr':                 {'min': 1e-3, 'max': 1e-1},
        'epochs':             {'value': 20},
        'use_residual':       {'values':[False]},
        'early_stop_patience':{'value': 5},
        'data_dir':           {'value': '/kaggle/input/inaturalist/inaturalist_12K'},
        'device':             {'value': 'cuda' if torch.cuda.is_available() else 'cpu'}
      }
    }

    sweep_id = wandb.sweep(sweep_config, project="inaturalist_cnn_from_scratch11")
    wandb.agent(sweep_id, function=train, count = 25)
