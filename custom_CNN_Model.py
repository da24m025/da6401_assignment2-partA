import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_filters, filter_size, activation, dense_neurons, batch_norm=False, dropout=0.0, num_classes=10, use_residual=False):
        """
        Custom CNN with 5 conv blocks, optional residual connections, and a dense classifier.
        
        Args:
            num_filters (list): List of 5 integers for the number of filters in each block.
            filter_size (int): Kernel size for each conv layer.
            activation (class): Activation function class (e.g. nn.ReLU).
            dense_neurons (int): Number of neurons in the dense layer.
            batch_norm (bool): Whether to use BatchNorm.
            dropout (float): Dropout rate.
            num_classes (int): Number of output classes.
            use_residual (bool): Whether to add residual (skip) connections in each block.
        """
        super(CustomCNN, self).__init__()
        self.use_residual = use_residual
        layers = []
        in_channels = 3  # RGB input

        for i in range(5):
            # Dynamic padding to maintain spatial dimensions
            padding = (filter_size - 1) // 2
            conv = nn.Conv2d(in_channels, num_filters[i], kernel_size=filter_size, padding=padding)
            bn = nn.BatchNorm2d(num_filters[i]) if batch_norm else nn.Identity()
            act = activation()
            # Save block in a sequential container but optionally add residual after activation if possible.
            # We will wrap this in a custom block that does: out = act(bn(conv(x))); if use_residual and in_channels == out_channels then out = out + x; then apply maxpool.
            block = ResidualBlock(conv, bn, act, use_residual)
            pool = nn.MaxPool2d(2, 2)
            layers.append(nn.Sequential(block, pool))
            in_channels = num_filters[i]
        
        self.features = nn.Sequential(*layers)
        # After 5 pooling layers, for 224 input, size becomes 224 / 32 = 7
        self.flatten_size = num_filters[-1] * 7 * 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, dense_neurons),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dense_neurons, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, conv, bn, activation, use_residual):
        super(ResidualBlock, self).__init__()
        self.conv = conv
        self.bn = bn
        self.activation = activation
        self.use_residual = use_residual

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        if self.use_residual and x.shape == out.shape:
            out = out + x
        return out
