import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_filters, filter_sizes, activations, dense_neurons, dense_activation):
        """
        Initialize the CNN model with flexible parameters.
        
        Args:
            num_filters (list): List of 5 integers specifying the number of filters for each conv layer.
            filter_sizes (list): List of 5 integers specifying the filter size for each conv layer.
            activations (list): List of 5 nn.Module activation functions for each conv layer.
            dense_neurons (int): Number of neurons in the hidden dense layer.
            dense_activation (nn.Module): Activation function for the dense layer.
        """
        super(CNNModel, self).__init__()

        # Input validation to ensure correct parameter lengths
        assert len(num_filters) == 5, "num_filters must be a list of 5 integers"
        assert len(filter_sizes) == 5, "filter_sizes must be a list of 5 integers"
        assert len(activations) == 5, "activations must be a list of 5 nn.Module instances"

        # Store convolutional blocks in a ModuleList
        self.blocks = nn.ModuleList()
        in_channels = 3  # RGB images have 3 input channels

        # Create 5 convolutional blocks
        for i in range(5):
            # Convolution layer: padding ensures output size matches input size before pooling
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filters[i],
                kernel_size=filter_sizes[i],
                padding=(filter_sizes[i] - 1) // 2,  # Padding = (k-1)/2 for odd k
                stride=1
            )
            activation = activations[i]  # Activation function for this layer
            pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 pooling halves spatial dimensions
            block = nn.Sequential(conv, activation, pool)  # Group layers into a block
            self.blocks.append(block)
            in_channels = num_filters[i]  # Update input channels for the next layer

        # Flattening layer
        self.flatten = nn.Flatten()

        # Dense layers: assumes input image size of 224x224
        # After 5 poolings (each halving the size): 224 -> 112 -> 56 -> 28 -> 14 -> 7
        flattened_size = 7 * 7 * num_filters[4]
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, dense_neurons),  # Hidden dense layer
            dense_activation,                          # Activation for dense layer
            nn.Linear(dense_neurons, 10)               # Output layer (10 classes)
        )

    def forward(self, x):
        """
        Define the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10)
        """
        # Pass through each convolutional block
        for block in self.blocks:
            x = block(x)
        
        # Flatten the feature maps
        x = self.flatten(x)
        
        # Pass through the classifier
        x = self.classifier(x)
        return x

# Example usage
num_filters = [16, 32, 64, 128, 256]  # Example filter counts
filter_sizes = [3, 3, 3, 3, 3]        # All 3x3 filters
activations = [nn.ReLU() for _ in range(5)]  # ReLU for all conv layers
dense_neurons = 512                   # 512 neurons in dense layer
dense_activation = nn.ReLU()          # ReLU for dense layer

# Instantiate the model
model = CNNModel(num_filters, filter_sizes, activations, dense_neurons, dense_activation)

# Test with a dummy input (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # Should be [1, 10]