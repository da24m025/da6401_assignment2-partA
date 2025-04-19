import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
import torch.nn as nn
import wandb

# Assume device is already set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# Optional Part 1: Visualize First-Layer Filters
#############################################
def visualize_first_layer_filters(model):
    """
    Visualizes filters in the first convolutional layer.
    If there are 64 filters, they are arranged in an 8x8 grid.
    Logs the figure to wandb.
    """
    first_conv = None
    # Find the first Conv2d layer
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    if first_conv is None:
        print("No Conv2d layer found!")
        return
    filters = first_conv.weight.data.clone().cpu()
    if filters.shape[0] != 64:
        print(f"Expected 64 filters, but found {filters.shape[0]}.")
        return
    # Normalize filters to [0, 1]
    filters = (filters - filters.min()) / (filters.max() - filters.min() + 1e-5)
    
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters[i].permute(1, 2, 0))
        ax.axis("off")
    plt.tight_layout()
    
    # Log the figure to wandb
    wandb.log({"First Layer Filters": wandb.Image(fig)})
    plt.show()

#############################################
# Optional Part 2: Guided Backpropagation Visualization
#############################################
class GuidedBackprop:
    """
    Implements guided backpropagation by modifying the backward pass of ReLU layers.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Register hooks on all ReLU layers
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.relu_backward_hook)

    def relu_backward_hook(self, module, grad_in, grad_out):
        # Only allow positive gradients to flow back
        return (torch.clamp(grad_in[0], min=0.0),)

    def generate_gradients(self, input_image, target_index):
        """
        Computes guided gradients with respect to the input image for the target output.
        target_index can be interpreted as a target neuron index (or class index).
        """
        self.model.zero_grad()
        input_image = input_image.unsqueeze(0).to(device)
        input_image.requires_grad = True
        output = self.model(input_image)
        # Select the target scalar (e.g., center pixel of the CONV5 feature map or output for a target class)
        target_score = output[0, target_index]
        target_score.backward(retain_graph=True)
        gradients = input_image.grad.data.cpu().numpy()[0]
        # Average gradients across color channels to form a single grayscale map
        grad_map = np.mean(gradients, axis=0)
        grad_map = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-5)
        return grad_map

def plot_guided_backprop(model, test_image, neuron_indices):
    """
    Applies guided backpropagation on the test image for each neuron index in neuron_indices.
    Plots the resulting gradient maps in a 10x1 grid.
    Logs the figure to wandb.
    """
    gbp = GuidedBackprop(model)
    guided_maps = []
    for idx in neuron_indices:
        grad_map = gbp.generate_gradients(test_image, idx)
        guided_maps.append(grad_map)
    
    # Plotting the guided gradient maps
    num_neurons = len(neuron_indices)
    fig, axes = plt.subplots(num_neurons, 1, figsize=(5, num_neurons * 3))
    if num_neurons == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(guided_maps[i], cmap="jet")
        ax.axis("off")
        ax.set_title(f"Neuron {neuron_indices[i]}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    wandb.log({"Guided Backpropagation Maps": wandb.Image(fig)})
    plt.show()

#############################################
# Main Execution for Optional Visualizations
#############################################
if __name__ == "__main__":
    # Ensure wandb is initialized for evaluation if not already done.
    if wandb.run is None:
        wandb.init(project="inaturalist_final_evaluation3", reinit=True)
    
    # Assume 'best_model' is already loaded from your final training code.
    # For example:
    # from your_model_module import CustomCNN, get_activation, generate_filters
    # best_model = CustomCNN(...).to(device)
    # best_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    # best_model.eval()

    # Optional Part 1: Visualize first-layer filters
    visualize_first_layer_filters(model)

    # Optional Part 2: Guided Backpropagation on 10 neurons in CONV5 layer.
    # Select 10 neuron indices (for example, 0 through 9)
    neuron_indices = list(range(10))
    # Choose a random test image from the test dataset for analysis.
    # Assume test_dataset is defined from your final evaluation code.
    test_dataset = datasets.ImageFolder(os.path.join("/kaggle/input/inaturalist/inaturalist_12K", "test"),
                                         transform=transforms.Compose([
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                         ]))
    test_image, _ = test_dataset[random.randint(0, len(test_dataset) - 1)]
    plot_guided_backprop(model, test_image, neuron_indices)
    
    wandb.finish()
