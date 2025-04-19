# da6401 Assignment 2 – Part A

**Author:** Teja Yelagandula 
**Institute ID:** DA24M025  
**GitHub Repository:** https://github.com/da24m025/da6401_assignment2-partA  
**W&B Report:** [ DA6401 Assignment 2 W&B Report](https://wandb.ai/fgbb66579-iit-madras-foundation/inaturalist_cnn_from_scratch3988/reports/Teja-Yelagandula-DA6401-Assignment-2--VmlldzoxMjE0MDAxNw?accessToken=a0qcojswservt8etlc43cdfwh0n5vrtchgehy3btabcth6eirhpcsgdl5l11133k)

---

## 📂 Repository Structure

```
├── CNN_Model.py                # Question 1: Flexible 5‑layer CNN from scratch
├── custom_CNN_Model.py         # Question 2: Extended CNN for hyperparameter sweeps
├── data_preparation.py         # Stratified train/val split and transforms
├── train.py                    # W&B sweep setup and training loop
├── Test.py                     # Load checkpoint and evaluate on test set
├── visualize_filters_and_backprop.py  # Filter & saliency visualizations
├── partA_notebook.ipynb        # Original notebook with narrative and prototyping
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### File Dependencies
- The `.py` modules are designed as independent components.  
- You may need to import functions or classes across files (e.g. `CNNModel` from `CNN_Model.py` in `train.py`).  
- All scripts assume the iNaturalist dataset is unpacked under a `train/` and `test/` folder structure.

---

##  Running in Kaggle Environment

1. **Create a new Kaggle Notebook** and upload this repository’s files.  
2. **Unzip** the iNaturalist 12K dataset into the notebook’s working directory (`/kaggle/working/data/train` and `/kaggle/working/data/test`).  
3. **Use the provided `partA_notebook.ipynb`** as your main driver to:
   - Prepare data (see `data_preparation.py`).  
   - Instantiate and train models (see `CNN_Model.py` and `custom_CNN_Model.py`).  
   - Launch hyperparameter sweeps with W&B (see `train.py`).  
   - Evaluate final checkpoints (see `Test.py`).  
   - Visualize filters and saliency maps (see `visualize_filters_and_backprop.py`).

_No command‑line steps are required; simply run cells in your Kaggle notebook to execute the scripts._

---

## 📋 Code Organization

- **`CNN_Model.py`**  
  Defines `CNNModel`: 5 convolution→activation→pool blocks + flatten + dense + output. Hyperparameters exposed via constructor arguments.

- **`custom_CNN_Model.py`**  
  Defines `CustomCNN`: extended API for per‑layer filter size, activation, batch‑norm, dropout, and optional residual connections.

- **`data_preparation.py`**  
  Implements a stratified 80/20 train/validation split on the `train/` folder and loads `test/`. Includes optional data augmentation transforms.

- **`train.py`**  
  Sets up W&B sweep with Bayesian optimization, logs all metrics (`train_loss`, `val_loss`, `val_accuracy`), hyperparameters, and automatically generates the required charts (accuracy vs. time, parallel coords, correlation table).

- **`Test.py`**  
  Loads a saved PyTorch checkpoint and computes final accuracy on the `test/` set.

- **`visualize_filters_and_backprop.py`**  
  Utility to render learned convolutional filters and class‑specific saliency maps via guided backprop.

- **`partA_notebook.ipynb`**  
  Interactive walkthrough: EDA, model building, debugging, and final results. Use this for reference.

---


---


