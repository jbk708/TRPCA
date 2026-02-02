# TRPCA: Transformer-based Robust Principal Component Analysis for Microbiome Data

[![Python](https://img.shields.io/badge/python-3.8--3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)]()

A PyTorch-based framework for analyzing microbiome data using normalized transformers with multi-task learning capabilities. This framework supports both single-task regression and multi-task learning (regression + classification) scenarios.

## Features

- **Normalized Transformer Architecture**
  - Custom transformer blocks with normalization layers
  - Learnable scaling parameters for attention and MLP updates
  - Positional encoding for sequence information

- **Multi-Task Learning Support**
  - Joint regression and classification tasks
  - Uncertainty-weighted loss functions
  - Automatic task balancing during training

- **Data Processing**
  - Built-in support for microbiome data
  - Robust preprocessing with RCLR transformation
  - PCA dimensionality reduction
  - Group-aware train/test splitting

- **Analysis Tools**
  - Feature importance analysis using SHAP
  - Performance visualization
  - Comprehensive evaluation metrics

## Prerequisites

- Python 3.8-3.11 (required due to gemelli/scikit-bio dependencies)
- PyTorch 1.8+

### Linux System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev pkg-config

# RHEL/CentOS/Fedora
sudo dnf install hdf5-devel pkgconfig
```

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/jbk708/TRPCA.git
cd TRPCA

# Create environment with Python 3.11
uv venv .venv --python 3.11
source .venv/bin/activate

# Install build dependencies first
uv pip install "numpy<2" "h5py<3.11" setuptools wheel Cython pkgconfig meson-python meson ninja

# Install TRPCA with all dependencies
uv pip install -e . --no-build-isolation
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/jbk708/TRPCA.git
cd TRPCA

# Ensure you're using Python 3.8-3.11
pip install -e .
```

## Command Line Interface

TRPCA includes a CLI for training and evaluation:

```bash
# Single-task regression
trpca train -f data/features.csv -m data/metadata.csv -t target_column -o output/

# Multi-task learning (regression + classification)
trpca train-mtl -f data/features.csv -m data/metadata.csv -r reg_col -c cls_col -o output/

# With hyperparameter optimization
trpca train -f data/features.csv -m data/metadata.csv -t target --use-optuna --n-trials 10
```

Run `trpca --help` for all available options.

## Quick Start

### Single-Task Regression

```python
import pandas as pd
from TRPCA.utils import preprocess, train_model, predict_and_evaluate

# Load your data
df = pd.read_csv('your_feature_table.csv', index_col=0)
target = pd.read_csv('your_metadata.csv', index_col=0)['target_column']

# Set model parameters
model_params = {
    'input_dim': 48,
    'hidden_dim': 256,
    'num_layers': 1,
    'output_dim': 1,
    'projection_dim': 4
}

# Preprocess data
train_loader, test_loader, pca = preprocess(
    df=df,
    series=target,
    num_pcs=48,
    test_split=0.1,
    batch_size=128
)

# Train model
model, history, training_fig = train_model(
    train_loader=train_loader,
    model_params=model_params,
    num_epochs=1000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    use_optuna=True
)

# Evaluate
metrics, predictions, eval_fig = predict_and_evaluate(
    model=model,
    test_loader=test_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

### Multi-Task Learning

```python
from TRPCA.utils import preprocess_mtl, train_model_mtl, predict_and_evaluate_mtl

# Load data for both tasks
regression_target = metadata['continuous_variable']
classification_target = metadata['categorical_variable']

# Define parameters
model_params = {
    'input_dim': 256,
    'hidden_dim': 256,
    'num_layers': 1,
    'num_classes': n_classes,
    'output_dim': 1,
    'projection_dim': 4
}

# Preprocess
train_loader, test_loader, pca, label_encoder = preprocess_mtl(
    df=df,
    reg_series=regression_target,
    cls_series=classification_target,
    num_pcs=256,
    test_split=0.1,
    batch_size=128
)

# Train
model, history, training_fig = train_model_mtl(
    train_loader=train_loader,
    model_params=model_params,
    num_epochs=1000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Evaluate
metrics, predictions, eval_fig = predict_and_evaluate_mtl(
    model=model,
    test_loader=test_loader,
    label_encoder=label_encoder,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

## Model Architecture

### Normalized Transformer
- **Input Processing**: PCA-reduced features are projected to a higher dimension
- **View Generation**: Creates multiple views of the data for robust learning
- **Transformer Blocks**: Self-attention with normalization and learnable scaling
- **Output Heads**: Task-specific layers for regression and classification

### Multi-Task Learning
- **Shared Backbone**: Common feature extraction through transformer layers
- **Uncertainty Weighting**: Automatic task balancing using learnable parameters
- **Task-Specific Heads**: Separate outputs for regression and classification

## Advanced Features

### Feature Importance Analysis
```python
from TRPCA.utils import analyze_features

results = analyze_features(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    pca=pca,
    original_features=df.columns
)

# Access results
feature_importance = results['feature_importance_df']
sample_importance = results['sample_importance_df']
```

### Model Comparison
```python
from TRPCA.utils import compare_regressors

results, comparison_plot = compare_regressors(
    X=df,
    y=target,
    train_loader=train_loader,
    test_loader=test_loader,
    pca=pca,
    regressors=regressors
)
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{your-paper,
    title={NA},
    author={NA},
    journal={Pending},
    year={2025}
}
```

