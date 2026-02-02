# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRPCA (Transformer-based Robust Principal Component Analysis) is a PyTorch framework for microbiome data analysis using normalized transformers with multi-task learning. It supports single-task regression and multi-task learning (regression + classification).

**Requirements:** Python 3.8+, PyTorch 1.8+

## Installation

```bash
# Development install (includes all dependencies)
pip install -e .
```

## CLI Commands

```bash
# Single-task regression training
trpca train -f data/features.csv -m data/metadata.csv -t target_column -o output/

# Multi-task learning (regression + classification)
trpca train-mtl -f data/features.csv -m data/metadata.csv -r reg_column -c cls_column -o output/

# Feature importance analysis (requires saved model and pickled data loaders)
trpca analyze -m output/model.pt --train-loader train.pkl --test-loader test.pkl -f data/features.csv

# Compare against sklearn regressors
trpca compare -f data/features.csv -m data/metadata.csv -t target_column --model-path output/model.pt
```

Key CLI options:
- `--use-optuna` / `--no-optuna`: Enable hyperparameter optimization
- `--device auto|cuda|mps|cpu`: Select compute device
- `--num-pcs`: Number of PCA components (default: 48 for single-task, 256 for MTL)
- `--epochs`: Max training epochs (default: 1000, uses early stopping)

## Architecture

### Core Modules (TRPCA/)

**trpca.py** - Model definitions:
- `NormalizedTransformerBlock`: Custom transformer with L2 normalization, learnable scaling (`alphaA`, `alphaM`)
- `NormalizedTransformer`: Single-task model (PCA → view generation → transformer blocks → regression head)
- `MTLNormalizedTransformer`: Multi-task variant with shared backbone + separate regression/classification heads

**losses.py** - Loss functions:
- `UncertaintyLoss`: Learnable uncertainty-weighted loss for MTL task balancing
- `compute_mtl_loss()`: Combines MSE (regression) and CrossEntropy (classification)

**utils.py** - Data processing and training (~1000 lines):
- `preprocess()` / `preprocess_mtl()`: RCLR transformation, PCA reduction, group-aware splitting
- `train_model()` / `train_model_mtl()`: Training with optional Optuna hyperparameter optimization
- `predict_and_evaluate()` / `predict_and_evaluate_mtl()`: Evaluation with metrics and plots
- `analyze_features()`: SHAP-based feature importance analysis
- `compare_regressors()`: Benchmark against sklearn models

### Data Flow

1. Input: CSV feature tables or BIOM files + metadata
2. Preprocessing: RCLR transformation (gemelli) → PCA reduction → DataLoader creation
3. Model: PCA features → view projection → transformer blocks → task heads
4. Output: Predictions, metrics (MAE, MSE, RMSE, R², accuracy, F1), visualizations

### Model Parameters

```python
model_params = {
    'input_dim': 48-256,      # PCA components
    'hidden_dim': 256,        # Transformer hidden dim
    'num_layers': 1-4,        # Transformer blocks
    'output_dim': 1,          # Regression output
    'projection_dim': 4,      # Number of data views
    'num_classes': N          # For MTL only
}
```

## Key Patterns

- Custom `IndexedDataset` class preserves sample indices through DataLoader
- Models return dictionaries: `{'regression_output': tensor}` or `{'regression_output': tensor, 'classification_output': tensor}`
- Random state 42 used throughout for reproducibility
- Device must be manually specified ('cuda' or 'cpu')

## Sample Data

`data/` contains skin microbiome datasets (WGS) in CSV and BIOM formats for testing.

## Example Usage

See `TRPCA_examples.ipynb` for complete workflows.

## Git Conventions

Do not add "Co-Authored-By: Claude" to commits or PRs in this repository.
