"""Command-line interface for TRPCA."""

import click
import pandas as pd
import torch
import json
from pathlib import Path


def get_device(device: str) -> str:
    """Resolve device string, handling 'auto' option."""
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """TRPCA: Transformer-based Robust Principal Component Analysis for Microbiome Data."""
    pass


@cli.command()
@click.option('--features', '-f', required=True, type=click.Path(exists=True),
              help='Path to feature table CSV (samples as rows, features as columns)')
@click.option('--metadata', '-m', required=True, type=click.Path(exists=True),
              help='Path to metadata CSV')
@click.option('--target', '-t', required=True, help='Target column name in metadata for regression')
@click.option('--output', '-o', default='.', type=click.Path(),
              help='Output directory for results')
@click.option('--num-pcs', default=48, help='Number of PCA components')
@click.option('--hidden-dim', default=256, help='Transformer hidden dimension')
@click.option('--num-layers', default=1, help='Number of transformer layers')
@click.option('--projection-dim', default=4, help='Number of data views')
@click.option('--test-split', default=0.1, help='Test set proportion')
@click.option('--val-split', default=0.1, help='Validation set proportion')
@click.option('--batch-size', default=128, help='Training batch size')
@click.option('--epochs', default=1000, help='Maximum training epochs')
@click.option('--stratify-col', default=None, help='Column name for stratified splitting')
@click.option('--group-col', default=None, help='Column name for group-aware splitting')
@click.option('--use-optuna/--no-optuna', default=False, help='Enable Optuna hyperparameter optimization')
@click.option('--n-trials', default=5, help='Number of Optuna trials')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cuda', 'mps', 'cpu']),
              help='Device for training')
@click.option('--save-model/--no-save-model', default=True, help='Save trained model')
def train(features, metadata, target, output, num_pcs, hidden_dim, num_layers,
          projection_dim, test_split, val_split, batch_size, epochs,
          stratify_col, group_col, use_optuna, n_trials, device, save_model):
    """Train a single-task regression model."""
    from TRPCA.utils import preprocess, train_model, predict_and_evaluate

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device)
    click.echo(f"Using device: {device}")

    # Load data
    click.echo("Loading data...")
    df = pd.read_csv(features, index_col=0)
    meta = pd.read_csv(metadata, index_col=0)
    y = meta[target]

    # Get stratify and group columns if specified
    stratify = meta[stratify_col] if stratify_col else None
    group = meta[group_col] if group_col else None

    # Model parameters
    model_params = {
        'input_dim': num_pcs,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'output_dim': 1,
        'projection_dim': projection_dim
    }

    # Preprocess
    click.echo("Preprocessing data...")
    train_loader, test_loader, pca = preprocess(
        df=df,
        series=y,
        num_pcs=num_pcs,
        test_split=test_split,
        stratify_col=stratify,
        group_col=group,
        batch_size=batch_size,
    )

    # Train
    click.echo("Training model...")
    model, history, training_fig = train_model(
        train_loader=train_loader,
        model_params=model_params,
        num_epochs=epochs,
        device=device,
        val_split=val_split,
        use_optuna=use_optuna,
        n_trials=n_trials,
    )

    # Save training figure
    training_fig.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    click.echo(f"Saved training history to {output_dir / 'training_history.png'}")

    # Evaluate
    click.echo("Evaluating model...")
    metrics, predictions, eval_fig = predict_and_evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    # Save evaluation figure
    eval_fig.savefig(output_dir / 'evaluation.png', dpi=150, bbox_inches='tight')
    click.echo(f"Saved evaluation plot to {output_dir / 'evaluation.png'}")

    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    click.echo(f"Saved metrics to {output_dir / 'metrics.json'}")

    # Save predictions
    predictions.to_csv(output_dir / 'predictions.csv')
    click.echo(f"Saved predictions to {output_dir / 'predictions.csv'}")

    # Save model
    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_params': model_params,
            'pca': pca,
        }, output_dir / 'model.pt')
        click.echo(f"Saved model to {output_dir / 'model.pt'}")

    # Print metrics
    click.echo("\nResults:")
    click.echo(f"  MAE:  {metrics['mae']:.4f}")
    click.echo(f"  MSE:  {metrics['mse']:.4f}")
    click.echo(f"  RMSE: {metrics['rmse']:.4f}")
    click.echo(f"  R²:   {metrics['r2']:.4f}")


@cli.command()
@click.option('--features', '-f', required=True, type=click.Path(exists=True),
              help='Path to feature table CSV (samples as rows, features as columns)')
@click.option('--metadata', '-m', required=True, type=click.Path(exists=True),
              help='Path to metadata CSV')
@click.option('--reg-target', '-r', required=True, help='Regression target column name')
@click.option('--cls-target', '-c', required=True, help='Classification target column name')
@click.option('--output', '-o', default='.', type=click.Path(),
              help='Output directory for results')
@click.option('--num-pcs', default=256, help='Number of PCA components')
@click.option('--hidden-dim', default=256, help='Transformer hidden dimension')
@click.option('--num-layers', default=1, help='Number of transformer layers')
@click.option('--projection-dim', default=4, help='Number of data views')
@click.option('--test-split', default=0.1, help='Test set proportion')
@click.option('--val-split', default=0.1, help='Validation set proportion')
@click.option('--batch-size', default=128, help='Training batch size')
@click.option('--epochs', default=1000, help='Maximum training epochs')
@click.option('--stratify-col', default=None, help='Column name for stratified splitting')
@click.option('--group-col', default=None, help='Column name for group-aware splitting')
@click.option('--use-optuna/--no-optuna', default=False, help='Enable Optuna hyperparameter optimization')
@click.option('--n-trials', default=5, help='Number of Optuna trials')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cuda', 'mps', 'cpu']),
              help='Device for training')
@click.option('--save-model/--no-save-model', default=True, help='Save trained model')
def train_mtl(features, metadata, reg_target, cls_target, output, num_pcs, hidden_dim,
              num_layers, projection_dim, test_split, val_split, batch_size, epochs,
              stratify_col, group_col, use_optuna, n_trials, device, save_model):
    """Train a multi-task learning model (regression + classification)."""
    from TRPCA.utils import preprocess_mtl, train_model_mtl, predict_and_evaluate_mtl

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device)
    click.echo(f"Using device: {device}")

    # Load data
    click.echo("Loading data...")
    df = pd.read_csv(features, index_col=0)
    meta = pd.read_csv(metadata, index_col=0)
    y_reg = meta[reg_target]
    y_cls = meta[cls_target]

    n_classes = y_cls.nunique()

    # Get stratify and group columns if specified
    stratify = meta[stratify_col] if stratify_col else None
    group = meta[group_col] if group_col else None

    # Model parameters
    model_params = {
        'input_dim': num_pcs,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_classes': n_classes,
        'output_dim': 1,
        'projection_dim': projection_dim
    }

    # Preprocess
    click.echo("Preprocessing data...")
    train_loader, test_loader, pca, label_encoder = preprocess_mtl(
        df=df,
        reg_series=y_reg,
        cls_series=y_cls,
        num_pcs=num_pcs,
        test_split=test_split,
        stratify_col=stratify,
        group_col=group,
        batch_size=batch_size,
    )

    # Train
    click.echo("Training model...")
    model, history, training_fig = train_model_mtl(
        train_loader=train_loader,
        model_params=model_params,
        num_epochs=epochs,
        device=device,
        val_split=val_split,
        use_optuna=use_optuna,
        n_trials=n_trials,
    )

    # Save training figure
    training_fig.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    click.echo(f"Saved training history to {output_dir / 'training_history.png'}")

    # Evaluate
    click.echo("Evaluating model...")
    metrics, predictions, eval_fig = predict_and_evaluate_mtl(
        model=model,
        test_loader=test_loader,
        label_encoder=label_encoder,
        device=device,
    )

    # Save evaluation figure
    eval_fig.savefig(output_dir / 'evaluation.png', dpi=150, bbox_inches='tight')
    click.echo(f"Saved evaluation plot to {output_dir / 'evaluation.png'}")

    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    click.echo(f"Saved metrics to {output_dir / 'metrics.json'}")

    # Save predictions
    predictions.to_csv(output_dir / 'predictions.csv')
    click.echo(f"Saved predictions to {output_dir / 'predictions.csv'}")

    # Save model
    if save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_params': model_params,
            'pca': pca,
            'label_encoder': label_encoder,
        }, output_dir / 'model.pt')
        click.echo(f"Saved model to {output_dir / 'model.pt'}")

    # Print metrics
    click.echo("\nResults:")
    click.echo(f"  Regression MAE:  {metrics['mae']:.4f}")
    click.echo(f"  Regression R²:   {metrics['r2']:.4f}")
    click.echo(f"  Classification Accuracy: {metrics['accuracy']:.4f}")
    click.echo(f"  Classification F1:       {metrics['f1']:.4f}")


@cli.command()
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='Path to saved model (.pt file)')
@click.option('--train-loader', required=True, type=click.Path(exists=True),
              help='Path to pickled train DataLoader')
@click.option('--test-loader', required=True, type=click.Path(exists=True),
              help='Path to pickled test DataLoader')
@click.option('--features', '-f', required=True, type=click.Path(exists=True),
              help='Path to original feature table CSV (for feature names)')
@click.option('--output', '-o', default='.', type=click.Path(),
              help='Output directory for results')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cuda', 'mps', 'cpu']),
              help='Device for computation')
def analyze(model, train_loader, test_loader, features, output, device):
    """Analyze feature importance using SHAP."""
    import pickle
    from TRPCA.utils import analyze_features
    from TRPCA.trpca import NormalizedTransformer

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device)
    click.echo(f"Using device: {device}")

    # Load model
    click.echo("Loading model...")
    checkpoint = torch.load(model, map_location=device)
    model_instance = NormalizedTransformer(**checkpoint['model_params'])
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    pca = checkpoint['pca']

    # Load data loaders
    click.echo("Loading data loaders...")
    with open(train_loader, 'rb') as f:
        train_dl = pickle.load(f)
    with open(test_loader, 'rb') as f:
        test_dl = pickle.load(f)

    # Load feature names
    df = pd.read_csv(features, index_col=0)

    # Analyze
    click.echo("Analyzing feature importance...")
    results = analyze_features(
        model=model_instance,
        train_loader=train_dl,
        test_loader=test_dl,
        pca=pca,
        original_features=df.columns,
        device=device
    )

    # Save results
    results['feature_importance_df'].to_csv(output_dir / 'feature_importance.csv')
    click.echo(f"Saved feature importance to {output_dir / 'feature_importance.csv'}")

    results['sample_importance_df'].to_csv(output_dir / 'sample_importance.csv')
    click.echo(f"Saved sample importance to {output_dir / 'sample_importance.csv'}")

    click.echo("\nTop 10 most important features:")
    top_features = results['feature_importance_df'].head(10)
    for idx, row in top_features.iterrows():
        click.echo(f"  {idx}: {row['mean_importance']:.4f}")


@cli.command()
@click.option('--features', '-f', required=True, type=click.Path(exists=True),
              help='Path to feature table CSV')
@click.option('--metadata', '-m', required=True, type=click.Path(exists=True),
              help='Path to metadata CSV')
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--model-path', required=True, type=click.Path(exists=True),
              help='Path to saved TRPCA model (.pt file)')
@click.option('--output', '-o', default='.', type=click.Path(),
              help='Output directory for results')
@click.option('--device', default='auto', type=click.Choice(['auto', 'cuda', 'mps', 'cpu']),
              help='Device for computation')
def compare(features, metadata, target, model_path, output, device):
    """Compare TRPCA against sklearn regressors."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from gemelli.preprocessing import matrix_rclr
    import matplotlib.pyplot as plt

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(device)

    # Load data
    click.echo("Loading data...")
    df = pd.read_csv(features, index_col=0)
    meta = pd.read_csv(metadata, index_col=0)
    y = meta[target]

    # Load saved model info
    checkpoint = torch.load(model_path, map_location=device)
    pca = checkpoint['pca']

    # Transform data
    click.echo("Transforming data...")
    X_clr = pd.DataFrame(
        matrix_rclr(df.values + 1),
        columns=df.columns,
        index=df.index
    )
    X_pca = pd.DataFrame(pca.transform(X_clr.values), index=df.index)

    # Define regressors
    regressors = {
        "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "MLP": MLPRegressor(hidden_layer_sizes=(100,), max_iter=2000, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    # Simple train/test split for comparison
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    results = {}
    for name, reg in regressors.items():
        click.echo(f"Training {name}...")
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        results[name] = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / 'comparison.csv')
    click.echo(f"Saved comparison to {output_dir / 'comparison.csv'}")

    # Print results
    click.echo("\nComparison Results:")
    click.echo(results_df.to_string())


def main():
    cli()


if __name__ == '__main__':
    main()
