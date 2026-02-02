#!/usr/bin/env python3
"""
Hyperparameter survey script for TRPCA.

This script systematically explores hyperparameter combinations and logs results.
Supports parallel execution on the same GPU.
"""

import subprocess
import json
import itertools
from pathlib import Path
from datetime import datetime
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os



def run_experiment(args_tuple):
    """Run a single training experiment with given hyperparameters."""
    config, features, metadata, target, device, base_output_dir, worker_id = args_tuple

    # Create unique output directory for this run
    run_name = (
        f"pcs{config['num_pcs']}_"
        f"hd{config['hidden_dim']}_"
        f"nl{config['num_layers']}_"
        f"pd{config['projection_dim']}_"
        f"bs{config['batch_size']}"
    )
    output_dir = Path(base_output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "trpca", "train",
        "-f", str(features),
        "-m", str(metadata),
        "-t", target,
        "-o", str(output_dir),
        "--num-pcs", str(config['num_pcs']),
        "--hidden-dim", str(config['hidden_dim']),
        "--num-layers", str(config['num_layers']),
        "--projection-dim", str(config['projection_dim']),
        "--batch-size", str(config['batch_size']),
        "--epochs", str(config['epochs']),
        "--device", device,
        "--no-optuna",  # Disable Optuna for survey (we're doing our own search)
    ]

    print(f"[Worker {worker_id}] Starting: {run_name}")

    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per experiment
        )

        # Check if metrics file was created
        metrics_file = output_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            print(f"[Worker {worker_id}] Completed: {run_name} | R²={metrics.get('r2', 'N/A'):.4f}")
            return {
                **config,
                **metrics,
                "status": "success",
                "output_dir": str(output_dir),
                "run_name": run_name
            }
        else:
            # Save stderr for debugging
            stderr_file = output_dir / "stderr.log"
            if result.stderr:
                with open(stderr_file, 'w') as f:
                    f.write(result.stderr)
            stdout_file = output_dir / "stdout.log"
            if result.stdout:
                with open(stdout_file, 'w') as f:
                    f.write(result.stdout)

            error_msg = result.stderr[-1000:] if result.stderr else "No metrics file created (check stderr.log)"
            print(f"[Worker {worker_id}] Failed: {run_name}")
            return {
                **config,
                "status": "failed",
                "error": error_msg,
                "output_dir": str(output_dir),
                "run_name": run_name
            }

    except subprocess.TimeoutExpired:
        print(f"[Worker {worker_id}] Timeout: {run_name}")
        return {
            **config,
            "status": "timeout",
            "output_dir": str(output_dir),
            "run_name": run_name
        }
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {run_name} - {e}")
        return {
            **config,
            "status": "error",
            "error": str(e),
            "output_dir": str(output_dir),
            "run_name": run_name
        }


def main():
    parser = argparse.ArgumentParser(description="TRPCA Hyperparameter Survey")
    parser.add_argument("-f", "--features", required=True, help="Path to feature table (CSV or BIOM)")
    parser.add_argument("-m", "--metadata", required=True, help="Path to metadata file (CSV or TSV)")
    parser.add_argument("-t", "--target", required=True, help="Target column name")
    parser.add_argument("-o", "--output", default="hyperparam_survey", help="Output directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu", "auto"], help="Device")

    # Parallelization
    parser.add_argument("-w", "--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, try 4-8 for underutilized GPU)")

    # Hyperparameter ranges
    parser.add_argument("--num-pcs", nargs="+", type=int, default=[32, 48, 64, 128],
                        help="PCA components to try")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 256, 512],
                        help="Hidden dimensions to try")
    parser.add_argument("--num-layers", nargs="+", type=int, default=[1, 2, 3],
                        help="Number of transformer layers to try")
    parser.add_argument("--projection-dims", nargs="+", type=int, default=[2, 4, 8],
                        help="Projection dimensions to try")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 128],
                        help="Batch sizes to try")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Epochs per run (default: 20, matching utils.py)")

    args = parser.parse_args()

    # Create output directory
    base_output_dir = Path(args.output)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all hyperparameter combinations
    param_grid = {
        'num_pcs': args.num_pcs,
        'hidden_dim': args.hidden_dims,
        'num_layers': args.num_layers,
        'projection_dim': args.projection_dims,
        'batch_size': args.batch_sizes,
        'epochs': [args.epochs],  # Fixed for all runs
    }

    # Create all combinations
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    configs = [dict(zip(keys, combo)) for combo in combinations]

    print(f"Total experiments to run: {len(configs)}")
    print(f"Parallel workers: {args.workers}")
    print(f"Parameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")

    # Save experiment config
    config_file = base_output_dir / "experiment_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "features": args.features,
            "metadata": args.metadata,
            "target": args.target,
            "device": args.device,
            "workers": args.workers,
            "param_grid": param_grid,
            "total_experiments": len(configs),
            "start_time": datetime.now().isoformat(),
        }, f, indent=2)

    # Prepare arguments for parallel execution
    task_args = [
        (config, args.features, args.metadata, args.target, args.device, str(base_output_dir), i % args.workers)
        for i, config in enumerate(configs)
    ]

    # Run experiments in parallel
    results = []
    completed = 0

    if args.workers == 1:
        # Sequential execution
        for i, task_arg in enumerate(task_args, 1):
            print(f"\n[{i}/{len(configs)}]")
            result = run_experiment(task_arg)
            results.append(result)

            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_df.to_csv(base_output_dir / "survey_results.csv", index=False)

            # Print current best
            successful = [r for r in results if r.get('status') == 'success']
            if successful:
                best = sorted(successful, key=lambda x: (-x.get('r2', -999), x.get('mae', 999)))[0]
                print(f"Current best: R²={best.get('r2', 'N/A'):.4f}, MAE={best.get('mae', 'N/A'):.4f}")
    else:
        # Parallel execution
        print(f"\nStarting parallel execution with {args.workers} workers...")
        print("="*60)

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_config = {executor.submit(run_experiment, arg): arg[0] for arg in task_args}

            for future in as_completed(future_to_config):
                completed += 1
                try:
                    result = future.result()
                    results.append(result)

                    # Save intermediate results (thread-safe via file system)
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(base_output_dir / "survey_results.csv", index=False)

                    # Progress update
                    successful = [r for r in results if r.get('status') == 'success']
                    if successful:
                        best = sorted(successful, key=lambda x: (-x.get('r2', -999), x.get('mae', 999)))[0]
                        print(f"[{completed}/{len(configs)}] Best so far: R²={best.get('r2', 'N/A'):.4f} "
                              f"(pcs={best['num_pcs']}, hd={best['hidden_dim']}, nl={best['num_layers']})")
                    else:
                        print(f"[{completed}/{len(configs)}] No successful runs yet")

                except Exception as e:
                    print(f"[{completed}/{len(configs)}] Task failed with exception: {e}")
                    results.append({
                        "status": "error",
                        "error": str(e)
                    })

    # Final summary
    print("\n" + "="*60)
    print("SURVEY COMPLETE")
    print("="*60)

    results_df = pd.DataFrame(results)
    results_df.to_csv(base_output_dir / "survey_results.csv", index=False)

    # Filter successful runs
    successful_df = results_df[results_df['status'] == 'success'].copy()

    if len(successful_df) > 0:
        # Sort by R² descending
        successful_df = successful_df.sort_values('r2', ascending=False)

        print(f"\nSuccessful runs: {len(successful_df)}/{len(configs)}")
        print("\nTop 10 configurations by R²:")
        print("-" * 60)

        top_cols = ['num_pcs', 'hidden_dim', 'num_layers', 'projection_dim', 'batch_size', 'r2', 'mae', 'rmse']
        available_cols = [c for c in top_cols if c in successful_df.columns]
        print(successful_df[available_cols].head(10).to_string(index=False))

        # Save top configurations
        successful_df.to_csv(base_output_dir / "best_configs.csv", index=False)
        print(f"\nResults saved to {base_output_dir / 'survey_results.csv'}")
        print(f"Best configurations saved to {base_output_dir / 'best_configs.csv'}")
    else:
        print("\nNo successful runs!")
        failed_df = results_df[results_df['status'] != 'success']
        print(f"Failed runs: {len(failed_df)}")
        if 'error' in failed_df.columns:
            print("\nErrors encountered:")
            for _, row in failed_df.iterrows():
                print(f"  {row.get('error', 'Unknown error')[:100]}")


if __name__ == "__main__":
    main()
