#!/usr/bin/env python3
"""
Learning Curve: Performance vs. Dataset Size
=============================================

Trains the model with increasing fractions of the training data
(e.g., 10%, 20%, 40%, 60%, 80%, 100%) and plots MAE/R² vs N_train.

Standard figure for ML-for-materials papers (JACS, Nature Comput. Sci., etc.)
Shows data efficiency and whether more data would improve performance.

Usage:
    python learning_curve.py best_config.json
    python learning_curve.py best_config.json --fractions 0.05 0.1 0.2 0.4 0.6 0.8 1.0
    python learning_curve.py best_config.json --repeats 3
"""

import os
import sys
import json
import copy
import gc
import logging
import datetime
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from e3nn_predict_v2 import (
    RunConfig, SimpleDataset,
    train_model, evaluate_model, compute_metrics,
    set_seed, setup_directories, setup_logging,
)


def run_learning_curve(config_path, fractions, repeats, epochs, patience):
    """
    Train with increasing data fractions, evaluate on FIXED test set.

    The test set is always the same (defined by seed + test_size in config).
    Only the training set size changes. Val set is a fixed fraction of
    whatever training data is available.
    """
    setup_directories()
    os.makedirs("data/learning_curve", exist_ok=True)
    setup_logging()

    cfg = RunConfig.from_json(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("=" * 60)
    logging.info("LEARNING CURVE: PERFORMANCE vs. DATASET SIZE")
    logging.info("=" * 60)
    logging.info(f"Config: {config_path}")
    logging.info(f"Fractions: {fractions}")
    logging.info(f"Repeats per fraction: {repeats}")
    logging.info(f"Epochs per run: {epochs}")
    logging.info(f"Device: {device}")

    # Create full dataset (defines the canonical test split)
    set_seed(cfg.data.seed)
    dataset = SimpleDataset(cfg.data)
    dataset.set_model_config(cfg.model)

    full_train_idx = list(dataset.train_idx)
    full_val_idx = list(dataset.val_idx)
    fixed_test_idx = list(dataset.test_idx)

    # Combine train + val as the pool to subsample from
    # (val will be re-split from the subsample)
    train_val_pool = full_train_idx + full_val_idx
    n_pool = len(train_val_pool)

    logging.info(f"Dataset: {len(dataset)} total, "
                 f"{n_pool} train+val pool, "
                 f"{len(fixed_test_idx)} fixed test")

    # Override epochs and patience
    train_cfg = copy.copy(cfg.train)
    train_cfg.num_epochs = epochs
    train_cfg.patience = patience
    # Recalculate warmup for the overridden epoch count
    # (copy.copy does NOT re-run __post_init__)
    train_cfg.warmup_epochs = max(1, epochs // 10)

    is_vector = dataset.target_stats.get('is_vector', False)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    for frac in fractions:
        n_subset = max(10, int(n_pool * frac))
        logging.info(f"\n{'='*60}")
        logging.info(f"FRACTION {frac:.0%}: {n_subset} molecules from pool")
        logging.info(f"{'='*60}")

        for rep in range(repeats):
            seed = cfg.data.seed + rep * 1000
            set_seed(seed)
            rng = np.random.RandomState(seed)

            # Subsample from train+val pool
            subset_idx = rng.choice(train_val_pool, size=n_subset, replace=False).tolist()

            # Split subset into train (90%) and val (10%) for early stopping
            n_val = max(1, int(n_subset * 0.1))
            rng.shuffle(subset_idx)
            sub_val_idx = subset_idx[:n_val]
            sub_train_idx = subset_idx[n_val:]

            # Assign to dataset
            dataset.train_idx = sub_train_idx
            dataset.val_idx = sub_val_idx
            dataset.test_idx = fixed_test_idx
            dataset._fit_normalization()

            logging.info(f"  Repeat {rep+1}/{repeats}: "
                         f"{len(sub_train_idx)} train, {len(sub_val_idx)} val, "
                         f"{len(fixed_test_idx)} test (seed={seed})")

            # Train
            try:
                model, history, _, _ = train_model(
                    dataset, cfg.model, train_cfg, device)

                # Evaluate on FIXED test set
                eval_results = evaluate_model(
                    model, dataset, device, is_vector, split="test")
                metrics = compute_metrics(eval_results, is_vector)

                best_epoch = history[-1].get("best_epoch", 0) if history else 0
                n_epochs_run = len(history)

                result = {
                    "fraction": frac,
                    "n_train": len(sub_train_idx),
                    "n_val": len(sub_val_idx),
                    "repeat": rep + 1,
                    "seed": seed,
                    "MAE": metrics.get("MAE", float('inf')),
                    "RMSE": metrics.get("RMSE", float('inf')),
                    "R2": metrics.get("R2", 0),
                    "Spearman": metrics.get("Spearman", 0),
                    "MedAE": metrics.get("MedAE", float('inf')),
                    "best_epoch": best_epoch,
                    "n_epochs_run": n_epochs_run,
                }

                logging.info(f"    MAE={result['MAE']:.4f}, "
                             f"R2={result['R2']:.4f}, "
                             f"Spearman={result['Spearman']:.4f}, "
                             f"best_epoch={best_epoch}")

                del model

            except Exception as e:
                logging.warning(f"    Failed: {e}")
                result = {
                    "fraction": frac, "n_train": len(sub_train_idx),
                    "n_val": len(sub_val_idx), "repeat": rep + 1,
                    "seed": seed, "MAE": float('inf'), "RMSE": float('inf'),
                    "R2": 0, "Spearman": 0, "MedAE": float('inf'),
                    "best_epoch": 0, "n_epochs_run": 0,
                }

            all_results.append(result)

            # Free GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save results CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join("data/learning_curve", f"learning_curve_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"\nResults saved: {csv_path}")

    # Filter out failed runs (inf values) before computing summary
    df_valid = df[df["MAE"] < float('inf')].copy()
    if len(df_valid) < len(df):
        n_dropped = len(df) - len(df_valid)
        logging.warning(f"Dropped {n_dropped} failed runs from summary")

    # Compute summary statistics (fillna(0) handles repeats=1 where std=NaN)
    summary = df_valid.groupby("fraction").agg(
        n_train=("n_train", "first"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        Spearman_mean=("Spearman", "mean"),
        Spearman_std=("Spearman", "std"),
    ).reset_index()
    summary = summary.fillna(0)  # std=NaN when repeats=1

    summary_path = os.path.join("data/learning_curve", f"learning_curve_summary_{timestamp}.csv")
    summary.to_csv(summary_path, index=False)
    logging.info(f"Summary saved: {summary_path}")

    logging.info("\n=== LEARNING CURVE SUMMARY ===")
    for _, row in summary.iterrows():
        logging.info(f"  {row['fraction']:.0%} ({int(row['n_train'])} train): "
                     f"MAE={row['MAE_mean']:.4f}+/-{row['MAE_std']:.4f}, "
                     f"R2={row['R2_mean']:.4f}+/-{row['R2_std']:.4f}, "
                     f"Spearman={row['Spearman_mean']:.4f}")

    # Generate plots
    plot_learning_curve(summary, timestamp)
    plot_learning_curve_combined(summary, timestamp)

    return df, summary


def plot_learning_curve(summary, timestamp):
    """MAE and R² vs dataset size (separate panels)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    n_train = summary["n_train"].values
    # MAE
    ax1.errorbar(n_train, summary["MAE_mean"], yerr=summary["MAE_std"],
                 fmt='o-', color='#1f77b4', lw=2, ms=8, capsize=5,
                 capthick=2, elinewidth=1.5)
    ax1.set_xlabel("Training Set Size", fontsize=14)
    ax1.set_ylabel("Test MAE (eV)", fontsize=14)
    ax1.set_title("MAE vs. Training Data", fontsize=16, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    # Log scale x-axis if range > 5x
    if max(n_train) / max(min(n_train), 1) > 5:
        ax1.set_xscale('log')

    # R²
    ax2.errorbar(n_train, summary["R2_mean"], yerr=summary["R2_std"],
                 fmt='s-', color='#2ca02c', lw=2, ms=8, capsize=5,
                 capthick=2, elinewidth=1.5)
    ax2.set_xlabel("Training Set Size", fontsize=14)
    ax2.set_ylabel("Test R²", fontsize=14)
    ax2.set_title("R² vs. Training Data", fontsize=16, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    if max(n_train) / max(min(n_train), 1) > 5:
        ax2.set_xscale('log')
    ax2.set_ylim(None, 1.05)

    plt.tight_layout()
    path = os.path.join("data/learning_curve", f"learning_curve_{timestamp}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Plot saved: {path}")


def plot_learning_curve_combined(summary, timestamp):
    """Publication-quality single panel: MAE with power-law fit."""
    fig, ax = plt.subplots(figsize=(8, 6))

    n_train = summary["n_train"].values.astype(float)
    mae_mean = summary["MAE_mean"].values
    mae_std = summary["MAE_std"].values

    # Data points with error bars
    ax.errorbar(n_train, mae_mean, yerr=mae_std,
                fmt='o', color='#1f77b4', lw=2, ms=10, capsize=6,
                capthick=2, elinewidth=2, zorder=5,
                label="Test MAE")

    # Power-law fit: MAE = a * N^(-b)
    # In log space: log(MAE) = log(a) - b*log(N)
    try:
        valid = mae_mean > 0
        if valid.sum() >= 3:
            log_n = np.log(n_train[valid])
            log_mae = np.log(mae_mean[valid])
            coeffs = np.polyfit(log_n, log_mae, 1)
            b = -coeffs[0]
            a = np.exp(coeffs[1])

            n_fit = np.linspace(min(n_train) * 0.8, max(n_train) * 1.5, 100)
            mae_fit = a * n_fit ** (-b)
            ax.plot(n_fit, mae_fit, '--', color='#ff7f0e', lw=2, alpha=0.8,
                    label=f"Fit: MAE ~ N$^{{-{b:.2f}}}$")
    except Exception:
        pass

    ax.set_xlabel("Training Set Size (N)", fontsize=14)
    ax.set_ylabel("Test MAE (eV)", fontsize=14)
    ax.set_title("Learning Curve", fontsize=16, fontweight="bold")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    path = os.path.join("data/learning_curve", f"learning_curve_publication_{timestamp}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Publication plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Learning Curve: Performance vs. Dataset Size")
    parser.add_argument("config", help="Config JSON (best from Optuna)")
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                        help="Data fractions to evaluate (default: 0.05 0.1 0.2 0.4 0.6 0.8 1.0)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeats per fraction for error bars (default: 3)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs per run (default: 100)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    args = parser.parse_args()

    run_learning_curve(args.config, args.fractions, args.repeats,
                       args.epochs, args.patience)


if __name__ == "__main__":
    main()
