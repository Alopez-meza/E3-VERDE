#!/usr/bin/env python3
"""
Visualization: all plotting functions for training, evaluation, and cross-validation.
"""

import os
import datetime
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import gaussian_kde


# Global style (applied on import)
sns.set_theme(style="ticks")
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
C1, C2, C_ERR = "#1f77b4", "#ff7f0e", "#d62728"


def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _save_fig(fig, name):
    path = os.path.join("data", "figures", f"{name}_{_timestamp()}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Plot saved: {path}")
    return path


def plot_loss_history(history, title_suffix=""):
    """Learning curve with LR schedule and best epoch marker."""
    if not history:
        return None
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    lrs = [h.get("lr", 0) for h in history]
    best_epoch = history[-1].get("best_epoch", 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})

    ax1.plot(epochs, train_loss, 'o-', color=C1, lw=2, ms=4, label="Train")
    ax1.plot(epochs, val_loss, 's-', color=C2, lw=2, ms=4, label="Validation")

    if best_epoch > 0 and best_epoch <= len(val_loss):
        best_val = val_loss[best_epoch - 1]
        ax1.axvline(best_epoch, color='green', ls=':', lw=1.5, alpha=0.7)
        ax1.scatter([best_epoch], [best_val], color='green', s=100, zorder=5,
                    marker='*', label=f"Best epoch {best_epoch}")

    min_loss = min(min(train_loss), min(val_loss))
    max_loss = max(max(train_loss), max(val_loss))
    if min_loss > 0 and max_loss / min_loss > 10:
        ax1.set_yscale('log')

    ax1.set_ylabel("Loss", fontsize=14)
    ax1.set_title(f"Learning Curve{title_suffix}", fontsize=16, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, lrs, '-', color='gray', lw=2)
    ax2.set_xlabel("Epoch", fontsize=14)
    ax2.set_ylabel("LR", fontsize=12)
    if all(lr > 0 for lr in lrs):
        ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return _save_fig(fig, f"loss_history{title_suffix.replace(' ', '_')}")


def plot_target_distribution(dataset, is_vector):
    targets = []
    for idx in dataset.train_idx:
        tv = dataset.database[idx].get(dataset.target_key)
        if tv is not None:
            if isinstance(tv, (list, np.ndarray)):
                targets.append(np.linalg.norm(tv) if is_vector else tv)
            else:
                targets.append(tv)
    if not targets:
        return None

    targets = np.array(targets)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.hist(targets, bins=50, color=C1, alpha=0.7, edgecolor="black", density=True)
    try:
        kde = gaussian_kde(targets)
        x_r = np.linspace(targets.min(), targets.max(), 200)
        ax.plot(x_r, kde(x_r), 'r-', lw=2, label="KDE")
    except Exception:
        pass
    ax.axvline(np.mean(targets), color='r', ls='--', lw=2, label=f'Mean: {np.mean(targets):.2f}')
    ax.axvline(np.median(targets), color='g', ls='--', lw=2, label=f'Median: {np.median(targets):.2f}')
    ax.set_xlabel("Target Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Target Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    try:
        stats.probplot(targets, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
    except Exception:
        ax.text(0.5, 0.5, "Q-Q Error", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    return _save_fig(fig, "target_distribution")


def plot_parity(eval_results, is_vector, target_name=None):
    if not eval_results:
        return None
    preds = np.array([r["pred"] for r in eval_results])
    truths = np.array([r["truth"] for r in eval_results])

    if is_vector:
        preds = preds.reshape(-1, preds.shape[-1])
        truths = truths.reshape(-1, truths.shape[-1])
        dim = preds.shape[1]
        labels = ['X', 'Y', 'Z'][:dim] if dim <= 3 else [f'C{i}' for i in range(dim)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 14)) if dim == 3 else \
            plt.subplots(1, dim + 1, figsize=(5 * (dim + 1), 5))
        axes = np.array(axes).flatten()

        pm, tm = np.linalg.norm(preds, axis=1), np.linalg.norm(truths, axis=1)
        _parity_subplot(axes[0], tm, pm, "Magnitude", C1)

        for i in range(min(dim, len(axes) - 1)):
            _parity_subplot(axes[i + 1], truths[:, i], preds[:, i], labels[i], C2)

        for j in range(dim + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        return _save_fig(fig, "parity_plot")
    else:
        preds, truths = preds.flatten(), truths.flatten()
        fig, ax = plt.subplots(figsize=(8, 8))
        _parity_subplot(ax, truths, preds, target_name or "Value", C1)
        plt.tight_layout()
        return _save_fig(fig, "parity_plot")


def _parity_subplot(ax, truths, preds, label, color):
    """Reusable parity subplot."""
    ax.scatter(truths, preds, alpha=0.6, s=50, color=color, edgecolor="black", lw=0.5)
    vmin, vmax = min(truths.min(), preds.min()), max(truths.max(), preds.max())
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2)

    mae = np.mean(np.abs(preds - truths))
    ss_res = np.sum((preds - truths) ** 2)
    ss_tot = np.sum((truths - truths.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0

    ax.text(0.05, 0.95, f"MAE = {mae:.4f}\nR² = {r2:.4f}", transform=ax.transAxes,
            fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    ax.set_xlabel(f"True {label}", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Predicted {label}", fontsize=12, fontweight="bold")
    ax.set_title(f"{label} (R² = {r2:.4f})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)


def plot_error_histogram(eval_results, is_vector):
    if not eval_results:
        return None
    preds = np.array([r["pred"] for r in eval_results])
    truths = np.array([r["truth"] for r in eval_results])

    if is_vector:
        errors = np.linalg.norm(preds - truths, axis=1)
    else:
        errors = np.abs(preds.flatten() - truths.flatten())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=30, color=C1, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(errors), color=C_ERR, ls='--', lw=2, label=f"Mean: {np.mean(errors):.4f}")
    ax.set_xlabel("Error" if not is_vector else "Magnitude Error", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_title("Error Distribution", fontsize=16, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, "error_histogram")


def plot_residuals(eval_results, is_vector):
    if not eval_results:
        return None
    preds = np.array([r["pred"] for r in eval_results])
    truths = np.array([r["truth"] for r in eval_results])

    if is_vector:
        residuals = np.linalg.norm(preds - truths, axis=1)
        fitted = np.linalg.norm(truths, axis=1)
    else:
        residuals = preds.flatten() - truths.flatten()
        fitted = truths.flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(fitted, residuals, alpha=0.6, s=50, color=C1, edgecolor="black", lw=0.5)
    ax.axhline(0, color=C_ERR, ls='--', lw=2)
    ax.set_xlabel("Fitted Values", fontsize=14)
    ax.set_ylabel("Residuals", fontsize=14)
    ax.set_title("Residual Plot", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _save_fig(fig, "residuals")


def plot_cv_metrics(all_metrics, avg_metrics):
    """CV metrics plot: MAE, R², Spearman with error bars."""
    folds = [m["Fold"] for m in all_metrics]
    mae_vals = [m.get("MAE", 0) for m in all_metrics]
    r2_vals = [m.get("R2", 0) for m in all_metrics]
    spearman_vals = [m.get("Spearman", 0) for m in all_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.bar(folds, mae_vals, color=C1, alpha=0.7, edgecolor="black")
    avg_mae = avg_metrics.get("MAE", 0)
    std_mae = avg_metrics.get("MAE_std", 0)
    ax.axhline(avg_mae, color=C_ERR, ls='--', lw=2,
               label=f"Mean: {avg_mae:.4f} ± {std_mae:.4f}")
    ax.fill_between([min(folds)-0.5, max(folds)+0.5], avg_mae - std_mae, avg_mae + std_mae,
                    color=C_ERR, alpha=0.1)
    ax.set_xlabel("Fold"); ax.set_ylabel("MAE")
    ax.set_title("MAE Across Folds", fontweight="bold"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(folds, r2_vals, color=C2, alpha=0.7, edgecolor="black")
    avg_r2 = avg_metrics.get("R2", 0)
    std_r2 = avg_metrics.get("R2_std", 0)
    ax.axhline(avg_r2, color=C_ERR, ls='--', lw=2,
               label=f"Mean: {avg_r2:.4f} ± {std_r2:.4f}")
    ax.set_xlabel("Fold"); ax.set_ylabel("R²")
    ax.set_title("R² Across Folds", fontweight="bold"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(folds, spearman_vals, color='#2ca02c', alpha=0.7, edgecolor="black")
    avg_sp = avg_metrics.get("Spearman", 0)
    std_sp = avg_metrics.get("Spearman_std", 0)
    ax.axhline(avg_sp, color=C_ERR, ls='--', lw=2,
               label=f"Mean: {avg_sp:.4f} ± {std_sp:.4f}")
    ax.set_xlabel("Fold"); ax.set_ylabel("Spearman ρ")
    ax.set_title("Spearman Across Folds", fontweight="bold"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return _save_fig(fig, "cv_metrics")


def plot_cv_learning_curves(all_histories, k):
    """Overlay learning curves from all CV folds to diagnose convergence consistency."""
    if not all_histories:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    for fold_idx, history in enumerate(all_histories):
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        color = cmap(fold_idx % 10)
        ax.plot(epochs, val_loss, '-', color=color, lw=1.5, alpha=0.8,
                label=f"Fold {fold_idx+1}")
        best_ep = history[-1].get("best_epoch", 0)
        if best_ep > 0 and best_ep <= len(val_loss):
            ax.scatter([best_ep], [val_loss[best_ep-1]], color=color, s=60, zorder=5, marker='*')

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation Loss", fontsize=14)
    ax.set_title(f"CV Learning Curves ({k} folds)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_vals = [h["val_loss"] for hist in all_histories for h in hist if hist]
    if all_vals and min(all_vals) > 0 and max(all_vals) / min(all_vals) > 10:
        ax.set_yscale('log')

    plt.tight_layout()
    return _save_fig(fig, "cv_learning_curves")
