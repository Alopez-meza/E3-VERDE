#!/usr/bin/env python3
"""
Optuna Bayesian Hyperparameter Optimization for E3NN Molecular Property Prediction
==================================================================================

Imports e3nn_predict_v2 as a module and calls train_model() directly.
Uses Optuna's TPE sampler with MedianPruner to efficiently explore the search space.

Usage:
    python optuna_optimize.py config.json                   # Run optimization
    python optuna_optimize.py config.json --n_trials 100    # Custom trial count
    python optuna_optimize.py config.json --resume           # Resume from checkpoint

The config.json provides the dataset settings (data section) and baseline values.
The Optuna script overrides model/train hyperparameters during search.

Outputs:
    data/optuna/study_{timestamp}.pkl           - Resumable study checkpoint
    data/optuna/best_config_{timestamp}.json    - Best config ready for production training
    data/optuna/results_{timestamp}.json        - Full results with all trial details
    data/optuna/optimization_history.png        - Optimization convergence plot
    data/optuna/param_importance.png            - Hyperparameter importance ranking
    data/optuna/parallel_coordinates.png        - Parameter correlation visualization
    data/optuna/slice_plot.png                  - Marginal parameter effects
"""

import os
import sys
import json
import copy
import time
import pickle
import logging
import datetime
import argparse

import numpy as np
import torch
import gc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import from e3nn_predict_v2 as module
from e3nn_predict_v2 import (
    RunConfig, ModelConfig, TrainConfig, DataConfig,
    SimpleDataset, PeriodicNetwork,
    train_model, evaluate_model, compute_metrics,
    set_seed, setup_directories, setup_logging,
    plot_loss_history, plot_parity, plot_error_histogram,
    _get_scaler_params,
)


# ============================================================================
# SEARCH SPACE DEFINITION
# ============================================================================
def suggest_hyperparameters(trial, base_cfg):
    """
    Define the Optuna search space.
    
    Ranges are based on NequIP, MACE, SchNet, and DimeNet++ papers.
    Only parameters with meaningful impact on performance are searched.
    Fixed parameters (e.g. use_rich_features=True) remain at their
    proven-good defaults.
    """

    # --- Architecture (biggest impact on expressivity) ---
    em_dim = trial.suggest_categorical("em_dim", [32, 64, 128])
    mul = trial.suggest_categorical("mul", [16, 32, 48])
    lmax = trial.suggest_int("lmax", 1, 2)
    layers = trial.suggest_int("layers", 1, 5)

    # Radial network
    number_of_basis = trial.suggest_categorical("number_of_basis", [8, 10, 16, 32, 64])
    radial_layers = trial.suggest_int("radial_layers", 1, 2)
    radial_neurons = trial.suggest_categorical("radial_neurons", [32, 64, 128])

    # Cutoff
    max_radius = trial.suggest_float("max_radius", 3.5, 8.0, step=0.5)

    # Regularization
    dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)

    # Readout architecture
    readout_type = trial.suggest_categorical("readout_type", ["mean", "attention"])
    output_mlp_layers = trial.suggest_int("output_mlp_layers", 1, 3)
    output_mlp_hidden = trial.suggest_categorical("output_mlp_hidden", [32, 64, 128])
    use_multiscale_readout = trial.suggest_categorical("use_multiscale_readout", [True, False])

    # --- Training (second biggest impact) ---
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    scheduler_type = trial.suggest_categorical("scheduler_type", ["cosine_warmup", "plateau"])
    #loss_function = trial.suggest_categorical("loss_function", ["l1", "mse", "huber"])
    loss_function = trial.suggest_categorical("loss_function", ["l1"])
    use_ema = trial.suggest_categorical("use_ema", [True, False])
    clip_grad_norm = trial.suggest_categorical("clip_grad_norm", [1.0, 5.0, 10.0])

    # Build configs
    model_cfg = ModelConfig(
        em_dim=em_dim,
        irreps_in=f"{em_dim}x0e",
        irreps_out=base_cfg.model.irreps_out,
        irreps_node_attr=base_cfg.model.irreps_node_attr,
        layers=layers,
        mul=mul,
        lmax=lmax,
        number_of_basis=number_of_basis,
        radial_layers=radial_layers,
        radial_neurons=radial_neurons,
        max_radius=max_radius,
        num_neighbors=-1,  # Always auto-compute
        reduce_output=base_cfg.model.reduce_output,
        dropout=dropout,
        use_layer_norm=False,
        # Fixed at proven-good defaults
        use_residual=True,
        use_self_interaction=True,
        use_rich_features=True,
        readout_type=readout_type,
        output_mlp_layers=output_mlp_layers,
        output_mlp_hidden=output_mlp_hidden,
        use_multiscale_readout=use_multiscale_readout,
    )

    train_cfg = TrainConfig(
        num_epochs=base_cfg.train.num_epochs,  # Controlled externally
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=base_cfg.train.patience,  # Keep user's patience
        scheduler_type=scheduler_type,
        warmup_epochs=-1,  # Auto
        clip_grad_norm=clip_grad_norm,
        label_smoothing=0.0,
        loss_function=loss_function,
        min_lr=1e-7,
        use_amp=base_cfg.train.use_amp,
        gradient_accumulation_steps=1,
        use_ema=use_ema,
        ema_decay=0.999,
        num_workers=base_cfg.train.num_workers,
    )

    return model_cfg, train_cfg


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================
def create_objective(dataset, base_cfg, device, hpo_epochs):
    """
    Factory that creates the Optuna objective function.
    
    The dataset is created ONCE outside and shared across all trials.
    This ensures:
    - All trials use the same train/val/test split
    - Val loss is the optimization target (val set for HPO)
    - Test set is NEVER touched during HPO (only for final evaluation)
    """

    def objective(trial):
        # Suggest hyperparameters
        model_cfg, train_cfg = suggest_hyperparameters(trial, base_cfg)

        # Override epochs for HPO (shorter than production)
        train_cfg.num_epochs = hpo_epochs
        # Shorter patience for HPO (no need to wait too long)
        train_cfg.patience = max(10, hpo_epochs // 4)

        # Update dataset cutoff for this trial's max_radius
        dataset.set_model_config(model_cfg)

        # Pruning callback: report val_loss to Optuna each epoch
        def epoch_callback(epoch, train_loss, val_loss):
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Train
        try:
            model, history, is_vector, model_cfg_used = train_model(
                dataset, model_cfg, train_cfg, device,
                epoch_callback=epoch_callback
            )
        except optuna.TrialPruned:
            raise  # Re-raise for Optuna to handle
        except Exception as e:
            logging.warning(f"Trial {trial.number} failed: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float('inf')  # Failed trials get worst score

        # Return best validation loss achieved during training
        if history:
            best_val_loss = min(h["val_loss"] for h in history)
        else:
            best_val_loss = float('inf')

        # Log key info for this trial
        n_params = sum(p.numel() for p in model.parameters())
        best_epoch = history[-1].get("best_epoch", 0) if history else 0
        logging.info(
            f"Trial {trial.number}: val_loss={best_val_loss:.4f}, "
            f"params={n_params:,}, best_epoch={best_epoch}, "
            f"mul={model_cfg.mul}, layers={model_cfg.layers}, "
            f"lmax={model_cfg.lmax}, lr={train_cfg.learning_rate:.1e}"
        )

        # Store extra info in trial
        trial.set_user_attr("n_params", n_params)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("n_epochs_run", len(history))

        # Free GPU memory between trials
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_val_loss

    return objective


# ============================================================================
# PLOTTING
# ============================================================================
def plot_optimization_results(study, output_dir):
    """Generate diagnostic plots for the optimization."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Optimization history
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if trials:
            trial_nums = [t.number for t in trials]
            values = [t.value for t in trials]
            ax.scatter(trial_nums, values, alpha=0.6, color='#1f77b4', edgecolor='black', s=40)
            # Running best
            running_best = []
            best_so_far = float('inf')
            for v in values:
                best_so_far = min(best_so_far, v)
                running_best.append(best_so_far)
            ax.plot(trial_nums, running_best, 'r-', lw=2, label='Best so far')
            ax.set_xlabel("Trial", fontsize=14)
            ax.set_ylabel("Validation Loss", fontsize=14)
            ax.set_title("Optimization History", fontsize=16, fontweight="bold")
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"optimization_history_{timestamp}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logging.warning(f"Could not plot optimization history: {e}")

    # 2. Parameter importance
    try:
        from optuna.importance import get_param_importances
        importances = get_param_importances(study)
        if importances:
            fig, ax = plt.subplots(figsize=(10, max(6, len(importances) * 0.4)))
            params = list(importances.keys())
            values = list(importances.values())
            # Sort by importance
            sorted_idx = np.argsort(values)
            ax.barh([params[i] for i in sorted_idx],
                    [values[i] for i in sorted_idx],
                    color='#2ca02c', alpha=0.8, edgecolor='black')
            ax.set_xlabel("Importance", fontsize=14)
            ax.set_title("Hyperparameter Importance", fontsize=16, fontweight="bold")
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"param_importance_{timestamp}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        logging.warning(f"Could not plot parameter importance: {e}")

    # 3. Parallel coordinates (top 20 trials)
    try:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) >= 5:
            # Sort by value, take top 20
            top_trials = sorted(completed, key=lambda t: t.value)[:20]
            params_to_plot = ["mul", "layers", "lmax", "learning_rate",
                              "max_radius", "readout_type", "batch_size"]
            available_params = [p for p in params_to_plot if p in top_trials[0].params]

            if available_params:
                fig, axes = plt.subplots(1, len(available_params), figsize=(3*len(available_params), 6),
                                         sharey=False)
                if len(available_params) == 1:
                    axes = [axes]

                for ax_idx, param in enumerate(available_params):
                    vals = []
                    scores = []
                    for t in top_trials:
                        v = t.params.get(param)
                        if v is not None:
                            vals.append(v if isinstance(v, (int, float)) else hash(str(v)) % 100)
                            scores.append(t.value)
                    if vals:
                        sc = axes[ax_idx].scatter(vals, scores, c=scores, cmap='RdYlGn_r',
                                                   alpha=0.7, edgecolor='black', s=50)
                        axes[ax_idx].set_xlabel(param, fontsize=10)
                        if ax_idx == 0:
                            axes[ax_idx].set_ylabel("Val Loss", fontsize=12)
                        axes[ax_idx].grid(True, alpha=0.3)

                fig.suptitle("Parameter vs. Validation Loss (Top 20)", fontsize=14, fontweight="bold")
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, f"parallel_coordinates_{timestamp}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
    except Exception as e:
        logging.warning(f"Could not plot parallel coordinates: {e}")

    # 4. Slice plot: key params vs objective
    try:
        key_params = ["mul", "layers", "lmax", "learning_rate", "max_radius"]
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) >= 5:
            available = [p for p in key_params if p in completed[0].params]
            n_plots = len(available)
            if n_plots > 0:
                fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
                if n_plots == 1:
                    axes = [axes]
                for idx, param in enumerate(available):
                    vals = [t.params[param] for t in completed if param in t.params]
                    scores = [t.value for t in completed if param in t.params]
                    axes[idx].scatter(vals, scores, alpha=0.5, color='#1f77b4', edgecolor='black', s=30)
                    axes[idx].set_xlabel(param, fontsize=12)
                    if idx == 0:
                        axes[idx].set_ylabel("Val Loss", fontsize=12)
                    axes[idx].set_title(param, fontsize=12, fontweight="bold")
                    axes[idx].grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, f"slice_plot_{timestamp}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
    except Exception as e:
        logging.warning(f"Could not plot slice plot: {e}")


# ============================================================================
# RESULTS SAVING
# ============================================================================
def save_best_config(study, base_cfg, output_dir, timestamp):
    """Save the best hyperparameters as a ready-to-use config JSON."""
    best = study.best_trial

    # Reconstruct full config with best params
    p = best.params
    best_config = {
        "mode": "train",
        "model": {
            "em_dim": p.get("em_dim", 64),
            "irreps_in": f"{p.get('em_dim', 64)}x0e",
            "irreps_out": base_cfg.model.irreps_out,
            "irreps_node_attr": base_cfg.model.irreps_node_attr,
            "layers": p.get("layers", 3),
            "mul": p.get("mul", 32),
            "lmax": p.get("lmax", 2),
            "number_of_basis": p.get("number_of_basis", 10),
            "radial_layers": p.get("radial_layers", 2),
            "radial_neurons": p.get("radial_neurons", 64),
            "max_radius": p.get("max_radius", 5.0),
            "num_neighbors": -1,
            "reduce_output": True,
            "dropout": p.get("dropout", 0.0),
            "use_layer_norm": False,
            "use_residual": True,
            "use_self_interaction": True,
            "use_rich_features": True,
            "readout_type": p.get("readout_type", "attention"),
            "output_mlp_layers": p.get("output_mlp_layers", 2),
            "output_mlp_hidden": p.get("output_mlp_hidden", 64),
            "use_multiscale_readout": p.get("use_multiscale_readout", True),
        },
        "train": {
            "num_epochs": 300,  # Full production epochs
            "batch_size": p.get("batch_size", 16),
            "learning_rate": p.get("learning_rate", 0.005),
            "weight_decay": p.get("weight_decay", 1e-5),
            "patience": 50,  # Production patience
            "scheduler_type": p.get("scheduler_type", "cosine_warmup"),
            "warmup_epochs": -1,
            "clip_grad_norm": p.get("clip_grad_norm", 10.0),
            "label_smoothing": 0.0,
            "loss_function": p.get("loss_function", "l1"),
            "min_lr": 1e-7,
            "use_amp": base_cfg.train.use_amp,
            "gradient_accumulation_steps": 1,
            "use_ema": p.get("use_ema", True),
            "ema_decay": 0.999,
            "num_workers": base_cfg.train.num_workers,
        },
        "data": {
            "dataset_path": base_cfg.data.dataset_path,
            "target_key": base_cfg.data.target_key,
            "structure_field": base_cfg.data.structure_field,
            "test_size": base_cfg.data.test_size,
            "val_size": base_cfg.data.val_size,
            "normalize_targets": base_cfg.data.normalize_targets,
            "normalize_features": base_cfg.data.normalize_features,
            "target_normalization": base_cfg.data.target_normalization,
            "seed": base_cfg.data.seed,
        },
        "pretrained_model_file": base_cfg.pretrained_model_file,
    }

    path = os.path.join(output_dir, f"best_config_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(best_config, f, indent=2)
    logging.info(f"Best config saved: {path}")
    return path, best_config


def save_results(study, output_dir, timestamp):
    """Save full optimization results."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    results = {
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "user_attrs": dict(study.best_trial.user_attrs),
        },
        "summary": {
            "n_trials": len(study.trials),
            "n_completed": len(completed),
            "n_pruned": len(pruned),
            "n_failed": len(failed),
            "best_value": study.best_value,
            "pruning_rate": len(pruned) / max(len(study.trials), 1),
        },
        "all_completed_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "n_params": t.user_attrs.get("n_params", 0),
                "best_epoch": t.user_attrs.get("best_epoch", 0),
                "n_epochs_run": t.user_attrs.get("n_epochs_run", 0),
            }
            for t in sorted(completed, key=lambda t: t.value)
        ],
    }

    path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Results saved: {path}")
    return results


# ============================================================================
# RETRAIN BEST
# ============================================================================
def retrain_best(best_config_path, device):
    """Retrain the best model with full production epochs."""
    logging.info("=" * 60)
    logging.info("RETRAINING BEST MODEL WITH FULL EPOCHS")
    logging.info("=" * 60)

    cfg = RunConfig.from_json(best_config_path)
    set_seed(cfg.data.seed)

    dataset = SimpleDataset(cfg.data)
    dataset.set_model_config(cfg.model)

    is_vector = dataset.target_stats.get('is_vector', False)
    logging.info(f"Retrain dataset: {len(dataset)} molecules, "
                 f"{len(dataset.train_idx)} train, {len(dataset.val_idx)} val, "
                 f"{len(dataset.test_idx)} test")

    model, history, is_vector, model_cfg_used = train_model(
        dataset, cfg.model, cfg.train, device)

    # Evaluate on test set
    test_results = evaluate_model(model, dataset, device, is_vector, split="test")
    if test_results:
        test_metrics = compute_metrics(test_results, is_vector)
        logging.info("=== Best Model Test Metrics ===")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                logging.info(f"  {k}: {v:.4f}")

        # Save plots
        plot_loss_history(history, title_suffix=" (Best HPO)")
        plot_parity(test_results, is_vector, target_name=f"{cfg.data.target_key} (Best HPO)")
        plot_error_histogram(test_results, is_vector)
    else:
        test_metrics = {}

    # Save checkpoint
    from dataclasses import asdict
    model_path = cfg.pretrained_model_file
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": asdict(model_cfg_used),
        "type_encoding": dataset.type_encoding,
        "target_stats": dataset.target_stats,
        "normalizer_state": {
            "strategy": dataset.normalizer.strategy,
            "stats": dataset.normalizer.stats,
            "is_vector": dataset.normalizer._is_vector,
            "scaler_params": _get_scaler_params(dataset.normalizer.scaler),
            "vector_mean": dataset.normalizer._vector_mean.tolist()
                if dataset.normalizer._vector_mean is not None else None,
            "vector_scale": float(dataset.normalizer._vector_scale)
                if dataset.normalizer._vector_scale is not None else None,
        },
        "is_vector": is_vector,
        "history": history,
        "test_metrics": test_metrics,
    }, model_path)
    logging.info(f"Best model saved: {model_path}")

    return test_metrics


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for E3NN")
    parser.add_argument("config", help="Base config JSON (provides dataset settings)")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--hpo_epochs", type=int, default=60,
                        help="Epochs per trial during HPO (default: 60)")
    parser.add_argument("--retrain_epochs", type=int, default=300,
                        help="Epochs for final retrain of best model (default: 300)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved study checkpoint")
    parser.add_argument("--study_path", type=str, default="",
                        help="Path to study .pkl file for resuming")
    parser.add_argument("--no_retrain", action="store_true",
                        help="Skip retraining best model after HPO")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n_startup_trials", type=int, default=10,
                        help="Random trials before TPE kicks in (default: 10)")
    parser.add_argument("--pruning_warmup", type=int, default=10,
                        help="Epochs before pruning can activate (default: 10)")
    args = parser.parse_args()

    # Setup
    setup_directories()
    os.makedirs("data/optuna", exist_ok=True)
    log_file = setup_logging(debug=False)
    set_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data/optuna"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("=" * 60)
    logging.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
    logging.info("=" * 60)
    logging.info(f"Device: {device}")
    logging.info(f"Config: {args.config}")
    logging.info(f"Trials: {args.n_trials}")
    logging.info(f"HPO epochs per trial: {args.hpo_epochs}")
    logging.info(f"Retrain epochs: {args.retrain_epochs}")
    logging.info(f"Startup trials (random): {args.n_startup_trials}")
    logging.info(f"Pruning warmup epochs: {args.pruning_warmup}")

    # Load base config (for dataset settings and defaults)
    base_cfg = RunConfig.from_json(args.config)

    # Create dataset ONCE (shared across all trials)
    logging.info("Loading dataset (shared across all trials)...")
    dataset = SimpleDataset(base_cfg.data)
    dataset.set_model_config(base_cfg.model)

    is_vector = dataset.target_stats.get('is_vector', False)
    logging.info(f"Dataset: {len(dataset)} molecules, "
                 f"{len(dataset.train_idx)} train, {len(dataset.val_idx)} val, "
                 f"{len(dataset.test_idx)} test, "
                 f"target={'vector' if is_vector else 'scalar'}")

    # Create or load study
    study_path = args.study_path or os.path.join(output_dir, f"study_{timestamp}.pkl")

    if args.resume and os.path.exists(study_path):
        logging.info(f"Resuming study from: {study_path}")
        with open(study_path, "rb") as f:
            study = pickle.load(f)
        remaining = max(0, args.n_trials - len(study.trials))
        logging.info(f"Loaded {len(study.trials)} trials, running {remaining} more")
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(
                seed=args.seed,
                n_startup_trials=args.n_startup_trials,
            ),
            pruner=MedianPruner(
                n_startup_trials=args.n_startup_trials,
                n_warmup_steps=args.pruning_warmup,
                interval_steps=5,  # Check pruning every 5 epochs
            ),
        )
        remaining = args.n_trials
        logging.info("Created new Optuna study")

    # Reduce Optuna logging noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Run optimization
    objective = create_objective(dataset, base_cfg, device, args.hpo_epochs)

    # Callback: save study after EVERY trial (protects against SLURM SIGKILL)
    # SIGKILL from SLURM time limit does NOT trigger finally: blocks,
    # so we must save after each completed/pruned trial.
    def save_study_callback(study, trial):
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        best_str = f", best={study.best_value:.4f}" if n_complete > 0 else ""
        logging.info(f"Study saved after trial {trial.number}: "
                     f"{n_complete} complete, {n_pruned} pruned{best_str}")

    t_start = time.time()
    try:
        study.optimize(
            objective,
            n_trials=remaining,
            show_progress_bar=True,
            gc_after_trial=True,  # Free GPU memory between trials
            callbacks=[save_study_callback],
        )
    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user")
    finally:
        # Final save (redundant but safe for non-SIGKILL cases)
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        logging.info(f"Study checkpoint saved: {study_path}")

    elapsed = time.time() - t_start
    logging.info(f"Optimization completed in {elapsed/60:.1f} minutes")

    # Results
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    logging.info("=" * 60)
    logging.info("OPTIMIZATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Completed: {len(completed)}, Pruned: {len(pruned)}, "
                 f"Failed: {len(study.trials) - len(completed) - len(pruned)}")
    logging.info(f"Pruning rate: {len(pruned)/max(len(study.trials),1):.1%}")

    if completed:
        best = study.best_trial
        logging.info(f"\nBest trial #{best.number}: val_loss = {best.value:.6f}")
        logging.info(f"  params = {best.n_params} parameters" if hasattr(best, 'n_params') else "")
        logging.info("  Hyperparameters:")
        for k, v in sorted(best.params.items()):
            logging.info(f"    {k}: {v}")

        # Save outputs
        best_config_path, best_config = save_best_config(study, base_cfg, output_dir, timestamp)
        save_results(study, output_dir, timestamp)
        plot_optimization_results(study, output_dir)

        # Retrain best model with full epochs
        if not args.no_retrain:
            # Update retrain epochs in best config
            best_config["train"]["num_epochs"] = args.retrain_epochs
            best_config["train"]["patience"] = max(50, args.retrain_epochs // 5)
            with open(best_config_path, "w") as f:
                json.dump(best_config, f, indent=2)

            retrain_metrics = retrain_best(best_config_path, device)

            logging.info("\n" + "=" * 60)
            logging.info("FINAL SUMMARY")
            logging.info("=" * 60)
            logging.info(f"HPO best val_loss: {best.value:.6f}")
            if retrain_metrics:
                for k, v in retrain_metrics.items():
                    if isinstance(v, float):
                        logging.info(f"Final test {k}: {v:.4f}")
            logging.info(f"Best config: {best_config_path}")
            logging.info(f"Run production: python e3nn_predict_v2.py {best_config_path}")
    else:
        logging.error("No trials completed successfully!")


if __name__ == "__main__":
    main()
