#!/usr/bin/env python3
"""
Training loop, evaluation, metrics, cross-validation, and batch prediction.

Library module used by ``e3verde.run``, ``e3verde.hpo``, and ``e3verde.learning_curve``.
For the command-line entry point (train / CV / predict), use ``python -m e3verde.run``.
"""

import os
import json
import copy
import logging
import datetime
from dataclasses import asdict
from io import StringIO
from typing import Dict, List, Any

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.stats as stats

from torch_geometric.loader import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import KFold, train_test_split
from ase.io import read
from e3nn.o3 import Irreps

from e3verde.config import ModelConfig, TrainConfig
from e3verde.data import SimpleDataset, prepare_batch, build_graph_from_atoms, check_nan_inf
from e3verde.model import PeriodicNetwork, EMA, LabelSmoothingLoss
from e3verde.plots import (
    plot_loss_history, plot_parity, plot_error_histogram,
    plot_cv_metrics, plot_cv_learning_curves,
)


def _get_scaler_params(scaler):
    """Extract scaler parameters for checkpoint serialization."""
    if scaler is None:
        return None
    if isinstance(scaler, StandardScaler):
        return {"type": "standard", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    if isinstance(scaler, RobustScaler):
        return {"type": "robust", "center": scaler.center_.tolist(), "scale": scaler.scale_.tolist()}
    if isinstance(scaler, QuantileTransformer):
        # Store full fitted state needed for inverse_transform in predict mode.
        return {
            "type": "quantile",
            "n_quantiles": int(getattr(scaler, "n_quantiles_", scaler.n_quantiles)),
            "output_distribution": scaler.output_distribution,
            "quantiles": scaler.quantiles_.tolist(),
            "references": scaler.references_.tolist(),
            "n_features_in": int(getattr(scaler, "n_features_in_", scaler.quantiles_.shape[1])),
        }
    return None


def auto_adjust_irreps_out(cfg: ModelConfig, is_vector: bool, vector_dim: int) -> str:
    """
    Auto-adjust output irreps to match target dimensionality.

    EQUIVARIANCE NOTE:
    - For 3D vectors: use 1x1o (proper SO(3) vector, transforms correctly under rotations)
    - For scalars: use 1x0e (invariant under rotations)
    - For multiple scalars: use Nx0e (N independent invariants)
    """
    output_irreps = Irreps(cfg.irreps_out)
    output_dim = output_irreps.dim

    if is_vector and output_dim != vector_dim:
        if vector_dim == 3:
            new_irreps = "1x1o"
            logging.info(f"Auto-adjusted irreps_out: '{cfg.irreps_out}' -> '{new_irreps}' (3D vector)")
        else:
            new_irreps = f"{vector_dim}x0e"
            logging.info(f"Auto-adjusted irreps_out: '{cfg.irreps_out}' -> '{new_irreps}' ({vector_dim}D)")
        return new_irreps
    return cfg.irreps_out


def train_model(dataset: SimpleDataset, model_cfg: ModelConfig,
                train_cfg: TrainConfig, device: torch.device,
                epoch_callback=None):
    """Train the equivariant model with optional AMP and EMA.

    Args:
        epoch_callback: Optional callable(epoch, train_loss, val_loss) called after each epoch.
            If it raises an exception (e.g. optuna.TrialPruned), training stops.
            Used by Optuna for trial pruning.

    Returns:
        (model, history, is_vector, model_cfg_adjusted)
    """
    is_vector = dataset.target_stats.get('is_vector', False)
    vector_dim = dataset.target_stats.get('vector_dim', 1)

    adjusted_irreps = auto_adjust_irreps_out(model_cfg, is_vector, vector_dim)
    model_cfg_adjusted = copy.copy(model_cfg)
    model_cfg_adjusted.irreps_out = adjusted_irreps

    if model_cfg_adjusted.num_neighbors <= 0:
        model_cfg_adjusted.num_neighbors = max(1.0, dataset.compute_avg_num_neighbors())

    in_dim = dataset.feature_dim
    model = PeriodicNetwork(in_dim, model_cfg_adjusted).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: {n_params:,} parameters, output irreps={model.irreps_out}")

    # Separate param groups: no weight decay on bias and LayerNorm.
    # Weight decay on bias/norm params acts as an unintended regularizer
    # that can hurt performance, especially for equivariant networks where
    # norm params control the scale of geometric features.
    # Reference: Loshchilov & Hutter, 2019; NequIP (Batzner et al., 2022).
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'layer_norm' in name or param.ndim == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': train_cfg.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=train_cfg.learning_rate)
    logging.info(f"Optimizer: AdamW, {len(decay_params)} params with WD={train_cfg.weight_decay}, "
                 f"{len(no_decay_params)} params without WD (bias/norm)")

    if train_cfg.scheduler_type == "cosine_warmup":
        warmup = train_cfg.warmup_epochs
        total = train_cfg.num_epochs
        base_lr = train_cfg.learning_rate
        min_lr_ratio = train_cfg.min_lr / base_lr
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return max(cosine, min_lr_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=train_cfg.patience // 3, factor=0.5,
            min_lr=train_cfg.min_lr)

    # EQUIVARIANCE NOTE: Label smoothing smooths targets toward batch mean.
    # For vectors, batch mean direction is arbitrary -> disable for vectors.
    if train_cfg.label_smoothing > 0 and not is_vector:
        criterion = LabelSmoothingLoss(train_cfg.label_smoothing)
    else:
        if train_cfg.loss_function == "mse":
            criterion = nn.MSELoss()
        elif train_cfg.loss_function == "huber":
            criterion = nn.HuberLoss()
        else:
            criterion = nn.L1Loss()
        if train_cfg.label_smoothing > 0 and is_vector:
            logging.warning("Label smoothing disabled for vector targets "
                            "(batch mean direction is arbitrary, would harm equivariance)")
    logging.info(f"Loss function: {criterion.__class__.__name__}")

    amp_enabled = train_cfg.use_amp and device.type == "cuda"
    grad_scaler = GradScaler(enabled=amp_enabled) if amp_enabled else None
    if amp_enabled:
        logging.info("Mixed precision training enabled (AMP)")
        logging.info("  Note: e3nn tensor products use fp32 CG coefficients internally. "
                     "AMP may slightly reduce numerical precision of equivariant operations.")

    ema = EMA(model, train_cfg.ema_decay) if train_cfg.use_ema else None
    if ema:
        logging.info(f"EMA enabled with decay={train_cfg.ema_decay}")

    train_loader = DataLoader(dataset.train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=train_cfg.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_data = dataset.val_dataset if len(dataset.val_idx) > 0 else dataset.test_dataset
    val_loader = DataLoader(val_data, batch_size=train_cfg.batch_size, shuffle=False,
                            num_workers=train_cfg.num_workers,
                            pin_memory=(device.type == "cuda"))
    val_source = "val" if len(dataset.val_idx) > 0 else "test"
    if val_source == "test":
        logging.warning("No validation split (val_size=0). Using TEST set for early stopping. "
                        "This causes information leakage. Set val_size > 0 for proper methodology.")
    logging.info(f"Early stopping monitored on: {val_source} set ({len(val_data)} samples)")

    accum_steps = train_cfg.gradient_accumulation_steps
    loss_name = train_cfg.loss_function.upper()
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(train_cfg.num_epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        optimizer.zero_grad()
        last_step = -1

        for step, batch in enumerate(train_loader):
            last_step = step
            batch, target = prepare_batch(batch, device, is_vector, vector_dim)

            if amp_enabled:
                with autocast(device_type="cuda"):
                    pred = model(batch)
                    check_nan_inf(pred, "predictions", raise_error=True)
                    loss = criterion(pred, target) / accum_steps
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error("NaN/Inf loss in AMP path, skipping batch")
                    optimizer.zero_grad()
                    continue
                grad_scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
            else:
                pred = model(batch)
                check_nan_inf(pred, "predictions", raise_error=True)
                loss = criterion(pred, target) / accum_steps
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error("NaN/Inf loss, skipping batch")
                    optimizer.zero_grad()
                    continue
                loss.backward()
                if (step + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

            batch_size = batch.batch.max().item() + 1
            train_loss += loss.item() * accum_steps * batch_size
            train_samples += batch_size

        # Handle remaining gradients from incomplete accumulation
        if train_samples > 0 and last_step >= 0 and (last_step + 1) % accum_steps != 0:
            if amp_enabled:
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

        train_loss /= max(train_samples, 1)

        if ema:
            ema.apply_shadow(model)

        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch, target = prepare_batch(batch, device, is_vector, vector_dim)
                if amp_enabled:
                    with autocast(device_type="cuda"):
                        pred = model(batch)
                        loss = criterion(pred, target)
                else:
                    pred = model(batch)
                    loss = criterion(pred, target)
                batch_size = batch.batch.max().item() + 1
                val_loss += loss.item() * batch_size
                val_samples += batch_size

        val_loss /= max(val_samples, 1)

        if ema:
            ema.restore(model)

        if train_cfg.scheduler_type == "cosine_warmup":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch+1:03d}: Train {loss_name}={train_loss:.4f}, "
                     f"Val {loss_name}={val_loss:.4f}, LR={lr:.2e}")

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "lr": lr, "best_epoch": best_epoch
        })

        if epoch_callback is not None:
            try:
                epoch_callback(epoch, train_loss, val_loss)
            except Exception:
                # CRITICAL: Re-raise so Optuna receives TrialPruned.
                # If we 'break' instead, train_model returns normally,
                # Optuna never sees the pruning, and MedianPruner is disabled.
                logging.info(f"Training stopped by callback at epoch {epoch+1}")
                raise

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            if ema:
                # Val loss was computed with EMA weights, so save EMA weights as best
                best_state = copy.deepcopy(ema.shadow)
            else:
                best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= train_cfg.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        if ema:
            for name, param in model.named_parameters():
                if name in best_state:
                    param.data.copy_(best_state[name])
        else:
            model.load_state_dict(best_state)
        logging.info(f"Loaded best model from epoch {best_epoch}")

    return model, history, is_vector, model_cfg_adjusted


def evaluate_model(model, dataset, device, is_vector, split="test"):
    """Evaluate model on a specific split (test, val, or train)."""
    model.eval()
    vector_dim = dataset.target_stats.get('vector_dim', 1)

    if split == "val":
        data_list = dataset.val_dataset
        idx_list = dataset.val_idx
    elif split == "train":
        data_list = dataset.train_dataset
        idx_list = dataset.train_idx
    else:
        data_list = dataset.test_dataset
        idx_list = dataset.test_idx

    if len(data_list) == 0:
        logging.warning(f"No data in '{split}' split for evaluation")
        return []

    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    results = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch, _ = prepare_batch(batch, device, is_vector, vector_dim)

            inchi_key = getattr(batch, "inchi_key", "")
            pred = model(batch)

            pred_np = pred.cpu().numpy()
            truth_np = batch.y_original.cpu().numpy() if hasattr(batch, 'y_original') else batch.y.cpu().numpy()
            pred_denorm = dataset.normalizer.inverse_transform(pred_np)

            pred_val = pred_denorm.squeeze()
            truth_val = truth_np.squeeze()

            num_atoms = 0
            atom_types = []
            try:
                if idx < len(idx_list):
                    mol_idx = idx_list[idx]
                    entry = dataset.database[mol_idx]
                    atoms = read(StringIO(entry[dataset.structure_field]), format="xyz")
                    num_atoms = len(atoms)
                    atom_types = atoms.get_chemical_symbols()
            except Exception:
                pass

            results.append({
                "inchi_key": inchi_key,
                "pred": pred_val.tolist() if isinstance(pred_val, np.ndarray) else pred_val,
                "truth": truth_val.tolist() if isinstance(truth_val, np.ndarray) else truth_val,
                "num_atoms": num_atoms,
                "atom_types": atom_types,
            })

    return results


def compute_metrics(eval_results: List[Dict], is_vector: bool) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    preds = np.array([r["pred"] for r in eval_results])
    truths = np.array([r["truth"] for r in eval_results])
    molecule_sizes = np.array([r["num_atoms"] for r in eval_results])

    if is_vector:
        return _compute_vector_metrics(preds, truths, molecule_sizes)
    return _compute_scalar_metrics(preds, truths, molecule_sizes)


def _compute_scalar_metrics(preds, truths, molecule_sizes=None):
    """Scalar metrics with bimodality-aware alternatives."""
    preds = preds.flatten()
    truths = truths.flatten()
    errors = np.abs(preds - truths)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    ss_res = np.sum((preds - truths) ** 2)
    ss_tot = np.sum((truths - truths.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0

    ss_tot_median = np.sum((truths - np.median(truths)) ** 2)
    r2_median = 1 - (ss_res / (ss_tot_median + 1e-10)) if ss_tot_median > 1e-10 else 0.0

    pearson = stats.pearsonr(preds, truths)[0] if len(preds) > 1 else 0.0
    spearman = stats.spearmanr(preds, truths)[0] if len(preds) > 1 else 0.0
    try:
        kendall = stats.kendalltau(preds, truths)[0] if len(preds) > 1 else 0.0
    except Exception:
        kendall = 0.0

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "R2_median": float(r2_median),
        "MedAE": float(np.median(errors)),
        "Max_Error": float(np.max(errors)),
        "Error_90th": float(np.percentile(errors, 90)),
        "Error_95th": float(np.percentile(errors, 95)),
        "Pearson": float(pearson),
        "Spearman": float(spearman),
        "Kendall_tau": float(kendall),
        "Mean_Signed_Error": float(np.mean(preds - truths)),
        "Std_Error": float(np.std(errors)),
    }

    if molecule_sizes is not None and len(molecule_sizes) == len(preds):
        for size in np.unique(molecule_sizes):
            if size == 0:
                continue
            mask = molecule_sizes == size
            if mask.sum() > 1:
                metrics[f"MAE_size_{size}"] = float(np.mean(errors[mask]))

    return metrics


def _compute_vector_metrics(preds, truths, molecule_sizes=None):
    """Vector metrics with per-component analysis."""
    preds = preds.reshape(-1, preds.shape[-1])
    truths = truths.reshape(-1, truths.shape[-1])
    vector_dim = preds.shape[1]
    comp_labels = ['X', 'Y', 'Z'][:vector_dim] if vector_dim <= 3 else \
        [f'C{i}' for i in range(vector_dim)]

    errors_mag = np.linalg.norm(preds - truths, axis=1)
    mae = float(np.mean(errors_mag))
    rmse = float(np.sqrt(np.mean(errors_mag ** 2)))

    r2_components = []
    for i in range(vector_dim):
        ss_res = np.sum((preds[:, i] - truths[:, i]) ** 2)
        ss_tot = np.sum((truths[:, i] - truths[:, i].mean()) ** 2)
        r2_c = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0
        r2_components.append(float(r2_c))
    r2 = float(np.mean(r2_components))

    pred_mags = np.linalg.norm(preds, axis=1)
    truth_mags = np.linalg.norm(truths, axis=1)
    try:
        spearman = float(stats.spearmanr(pred_mags, truth_mags)[0]) if len(pred_mags) > 1 else 0.0
        pearson = float(stats.pearsonr(pred_mags, truth_mags)[0]) if len(pred_mags) > 1 else 0.0
    except Exception:
        spearman, pearson = 0.0, 0.0

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "R2_per_component": {comp_labels[i]: r2_components[i] for i in range(vector_dim)},
        "MAE_per_component": {comp_labels[i]: float(np.mean(np.abs(preds[:, i] - truths[:, i])))
                              for i in range(vector_dim)},
        "MedAE": float(np.median(errors_mag)),
        "Max_Error": float(np.max(errors_mag)),
        "Error_90th": float(np.percentile(errors_mag, 90)),
        "Error_95th": float(np.percentile(errors_mag, 95)),
        "Spearman": spearman,
        "Pearson": pearson,
        "Mean_Signed_Error": np.mean(preds - truths, axis=0).tolist(),
        "Std_Error": float(np.std(errors_mag)),
    }

    if vector_dim == 3:
        try:
            dot = np.sum(preds * truths, axis=1)
            pn = np.linalg.norm(preds, axis=1)
            tn = np.linalg.norm(truths, axis=1)
            valid = (pn > 1e-10) & (tn > 1e-10)
            if valid.sum() > 0:
                cosines = np.clip(dot[valid] / (pn[valid] * tn[valid]), -1.0, 1.0)
                metrics["Angular_Error_deg"] = float(np.mean(np.arccos(cosines) * 180 / np.pi))
        except Exception:
            pass

    return metrics


class CSVWriter:
    """
    Centralized CSV output.

    Produces:
      - {run_id}__predictions_{split}.csv
      - {run_id}__metrics_{split}.csv
      - {run_id}__training_history.csv
      - {run_id}__splits.csv
      - {run_id}__run_config.csv
    """

    def __init__(self, run_id: str, output_dir: str = "data/csv"):
        self.run_id = run_id
        self.output_dir = output_dir

    def _path(self, suffix):
        return os.path.join(self.output_dir, f"{self.run_id}__{suffix}.csv")

    def save_predictions(self, eval_results, is_vector, split="test"):
        rows = []
        for r in eval_results:
            pred, truth = np.array(r["pred"]), np.array(r["truth"])
            if is_vector:
                error = float(np.linalg.norm(pred - truth))
            else:
                pred_s = float(pred.item() if pred.size == 1 else pred.flat[0])
                truth_s = float(truth.item() if truth.size == 1 else truth.flat[0])
                error = pred_s - truth_s

            rows.append({
                "molecule_id": r["inchi_key"],
                "num_atoms": r["num_atoms"],
                "true_value": r["truth"] if not is_vector else str(r["truth"]),
                "predicted_value": r["pred"] if not is_vector else str(r["pred"]),
                "error": float(error),
                "absolute_error": float(abs(error)),
            })

        path = self._path(f"predictions_{split}")
        pd.DataFrame(rows).to_csv(path, index=False)
        logging.info(f"Predictions ({split}) saved: {path}")

    def save_metrics(self, metrics, split="test", extra=None):
        data = {**metrics}
        data["split"] = split
        if extra:
            data.update(extra)
        path = self._path(f"metrics_{split}")
        pd.DataFrame([data]).to_csv(path, index=False)
        logging.info(f"Metrics ({split}) saved: {path}")

    def save_history(self, history):
        path = self._path("training_history")
        pd.DataFrame(history).to_csv(path, index=False)
        logging.info(f"History saved: {path}")

    def save_splits(self, dataset):
        rows = []
        for split_name, indices in [("train", dataset.train_idx),
                                     ("val", dataset.val_idx),
                                     ("test", dataset.test_idx)]:
            for i, idx in enumerate(indices):
                entry = dataset.database[idx]
                rows.append({
                    "split": split_name,
                    "global_index": idx,
                    "inchi_key": entry.get("inchi_key", f"molecule_{idx}"),
                    "smiles": entry.get("smiles", ""),
                    "formula": entry.get("formula", ""),
                    "target_value": entry.get(dataset.target_key),
                })

        path = self._path("splits")
        pd.DataFrame(rows).to_csv(path, index=False)
        logging.info(f"Splits saved: {path}")

    def save_config(self, run_config):
        flat = {
            "mode": run_config.mode,
            **{f"model.{k}": v for k, v in asdict(run_config.model).items()},
            **{f"train.{k}": v for k, v in asdict(run_config.train).items()},
            **{f"data.{k}": v for k, v in asdict(run_config.data).items()},
        }
        path = self._path("run_config")
        pd.DataFrame([flat]).to_csv(path, index=False)
        logging.info(f"Config saved: {path}")


def cross_validate(dataset, model_cfg, train_cfg, device, k=5):
    """
    K-fold cross-validation with proper inner validation split.

    For each fold:
    - fold test set: used for evaluation metrics
    - fold train set is further split into train + val
    - val is used for early stopping within the fold
    - This prevents information leakage from test into early stopping

    Saves per-fold: predictions, training history, learning curve plot.
    Saves combined: all predictions, metrics, learning curve overlay.
    """
    is_vector = dataset.target_stats.get('is_vector', False)
    vector_dim = dataset.target_stats.get('vector_dim', 1)

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = list(range(len(dataset.database)))
    all_metrics = []
    all_results = []
    all_histories = []

    inner_val_fraction = 0.1
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = "data/cross_validation"

    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(indices), 1):
        logging.info(f"--- Fold {fold}/{k} ---")

        inner_train_idx, inner_val_idx = train_test_split(
            list(train_val_idx), test_size=inner_val_fraction, random_state=42 + fold)

        dataset.train_idx = inner_train_idx
        dataset.val_idx = inner_val_idx
        dataset.test_idx = list(test_idx)
        dataset._fit_normalization()

        logging.info(f"  Fold {fold}: {len(inner_train_idx)} train, "
                     f"{len(inner_val_idx)} val (early stopping), {len(test_idx)} test")

        model, history, _, _model_cfg_used = train_model(dataset, model_cfg, train_cfg, device)
        all_histories.append(history)

        fold_history_df = pd.DataFrame(history)
        fold_history_df["fold"] = fold
        fold_history_df.to_csv(
            os.path.join(cv_dir, f"cv_{timestamp}__fold{fold:02d}_history.csv"), index=False)

        plot_loss_history(history, title_suffix=f" (Fold {fold})")

        eval_results = evaluate_model(model, dataset, device, is_vector, split="test")
        metrics = compute_metrics(eval_results, is_vector)
        metrics["Fold"] = fold
        metrics["n_train"] = len(inner_train_idx)
        metrics["n_val"] = len(inner_val_idx)
        metrics["n_test"] = len(test_idx)
        metrics["best_epoch"] = history[-1].get("best_epoch", len(history)) if history else 0
        metrics["total_epochs"] = len(history)
        all_metrics.append(metrics)

        fold_rows = []
        for r in eval_results:
            pred, truth = np.array(r["pred"]), np.array(r["truth"])
            if is_vector:
                error = float(np.linalg.norm(pred - truth))
            else:
                error = float(pred.flat[0]) - float(truth.flat[0])
            fold_rows.append({
                "fold": fold, "molecule_id": r["inchi_key"],
                "true_value": r["truth"] if not is_vector else str(r["truth"]),
                "predicted_value": r["pred"] if not is_vector else str(r["pred"]),
                "error": error, "absolute_error": abs(error),
            })
        pd.DataFrame(fold_rows).to_csv(
            os.path.join(cv_dir, f"cv_{timestamp}__fold{fold:02d}_predictions.csv"), index=False)

        all_results.extend(eval_results)

    # Combined outputs
    combined_rows = []
    for r in all_results:
        pred, truth = np.array(r["pred"]), np.array(r["truth"])
        if is_vector:
            error = float(np.linalg.norm(pred - truth))
        else:
            error = float(pred.flat[0]) - float(truth.flat[0])
        combined_rows.append({
            "molecule_id": r["inchi_key"],
            "true_value": r["truth"] if not is_vector else str(r["truth"]),
            "predicted_value": r["pred"] if not is_vector else str(r["pred"]),
            "error": error, "absolute_error": abs(error),
        })
    pd.DataFrame(combined_rows).to_csv(
        os.path.join(cv_dir, f"cv_{timestamp}__all_predictions.csv"), index=False)
    logging.info(f"Combined CV predictions: {len(combined_rows)} samples")

    _metadata_keys = {"Fold", "n_train", "n_val", "n_test", "best_epoch", "total_epochs"}
    numeric_keys = {k for m in all_metrics for k, v in m.items()
                    if isinstance(v, (int, float)) and k not in _metadata_keys
                    and not np.isnan(v)}
    avg_metrics = {}
    for key in numeric_keys:
        vals = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
        if vals:
            avg_metrics[key] = float(np.mean(vals))
            avg_metrics[f"{key}_std"] = float(np.std(vals))
    avg_metrics["n_folds"] = k

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(cv_dir, f"cv_{timestamp}__metrics_per_fold.csv"), index=False)
    pd.DataFrame([avg_metrics]).to_csv(
        os.path.join(cv_dir, f"cv_{timestamp}__metrics_averaged.csv"), index=False)

    plot_cv_metrics(all_metrics, avg_metrics)
    plot_cv_learning_curves(all_histories, k)

    if all_results:
        plot_parity(all_results, is_vector, target_name=f"{dataset.target_key} (CV)")

    logging.info("=== Cross-Validation Summary ===")
    for key in sorted(avg_metrics):
        if not key.endswith("_std") and key not in ("n_folds",):
            std = avg_metrics.get(f"{key}_std", 0)
            if isinstance(avg_metrics[key], float):
                logging.info(f"  {key}: {avg_metrics[key]:.4f} +/- {std:.4f}")

    return all_metrics, avg_metrics


def predict_new_molecules(model, dataset, molecules_path, device, is_vector, fmt="xyz"):
    model.eval()
    atoms_list, file_names = _load_molecules(molecules_path, dataset.structure_field, fmt)

    if not atoms_list:
        logging.warning("No valid molecules found")
        return pd.DataFrame()

    results = []
    with torch.no_grad():
        for i, atoms in enumerate(atoms_list):
            symbols = atoms.get_chemical_symbols()
            unknown = set(symbols) - set(dataset.type_encoding.keys())
            if unknown:
                logging.warning(f"Molecule {i+1}: unknown atoms {unknown}, skipping")
                continue

            data = build_graph_from_atoms(
                atoms, dataset.type_encoding, dataset.type_onehot, dataset.cutoff,
                dataset.normalize_features, dataset.mass_mean, dataset.mass_std,
                use_rich_features=dataset.use_rich_features)
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
            data = data.to(device)

            pred = model(data)
            pred_np = pred.cpu().numpy().squeeze()
            pred_denorm = dataset.normalizer.inverse_transform(pred_np.reshape(1, -1)).squeeze()

            mol_id = atoms.info.get('inchi_key', f"molecule_{i+1}") if hasattr(atoms, 'info') else f"molecule_{i+1}"

            result = {
                "molecule_id": mol_id,
                "xyz_file": file_names[i] if i < len(file_names) else f"unknown_{i+1}",
                "formula": atoms.get_chemical_formula(),
                "num_atoms": len(atoms),
            }

            if is_vector:
                pred_arr = np.array(pred_denorm)
                if len(pred_arr) >= 3:
                    result.update({
                        "predicted_x": float(pred_arr[0]),
                        "predicted_y": float(pred_arr[1]),
                        "predicted_z": float(pred_arr[2]),
                        "predicted_magnitude": float(np.linalg.norm(pred_arr[:3])),
                    })
                else:
                    result["predicted_value"] = pred_arr.tolist()
            else:
                result["predicted_value"] = float(pred_denorm) if np.ndim(pred_denorm) == 0 else float(pred_denorm[0])

            results.append(result)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    if len(df) > 0:
        path = os.path.join("data", "predictions", f"predictions_{ts}.csv")
        df.to_csv(path, index=False)
        logging.info(f"Predictions saved: {path}")
    return df


def _load_molecules(path, structure_field, fmt):
    """Load molecules from file, directory, or JSON database."""
    atoms_list, file_names = [], []

    if fmt.lower() == "json" and os.path.isfile(path):
        with open(path, "r") as f:
            db = json.load(f)
        for idx, entry in enumerate(db):
            if structure_field in entry:
                try:
                    atoms = read(StringIO(entry[structure_field]), format="xyz")
                    atoms_list.append(atoms)
                    file_names.append(entry.get("inchi_key", f"entry_{idx}"))
                except Exception:
                    pass
    elif os.path.isfile(path):
        fname = os.path.basename(path)
        loaded = read(path, index=":", format=fmt)
        atoms_list = loaded if isinstance(loaded, list) else [loaded]
        file_names = [fname] * len(atoms_list)
    elif os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if fname.endswith(f".{fmt}"):
                try:
                    atoms_list.append(read(os.path.join(path, fname), format=fmt))
                    file_names.append(fname)
                except Exception:
                    pass
    else:
        raise ValueError(f"Path not found: {path}")

    return atoms_list, file_names
