#!/usr/bin/env python3
"""
CLI: run train, cross-validate, or predict from a config JSON.

Usage:
    python -m e3verde.run config.json
    python -m e3verde.run config.json --mode train
    python -m e3verde.run config.json --mode cross_validate
    python -m e3verde.run config.json --mode predict
    python -m e3verde.run config.json --debug

Library code (training loop, metrics, etc.) lives in ``e3verde.training``.
"""

import os
import sys
import logging
import argparse
import traceback
from dataclasses import asdict

import torch
import numpy as np

from e3verde import (
    RunConfig, ModelConfig,
    SimpleDataset,
    PeriodicNetwork,
    set_seed, setup_directories, setup_logging,
    train_model, evaluate_model, compute_metrics,
    cross_validate, predict_new_molecules,
    diagnose_target_distribution,
    CSVWriter,
    plot_loss_history, plot_parity, plot_error_histogram,
    plot_target_distribution, plot_residuals,
    ATOMIC_PROP_DIM,
    _get_scaler_params,
)
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
import datetime


def main():
    parser = argparse.ArgumentParser(description="E(3)-VERDE molecular property prediction")
    parser.add_argument("config", help="Path to config JSON")
    parser.add_argument("--mode", choices=["train", "cross_validate", "predict"],
                        help="Override the mode in the config JSON")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_directories()
    setup_logging(debug=args.debug)

    if not os.path.exists(args.config):
        logging.error(f"Config not found: {args.config}")
        sys.exit(1)

    cfg = RunConfig.from_json(args.config)
    if args.mode:
        cfg.mode = args.mode

    set_seed(cfg.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    if not cfg.data.dataset_path or not cfg.data.target_key or not cfg.data.structure_field:
        logging.error("Missing required data config: dataset_path, target_key, structure_field")
        sys.exit(1)

    logging.info("Loading dataset...")
    try:
        dataset = SimpleDataset(cfg.data)
        dataset.set_model_config(cfg.model)
    except Exception as e:
        logging.error(f"Dataset error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

    is_vector = dataset.target_stats.get('is_vector', False)
    logging.info(f"Dataset: {len(dataset)} molecules "
                 f"({len(dataset.train_idx)} train, {len(dataset.val_idx)} val, "
                 f"{len(dataset.test_idx)} test), "
                 f"target={'vector' if is_vector else 'scalar'}")

    train_targets = []
    for idx in dataset.train_idx:
        tv = dataset.database[idx][dataset.target_key]
        train_targets.append(tv if isinstance(tv, (list, np.ndarray)) else [tv])
    if train_targets:
        diagnose_target_distribution(np.array(train_targets))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg.mode}_{timestamp}"
    writer = CSVWriter(run_id)

    if cfg.mode == "train":
        logging.info("Starting training...")
        model, history, is_vector, model_cfg_used = train_model(dataset, cfg.model, cfg.train, device)

        val_results = evaluate_model(model, dataset, device, is_vector, split="val")
        if val_results:
            val_metrics = compute_metrics(val_results, is_vector)
            logging.info("Validation Metrics:")
            for k, v in val_metrics.items():
                if isinstance(v, float):
                    logging.info(f"  {k}: {v:.4f}" if abs(v) < 1000 else f"  {k}: {v:.2e}")
            writer.save_predictions(val_results, is_vector, split="val")
            writer.save_metrics(val_metrics, split="val", extra={
                "run_id": run_id, "n_samples": len(dataset.val_idx)})
        else:
            val_metrics = {}

        test_results = evaluate_model(model, dataset, device, is_vector, split="test")
        if test_results:
            test_metrics = compute_metrics(test_results, is_vector)
            logging.info("Test Metrics (held-out, unbiased):")
            for k, v in test_metrics.items():
                if isinstance(v, float):
                    logging.info(f"  {k}: {v:.4f}" if abs(v) < 1000 else f"  {k}: {v:.2e}")
                elif isinstance(v, dict):
                    logging.info(f"  {k}: {v}")
                else:
                    logging.info(f"  {k}: {v}")
            writer.save_predictions(test_results, is_vector, split="test")
            writer.save_metrics(test_metrics, split="test", extra={
                "run_id": run_id, "target": dataset.target_key,
                "n_train": len(dataset.train_idx),
                "n_val": len(dataset.val_idx),
                "n_test": len(dataset.test_idx),
                "best_epoch": history[-1].get("best_epoch", len(history)),
                "total_epochs": len(history),
            })
        else:
            test_metrics = {}
            logging.warning("No test data available for final evaluation")

        writer.save_history(history)
        writer.save_splits(dataset)
        writer.save_config(cfg)

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
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }, model_path)
        logging.info(f"Model saved: {model_path}")

        plot_results = test_results if test_results else val_results
        logging.info("Generating plots...")
        plot_loss_history(history)
        plot_target_distribution(dataset, is_vector)
        if plot_results:
            plot_parity(plot_results, is_vector, target_name=dataset.target_key)
            plot_error_histogram(plot_results, is_vector)
            plot_residuals(plot_results, is_vector)

    elif cfg.mode == "cross_validate":
        logging.info("Starting cross-validation...")
        all_metrics, avg_metrics = cross_validate(
            dataset, cfg.model, cfg.train, device, k=cfg.cv_folds)
        writer.save_config(cfg)

    elif cfg.mode == "predict":
        if not cfg.molecules_path:
            logging.error("molecules_path not specified in config")
            sys.exit(1)
        if not cfg.pretrained_model_file or not os.path.exists(cfg.pretrained_model_file):
            logging.error(f"Model file not found: {cfg.pretrained_model_file}")
            sys.exit(1)

        checkpoint = torch.load(cfg.pretrained_model_file, map_location=device, weights_only=False)

        saved_model_cfg = ModelConfig(**{
            k: v for k, v in checkpoint.get("model_config", {}).items()
            if k in ModelConfig.__dataclass_fields__
        }) if "model_config" in checkpoint else cfg.model

        in_dim = len(checkpoint["type_encoding"]) + (
            ATOMIC_PROP_DIM if saved_model_cfg.use_rich_features else 1)
        model = PeriodicNetwork(in_dim, saved_model_cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        is_vector = checkpoint.get("is_vector", False)
        dataset.use_rich_features = saved_model_cfg.use_rich_features

        norm_state = checkpoint.get("normalizer_state", {})
        if norm_state:
            dataset.normalizer.stats = norm_state.get("stats", dataset.normalizer.stats)
            dataset.normalizer._is_vector = norm_state.get("is_vector", False)
            dataset.normalizer._fitted = True
            if norm_state.get("vector_scale") is not None:
                vm = norm_state.get("vector_mean")
                dataset.normalizer._vector_mean = np.array(vm) if vm is not None else None
                dataset.normalizer._vector_scale = norm_state["vector_scale"]
                dataset.normalizer.scaler = None
                logging.info(f"Restored vector normalizer: scale={dataset.normalizer._vector_scale}"
                             f"{', mean=' + str(dataset.normalizer._vector_mean) if vm is not None else ' (scale-only)'}")
            elif norm_state.get("scaler_params") is not None:
                sp = norm_state["scaler_params"]
                if sp["type"] == "standard":
                    dataset.normalizer.scaler = StandardScaler()
                    dataset.normalizer.scaler.mean_ = np.array(sp["mean"])
                    dataset.normalizer.scaler.scale_ = np.array(sp["scale"])
                    dataset.normalizer.scaler.var_ = np.array(sp["scale"]) ** 2
                    dataset.normalizer.scaler.n_features_in_ = len(sp["mean"])
                elif sp["type"] == "robust":
                    dataset.normalizer.scaler = RobustScaler()
                    dataset.normalizer.scaler.center_ = np.array(sp["center"])
                    dataset.normalizer.scaler.scale_ = np.array(sp["scale"])
                    dataset.normalizer.scaler.n_features_in_ = len(sp["center"])
                elif sp["type"] == "quantile":
                    dataset.normalizer.scaler = QuantileTransformer(
                        n_quantiles=sp.get("n_quantiles", 1000),
                        output_distribution=sp.get("output_distribution", "uniform"),
                        random_state=42,
                    )
                    dataset.normalizer.scaler.quantiles_ = np.array(sp["quantiles"])
                    dataset.normalizer.scaler.references_ = np.array(sp["references"])
                    dataset.normalizer.scaler.n_quantiles_ = int(sp.get("n_quantiles", dataset.normalizer.scaler.quantiles_.shape[0]))
                    dataset.normalizer.scaler.n_features_in_ = int(sp.get("n_features_in", dataset.normalizer.scaler.quantiles_.shape[1]))
                logging.info(f"Restored scalar normalizer: type={sp['type']}")
        dataset.type_encoding = checkpoint["type_encoding"]
        dataset.type_onehot = torch.eye(len(dataset.type_encoding), dtype=torch.float32)

        results_df = predict_new_molecules(
            model, dataset, cfg.molecules_path, device, is_vector,
            fmt=cfg.molecules_format)
        logging.info(f"Predictions: {len(results_df)} molecules")
        print(results_df)

    else:
        logging.error(f"Unknown mode: {cfg.mode}. Use 'train', 'cross_validate', or 'predict'")
        sys.exit(1)

    logging.info("Completed successfully!")


if __name__ == "__main__":
    main()
