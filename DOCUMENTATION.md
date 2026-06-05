# e3verde — Technical documentation

> **Start here:** [README.md](README.md) for installation, quick start, and workflow overview.  
> This file is the full reference for configuration, dataset format, and outputs.

## 1. Overview

**E(3)-VERDE** trains E(3)-equivariant graph neural networks to predict molecular properties from 3D structures. All Python code lives in the `e3verde` package:

- **CLI modules** (`run`, `hpo`, `learning_curve`) — parse JSON configs and invoke workflows
- **Library modules** (`training`, `data`, `model`, `config`, `plots`) — implementation

Scalar targets use invariant readouts; vector targets (e.g. dipole vectors) use equivariant outputs with scale-only normalization (no per-component mean centering).

---

## 2. Installation

### Dependencies

The code imports the following packages (install a PyTorch build appropriate for your CUDA setup first):

| Package | Role |
|---------|------|
| `torch` | Training, checkpoints, AMP |
| `e3nn` | Equivariant layers, irreps, spherical harmonics |
| `torch_geometric` | `Data`, `Dataset`, `DataLoader` |
| `torch_scatter` | Graph aggregation in the model |
| `optuna` | HPO (`e3verde.hpo`) |
| `scikit-learn` | Splits, scalers, `KFold` |
| `ase` | Read XYZ structures from strings/files |
| `numpy`, `pandas` | Arrays and CSV I/O |
| `matplotlib`, `seaborn` | Plots |
| `scipy` | Metrics and distribution diagnostics |

### Install

From the project root (with PyTorch already installed):

```bash
pip install -r requirements.txt
```

`torch-scatter` and `torch-geometric` wheels must match your PyTorch and CUDA versions; follow the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the default pip install fails.

Run commands from the repository root so imports resolve (`e3verde` package and config paths).

---

## 3. Project structure

### Repository root

| Path | Purpose |
|------|---------|
| `README.md` | Quick start, install, usage |
| `DOCUMENTATION.md` | This file — full technical reference |
| `data_/README.md` | Paper checkpoints and per-target configs |
| `LICENSE` | MIT license |
| `CITATION.cff` | Citation metadata |
| `requirements.txt` | Python dependencies (PyTorch installed separately) |
| `Picture1.png` | Graphical abstract |
| `configs/` | Generic example JSON configs |
| `tutorial/` | Step-by-step tutorial; demo configs and sample XYZ structures |
| `data_/` | Paper archive: train, predict, hpo, learning_curve per target |
| `data/` | Runtime outputs (created on first run; most subdirs gitignored) |
| `verde+PCs.json` | Training database (not in repo; see paper / SI) |

### Package (`e3verde/`)

| Module | Type | Purpose |
|--------|------|---------|
| `run.py` | CLI | Train, cross-validate, or predict from a `RunConfig` JSON |
| `hpo.py` | CLI | Optuna TPE + MedianPruner search; optional retrain of best trial |
| `learning_curve.py` | CLI | Train on increasing data fractions; MAE/R² vs training size |
| `training.py` | Library | `train_model`, `evaluate_model`, `compute_metrics`, `cross_validate`, `predict_new_molecules`, `CSVWriter` |
| `config.py` | Library | `ModelConfig`, `TrainConfig`, `DataConfig`, `RunConfig`; `set_seed`, `setup_directories`, `setup_logging` |
| `data.py` | Library | `SimpleDataset`, `TargetNormalizer`, graph build, splits |
| `model.py` | Library | `PeriodicNetwork`, `MessagePassingBlock`, `EMA`, `LabelSmoothingLoss` |
| `plots.py` | Library | Training, evaluation, and CV figures |
| `__init__.py` | Library | Re-exports the public API for `from e3verde import …` |

**Note:** `run.py` is the CLI entry point; `training.py` holds the training loop and related logic. They are not duplicates.

### Paper archive (`data_/`)

Per-target folders (`e00S1`, `e00T1`, `oxS0`, `oxS1`, `oxT1`, `redS0`, `redS1`, `redT1`) under:

| Subfolder | Files |
|-----------|-------|
| `data_/train/<target>/` | `config_train.json` (saves to `data_/predict/<target>/verde_best.pt`) |
| `data_/predict/<target>/` | `config_predict.json`, `verde_best.pt` |
| `data_/predict/xyz/` | Benchmark XYZ structures for inference |
| `data_/hpo/<target>/` | `config_hpo.json`, `best_config.json` |
| `data_/learning_curve/<target>/` | `config_learning_curve.json` |

See [data_/README.md](data_/README.md) for target keys and example commands.

---

## 4. Configuration

Configs are JSON files loaded with `RunConfig.from_json(path)`. Two layouts are supported:

- **Nested (recommended):** top-level `mode`, `model`, `train`, `data`, plus optional run fields.
- **Flat:** model/train/data fields at the top level (backward-compatible).

Unknown keys in each section are ignored when loading.

### Top-level `RunConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `mode` | `"train"` | `"train"`, `"cross_validate"`, or `"predict"` |
| `model` | `ModelConfig()` | Architecture (see below) |
| `train` | `TrainConfig()` | Optimization (see below) |
| `data` | `DataConfig()` | Dataset and splits (see below) |
| `molecules_path` | `""` | Predict mode: file, directory, or JSON path |
| `molecules_format` | `"xyz"` | ASE format for `molecules_path` (`"json"` supported for JSON databases) |
| `pretrained_model_file` | `"data/models/model.pt"` | Checkpoint for predict; save path after training |
| `cv_folds` | `5` | K-fold count when `mode` is `"cross_validate"` |

### `model` — `ModelConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `em_dim` | `64` | Embedding dimension; `irreps_in` is synced to `{em_dim}x0e` if mismatched |
| `irreps_in` | `"64x0e"` | Input node irreps |
| `irreps_out` | `"1x0e"` | Output irreps; auto-adjusted in training for vector dimension |
| `irreps_node_attr` | `"1x0e"` | Node attribute irreps for convolutions |
| `layers` | `3` | Message-passing layers |
| `mul` | `32` | Multiplicity per irrep in hidden layers |
| `lmax` | `2` | Max spherical harmonic degree on edges |
| `number_of_basis` | `10` | Radial basis count |
| `radial_layers` | `1` | Layers in radial MLP |
| `radial_neurons` | `64` | Hidden width in radial MLP |
| `max_radius` | `5.0` | Cutoff (Å); also used as graph cutoff in `SimpleDataset` |
| `num_neighbors` | `-1` | e3nn neighbor normalization; `-1` = auto from training data |
| `reduce_output` | `true` | Graph-level pooling vs per-node output |
| `dropout` | `0.0` | Equivariant channel dropout probability |
| `use_layer_norm` | `false` | LayerNorm on l=0 scalar channels only |
| `use_residual` | `true` | Residual when input/output shapes match |
| `use_self_interaction` | `true` | Equivariant linear after each gate |
| `use_rich_features` | `true` | 7-D atomic property features vs mass only |
| `readout_type` | `"attention"` | `"mean"` or `"attention"` (scalar outputs) |
| `output_mlp_layers` | `2` | Scalar output MLP depth (`0` disables MLP stack) |
| `output_mlp_hidden` | `64` | Hidden width in output MLP |
| `use_multiscale_readout` | `true` | Concatenate scalar features from all layers (scalar outputs) |

### `train` — `TrainConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `num_epochs` | `100` | Maximum epochs |
| `batch_size` | `32` | Batch size |
| `learning_rate` | `1e-3` | AdamW learning rate |
| `weight_decay` | `1e-5` | Weight decay (bias/norm excluded) |
| `patience` | `30` | Early stopping on validation loss |
| `scheduler_type` | `"cosine_warmup"` | `"cosine_warmup"` or `"plateau"` |
| `warmup_epochs` | `-1` | Warmup length; `-1` → `max(1, num_epochs // 10)` |
| `clip_grad_norm` | `1.0` | Gradient clipping max norm |
| `label_smoothing` | `0.0` | Regression label smoothing (disabled for vectors) |
| `loss_function` | `"l1"` | `"l1"`, `"mse"`, or `"huber"` |
| `min_lr` | `1e-7` | Minimum LR (cosine / plateau) |
| `use_amp` | `false` | Mixed precision on CUDA |
| `gradient_accumulation_steps` | `1` | Accumulation before optimizer step |
| `use_ema` | `false` | Exponential moving average of weights |
| `ema_decay` | `0.999` | EMA decay if `use_ema` is true |
| `num_workers` | `0` | `DataLoader` workers |

### `data` — `DataConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `dataset_path` | `""` | Path to JSON array of molecules |
| `target_key` | `""` | JSON field name for the target property |
| `structure_field` | `""` | JSON field with XYZ text (ASE-readable) |
| `test_size` | `0.1` | Fraction held out for test |
| `val_size` | `0.1` | Fraction of remainder used for validation |
| `normalize_targets` | `true` | Fit `TargetNormalizer` on train targets |
| `normalize_features` | `true` | Normalize legacy mass feature (if not rich features) |
| `target_normalization` | `"auto"` | `"auto"`, `"standard"`, `"robust"`, or `"quantile"` |
| `seed` | `42` | Split and reproducibility seed |

Constraint: `test_size + val_size < 1.0`.

---

## 5. Usage

Set `data.dataset_path`, `data.target_key`, and `data.structure_field` before running.

**Generic configs** (`configs/`):

| File | Mode / use |
|------|------------|
| `configs/train.json` | Train on `verde+PCs.json` |
| `configs/cross_validate.json` | 5-fold CV |
| `configs/predict.json` | Inference from a checkpoint |
| `configs/hpo.json` | Optuna baseline (`e3verde.hpo`; model/train searched per trial) |
| `configs/learning_curve.json` | Learning curve (`e3verde.learning_curve`; fixed test set) |

**Paper configs** (`data_/` — one folder per target, e.g. `oxS0`):

```bash
python -m e3verde.run data_/train/oxS0/config_train.json
python -m e3verde.run data_/predict/oxS0/config_predict.json
python -m e3verde.hpo data_/hpo/oxS0/config_hpo.json --n_trials 50 --hpo_epochs 60
python -m e3verde.learning_curve data_/learning_curve/oxS0/config_learning_curve.json
```

### Train

```bash
python -m e3verde.run configs/train.json
```

Optional: `--mode train` (override JSON mode), `--debug` (verbose logging).

### Cross-validation

```bash
python -m e3verde.run configs/cross_validate.json
```

Set `"mode": "cross_validate"` and `cv_folds` in the JSON (or `python -m e3verde.run config.json --mode cross_validate`).

### Predict

Requires a checkpoint at `pretrained_model_file`.

```bash
python -m e3verde.run configs/predict.json
```

Predict configs need `molecules_path` and `molecules_format` (e.g. `"json"` with a database file, or `"xyz"` for a file/directory). Paper predict configs use `data_/predict/xyz/`.

### Hyperparameter optimization

```bash
python -m e3verde.hpo configs/hpo.json
python -m e3verde.hpo configs/hpo.json --n_trials 50 --hpo_epochs 60
python -m e3verde.hpo configs/hpo.json --resume --study_path data/optuna/study_YYYYMMDD_HHMMSS.pkl
python -m e3verde.hpo configs/hpo.json --no_retrain
```

Use `data_/hpo/<target>/best_config.json` or `data/optuna/best_config_*.json` for production training or learning curves after a search completes.

| Flag | Default | Description |
|------|---------|-------------|
| `--n_trials` | `50` | Optuna trials |
| `--hpo_epochs` | `60` | Epochs per trial |
| `--retrain_epochs` | `300` | Epochs for post-HPO retrain |
| `--resume` | off | Resume from `--study_path` or new study file |
| `--study_path` | auto timestamped | Pickle study path |
| `--no_retrain` | off | Skip `retrain_best` after search |
| `--seed` | `42` | Random seed |
| `--n_startup_trials` | `10` | Random trials before TPE |
| `--pruning_warmup` | `10` | Epochs before MedianPruner can prune |

### Learning curve

```bash
python -m e3verde.learning_curve configs/learning_curve.json
python -m e3verde.learning_curve configs/learning_curve.json --fractions 0.1 0.2 0.5 1.0 --repeats 3 --epochs 100 --patience 20
python -m e3verde.learning_curve data/optuna/best_config_YYYYMMDD_HHMMSS.json --epochs 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--fractions` | `0.05 … 1.0` | Fractions of train+val pool to subsample |
| `--repeats` | `3` | Repeats per fraction |
| `--epochs` | `100` | Max epochs per run |
| `--patience` | `20` | Early stopping patience |

---

## 6. Dataset format

### JSON layout

`dataset_path` must point to a **JSON array**. Each element is one molecule object.

### Required fields (via config)

| Config key | Meaning |
|------------|---------|
| `data.target_key` | Property to predict (must be present and parseable on each entry) |
| `data.structure_field` | XYZ string readable by ASE (`read(StringIO(...), format="xyz")`) |

Entries missing either field, with empty structures, or with unparseable targets are skipped. If none remain, `SimpleDataset` raises `ValueError`.

### Target values (`parse_target_value`)

Supported formats for `target_key`:

- Number: `1.43`
- List: `[0.1, 0.0, 0.2]` (vector target; dimension > 1 triggers equivariant scale-only normalization)
- Comma-separated string: `"5.17, -4.83, 0.0"`
- Single-value string: `"-2.33"`

### Structure field

XYZ text in standard format (line 1: atom count, line 2: comment, then `Symbol x y z` rows). Multi-line strings are stored as JSON string values.

### Optional metadata

Used when present: `inchi_key` (molecule ID in outputs), `smiles`, `formula`. Not required for training.

### Example entry

```json
{
  "inchi_key": "AAEBJCSCYHPBRF-FZSIALSZSA-N",
  "smiles": "O=C1C(C=NC(F)(F)F)=CC(=O)c2ncccc21",
  "oxidation_potential_S0 (eV)": 3.12,
  "xyz_S0": "23\n...\nO 0.409  -2.276  0.000\n..."
}
```

### Splits

`SimpleDataset` holds out `test_size`, then splits the remainder into train and val using `sklearn.model_selection.train_test_split` with `data.seed`. Normalization is fit on **train indices only**.

---

## 7. Outputs

`setup_directories()` (called by CLIs) creates these directories under `data/`:

| Directory | Created by | Contents |
|-----------|------------|----------|
| `data/logs/` | `setup_logging()` | `log_YYYYMMDD_HHMMSS.log` per run |
| `data/csv/` | `e3verde.run` (`CSVWriter`) | `{run_id}__predictions_{split}.csv`, `{run_id}__metrics_{split}.csv`, `{run_id}__training_history.csv`, `{run_id}__splits.csv`, `{run_id}__run_config.csv` |
| `data/figures/` | `plots.py` / training | Timestamped PNGs: `loss_history_*.png`, `parity_plot_*.png`, `target_distribution_*.png`, `error_histogram_*.png`, `residuals_*.png`, `cv_metrics_*.png`, `cv_learning_curves_*.png` |
| `data/models/` | `e3verde.run`, `e3verde.hpo` | PyTorch checkpoint at `pretrained_model_file` (default `data/models/model.pt`) |
| `data/cross_validation/` | `cross_validate()` | `cv_{timestamp}__foldNN_history.csv`, `cv_{timestamp}__foldNN_predictions.csv`, `cv_{timestamp}__all_predictions.csv`, `cv_{timestamp}__metrics_per_fold.csv`, `cv_{timestamp}__metrics_averaged.csv` |
| `data/predictions/` | `predict_new_molecules()` | `predictions_YYYYMMDD_HHMMSS.csv` with `molecule_id`, `xyz_file`, `formula`, `num_atoms`, and scalar or vector prediction columns |
| `data/learning_curve/` | `e3verde.learning_curve` | `learning_curve_{timestamp}.csv`, `learning_curve_summary_{timestamp}.csv`, `learning_curve_{timestamp}.png`, `learning_curve_publication_{timestamp}.png` |
| `data/optuna/` | `e3verde.hpo` | `study_{timestamp}.pkl`, `best_config_{timestamp}.json`, `results_{timestamp}.json`, diagnostic plots (`optimization_history_*.png`, `param_importance_*.png`, etc.) |

### Training checkpoint (`torch.save`)

Written to `pretrained_model_file` after training (and after HPO retrain):

- `model_state`, `model_config` (dict), `type_encoding`, `target_stats`, `normalizer_state` (scaler params, vector scale), `is_vector`, `history`, `val_metrics`, `test_metrics` (train mode)

Predict mode loads this file and restores the normalizer before inference.
