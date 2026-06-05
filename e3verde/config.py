#!/usr/bin/env python3
"""
Configuration: dataclasses, constants, seed, and logging setup.
"""

import os
import json
import logging
import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from e3nn.o3 import Irreps


ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078
}

# Rich atomic properties for node embeddings (electronegativity, IE, EA, etc.).
# Values normalized to ~[0, 1]. Sources: CRC Handbook, NIST.
# Feature order: mass/100, electronegativity/4, ionization_energy/25,
# electron_affinity/4, covalent_radius/2, polarizability/60, n_valence/8
ATOMIC_PROPERTIES = {
    #        mass    eneg   IE     EA     r_cov  polar  n_val
    'H':  [0.0101, 0.550, 0.544, 0.183, 0.155, 0.011, 0.125],
    'He': [0.0400, 0.000, 0.985, 0.000, 0.140, 0.003, 0.000],
    'Li': [0.0694, 0.245, 0.215, 0.152, 0.640, 0.400, 0.125],
    'Be': [0.0901, 0.393, 0.373, 0.000, 0.530, 0.094, 0.250],
    'B':  [0.1081, 0.510, 0.333, 0.068, 0.420, 0.050, 0.375],
    'C':  [0.1201, 0.638, 0.452, 0.309, 0.385, 0.029, 0.500],
    'N':  [0.1401, 0.760, 0.581, 0.000, 0.360, 0.019, 0.625],
    'O':  [0.1600, 0.863, 0.545, 0.365, 0.350, 0.013, 0.750],
    'F':  [0.1900, 0.995, 0.698, 0.820, 0.360, 0.009, 0.875],
    'Ne': [0.2018, 0.000, 0.864, 0.000, 0.290, 0.007, 1.000],
    'Na': [0.2299, 0.233, 0.206, 0.133, 0.830, 0.400, 0.125],
    'Mg': [0.2431, 0.328, 0.305, 0.000, 0.720, 0.178, 0.250],
    'Al': [0.2698, 0.403, 0.239, 0.108, 0.640, 0.110, 0.375],
    'Si': [0.2809, 0.450, 0.327, 0.332, 0.555, 0.091, 0.500],
    'P':  [0.3097, 0.528, 0.419, 0.182, 0.530, 0.062, 0.625],
    'S':  [0.3207, 0.645, 0.416, 0.503, 0.510, 0.048, 0.750],
    'Cl': [0.3545, 0.790, 0.521, 0.879, 0.510, 0.038, 0.875],
    'Ar': [0.3995, 0.000, 0.631, 0.000, 0.530, 0.028, 1.000],
    'K':  [0.3910, 0.205, 0.174, 0.122, 1.100, 0.717, 0.125],
    'Ca': [0.4008, 0.250, 0.244, 0.005, 0.990, 0.420, 0.250],
}
ATOMIC_PROP_DIM = 7

OUTPUT_DIRS = [
    "data/logs", "data/csv", "data/figures",
    "data/models", "data/cross_validation", "data/predictions",
    "data/learning_curve", "data/optuna"
]


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    em_dim: int = 64
    irreps_in: str = "64x0e"
    irreps_out: str = "1x0e"
    irreps_node_attr: str = "1x0e"
    layers: int = 3
    mul: int = 32
    lmax: int = 2
    number_of_basis: int = 10
    radial_layers: int = 1
    radial_neurons: int = 64
    max_radius: float = 5.0
    num_neighbors: float = -1  # -1 = auto-compute from data (recommended)
    reduce_output: bool = True
    dropout: float = 0.0
    use_layer_norm: bool = False
    use_residual: bool = True
    use_self_interaction: bool = True
    use_rich_features: bool = True
    readout_type: str = "attention"
    output_mlp_layers: int = 2
    output_mlp_hidden: int = 64
    use_multiscale_readout: bool = True

    def __post_init__(self):
        assert self.em_dim > 0, "em_dim must be positive"
        assert self.layers > 0, "layers must be positive"
        assert self.mul > 0, "mul must be positive"
        assert self.lmax >= 0, "lmax must be non-negative"
        assert self.max_radius > 0, "max_radius must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"
        assert self.readout_type in ("mean", "attention"), "readout_type must be 'mean' or 'attention'"
        assert self.output_mlp_layers >= 0, "output_mlp_layers must be >= 0"
        irreps_in_dim = Irreps(self.irreps_in).dim
        if irreps_in_dim != self.em_dim:
            self.irreps_in = f"{self.em_dim}x0e"


@dataclass
class TrainConfig:
    """Training configuration."""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 30
    scheduler_type: str = "cosine_warmup"
    warmup_epochs: int = -1  # -1 = auto (num_epochs // 10)
    clip_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    loss_function: str = "l1"  # "l1", "mse", "huber"
    min_lr: float = 1e-7
    use_amp: bool = False
    gradient_accumulation_steps: int = 1
    use_ema: bool = False
    ema_decay: float = 0.999
    num_workers: int = 0

    def __post_init__(self):
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.patience > 0, "patience must be positive"
        assert self.gradient_accumulation_steps >= 1
        assert self.loss_function in ("l1", "mse", "huber"), "loss_function must be 'l1', 'mse', or 'huber'"
        if self.warmup_epochs == -1:
            self.warmup_epochs = max(1, self.num_epochs // 10)


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_path: str = ""
    target_key: str = ""
    structure_field: str = ""
    test_size: float = 0.1
    val_size: float = 0.1
    normalize_targets: bool = True
    normalize_features: bool = True
    target_normalization: str = "auto"  # "auto", "standard", "robust", "quantile"
    seed: int = 42

    def __post_init__(self):
        assert 0.0 <= self.test_size < 1.0, "test_size must be in [0, 1)"
        assert 0.0 <= self.val_size < 1.0, "val_size must be in [0, 1)"
        assert self.test_size + self.val_size < 1.0, "test_size + val_size must be < 1.0"
        assert self.target_normalization in ("auto", "standard", "robust", "quantile")


@dataclass
class RunConfig:
    """Full run configuration."""
    mode: str = "train"  # "train", "cross_validate", "predict"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    molecules_path: str = ""
    molecules_format: str = "xyz"
    pretrained_model_file: str = "data/models/model.pt"
    cv_folds: int = 5

    @classmethod
    def from_json(cls, path: str) -> "RunConfig":
        """Load from JSON config file (backward-compatible with flat format)."""
        with open(path, "r") as f:
            raw = json.load(f)

        if "model" in raw and isinstance(raw["model"], dict):
            return cls(
                mode=raw.get("mode", "train"),
                model=ModelConfig(**{k: v for k, v in raw.get("model", {}).items()
                                     if k in ModelConfig.__dataclass_fields__}),
                train=TrainConfig(**{k: v for k, v in raw.get("train", {}).items()
                                     if k in TrainConfig.__dataclass_fields__}),
                data=DataConfig(**{k: v for k, v in raw.get("data", {}).items()
                                   if k in DataConfig.__dataclass_fields__}),
                molecules_path=raw.get("molecules_path", ""),
                molecules_format=raw.get("molecules_format", "xyz"),
                pretrained_model_file=raw.get("pretrained_model_file", "data/models/model.pt"),
                cv_folds=raw.get("cv_folds", 5),
            )
        else:
            # Flat format (backward-compatible with V1 config.json)
            model_fields = {k for k in ModelConfig.__dataclass_fields__}
            train_fields = {k for k in TrainConfig.__dataclass_fields__}
            data_fields = {k for k in DataConfig.__dataclass_fields__}

            model_kwargs = {k: raw[k] for k in model_fields if k in raw}
            train_kwargs = {k: raw[k] for k in train_fields if k in raw}
            data_kwargs = {k: raw[k] for k in data_fields if k in raw}

            return cls(
                mode=raw.get("mode", "train"),
                model=ModelConfig(**model_kwargs),
                train=TrainConfig(**train_kwargs),
                data=DataConfig(**data_kwargs),
                molecules_path=raw.get("molecules_path", ""),
                molecules_format=raw.get("molecules_format", "xyz"),
                pretrained_model_file=raw.get("pretrained_model_file", "data/models/model.pt"),
                cv_folds=raw.get("cv_folds", 5),
            )


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


def setup_directories():
    """Create output directories listed in OUTPUT_DIRS if they do not exist."""
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)


def setup_logging(debug=False):
    """Configure file + console logging under data/logs/."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("data", "logs", f"log_{timestamp}.log")
    level = logging.DEBUG if debug else logging.INFO
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logging.info(f"Logging initialized: {log_file}")
    return log_file
