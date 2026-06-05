#!/usr/bin/env python3
"""
E3NN Molecular Property Prediction - Refactored Architecture (V2)
=================================================================
Improvements over V1:
  - Dataclass-based configuration with validation
  - MessagePassingBlock as top-level class (pre-computed scalar mask)
  - Removed dead code in compute_metrics
  - Extracted prepare_batch helper to eliminate duplication
  - Consolidated CSV output (no redundant files)
  - Mixed precision training support (AMP)
  - Gradient accumulation support
  - EMA (Exponential Moving Average) for model weights
  - Reproducibility via global seed setting
  - Preserved E(3)-equivariance: LayerNorm only on l=0 scalars
"""

import os
import sys
import json
import logging
import datetime
import copy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from io import StringIO

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.signal import find_peaks

# PyTorch Geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean, scatter_add, scatter_max

# ASE
from ase.io import read
from ase import Atoms
from ase.neighborlist import neighbor_list

# e3nn (equivariant)
from e3nn.o3 import Irreps, spherical_harmonics, Linear as E3NNLinear
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists
from e3nn.math import soft_one_hot_linspace

# Sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import gaussian_kde

# Mixed precision
from torch.amp import autocast, GradScaler


# ============================================================================
# CONFIGURATION (dataclass-based, validated)
# ============================================================================
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
    num_neighbors: float = -1  # -1 = auto-compute from data (recommended). Controls 1/sqrt(N) normalization in Convolution.
    reduce_output: bool = True
    dropout: float = 0.0
    use_layer_norm: bool = False
    # === Improvements (all backward-compatible defaults) ===
    use_residual: bool = True           # #1: Residual connections between MP layers
    use_self_interaction: bool = True    # #2: Linear self-interaction after gate
    use_rich_features: bool = True       # #3: Rich atomic properties (eneg, IE, EA, etc.)
    readout_type: str = "attention"      # #4: "mean", "attention" (learned scalar attention)
    output_mlp_layers: int = 2           # #5: FC layers after aggregation (0 = linear only)
    output_mlp_hidden: int = 64         # #5: Hidden dim for output MLP
    use_multiscale_readout: bool = True  # #6: Concatenate readouts from all layers

    def __post_init__(self):
        assert self.em_dim > 0, "em_dim must be positive"
        assert self.layers > 0, "layers must be positive"
        assert self.mul > 0, "mul must be positive"
        assert self.lmax >= 0, "lmax must be non-negative"
        assert self.max_radius > 0, "max_radius must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"
        assert self.readout_type in ("mean", "attention"), "readout_type must be 'mean' or 'attention'"
        assert self.output_mlp_layers >= 0, "output_mlp_layers must be >= 0"
        # Validate irreps_in matches em_dim (embedding output feeds into first conv)
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
    min_lr: float = 1e-7  # Minimum LR for cosine scheduler (avoids lr=0)
    use_amp: bool = False  # Mixed precision
    gradient_accumulation_steps: int = 1
    use_ema: bool = False
    ema_decay: float = 0.999
    num_workers: int = 0  # DataLoader workers (0 = main process, >0 for parallel loading)

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
    val_size: float = 0.1  # Validation split for early stopping
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
    # Prediction mode
    molecules_path: str = ""
    molecules_format: str = "xyz"
    pretrained_model_file: str = "data/models/model.pt"
    # Cross-validation
    cv_folds: int = 5

    @classmethod
    def from_json(cls, path: str) -> "RunConfig":
        """Load from JSON config file (backward-compatible with flat format)."""
        with open(path, "r") as f:
            raw = json.load(f)

        # Support both nested and flat config formats
        if "model" in raw and isinstance(raw["model"], dict):
            # Nested format
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


# ============================================================================
# REPRODUCIBILITY
# ============================================================================
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


# ============================================================================
# SETUP
# ============================================================================
OUTPUT_DIRS = [
    "data/logs", "data/csv", "data/figures",
    "data/models", "data/cross_validation", "data/predictions"
]

def setup_directories():
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)

def setup_logging(debug=False):
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


# ============================================================================
# CONSTANTS
# ============================================================================
ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
    'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078
}

# Rich atomic properties for chemical embedding (Improvement #3)
# All values normalized to ~[0, 1] range by dividing by reasonable max values.
# Sources: CRC Handbook, NIST. For elements not listed, defaults are provided.
# Keys: mass/100, electronegativity/4, ionization_energy/25, electron_affinity/4,
#        covalent_radius/2, polarizability/60, n_valence/8
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
ATOMIC_PROP_DIM = 7  # Number of properties per atom


# ============================================================================
# UTILITIES
# ============================================================================
def check_nan_inf(tensor, name="tensor", raise_error=True) -> bool:
    """Check for NaN/Inf values in tensor."""
    if tensor is None:
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        nan_c = torch.isnan(tensor).sum().item() if has_nan else 0
        inf_c = torch.isinf(tensor).sum().item() if has_inf else 0
        msg = f"{name} contains NaN: {nan_c}, Inf: {inf_c}"
        logging.error(msg)
        if raise_error:
            raise ValueError(msg)
        return True
    return False


def build_graph_from_atoms(atoms, type_encoding, type_onehot, cutoff,
                           normalize_features=False, mass_mean=None, mass_std=None,
                           use_rich_features=False):
    """Build PyG Data object from ASE Atoms, preserving equivariance.
    
    Node features are invariant scalars (0e): one-hot encoding + atomic properties.
    These don't transform under rotation, so they don't affect equivariance.
    """
    symbols = atoms.get_chemical_symbols()

    # Node features: one-hot encoding
    onehot = [type_onehot[type_encoding.get(sym, 0)] for sym in symbols]
    x = torch.stack(onehot, dim=0)

    if use_rich_features:
        # Rich atomic properties: pre-normalized ~[0,1] (Improvement #3)
        # All scalar invariants, safe for equivariance
        default_props = [0.1, 0.5, 0.3, 0.1, 0.4, 0.03, 0.5]
        props = np.array([ATOMIC_PROPERTIES.get(sym, default_props) for sym in symbols])
        x = torch.cat([x, torch.tensor(props, dtype=torch.float32)], dim=1)
    else:
        # Legacy: mass only
        masses = np.array([ATOMIC_MASSES.get(sym, 1.0) for sym in symbols])
        if normalize_features and mass_mean is not None:
            masses = (masses - mass_mean) / (mass_std + 1e-8)
        else:
            masses = masses / 100.0
        x = torch.cat([x, torch.tensor(masses, dtype=torch.float32).unsqueeze(1)], dim=1)

    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    if len(atoms) > 1:
        i_idx, j_idx, _ = neighbor_list("ijS", atoms, cutoff)
        edge_index = torch.stack([torch.LongTensor(i_idx), torch.LongTensor(j_idx)], dim=0)
        edge_vec = pos[torch.LongTensor(j_idx)] - pos[torch.LongTensor(i_idx)]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_vec = torch.zeros((0, 3), dtype=torch.float32)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_vec=edge_vec)
    check_nan_inf(data.x, "node_features", raise_error=False)
    check_nan_inf(data.pos, "positions", raise_error=False)
    return data


def prepare_batch(batch, device, is_vector, vector_dim):
    """Prepare a batch: move to device and reshape targets. Eliminates duplication."""
    if not hasattr(batch, "batch"):
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
    batch = batch.to(device)
    if is_vector:
        target = batch.y.view(-1, vector_dim) if len(batch.y.shape) == 1 else batch.y
    else:
        target = batch.y.view(-1, 1)
    return batch, target


def parse_target_value(tv):
    """
    Parse a target value that may be stored as:
    - float/int: 2.33
    - list: [0.5, -1.2, 0.8]
    - numpy array
    - STRING of comma-separated values: "-8.35, 4.69, -0.0"
    - STRING of a single number: "-2.33"
    
    Returns a Python float or list of floats.
    """
    if tv is None:
        return None
    if isinstance(tv, (int, float, np.integer, np.floating)):
        return float(tv)
    if isinstance(tv, (list, tuple)):
        return [float(x) for x in tv]
    if isinstance(tv, np.ndarray):
        return tv.tolist()
    if isinstance(tv, str):
        tv = tv.strip()
        # Remove surrounding brackets if present: "[1.0, 2.0]" -> "1.0, 2.0"
        if tv.startswith('[') or tv.startswith('('):
            tv = tv.strip('[]() ')
        if ',' in tv:
            # Vector stored as comma-separated string
            parts = [x.strip() for x in tv.split(',')]
            return [float(x) for x in parts if x]
        else:
            # Single scalar as string
            return float(tv)
    # Fallback: try direct conversion
    return float(tv)


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================
class EMA:
    """
    Exponential Moving Average of model parameters.
    
    This does NOT affect equivariance: it only smooths scalar weight values,
    not the geometric structure of the tensor product operations.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """Replace model params with EMA params (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original params (after evaluation)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ============================================================================
# EQUIVARIANT MESSAGE PASSING BLOCK (top-level class)
# ============================================================================
class MessagePassingBlock(nn.Module):
    """
    Equivariant message passing block with residual connections and self-interaction.
    
    Architecture: Conv -> Gate -> [Self-Interaction] -> [LayerNorm] -> [Dropout] -> [+ Residual]
    
    EQUIVARIANCE NOTES:
    - Residual connection (#1): x_out + x_in is equivariant because addition of
      tensors in the same irrep space commutes with rotation: R(a+b) = Ra + Rb.
      Only applied when input/output irreps match.
    - Self-interaction (#2): E3NNLinear maps irreps -> irreps equivariantly.
      It mixes multiplicities within each (l,p) channel without involving neighbors,
      increasing expressivity per NequIP (Batzner et al., 2022).
    - LayerNorm: applied ONLY to l=0 (scalar) components.
    - Dropout: drops entire irrep channels together (equivariant).
    """

    def __init__(self, conv: Convolution, gate: Gate,
                 dropout_prob: float = 0.0, use_layer_norm: bool = False,
                 use_residual: bool = True, use_self_interaction: bool = True):
        super().__init__()
        self.conv = conv
        self.gate = gate
        self.dropout_prob = dropout_prob
        self.use_residual = use_residual

        # Self-interaction: equivariant linear mixing of channels (Improvement #2)
        if use_self_interaction:
            self.self_interaction = E3NNLinear(gate.irreps_out, gate.irreps_out)
        else:
            self.self_interaction = None

        # Pre-compute irrep block structure from gate output
        self._irrep_slices: List[Tuple[int, int, int]] = []
        self._scalar_indices: List[int] = []
        idx = 0
        for mul, ir in gate.irreps_out:
            if ir.l == 0:
                self._scalar_indices.extend(range(idx, idx + mul))
            for m in range(mul):
                self._irrep_slices.append((idx, idx + ir.dim, ir.dim))
                idx += ir.dim

        # LayerNorm on scalars only
        if use_layer_norm and len(self._scalar_indices) > 0:
            self.layer_norm = nn.LayerNorm(len(self._scalar_indices), elementwise_affine=False)
        else:
            self.layer_norm = None

    def _equivariant_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Equivariant dropout: drops entire irrep channels together."""
        if not self.training or self.dropout_prob == 0.0:
            return x
        device = x.device
        n_channels = len(self._irrep_slices)
        keep_prob = 1.0 - self.dropout_prob
        channel_mask = torch.bernoulli(
            torch.full((n_channels,), keep_prob, device=device)
        ) / keep_prob
        x_out = x.clone()
        for ch_idx, (start, end, dim) in enumerate(self._irrep_slices):
            x_out[..., start:end] = x[..., start:end] * channel_mask[ch_idx]
        return x_out

    def forward(self, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded):
        x_in = x  # Save for residual

        x = self.conv(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
        x = self.gate(x)

        # Self-interaction: mix channels within each node (Improvement #2)
        if self.self_interaction is not None:
            x = self.self_interaction(x)

        # LayerNorm only on l=0 scalars
        if self.layer_norm is not None and len(self._scalar_indices) > 0:
            x = x.clone()
            x[..., self._scalar_indices] = self.layer_norm(x[..., self._scalar_indices])

        x = self._equivariant_dropout(x)

        # Residual connection (Improvement #1)
        # Only when input/output dimensions match (same irreps)
        if self.use_residual and x_in.shape[-1] == x.shape[-1]:
            x = x + x_in

        return x


# ============================================================================
# EQUIVARIANT NETWORK
# ============================================================================
class PeriodicNetwork(nn.Module):
    """
    E(3)-equivariant Graph Neural Network for molecular property prediction.
    
    Architecture improvements over vanilla gate_points_2101:
    #1 Residual connections between message passing layers
    #2 Self-interaction linear after each gate
    #3 Rich atomic features (electronegativity, IE, EA, covalent radius, etc.)
    #4 Attention-based readout (learned scalar weights per atom)
    #5 Multi-layer output MLP after aggregation
    #6 Multi-scale readout (features from all layers concatenated)
    
    All improvements preserve E(3) equivariance:
    - Attention weights are computed from l=0 scalars only (invariant)
    - Output MLP acts on aggregated invariant scalars only
    - Multi-scale concatenation of features in same irrep space is equivariant
    """

    def __init__(self, in_dim: int, cfg: ModelConfig):
        super().__init__()
        self.em = E3NNLinear(Irreps(f"{in_dim}x0e"), Irreps(f"{cfg.em_dim}x0e"))
        self.em_dim = cfg.em_dim

        self.irreps_in = Irreps(cfg.irreps_in)
        self.irreps_node_attr = Irreps(cfg.irreps_node_attr)
        self.irreps_out = Irreps(cfg.irreps_out)
        self.irreps_hidden = Irreps([(cfg.mul, (l, p)) for l in range(cfg.lmax + 1) for p in [-1, 1]])
        self.irreps_edge_attr = Irreps.spherical_harmonics(cfg.lmax)

        self.max_radius = cfg.max_radius
        self.number_of_basis = cfg.number_of_basis
        self.reduce_output = cfg.reduce_output

        # Activation functions respecting parity
        act = {1: nn.functional.leaky_relu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        # Build equivariant message passing layers
        self.mp_layers = nn.ModuleList()
        self._layer_irreps_out = []  # Track each layer's actual output irreps
        irreps = self.irreps_in

        for layer_idx in range(cfg.layers):
            irreps_scalars = Irreps([
                (m, ir) for m, ir in self.irreps_hidden
                if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
            ])
            irreps_gated = Irreps([
                (m, ir) for m, ir in self.irreps_hidden
                if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
            ])
            gate_ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = Irreps([(m, gate_ir) for m, _ in irreps_gated])

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated
            )
            conv = Convolution(
                irreps, self.irreps_node_attr, self.irreps_edge_attr,
                gate.irreps_in, cfg.number_of_basis, cfg.radial_layers,
                cfg.radial_neurons, cfg.num_neighbors
            )

            block = MessagePassingBlock(
                conv, gate, cfg.dropout, cfg.use_layer_norm,
                use_residual=cfg.use_residual,
                use_self_interaction=cfg.use_self_interaction
            )
            self.mp_layers.append(block)
            self._layer_irreps_out.append(gate.irreps_out)

            if layer_idx > 0 and gate.irreps_out.dim != irreps.dim:
                logging.info(f"  Layer {layer_idx}: irreps changed {irreps} -> {gate.irreps_out} "
                             f"(residual will be skipped for this layer)")

            irreps = gate.irreps_out

        self.irreps_hidden_out = irreps  # irreps after LAST MP layer

        # Detect if output requires equivariant (l>0) features
        self._is_scalar_output = all(ir.l == 0 for _, ir in self.irreps_out)
        if not self._is_scalar_output:
            logging.info(f"Vector output detected ({self.irreps_out}): "
                         f"using equivariant final conv (no MLP, no multi-scale)")

        # Count scalar dim from LAST layer's hidden irreps (for single-scale path)
        self._hidden_scalar_dim = sum(mul for mul, ir in self.irreps_hidden_out if ir.l == 0)

        # --- Multi-scale readout (#6) - ONLY for scalar outputs ---
        # CRITICAL: Each layer may have different output irreps (layer 0 has fewer
        # irrep types because 0e-only input limits tensor product paths).
        # Each projection must use its own layer's actual irreps.
        if self._is_scalar_output and cfg.use_multiscale_readout and cfg.layers > 1:
            self.layer_readouts = nn.ModuleList()
            per_layer_scalar_dims = []
            for li in range(cfg.layers):
                layer_irreps = self._layer_irreps_out[li]
                # Project from this layer's specific irreps to scalars
                n_scalars = sum(mul for mul, ir in layer_irreps if ir.l == 0)
                if n_scalars == 0:
                    n_scalars = 1  # Fallback: at minimum extract 1 scalar
                self.layer_readouts.append(
                    E3NNLinear(layer_irreps, Irreps(f"{n_scalars}x0e"))
                )
                per_layer_scalar_dims.append(n_scalars)
            total_scalar_dim = sum(per_layer_scalar_dims)
            self.use_multiscale = True
            logging.info(f"Multi-scale readout: per-layer scalar dims = {per_layer_scalar_dims}, "
                         f"total = {total_scalar_dim}")
        else:
            self.layer_readouts = None
            self.use_multiscale = False
            total_scalar_dim = self._hidden_scalar_dim

        # --- Final convolution ---
        if self._is_scalar_output and not self.use_multiscale:
            # Scalar path: conv to scalars for MLP
            self.final_conv = Convolution(
                irreps, self.irreps_node_attr, self.irreps_edge_attr,
                Irreps(f"{self._hidden_scalar_dim}x0e"),
                cfg.number_of_basis, cfg.radial_layers,
                cfg.radial_neurons, cfg.num_neighbors
            )
        elif not self._is_scalar_output:
            # Vector path: equivariant conv directly to target irreps
            self.final_conv = Convolution(
                irreps, self.irreps_node_attr, self.irreps_edge_attr,
                self.irreps_out, cfg.number_of_basis, cfg.radial_layers,
                cfg.radial_neurons, cfg.num_neighbors
            )
        else:
            self.final_conv = None

        # --- Attention readout (#4) ---
        if cfg.readout_type == "attention" and self.reduce_output and self._is_scalar_output:
            self.attn_gate = nn.Sequential(
                nn.Linear(total_scalar_dim, max(total_scalar_dim // 2, 1)),
                nn.SiLU(),
                nn.Linear(max(total_scalar_dim // 2, 1), 1),
            )
            logging.info("Attention readout enabled (learned scalar weights per atom)")
        else:
            self.attn_gate = None

        # --- Output MLP head (#5) - ONLY for scalar outputs ---
        out_dim = self.irreps_out.dim
        if cfg.output_mlp_layers > 0 and self.reduce_output and self._is_scalar_output:
            mlp_layers = []
            in_d = total_scalar_dim
            for i in range(cfg.output_mlp_layers):
                out_d = cfg.output_mlp_hidden if i < cfg.output_mlp_layers - 1 else out_dim
                mlp_layers.append(nn.Linear(in_d, out_d))
                if i < cfg.output_mlp_layers - 1:
                    mlp_layers.append(nn.SiLU())
                in_d = out_d
            self.output_mlp = nn.Sequential(*mlp_layers)
            logging.info(f"Output MLP: {total_scalar_dim} -> ... -> {out_dim} "
                         f"({cfg.output_mlp_layers} layers)")
        elif self.reduce_output and self._is_scalar_output and total_scalar_dim != out_dim:
            # No MLP requested, but dimensions don't match: add a single linear projection
            self.output_mlp = nn.Linear(total_scalar_dim, out_dim)
            logging.info(f"Output projection: {total_scalar_dim} -> {out_dim} (linear, no MLP)")
        else:
            self.output_mlp = None

    def forward(self, data):
        batch = data.batch if hasattr(data, "batch") else data.pos.new_zeros(data.pos.shape[0], dtype=torch.long)
        edge_src, edge_dst = data.edge_index[0], data.edge_index[1]
        edge_vec = data.edge_vec

        # Spherical harmonics (equivariant edge attributes)
        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec,
                                      normalize=True, normalization="component")
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length, start=0.0, end=self.max_radius,
            number=self.number_of_basis, basis="smooth_finite", cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        # Node embedding
        x = self.em(data.x) if hasattr(data, "x") and data.x is not None else \
            data.pos.new_ones((data.pos.shape[0], self.em_dim))
        z = data.z if hasattr(data, "z") and data.z is not None else \
            data.pos.new_ones((data.pos.shape[0], self.irreps_node_attr.dim))

        # Message passing with optional multi-scale collection (#6)
        layer_outputs = []
        for i, layer in enumerate(self.mp_layers):
            x = layer(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
            if self.training:
                check_nan_inf(x, f"layer_{i}_output", raise_error=False)
            if self.use_multiscale and self.layer_readouts is not None:
                layer_outputs.append(x)

        # --- Node-level output (no aggregation) ---
        if not self.reduce_output:
            if self.final_conv is not None:
                x = self.final_conv(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
            check_nan_inf(x, "model_output", raise_error=True)
            return x

        # --- Graph-level output ---
        if not self._is_scalar_output:
            # VECTOR PATH: equivariant final conv -> weighted mean aggregation
            # The final conv maps to target irreps (e.g. 1x1o) equivariantly
            node_out = self.final_conv(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
            out = scatter_mean(node_out, batch, dim=0)
        else:
            # SCALAR PATH: multi-scale/single-scale -> attention -> MLP
            if self.use_multiscale and self.layer_readouts is not None and len(layer_outputs) > 0:
                scalar_parts = [proj(feats) for feats, proj in zip(layer_outputs, self.layer_readouts)]
                node_scalars = torch.cat(scalar_parts, dim=-1)
            else:
                node_scalars = self.final_conv(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

            # Aggregation: attention or mean
            if self.attn_gate is not None:
                # Attention pooling with numerically stable per-graph softmax
                attn_logits = self.attn_gate(node_scalars).squeeze(-1)  # [n_atoms]
                # Stability: subtract per-graph max
                attn_max, _ = scatter_max(attn_logits, batch, dim=0)
                attn_logits = attn_logits - attn_max[batch]
                attn_exp = torch.exp(attn_logits)  # [n_atoms]
                attn_sum = scatter_add(attn_exp, batch, dim=0)  # [n_graphs]
                attn_weights = (attn_exp / (attn_sum[batch] + 1e-10)).unsqueeze(-1)  # [n_atoms, 1]
                out = scatter_add(attn_weights * node_scalars, batch, dim=0)  # [n_graphs, dim]
            else:
                out = scatter_mean(node_scalars, batch, dim=0)

            # Output MLP (#5)
            if self.output_mlp is not None:
                out = self.output_mlp(out)

        check_nan_inf(out, "model_output", raise_error=True)
        return out


# ============================================================================
# TARGET NORMALIZER (separated responsibility)
# ============================================================================
class TargetNormalizer:
    """
    Handles target normalization with equivariance-aware vector handling.
    
    EQUIVARIANCE CRITICAL:
    For vector targets (e.g., dipole moments, forces), per-component normalization
    (different scale for x, y, z) BREAKS equivariance because it applies a
    non-uniform scaling that doesn't commute with rotations.
    
    Solution: For vectors, use a single scalar (RMS of all components) to scale
    all components equally, preserving direction and equivariance.
    
    For scalar targets: standard per-component normalization is fine.
    """

    def __init__(self, strategy: str = "auto"):
        self.strategy = strategy
        self.scaler = None
        self.stats: Dict[str, Any] = {}
        self._fitted = False
        self._is_vector = False
        # For vector normalization (equivariance-preserving)
        self._vector_mean = None   # shape (dim,) - subtracted from each component
        self._vector_scale = None  # single scalar - divides all components equally

    def fit(self, targets: np.ndarray) -> "TargetNormalizer":
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)

        self._is_vector = targets.shape[1] > 1
        actual_strategy = 'equivariant_uniform'  # Default for vectors

        if self._is_vector:
            # EQUIVARIANT normalization for vectors: SCALE ONLY, NO CENTERING.
            #
            # WHY NO CENTERING:
            # With 1x1o output, the model produces equivariant vectors: R*v under rotation.
            # If targets = (v - mean_vec) / scale, they transform as R*v - mean_vec (NOT equivariant).
            # The model's equivariant output CANNOT learn this non-equivariant mapping.
            #
            # Scale-only is safe: R*(v/s) = (R*v)/s, so equivariance is preserved.
            # We use the RMS of all components across all samples as a single scalar scale.
            self._vector_mean = None  # NO centering for equivariance
            rms = np.sqrt(np.mean(targets ** 2)) + 1e-10
            self._vector_scale = rms
            self.scaler = None  # Don't use sklearn scaler for vectors
            logging.info(f"TargetNormalizer (vector, equivariant): scale_only, "
                         f"rms_scale={self._vector_scale:.6f} (no centering)")
        else:
            # Scalar normalization: use sklearn scalers as before
            actual_strategy = self.strategy
            if actual_strategy == "auto":
                actual_strategy = self._detect_best_scaler(targets)

            if actual_strategy == "robust":
                self.scaler = RobustScaler()
            elif actual_strategy == "quantile":
                self.scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
            else:
                self.scaler = StandardScaler()

            self.scaler.fit(targets)
            logging.info(f"TargetNormalizer (scalar): strategy={actual_strategy}")

        self.stats = {
            'mean': np.mean(targets, axis=0),
            'std': np.std(targets, axis=0),
            'median': np.median(targets, axis=0),
            'is_vector': self._is_vector,
            'vector_dim': targets.shape[1],
            'normalization_type': 'equivariant_uniform' if self._is_vector else actual_strategy,
        }
        self._fitted = True
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return targets
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)

        if self._is_vector:
            # Equivariant: scale only, no centering (preserves R*v/s = R*(v/s))
            # Backward-compat: if _vector_mean exists from old checkpoint, apply it
            if self._vector_mean is not None:
                return (targets - self._vector_mean) / self._vector_scale
            return targets / self._vector_scale
        elif self.scaler is not None:
            return self.scaler.transform(targets)
        return targets

    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return predictions
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)

        if self._is_vector:
            # Equivariant inverse: scale only (no centering was applied)
            # Backward-compat: if _vector_mean exists from old checkpoint, apply it
            result = predictions * self._vector_scale
            if self._vector_mean is not None:
                result = result + self._vector_mean
            return result
        elif self.scaler is not None:
            return self.scaler.inverse_transform(predictions)
        return predictions

    def _detect_best_scaler(self, targets: np.ndarray) -> str:
        """Detect bimodality using KDE peak detection."""
        try:
            vals = targets.flatten()
            if len(vals) < 10:
                return "robust"
            kde = gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 100)
            density = kde(x_range)
            peaks, _ = find_peaks(density, height=np.max(density) * 0.1)
            if len(peaks) >= 2:
                logging.info("Auto-detection: bimodal distribution -> RobustScaler")
                return "robust"
            logging.info("Auto-detection: unimodal distribution -> StandardScaler")
            return "standard"
        except Exception as e:
            logging.warning(f"Auto-detection failed ({e}), defaulting to RobustScaler")
            return "robust"


# ============================================================================
# DISTRIBUTION DIAGNOSTICS
# ============================================================================
def diagnose_target_distribution(targets: np.ndarray) -> Dict[str, Any]:
    """
    Diagnose target distribution for bimodality.
    
    Bimodal distributions cause:
    - StandardScaler mean falls between peaks (poor centering)
    - Inflated SS_tot -> artificially low R²
    - Model may collapse to predicting intermediate values
    """
    if len(targets) == 0:
        return {}

    targets = targets.flatten() if len(targets.shape) > 1 and targets.shape[1] == 1 else targets
    if len(targets.shape) > 1:
        targets = np.linalg.norm(targets, axis=1)

    mean_val = np.mean(targets)
    median_val = np.median(targets)
    std_val = np.std(targets)

    num_peaks = 0
    is_bimodal = False
    mode_separation = None

    try:
        kde = gaussian_kde(targets)
        x_range = np.linspace(targets.min(), targets.max(), 200)
        density = kde(x_range)
        peaks, _ = find_peaks(density, height=np.max(density) * 0.1)
        num_peaks = len(peaks)
        is_bimodal = num_peaks >= 2
        if num_peaks >= 2 and std_val > 0:
            mode_separation = float(np.abs(x_range[peaks[1]] - x_range[peaks[0]]) / std_val)
    except Exception as e:
        logging.warning(f"Bimodality test error: {e}")

    diagnosis = {
        'n_samples': len(targets), 'mean': float(mean_val), 'median': float(median_val),
        'std': float(std_val), 'skewness': float(stats.skew(targets)),
        'kurtosis': float(stats.kurtosis(targets)), 'num_peaks': num_peaks,
        'is_bimodal': is_bimodal, 'mode_separation': mode_separation,
    }

    logging.info(f"Distribution diagnosis: N={diagnosis['n_samples']}, "
                 f"mean={diagnosis['mean']:.4f}, median={diagnosis['median']:.4f}, "
                 f"std={diagnosis['std']:.4f}, peaks={num_peaks}, bimodal={is_bimodal}")
    if is_bimodal:
        logging.warning("BIMODAL DISTRIBUTION DETECTED - R² may be artificially low. "
                        "Use Spearman correlation as a more robust metric.")

    return diagnosis


# ============================================================================
# DATASET
# ============================================================================
class SimpleDataset(Dataset):
    """Molecular dataset with train/val/test three-way split.
    
    Split strategy:
    - train: used for gradient updates
    - val: used for early stopping and hyperparameter selection
    - test: held-out, only used for final unbiased evaluation
    
    Normalization is fitted ONLY on train data.
    """

    def __init__(self, data_cfg: DataConfig):
        super().__init__()

        with open(data_cfg.dataset_path, "r") as f:
            raw_db = json.load(f)

        self.target_key = data_cfg.target_key
        self.structure_field = data_cfg.structure_field
        self.cutoff = 5.0  # Will be overridden by model max_radius
        self.normalize_features = data_cfg.normalize_features
        self.normalize_targets = data_cfg.normalize_targets
        self.target_normalization = data_cfg.target_normalization
        self._seed = data_cfg.seed
        self.use_rich_features = False  # Set by set_model_config() after init

        # Filter valid entries
        self.database = self._filter_valid_entries(raw_db)
        if len(self.database) == 0:
            raise ValueError(f"No valid entries with target '{data_cfg.target_key}' "
                             f"and structure '{data_cfg.structure_field}'")
        logging.info(f"Dataset: {len(self.database)} valid entries")

        # Atom type encoding
        self.type_encoding, self.type_onehot = self._build_type_encoding()

        # Three-way split: train / val / test
        self.train_idx, self.val_idx, self.test_idx = self._make_splits(
            data_cfg.test_size, data_cfg.val_size, data_cfg.seed)

        # Normalization (fitted on train only)
        self.normalizer = TargetNormalizer(data_cfg.target_normalization)
        self.mass_mean = None
        self.mass_std = None
        self._fit_normalization()

    def _make_splits(self, test_size, val_size, seed):
        """Create train/val/test splits.
        
        Strategy: first hold out test, then split remainder into train+val.
        This ensures test is never seen during training or early stopping.
        """
        indices = list(range(len(self.database)))

        if test_size > 0 and val_size > 0:
            # First split: hold out test
            trainval_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=seed)
            # Second split: split trainval into train + val
            # val_size is relative to total, adjust for remaining data
            val_fraction_of_trainval = val_size / (1.0 - test_size)
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=val_fraction_of_trainval, random_state=seed)
        elif test_size > 0:
            # No val: val = subset of train (backward-compatible behavior)
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=seed)
            val_idx = []
        elif val_size > 0:
            train_idx, val_idx = train_test_split(
                indices, test_size=val_size, random_state=seed)
            test_idx = []
        else:
            train_idx, val_idx, test_idx = indices, [], []

        logging.info(f"Splits: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        return train_idx, val_idx, test_idx

    @property
    def target_stats(self):
        return self.normalizer.stats

    def set_model_config(self, model_cfg: "ModelConfig"):
        """Apply model-dependent settings to dataset."""
        self.cutoff = model_cfg.max_radius
        self.use_rich_features = model_cfg.use_rich_features

    def compute_avg_num_neighbors(self, max_samples: int = 200) -> float:
        """Compute average number of neighbors from training data.
        
        Used to set num_neighbors for e3nn Convolution normalization.
        Convolution divides by sqrt(num_neighbors), so wrong values cause
        feature magnitudes to be off, hurting training stability.
        """
        sample_indices = self.train_idx[:max_samples] if len(self.train_idx) > max_samples else self.train_idx
        total_edges = 0
        total_nodes = 0
        for idx in sample_indices:
            try:
                atoms = read(StringIO(self.database[idx][self.structure_field]), format="xyz")
                if len(atoms) > 1:
                    i_idx, _, _ = neighbor_list("ijS", atoms, self.cutoff)
                    total_edges += len(i_idx)
                    total_nodes += len(atoms)
            except:
                continue
        avg = total_edges / max(total_nodes, 1)
        logging.info(f"Computed avg_num_neighbors={avg:.1f} from {len(sample_indices)} molecules "
                     f"(cutoff={self.cutoff})")
        return avg

    @property
    def feature_dim(self):
        """Input feature dimension per atom (for model construction)."""
        extra = ATOMIC_PROP_DIM if self.use_rich_features else 1
        return len(self.type_encoding) + extra

    def _filter_valid_entries(self, raw_db):
        valid = []
        for entry in raw_db:
            tv = entry.get(self.target_key)
            if tv is None:
                continue
            # Parse target: handle string vectors like "-8.35, 4.69, -0.0"
            try:
                parsed = parse_target_value(tv)
                if parsed is None:
                    continue
                entry[self.target_key] = parsed  # Store parsed value back
            except (ValueError, TypeError):
                logging.warning(f"Cannot parse target '{tv}' for key '{self.target_key}', skipping")
                continue
            sd = entry.get(self.structure_field)
            if sd is None or not sd or (isinstance(sd, str) and not sd.strip()):
                continue
            try:
                atoms = read(StringIO(sd), format="xyz")
                if len(atoms) == 0:
                    continue
                valid.append(entry)
            except (StopIteration, ValueError, KeyError, IndexError, TypeError):
                continue
        return valid

    def _build_type_encoding(self):
        all_symbols = set(ATOMIC_MASSES.keys())
        for entry in self.database:
            try:
                atoms = read(StringIO(entry[self.structure_field]), format="xyz")
                all_symbols.update(atoms.get_chemical_symbols())
            except:
                continue
        encoding = {sym: i for i, sym in enumerate(sorted(all_symbols))}
        onehot = torch.eye(len(encoding), dtype=torch.float32)
        return encoding, onehot

    def _fit_normalization(self):
        """Fit normalizer on training targets and compute mass statistics."""
        if len(self.train_idx) == 0:
            return

        # Target normalization (can be disabled via normalize_targets=False)
        if self.normalize_targets:
            train_targets = []
            for idx in self.train_idx:
                tv = self.database[idx][self.target_key]
                train_targets.append(tv if isinstance(tv, (list, tuple, np.ndarray)) else [tv])
            train_targets = np.array(train_targets)
            self.normalizer.fit(train_targets)
        else:
            # Still compute stats for logging, but don't fit scaler
            train_targets = []
            for idx in self.train_idx:
                tv = self.database[idx][self.target_key]
                train_targets.append(tv if isinstance(tv, (list, tuple, np.ndarray)) else [tv])
            train_targets = np.array(train_targets)
            self.normalizer.stats = {
                'mean': np.mean(train_targets, axis=0),
                'std': np.std(train_targets, axis=0),
                'median': np.median(train_targets, axis=0),
                'is_vector': train_targets.shape[1] > 1,
                'vector_dim': train_targets.shape[1],
                'normalization_type': 'none',
            }
            logging.info("Target normalization DISABLED (normalize_targets=False)")

        # Feature normalization (mass)
        if self.normalize_features:
            all_masses = []
            for idx in self.train_idx:
                try:
                    atoms = read(StringIO(self.database[idx][self.structure_field]), format="xyz")
                    for sym in atoms.get_chemical_symbols():
                        if sym in ATOMIC_MASSES:
                            all_masses.append(ATOMIC_MASSES[sym])
                except:
                    continue
            if all_masses:
                self.mass_mean = np.mean(all_masses)
                self.mass_std = np.std(all_masses)

    def len(self):
        return len(self.database)

    def get(self, idx):
        entry = self.database[idx]
        atoms = read(StringIO(entry[self.structure_field]), format="xyz")

        data = build_graph_from_atoms(
            atoms, self.type_encoding, self.type_onehot, self.cutoff,
            self.normalize_features, self.mass_mean, self.mass_std,
            use_rich_features=self.use_rich_features
        )

        # Target
        tv = entry[self.target_key]
        target_array = np.array(tv if isinstance(tv, (list, tuple, np.ndarray)) else [tv]).reshape(1, -1)
        data.y_original = torch.tensor(target_array[0], dtype=torch.float32)

        if self.normalizer._fitted:
            data.y = torch.tensor(self.normalizer.transform(target_array)[0], dtype=torch.float32)
        else:
            data.y = data.y_original.clone()

        data.inchi_key = entry.get("inchi_key", f"molecule_{idx}")
        return data

    @property
    def train_dataset(self):
        return [self.get(i) for i in self.train_idx]

    @property
    def val_dataset(self):
        return [self.get(i) for i in self.val_idx]

    @property
    def test_dataset(self):
        return [self.get(i) for i in self.test_idx]


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class LabelSmoothingLoss(nn.Module):
    """Label smoothing for regression (smooths target towards mean)."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.base = nn.L1Loss()

    def forward(self, pred, target):
        target_smooth = (1 - self.smoothing) * target + self.smoothing * target.mean()
        return self.base(pred, target_smooth)


# ============================================================================
# TRAINING
# ============================================================================
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
    """
    is_vector = dataset.target_stats.get('is_vector', False)
    vector_dim = dataset.target_stats.get('vector_dim', 1)

    # Auto-adjust output irreps
    adjusted_irreps = auto_adjust_irreps_out(model_cfg, is_vector, vector_dim)
    model_cfg_adjusted = copy.copy(model_cfg)
    model_cfg_adjusted.irreps_out = adjusted_irreps

    # Auto-compute num_neighbors if set to -1
    if model_cfg_adjusted.num_neighbors <= 0:
        model_cfg_adjusted.num_neighbors = max(1.0, dataset.compute_avg_num_neighbors())

    # Build model
    in_dim = dataset.feature_dim
    model = PeriodicNetwork(in_dim, model_cfg_adjusted).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model: {n_params:,} parameters, output irreps={model.irreps_out}")

    # Optimizer - separate param groups: no weight decay on bias and LayerNorm
    # This is ML best practice (Loshchilov & Hutter, 2019) and used in NequIP.
    # Weight decay on bias/norm params acts as an unintended regularizer 
    # that can hurt performance, especially for equivariant networks where
    # norm params control the scale of geometric features.
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

    # Scheduler
    if train_cfg.scheduler_type == "cosine_warmup":
        warmup = train_cfg.warmup_epochs
        total = train_cfg.num_epochs
        base_lr = train_cfg.learning_rate
        min_lr_ratio = train_cfg.min_lr / base_lr  # Floor as fraction of base LR
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, total - warmup)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return max(cosine, min_lr_ratio)  # Never go below min_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=train_cfg.patience // 3, factor=0.5,
            min_lr=train_cfg.min_lr)

    # Loss function (configurable)
    # EQUIVARIANCE NOTE: Label smoothing smooths targets toward batch mean.
    # For vectors, batch mean direction is arbitrary -> disable for vectors.
    if train_cfg.label_smoothing > 0 and not is_vector:
        criterion = LabelSmoothingLoss(train_cfg.label_smoothing)
    else:
        if train_cfg.loss_function == "mse":
            criterion = nn.MSELoss()
        elif train_cfg.loss_function == "huber":
            criterion = nn.HuberLoss()
        else:  # "l1" default
            criterion = nn.L1Loss()
        if train_cfg.label_smoothing > 0 and is_vector:
            logging.warning("Label smoothing disabled for vector targets "
                            "(batch mean direction is arbitrary, would harm equivariance)")
    logging.info(f"Loss function: {criterion.__class__.__name__}")

    # AMP scaler
    amp_enabled = train_cfg.use_amp and device.type == "cuda"
    grad_scaler = GradScaler(enabled=amp_enabled) if amp_enabled else None
    if amp_enabled:
        logging.info("Mixed precision training enabled (AMP)")
        logging.info("  Note: e3nn tensor products use fp32 CG coefficients internally. "
                     "AMP may slightly reduce numerical precision of equivariant operations.")

    # EMA
    ema = EMA(model, train_cfg.ema_decay) if train_cfg.use_ema else None
    if ema:
        logging.info(f"EMA enabled with decay={train_cfg.ema_decay}")

    # Data loaders
    train_loader = DataLoader(dataset.train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=train_cfg.num_workers,
                              pin_memory=(device.type == "cuda"))
    # Validation: use val set for early stopping. Fall back to test if no val set.
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
        # --- Training ---
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
                        ema.update(model)  # EMA only after actual weight update
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
                        ema.update(model)  # EMA only after actual weight update

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
                ema.update(model)  # EMA after final weight update of epoch

        train_loss /= max(train_samples, 1)

        # --- Evaluation ---
        # Use EMA weights for evaluation if available
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

        # Scheduler step
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

        # Epoch callback (used by Optuna for pruning)
        if epoch_callback is not None:
            try:
                epoch_callback(epoch, train_loss, val_loss)
            except Exception:
                # CRITICAL: Re-raise so Optuna receives TrialPruned.
                # If we 'break' instead, train_model returns normally,
                # Optuna never sees the pruning, and MedianPruner is disabled.
                logging.info(f"Training stopped by callback at epoch {epoch+1}")
                raise

        # Early stopping (on validation loss)
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

    # Load best model
    if best_state is not None:
        if ema:
            # best_state contains EMA shadow params (keyed by param name)
            for name, param in model.named_parameters():
                if name in best_state:
                    param.data.copy_(best_state[name])
        else:
            model.load_state_dict(best_state)
        logging.info(f"Loaded best model from epoch {best_epoch}")

    return model, history, is_vector, model_cfg_adjusted  # 4 values: callers must unpack all 4


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model, dataset, device, is_vector, split="test"):
    """
    Evaluate model on a specific split.
    
    Args:
        split: "test", "val", or "train"
    """
    model.eval()
    vector_dim = dataset.target_stats.get('vector_dim', 1)

    if split == "val":
        data_list = dataset.val_dataset
        idx_list = dataset.val_idx
    elif split == "train":
        data_list = dataset.train_dataset
        idx_list = dataset.train_idx
    else:  # "test"
        data_list = dataset.test_dataset
        idx_list = dataset.test_idx

    if len(data_list) == 0:
        logging.warning(f"No data in '{split}' split for evaluation")
        return []

    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    results = []  # Unified result list

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch, _ = prepare_batch(batch, device, is_vector, vector_dim)

            inchi_key = getattr(batch, "inchi_key", "")
            pred = model(batch)

            # Denormalize
            pred_np = pred.cpu().numpy()
            truth_np = batch.y_original.cpu().numpy() if hasattr(batch, 'y_original') else batch.y.cpu().numpy()
            pred_denorm = dataset.normalizer.inverse_transform(pred_np)

            pred_val = pred_denorm.squeeze()
            truth_val = truth_np.squeeze()

            # Molecule metadata
            num_atoms = 0
            atom_types = []
            try:
                if idx < len(idx_list):
                    mol_idx = idx_list[idx]
                    entry = dataset.database[mol_idx]
                    atoms = read(StringIO(entry[dataset.structure_field]), format="xyz")
                    num_atoms = len(atoms)
                    atom_types = atoms.get_chemical_symbols()
            except:
                pass

            results.append({
                "inchi_key": inchi_key,
                "pred": pred_val.tolist() if isinstance(pred_val, np.ndarray) else pred_val,
                "truth": truth_val.tolist() if isinstance(truth_val, np.ndarray) else truth_val,
                "num_atoms": num_atoms,
                "atom_types": atom_types,
            })

    return results


# ============================================================================
# METRICS (dead code removed, consolidated)
# ============================================================================
def compute_metrics(eval_results: List[Dict], is_vector: bool) -> Dict[str, Any]:
    """Compute evaluation metrics. Clean implementation without dead code."""
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

    # R² using median (more robust to bimodality)
    ss_tot_median = np.sum((truths - np.median(truths)) ** 2)
    r2_median = 1 - (ss_res / (ss_tot_median + 1e-10)) if ss_tot_median > 1e-10 else 0.0

    # Correlations
    pearson = stats.pearsonr(preds, truths)[0] if len(preds) > 1 else 0.0
    spearman = stats.spearmanr(preds, truths)[0] if len(preds) > 1 else 0.0
    try:
        kendall = stats.kendalltau(preds, truths)[0] if len(preds) > 1 else 0.0
    except:
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

    # Per-size analysis
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

    # Magnitude metrics
    errors_mag = np.linalg.norm(preds - truths, axis=1)
    mae = float(np.mean(errors_mag))
    rmse = float(np.sqrt(np.mean(errors_mag ** 2)))

    # R² per component (averaged)
    r2_components = []
    for i in range(vector_dim):
        ss_res = np.sum((preds[:, i] - truths[:, i]) ** 2)
        ss_tot = np.sum((truths[:, i] - truths[:, i].mean()) ** 2)
        r2_c = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0
        r2_components.append(float(r2_c))
    r2 = float(np.mean(r2_components))

    # Magnitude correlations
    pred_mags = np.linalg.norm(preds, axis=1)
    truth_mags = np.linalg.norm(truths, axis=1)
    try:
        spearman = float(stats.spearmanr(pred_mags, truth_mags)[0]) if len(pred_mags) > 1 else 0.0
        pearson = float(stats.pearsonr(pred_mags, truth_mags)[0]) if len(pred_mags) > 1 else 0.0
    except:
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

    # Angular error for 3D vectors
    if vector_dim == 3:
        try:
            dot = np.sum(preds * truths, axis=1)
            pn = np.linalg.norm(preds, axis=1)
            tn = np.linalg.norm(truths, axis=1)
            valid = (pn > 1e-10) & (tn > 1e-10)
            if valid.sum() > 0:
                cosines = np.clip(dot[valid] / (pn[valid] * tn[valid]), -1.0, 1.0)
                metrics["Angular_Error_deg"] = float(np.mean(np.arccos(cosines) * 180 / np.pi))
        except:
            pass

    return metrics


# ============================================================================
# PLOTTING (consolidated)
# ============================================================================
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

    # Top: loss curves (log scale if range > 10x)
    ax1.plot(epochs, train_loss, 'o-', color=C1, lw=2, ms=4, label="Train")
    ax1.plot(epochs, val_loss, 's-', color=C2, lw=2, ms=4, label="Validation")

    # Best epoch marker
    if best_epoch > 0 and best_epoch <= len(val_loss):
        best_val = val_loss[best_epoch - 1]
        ax1.axvline(best_epoch, color='green', ls=':', lw=1.5, alpha=0.7)
        ax1.scatter([best_epoch], [best_val], color='green', s=100, zorder=5,
                    marker='*', label=f"Best epoch {best_epoch}")

    # Auto log scale if loss range > 10x (and all values positive)
    min_loss = min(min(train_loss), min(val_loss))
    max_loss = max(max(train_loss), max(val_loss))
    if min_loss > 0 and max_loss / min_loss > 10:
        ax1.set_yscale('log')

    ax1.set_ylabel("Loss", fontsize=14)
    ax1.set_title(f"Learning Curve{title_suffix}", fontsize=16, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Bottom: LR schedule
    ax2.plot(epochs, lrs, '-', color='gray', lw=2)
    ax2.set_xlabel("Epoch", fontsize=14)
    ax2.set_ylabel("LR", fontsize=12)
    if all(lr > 0 for lr in lrs):
        ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return _save_fig(fig, f"loss_history{title_suffix.replace(' ', '_')}")


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

        # Magnitude
        pm, tm = np.linalg.norm(preds, axis=1), np.linalg.norm(truths, axis=1)
        _parity_subplot(axes[0], tm, pm, "Magnitude", C1)

        # Components
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

    # Histogram + KDE
    ax = axes[0]
    ax.hist(targets, bins=50, color=C1, alpha=0.7, edgecolor="black", density=True)
    try:
        kde = gaussian_kde(targets)
        x_r = np.linspace(targets.min(), targets.max(), 200)
        ax.plot(x_r, kde(x_r), 'r-', lw=2, label="KDE")
    except:
        pass
    ax.axvline(np.mean(targets), color='r', ls='--', lw=2, label=f'Mean: {np.mean(targets):.2f}')
    ax.axvline(np.median(targets), color='g', ls='--', lw=2, label=f'Median: {np.median(targets):.2f}')
    ax.set_xlabel("Target Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Target Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1]
    try:
        stats.probplot(targets, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
    except:
        ax.text(0.5, 0.5, "Q-Q Error", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    return _save_fig(fig, "target_distribution")


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

    # MAE
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

    # R²
    ax = axes[1]
    ax.bar(folds, r2_vals, color=C2, alpha=0.7, edgecolor="black")
    avg_r2 = avg_metrics.get("R2", 0)
    std_r2 = avg_metrics.get("R2_std", 0)
    ax.axhline(avg_r2, color=C_ERR, ls='--', lw=2,
               label=f"Mean: {avg_r2:.4f} ± {std_r2:.4f}")
    ax.set_xlabel("Fold"); ax.set_ylabel("R²")
    ax.set_title("R² Across Folds", fontweight="bold"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Spearman (robust to bimodality)
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
        # Mark best epoch
        best_ep = history[-1].get("best_epoch", 0)
        if best_ep > 0 and best_ep <= len(val_loss):
            ax.scatter([best_ep], [val_loss[best_ep-1]], color=color, s=60, zorder=5, marker='*')

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation Loss", fontsize=14)
    ax.set_title(f"CV Learning Curves ({k} folds)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Auto log scale (only if all values positive and range > 10x)
    all_vals = [h["val_loss"] for hist in all_histories for h in hist if hist]
    if all_vals and min(all_vals) > 0 and max(all_vals) / min(all_vals) > 10:
        ax.set_yscale('log')

    plt.tight_layout()
    return _save_fig(fig, "cv_learning_curves")


# ============================================================================
# CSV OUTPUT (consolidated, no redundancy)
# ============================================================================
class CSVWriter:
    """
    Centralized CSV output. Eliminates redundant files from V1.
    
    V2 produces exactly:
      - {run_id}__predictions_{split}.csv   (predictions per split with errors)
      - {run_id}__metrics_{split}.csv       (metrics per split)
      - {run_id}__training_history.csv      (epoch-level loss/lr)
      - {run_id}__splits.csv                (train/val/test molecule assignments)
      - {run_id}__run_config.csv            (full configuration for reproducibility)
    """

    def __init__(self, run_id: str, output_dir: str = "data/csv"):
        self.run_id = run_id
        self.output_dir = output_dir

    def _path(self, suffix):
        return os.path.join(self.output_dir, f"{self.run_id}__{suffix}.csv")

    def save_predictions(self, eval_results, is_vector, split="test"):
        """Predictions CSV for a given split."""
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
        """Metrics CSV for a given split."""
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
        """Single splits CSV with train/val/test column."""
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

    def save_config(self, run_config: RunConfig):
        """Save full config for reproducibility."""
        flat = {
            "mode": run_config.mode,
            **{f"model.{k}": v for k, v in asdict(run_config.model).items()},
            **{f"train.{k}": v for k, v in asdict(run_config.train).items()},
            **{f"data.{k}": v for k, v in asdict(run_config.data).items()},
        }
        path = self._path("run_config")
        pd.DataFrame([flat]).to_csv(path, index=False)
        logging.info(f"Config saved: {path}")


# ============================================================================
# CROSS-VALIDATION
# ============================================================================
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
    all_histories = []  # Store per-fold training histories

    # Inner val fraction (10% of fold's training data)
    inner_val_fraction = 0.1

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_dir = "data/cross_validation"

    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(indices), 1):
        logging.info(f"--- Fold {fold}/{k} ---")

        # Inner split: train_val -> train + val (for early stopping)
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

        # Save per-fold training history
        fold_history_df = pd.DataFrame(history)
        fold_history_df["fold"] = fold
        fold_history_df.to_csv(
            os.path.join(cv_dir, f"cv_{timestamp}__fold{fold:02d}_history.csv"), index=False)

        # Plot per-fold learning curve
        plot_loss_history(history, title_suffix=f" (Fold {fold})")

        # Evaluate on the fold's test set (held-out from this fold)
        eval_results = evaluate_model(model, dataset, device, is_vector, split="test")
        metrics = compute_metrics(eval_results, is_vector)
        metrics["Fold"] = fold
        metrics["n_train"] = len(inner_train_idx)
        metrics["n_val"] = len(inner_val_idx)
        metrics["n_test"] = len(test_idx)
        metrics["best_epoch"] = history[-1].get("best_epoch", len(history)) if history else 0
        metrics["total_epochs"] = len(history)
        all_metrics.append(metrics)

        # Save fold predictions
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

    # --- Combined outputs ---

    # Save all predictions combined
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

    # Average metrics (exclude metadata fields that shouldn't be averaged)
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

    # Save metrics CSVs
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(cv_dir, f"cv_{timestamp}__metrics_per_fold.csv"), index=False)
    pd.DataFrame([avg_metrics]).to_csv(
        os.path.join(cv_dir, f"cv_{timestamp}__metrics_averaged.csv"), index=False)

    # Plots
    plot_cv_metrics(all_metrics, avg_metrics)
    plot_cv_learning_curves(all_histories, k)

    # Parity plot from combined predictions
    if all_results:
        plot_parity(all_results, is_vector, target_name=f"{dataset.target_key} (CV)")

    logging.info("=== Cross-Validation Summary ===")
    for key in sorted(avg_metrics):
        if not key.endswith("_std") and key not in ("n_folds",):
            std = avg_metrics.get(f"{key}_std", 0)
            if isinstance(avg_metrics[key], float):
                logging.info(f"  {key}: {avg_metrics[key]:.4f} +/- {std:.4f}")

    return all_metrics, avg_metrics


# ============================================================================
# PREDICTIONS
# ============================================================================
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
                except:
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
                except:
                    pass
    else:
        raise ValueError(f"Path not found: {path}")

    return atoms_list, file_names


# ============================================================================
# MAIN
# ============================================================================
def main():
    setup_directories()
    setup_logging(debug=False)

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    if not os.path.exists(config_path):
        logging.error(f"Config not found: {config_path}")
        return

    cfg = RunConfig.from_json(config_path)
    set_seed(cfg.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Validate required fields
    if not cfg.data.dataset_path or not cfg.data.target_key or not cfg.data.structure_field:
        logging.error("Missing required data config: dataset_path, target_key, structure_field")
        return

    # Create dataset
    logging.info("Loading dataset...")
    try:
        dataset = SimpleDataset(cfg.data)
        dataset.set_model_config(cfg.model)
    except Exception as e:
        logging.error(f"Dataset error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return

    is_vector = dataset.target_stats.get('is_vector', False)
    logging.info(f"Dataset: {len(dataset)} molecules "
                 f"({len(dataset.train_idx)} train, {len(dataset.val_idx)} val, "
                 f"{len(dataset.test_idx)} test), "
                 f"target={'vector' if is_vector else 'scalar'}")

    # Distribution diagnosis
    train_targets = []
    for idx in dataset.train_idx:
        tv = dataset.database[idx][dataset.target_key]
        train_targets.append(tv if isinstance(tv, (list, np.ndarray)) else [tv])
    if train_targets:
        diagnose_target_distribution(np.array(train_targets))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg.mode}_{timestamp}"
    writer = CSVWriter(run_id)

    # ========== TRAIN ==========
    if cfg.mode == "train":
        logging.info("Starting training...")
        model, history, is_vector, model_cfg_used = train_model(dataset, cfg.model, cfg.train, device)

        # Evaluate on validation set (used for early stopping)
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

        # Evaluate on held-out test set (unbiased final evaluation)
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

        # Save outputs
        writer.save_history(history)
        writer.save_splits(dataset)
        writer.save_config(cfg)

        # Save model checkpoint
        model_path = cfg.pretrained_model_file
        torch.save({
            "model_state": model.state_dict(),
            "model_config": asdict(model_cfg_used),  # Save adjusted config (correct irreps_out)
            "type_encoding": dataset.type_encoding,
            "target_stats": dataset.target_stats,
            "normalizer_state": {
                "strategy": dataset.normalizer.strategy,
                "stats": dataset.normalizer.stats,
                "is_vector": dataset.normalizer._is_vector,
                "scaler_params": _get_scaler_params(dataset.normalizer.scaler),
                # Vector normalization params (equivariance-preserving)
                "vector_mean": dataset.normalizer._vector_mean.tolist()
                    if dataset.normalizer._vector_mean is not None else None,
                "vector_scale": float(dataset.normalizer._vector_scale)
                    if dataset.normalizer._vector_scale is not None else None,
            },
            "is_vector": is_vector,
            "history": history,
            "val_metrics": val_metrics if val_results else {},
            "test_metrics": test_metrics,
        }, model_path)
        logging.info(f"Model saved: {model_path}")

        # Plots (use test results if available, otherwise val)
        plot_results = test_results if test_results else val_results
        logging.info("Generating plots...")
        plot_loss_history(history)
        plot_target_distribution(dataset, is_vector)
        if plot_results:
            plot_parity(plot_results, is_vector, target_name=dataset.target_key)
            plot_error_histogram(plot_results, is_vector)
            plot_residuals(plot_results, is_vector)

    # ========== CROSS-VALIDATE ==========
    elif cfg.mode == "cross_validate":
        logging.info("Starting cross-validation...")
        all_metrics, avg_metrics = cross_validate(
            dataset, cfg.model, cfg.train, device, k=cfg.cv_folds)
        writer.save_config(cfg)

    # ========== PREDICT ==========
    elif cfg.mode == "predict":
        if not cfg.molecules_path:
            logging.error("molecules_path not specified")
            return
        if not cfg.pretrained_model_file or not os.path.exists(cfg.pretrained_model_file):
            logging.error(f"Model file not found: {cfg.pretrained_model_file}")
            return

        checkpoint = torch.load(cfg.pretrained_model_file, map_location=device, weights_only=False)

        # Reconstruct model config from checkpoint
        saved_model_cfg = ModelConfig(**{
            k: v for k, v in checkpoint.get("model_config", {}).items()
            if k in ModelConfig.__dataclass_fields__
        }) if "model_config" in checkpoint else cfg.model

        in_dim = len(checkpoint["type_encoding"]) + (
            ATOMIC_PROP_DIM if saved_model_cfg.use_rich_features else 1)
        model = PeriodicNetwork(in_dim, saved_model_cfg).to(device)
        model.load_state_dict(checkpoint["model_state"])
        is_vector = checkpoint.get("is_vector", False)
        # Apply saved model config to dataset
        dataset.use_rich_features = saved_model_cfg.use_rich_features

        # Restore normalizer from checkpoint (critical for correct denormalization)
        norm_state = checkpoint.get("normalizer_state", {})
        if norm_state:
            dataset.normalizer.stats = norm_state.get("stats", dataset.normalizer.stats)
            dataset.normalizer._is_vector = norm_state.get("is_vector", False)
            dataset.normalizer._fitted = True
            # Restore vector normalization params
            if norm_state.get("vector_scale") is not None:
                # Scale-only normalization (equivariance-preserving)
                vm = norm_state.get("vector_mean")
                dataset.normalizer._vector_mean = np.array(vm) if vm is not None else None
                dataset.normalizer._vector_scale = norm_state["vector_scale"]
                dataset.normalizer.scaler = None  # Vectors don't use sklearn scaler
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
                logging.info(f"Restored scalar normalizer: type={sp['type']}")
        # Also restore type_encoding from checkpoint
        dataset.type_encoding = checkpoint["type_encoding"]
        dataset.type_onehot = torch.eye(len(dataset.type_encoding), dtype=torch.float32)

        results_df = predict_new_molecules(
            model, dataset, cfg.molecules_path, device, is_vector,
            fmt=cfg.molecules_format)
        logging.info(f"Predictions: {len(results_df)} molecules")
        print(results_df)

    else:
        logging.error(f"Unknown mode: {cfg.mode}. Use 'train', 'cross_validate', or 'predict'")

    logging.info("Completed successfully!")


def _get_scaler_params(scaler):
    """Extract scaler parameters for checkpoint serialization."""
    if scaler is None:
        return None
    if isinstance(scaler, StandardScaler):
        return {"type": "standard", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    if isinstance(scaler, RobustScaler):
        return {"type": "robust", "center": scaler.center_.tolist(), "scale": scaler.scale_.tolist()}
    if isinstance(scaler, QuantileTransformer):
        return {"type": "quantile"}
    return None


if __name__ == "__main__":
    main()
