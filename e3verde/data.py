#!/usr/bin/env python3
"""
Data loading, graph construction, normalization, and dataset splitting.
"""

import json
import logging
from io import StringIO
from typing import Dict, Any

import torch
import numpy as np
import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

from torch_geometric.data import Data, Dataset
from ase.io import read
from ase.neighborlist import neighbor_list
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split

from e3verde.config import (
    DataConfig, ModelConfig,
    ATOMIC_MASSES, ATOMIC_PROPERTIES, ATOMIC_PROP_DIM,
)


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
        if tv.startswith('[') or tv.startswith('('):
            tv = tv.strip('[]() ')
        if ',' in tv:
            parts = [x.strip() for x in tv.split(',')]
            return [float(x) for x in parts if x]
        else:
            return float(tv)
    return float(tv)


def build_graph_from_atoms(atoms, type_encoding, type_onehot, cutoff,
                           normalize_features=False, mass_mean=None, mass_std=None,
                           use_rich_features=False):
    """Build PyG Data object from ASE Atoms, preserving equivariance.

    Node features are invariant scalars (0e): one-hot encoding + atomic properties.
    These don't transform under rotation, so they don't affect equivariance.
    """
    symbols = atoms.get_chemical_symbols()

    onehot = [type_onehot[type_encoding.get(sym, 0)] for sym in symbols]
    x = torch.stack(onehot, dim=0)

    if use_rich_features:
        # Rich atomic properties (pre-normalized scalars; invariant under rotation).
        default_props = [0.1, 0.5, 0.3, 0.1, 0.4, 0.03, 0.5]
        props = np.array([ATOMIC_PROPERTIES.get(sym, default_props) for sym in symbols])
        x = torch.cat([x, torch.tensor(props, dtype=torch.float32)], dim=1)
    else:
        # Mass-only node features (legacy mode).
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
    """Prepare a batch: move to device and reshape targets."""
    if getattr(batch, "batch", None) is None:
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)
    batch = batch.to(device)
    if is_vector:
        target = batch.y.view(-1, vector_dim) if len(batch.y.shape) == 1 else batch.y
    else:
        target = batch.y.view(-1, 1)
    return batch, target


class TargetNormalizer:
    """
    Handles target normalization with equivariance-aware vector handling.

    EQUIVARIANCE CRITICAL:
    For vector targets (e.g., dipole moments, forces), per-component normalization
    BREAKS equivariance. Solution: for vectors, use a single scalar (RMS of all
    components) to scale all components equally, preserving direction.
    """

    def __init__(self, strategy: str = "auto"):
        self.strategy = strategy
        self.scaler = None
        self.stats: Dict[str, Any] = {}
        self._fitted = False
        self._is_vector = False
        self._vector_mean = None
        self._vector_scale = None

    def fit(self, targets: np.ndarray) -> "TargetNormalizer":
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)

        self._is_vector = targets.shape[1] > 1
        actual_strategy = 'equivariant_uniform'

        if self._is_vector:
            # EQUIVARIANT normalization for vectors: SCALE ONLY, NO CENTERING.
            #
            # WHY NO CENTERING:
            # With 1x1o output, the model produces equivariant vectors: R*v under rotation.
            # If targets = (v - mean_vec) / scale, they transform as R*v - mean_vec (NOT equivariant).
            # Scale-only is safe: R*(v/s) = (R*v)/s.
            self._vector_mean = None
            rms = np.sqrt(np.mean(targets ** 2)) + 1e-10
            self._vector_scale = rms
            self.scaler = None
            logging.info(f"TargetNormalizer (vector, equivariant): scale_only, "
                         f"rms_scale={self._vector_scale:.6f} (no centering)")
        else:
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
            # Equivariant: scale only, no centering
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

        self.database = self._filter_valid_entries(raw_db)
        if len(self.database) == 0:
            raise ValueError(f"No valid entries with target '{data_cfg.target_key}' "
                             f"and structure '{data_cfg.structure_field}'")
        logging.info(f"Dataset: {len(self.database)} valid entries")

        self.type_encoding, self.type_onehot = self._build_type_encoding()

        self.train_idx, self.val_idx, self.test_idx = self._make_splits(
            data_cfg.test_size, data_cfg.val_size, data_cfg.seed)

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
            trainval_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=seed)
            val_fraction_of_trainval = val_size / (1.0 - test_size)
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=val_fraction_of_trainval, random_state=seed)
        elif test_size > 0:
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

    def set_model_config(self, model_cfg: ModelConfig):
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
            except Exception:
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
            try:
                parsed = parse_target_value(tv)
                if parsed is None:
                    continue
                entry[self.target_key] = parsed
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
            except Exception:
                continue
        encoding = {sym: i for i, sym in enumerate(sorted(all_symbols))}
        onehot = torch.eye(len(encoding), dtype=torch.float32)
        return encoding, onehot

    def _fit_normalization(self):
        """Fit normalizer on training targets and compute mass statistics."""
        if len(self.train_idx) == 0:
            return

        if self.normalize_targets:
            train_targets = []
            for idx in self.train_idx:
                tv = self.database[idx][self.target_key]
                train_targets.append(tv if isinstance(tv, (list, tuple, np.ndarray)) else [tv])
            train_targets = np.array(train_targets)
            self.normalizer.fit(train_targets)
        else:
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

        if self.normalize_features:
            all_masses = []
            for idx in self.train_idx:
                try:
                    atoms = read(StringIO(self.database[idx][self.structure_field]), format="xyz")
                    for sym in atoms.get_chemical_symbols():
                        if sym in ATOMIC_MASSES:
                            all_masses.append(ATOMIC_MASSES[sym])
                except Exception:
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
