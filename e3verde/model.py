#!/usr/bin/env python3
"""
Equivariant model: MessagePassingBlock, PeriodicNetwork, EMA, and loss functions.
"""

import copy
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np

from e3nn.o3 import Irreps, spherical_harmonics, Linear as E3NNLinear
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists
from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter_mean, scatter_add, scatter_max

from e3verde.config import ModelConfig
from e3verde.data import check_nan_inf


class EMA:
    """
    Exponential Moving Average of model parameters.

    Does NOT affect equivariance: it only smooths scalar weight values,
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


class MessagePassingBlock(nn.Module):
    """
    Equivariant message passing block with residual connections and self-interaction.

    Architecture: Conv -> Gate -> [Self-Interaction] -> [LayerNorm] -> [Dropout] -> [+ Residual]

    EQUIVARIANCE NOTES:
    - Residual connection: equivariant when input/output irreps match (R(a+b) = Ra + Rb).
    - Self-interaction: E3NNLinear maps irreps -> irreps equivariantly (NequIP-style).
    - LayerNorm: applied only to l=0 (scalar) components.
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

        if use_self_interaction:
            self.self_interaction = E3NNLinear(gate.irreps_out, gate.irreps_out)
        else:
            self.self_interaction = None

        self._irrep_slices: List[Tuple[int, int, int]] = []
        self._scalar_indices: List[int] = []
        idx = 0
        for mul, ir in gate.irreps_out:
            if ir.l == 0:
                self._scalar_indices.extend(range(idx, idx + mul))
            for m in range(mul):
                self._irrep_slices.append((idx, idx + ir.dim, ir.dim))
                idx += ir.dim

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
        x_in = x

        x = self.conv(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
        x = self.gate(x)

        if self.self_interaction is not None:
            x = self.self_interaction(x)

        if self.layer_norm is not None and len(self._scalar_indices) > 0:
            x = x.clone()
            x[..., self._scalar_indices] = self.layer_norm(x[..., self._scalar_indices])

        x = self._equivariant_dropout(x)

        if self.use_residual and x_in.shape[-1] == x.shape[-1]:
            x = x + x_in

        return x


class PeriodicNetwork(nn.Module):
    """
    E(3)-equivariant Graph Neural Network for molecular property prediction.

    Extensions over vanilla gate_points_2101:
    - Residual message-passing connections
    - Self-interaction linear layers after each gate
    - Rich atomic node features (see config.ATOMIC_PROPERTIES)
    - Attention readout (weights from l=0 scalars only)
    - Optional multi-layer output MLP on aggregated scalars
    - Optional multi-scale readout (concatenate layer-wise scalar features)

    All of the above preserve E(3) equivariance for scalar and vector targets.
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

        act = {1: nn.functional.leaky_relu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        self.mp_layers = nn.ModuleList()
        self._layer_irreps_out = []
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

        self.irreps_hidden_out = irreps

        self._is_scalar_output = all(ir.l == 0 for _, ir in self.irreps_out)
        if not self._is_scalar_output:
            logging.info(f"Vector output detected ({self.irreps_out}): "
                         f"using equivariant final conv (no MLP, no multi-scale)")

        self._hidden_scalar_dim = sum(mul for mul, ir in self.irreps_hidden_out if ir.l == 0)

        if self._is_scalar_output and cfg.use_multiscale_readout and cfg.layers > 1:
            self.layer_readouts = nn.ModuleList()
            per_layer_scalar_dims = []
            for li in range(cfg.layers):
                layer_irreps = self._layer_irreps_out[li]
                n_scalars = sum(mul for mul, ir in layer_irreps if ir.l == 0)
                if n_scalars == 0:
                    n_scalars = 1
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

        if self._is_scalar_output and not self.use_multiscale:
            self.final_conv = Convolution(
                irreps, self.irreps_node_attr, self.irreps_edge_attr,
                Irreps(f"{self._hidden_scalar_dim}x0e"),
                cfg.number_of_basis, cfg.radial_layers,
                cfg.radial_neurons, cfg.num_neighbors
            )
        elif not self._is_scalar_output:
            self.final_conv = Convolution(
                irreps, self.irreps_node_attr, self.irreps_edge_attr,
                self.irreps_out, cfg.number_of_basis, cfg.radial_layers,
                cfg.radial_neurons, cfg.num_neighbors
            )
        else:
            self.final_conv = None

        if cfg.readout_type == "attention" and self.reduce_output and self._is_scalar_output:
            self.attn_gate = nn.Sequential(
                nn.Linear(total_scalar_dim, max(total_scalar_dim // 2, 1)),
                nn.SiLU(),
                nn.Linear(max(total_scalar_dim // 2, 1), 1),
            )
            logging.info("Attention readout enabled (learned scalar weights per atom)")
        else:
            self.attn_gate = None

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
            self.output_mlp = nn.Linear(total_scalar_dim, out_dim)
            logging.info(f"Output projection: {total_scalar_dim} -> {out_dim} (linear, no MLP)")
        else:
            self.output_mlp = None

    def forward(self, data):
        batch = data.batch if getattr(data, "batch", None) is not None else \
            data.pos.new_zeros(data.pos.shape[0], dtype=torch.long)
        edge_src, edge_dst = data.edge_index[0], data.edge_index[1]
        edge_vec = data.edge_vec

        edge_sh = spherical_harmonics(self.irreps_edge_attr, edge_vec,
                                      normalize=True, normalization="component")
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length, start=0.0, end=self.max_radius,
            number=self.number_of_basis, basis="smooth_finite", cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        x = self.em(data.x) if hasattr(data, "x") and data.x is not None else \
            data.pos.new_ones((data.pos.shape[0], self.em_dim))
        z = data.z if hasattr(data, "z") and data.z is not None else \
            data.pos.new_ones((data.pos.shape[0], self.irreps_node_attr.dim))

        layer_outputs = []
        for i, layer in enumerate(self.mp_layers):
            x = layer(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
            if self.training:
                check_nan_inf(x, f"layer_{i}_output", raise_error=False)
            if self.use_multiscale and self.layer_readouts is not None:
                layer_outputs.append(x)

        if not self.reduce_output:
            if self.final_conv is not None:
                x = self.final_conv(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
            check_nan_inf(x, "model_output", raise_error=True)
            return x

        if not self._is_scalar_output:
            # VECTOR PATH: equivariant final conv -> weighted mean aggregation
            node_out = self.final_conv(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
            out = scatter_mean(node_out, batch, dim=0)
        else:
            # SCALAR PATH: multi-scale/single-scale -> attention -> MLP
            if self.use_multiscale and self.layer_readouts is not None and len(layer_outputs) > 0:
                scalar_parts = [proj(feats) for feats, proj in zip(layer_outputs, self.layer_readouts)]
                node_scalars = torch.cat(scalar_parts, dim=-1)
            else:
                node_scalars = self.final_conv(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

            if self.attn_gate is not None:
                # Attention pooling with numerically stable per-graph softmax
                attn_logits = self.attn_gate(node_scalars).squeeze(-1)
                attn_max, _ = scatter_max(attn_logits, batch, dim=0)
                attn_logits = attn_logits - attn_max[batch]
                attn_exp = torch.exp(attn_logits)
                attn_sum = scatter_add(attn_exp, batch, dim=0)
                attn_weights = (attn_exp / (attn_sum[batch] + 1e-10)).unsqueeze(-1)
                out = scatter_add(attn_weights * node_scalars, batch, dim=0)
            else:
                out = scatter_mean(node_scalars, batch, dim=0)

            if self.output_mlp is not None:
                out = self.output_mlp(out)

        check_nan_inf(out, "model_output", raise_error=True)
        return out


class LabelSmoothingLoss(nn.Module):
    """Label smoothing for regression (smooths target towards mean)."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.base = nn.L1Loss()

    def forward(self, pred, target):
        target_smooth = (1 - self.smoothing) * target + self.smoothing * target.mean()
        return self.base(pred, target_smooth)
