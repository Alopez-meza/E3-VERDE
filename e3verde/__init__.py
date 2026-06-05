"""
e3verde: E(3)-VERDE equivariant molecular property prediction.

CLI entry points (run from project root):

    python -m e3verde.run CONFIG.json      train / CV / predict
    python -m e3verde.hpo CONFIG.json
    python -m e3verde.learning_curve CONFIG.json

Implementation modules: ``training`` (loops & metrics), ``data``, ``model``, ``config``, ``plots``.
"""

from e3verde.config import (
    ModelConfig,
    TrainConfig,
    DataConfig,
    RunConfig,
    ATOMIC_MASSES,
    ATOMIC_PROPERTIES,
    ATOMIC_PROP_DIM,
    OUTPUT_DIRS,
    set_seed,
    setup_directories,
    setup_logging,
)

from e3verde.data import (
    SimpleDataset,
    TargetNormalizer,
    check_nan_inf,
    parse_target_value,
    build_graph_from_atoms,
    prepare_batch,
    diagnose_target_distribution,
)

from e3verde.model import (
    PeriodicNetwork,
    MessagePassingBlock,
    EMA,
    LabelSmoothingLoss,
)

from e3verde.training import (
    train_model,
    evaluate_model,
    compute_metrics,
    cross_validate,
    predict_new_molecules,
    CSVWriter,
    auto_adjust_irreps_out,
    _get_scaler_params,
)

from e3verde.plots import (
    plot_loss_history,
    plot_parity,
    plot_error_histogram,
    plot_target_distribution,
    plot_residuals,
    plot_cv_metrics,
    plot_cv_learning_curves,
)

__all__ = [
    # Config
    "ModelConfig", "TrainConfig", "DataConfig", "RunConfig",
    "ATOMIC_MASSES", "ATOMIC_PROPERTIES", "ATOMIC_PROP_DIM", "OUTPUT_DIRS",
    "set_seed", "setup_directories", "setup_logging",
    # Data
    "SimpleDataset", "TargetNormalizer", "check_nan_inf", "parse_target_value",
    "build_graph_from_atoms", "prepare_batch", "diagnose_target_distribution",
    # Model
    "PeriodicNetwork", "MessagePassingBlock", "EMA", "LabelSmoothingLoss",
    # Training
    "train_model", "evaluate_model", "compute_metrics", "cross_validate",
    "predict_new_molecules", "CSVWriter", "auto_adjust_irreps_out", "_get_scaler_params",
    # Plots
    "plot_loss_history", "plot_parity", "plot_error_histogram",
    "plot_target_distribution", "plot_residuals", "plot_cv_metrics", "plot_cv_learning_curves",
]
