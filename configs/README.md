# Config templates (`configs/`)

Generic JSON **starting points** when you train a **new target** or workflow from scratch — not tied to a specific paper property folder.

## When to use this folder

| You want to… | Use instead |
|--------------|-------------|
| Predict on **your molecules** with paper models | [`tutorial/configs/new_molecules/`](../tutorial/configs/new_molecules/) |
| Reproduce **paper** train/predict/HPO/LC (8 targets) | [`data_/`](../data_/README.md) |
| Build a config for a **custom** target or database | **This folder** — copy and edit |

## Files

| File | `mode` | Purpose |
|------|--------|---------|
| `train.json` | train | Train; saves to `data/models/verde_best.pt`; expects `verde+PCs.json` |
| `predict.json` | predict | Predict with `data_/predict/oxS0/verde_best.pt` and `data_/predict/xyz/` |
| `cross_validate.json` | cross_validate | K-fold CV |
| `hpo.json` | train (baseline for search) | Optuna via `python -m e3verde.hpo configs/hpo.json` |
| `learning_curve.json` | train (baseline) | LC via `python -m e3verde.learning_curve configs/learning_curve.json` |

Copy a file, change `data.target_key`, `data.dataset_path`, and `pretrained_model_file` for your project.

See [DOCUMENTATION.md](../DOCUMENTATION.md) for all JSON fields.
