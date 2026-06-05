# Paper archive (`data_`)

Exact configs for the **eight manuscript targets**. Checkpoints live in **`data_/predict/<target>/verde_best.pt`** only (one copy per property).

> **New users predicting on their own XYZ:** use [`tutorial/`](../tutorial/README.md) instead — no full database required.

## Layout

```
data_/
  predict/<target>/     config_predict.json + verde_best.pt   ← models here
  train/<target>/       config_train.json only               ← saves to predict/ path
  hpo/<target>/           config_hpo.json + best_config.json
  learning_curve/<target>/ config_learning_curve.json
  predict/xyz/            40 benchmark XYZ
```

Target folders: `e00S1`, `e00T1`, `oxS0`, `oxS1`, `oxT1`, `redS0`, `redS1`, `redT1`.

## Targets

| Folder | `target_key` |
|--------|----------------|
| `oxS0` | `oxidation_potential_S0 (eV)` |
| `oxS1` | `oxidation_potential_S1 (eV)` |
| `oxT1` | `oxidation_potential_T1 (eV)` |
| `redS0` | `reduction_potential_S0 (eV)` |
| `redS1` | `reduction_potential_S1 (eV)` |
| `redT1` | `reduction_potential_T1 (eV)` |
| `e00S1` | `0-0_S1 (eV)` |
| `e00T1` | `0-0_T1 (eV)` |

All models use `structure_field`: `"xyz_S0"`.

## Commands (from repo root)

Predict and paper benchmark (no `verde+PCs.json` needed for predict on `predict/xyz/`):

```bash
python -m e3verde.run data_/predict/oxS0/config_predict.json
```

Full retrain / HPO / learning curve (requires `verde+PCs.json` in project root):

```bash
python -m e3verde.run data_/train/oxS0/config_train.json
python -m e3verde.hpo data_/hpo/oxS0/config_hpo.json --n_trials 50 --hpo_epochs 60
python -m e3verde.learning_curve data_/learning_curve/oxS0/config_learning_curve.json
```

Train configs write to `data_/predict/<target>/verde_best.pt`.

Generic templates: [`configs/`](../configs/README.md). Full reference: [DOCUMENTATION.md](../DOCUMENTATION.md).
