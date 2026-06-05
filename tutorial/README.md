# How to use E(3)-VERDE

This is the **only step-by-step tutorial** in the repository. Other folders (`configs/`, `data_/`) hold JSON configs for different audiences — see [Which folder should I use?](../README.md#which-folder-should-i-use) in the main README if that is confusing.

This guide is **self-contained**: follow the steps below from the project root (the folder that contains `e3verde/`, `tutorial/`, and `data_/`). No extra scripts or files are required beyond what ships in the repository.

**Goal:** predict redox properties (eV) on your 3D structures using the pre-trained paper models in `data_/`.

---

## What you need before starting

| Requirement | Included in repo? |
|-------------|-------------------|
| Python 3.10+ | You install |
| PyTorch + PyG + dependencies | `pip install -r requirements.txt` |
| **CPU or GPU** | Both supported — GPU used automatically if CUDA is available |
| Pre-trained models (`verde_best.pt`) | Yes — `data_/predict/<target>/` |
| Example input molecules | Yes — `tutorial/my_molecules/*.xyz` (3 files) |
| Predict configs (JSON) | Yes — `tutorial/configs/new_molecules/*.json` |
| Full training database (`verde+PCs.json`) | **No** — not needed for prediction with paper models |
| Your own JSON database | **No** — only if you train or retrain models (see below) |

**Two use cases**

| Goal | What you need |
|------|----------------|
| **Predict with paper models** (Steps 1–6 below) | XYZ files + configs in `tutorial/configs/new_molecules/` |
| **Train your own model** or use **your own database** | JSON database + train config — see [Your own database and custom models](#your-own-database-and-custom-models) |

---

## Quick reference — the six steps

| Step | Action | Location |
|------|--------|----------|
| 1 | Install | `pip install -r requirements.txt` |
| 2 | Add your S₀ XYZ files (or keep the examples) | `tutorial/my_molecules/` |
| 3 | Choose the property to predict | `tutorial/configs/new_molecules/<target>.json` |
| 4 | Edit the JSON only if your XYZ are elsewhere | field `molecules_path` |
| 5 | Run predict | `python -m e3verde.run tutorial/configs/new_molecules/oxS0.json` |
| 6 | Open the results CSV | `data/predictions/predictions_*.csv` |

---

## Example — first run without editing anything

The repo includes three example `.xyz` files and ready-to-use configs. Try this immediately after installing:

```bash
pip install -r requirements.txt
python -m e3verde.run tutorial/configs/new_molecules/oxS0.json
```

**What you should see**

- `Device: cpu` or `Device: cuda` (GPU is selected automatically when available)
- Log lines such as `Dataset: 73 valid entries` (internal reference data loading)
- `Predictions: 3 molecules`
- A table in the terminal with columns `molecule_id`, `xyz_file`, `formula`, `num_atoms`, `predicted_value`
- Final line: `Completed successfully!`

**Where the results go**

| File | Content |
|------|---------|
| `data/predictions/predictions_YYYYMMDD_HHMMSS.csv` | One row per molecule, values in eV |
| `data/logs/log_YYYYMMDD_HHMMSS.log` | Full log of the run |

The CSV is the file you keep. The newest file in `data/predictions/` is from your latest run.

Example CSV (three example molecules, oxidation S₀):

| molecule_id | xyz_file | formula | num_atoms | predicted_value |
|-------------|----------|---------|-----------|-----------------|
| molecule_1 | AENVNKCHSWUVSH-UHFFFAOYSA.xyz | C17H10O2 | 29 | ~1.6 eV |
| molecule_2 | RCBSZGSHSLQLBI-UHFFFAOYSA.xyz | … | 44 | … |
| molecule_3 | XNFHFFWRUFLEQE-UHFFFAOYSA.xyz | … | 52 | … |

Exact numbers vary slightly by platform; the important check is **3 rows** and **no error** at the end.

---

## Repository map — folders and files

```
e3verde-clean/
├── e3verde/                              code (do not edit for normal use)
│   └── run.py                            CLI: python -m e3verde.run
│
├── data_/                                paper archive (read only)
│   └── predict/
│       ├── oxS0/   verde_best.pt
│       ├── oxS1/   verde_best.pt
│       ├── oxT1/   verde_best.pt
│       ├── redS0/  verde_best.pt
│       ├── redS1/  verde_best.pt
│       ├── redT1/  verde_best.pt
│       ├── e00S1/  verde_best.pt
│       └── e00T1/  verde_best.pt
│
├── tutorial/                             your working area
│   ├── my_molecules/                     ← input XYZ
│   ├── configs/
│   │   ├── new_molecules/                ← predict configs (8 properties)
│   │   ├── train_demo.json               optional: train a small model
│   │   ├── predict_trained.json          optional: predict with demo model
│   │   ├── hpo_demo.json                 optional: hyperparameter search
│   │   └── learning_curve_demo.json      optional: learning curve
│   └── data/
│       └── verde_mini.json               internal reference DB (do not remove)
│
└── data/                                 outputs (created when you run)
    ├── predictions/   *.csv
    ├── logs/          *.log
    ├── models/        *.pt   (only if you train)
    ├── csv/           *.csv  (only if you train)
    ├── optuna/        (only if you run HPO)
    └── learning_curve/ (only if you run learning curve)
```

### What you change vs what you keep

| Path | Role | Edit? |
|------|------|-------|
| `tutorial/my_molecules/*.xyz` | Structures to predict | **Yes** — add or replace your molecules |
| `tutorial/configs/new_molecules/*.json` | Property + paths | **Sometimes** — mainly `molecules_path` |
| `data_/predict/*/verde_best.pt` | Paper-trained weights | **No** |
| `tutorial/data/verde_mini.json` | Internal reference DB | **No** |
| `e3verde/` | Source code | **No** |
| `data/` | Run outputs | **No** — created automatically |

Nothing is written to `tutorial/my_molecules/` or `data_/` when you predict.

---

## Step 1 — Install

From the project root:

```bash
pip install -r requirements.txt
```

Install [PyTorch](https://pytorch.org/) first, then matching [PyG wheels](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), if the command fails on `torch_geometric` or `torch_scatter`.

### CPU and GPU

E(3)-VERDE runs on **CPU and GPU**. No config change is needed:

| Hardware | How it is selected |
|----------|-------------------|
| **GPU (CUDA)** | Used automatically if PyTorch detects a CUDA device |
| **CPU** | Used when no GPU is available |

At the start of each run the log prints `Device: cuda` or `Device: cpu`. Predict, train, HPO, and learning-curve all use the same rule.

**Install notes**

- **CPU only:** install the default PyTorch build from [pytorch.org](https://pytorch.org/).
- **GPU (NVIDIA):** install PyTorch with CUDA support (same page — choose your CUDA version), then install matching PyG wheels.
- **Mixed precision (AMP)** during training is enabled only on GPU (`use_amp: true` in configs); on CPU training runs in full precision.

The tutorial works on a laptop without a GPU. Times below are for CPU; GPU is typically faster for train and large batches.

All later commands are also run from the **project root**.

---

## Step 2 — Prepare input structures

### Where to put files

```
tutorial/my_molecules/
  your_molecule_1.xyz
  your_molecule_2.xyz
  ...
```

**One file = one molecule.** Three examples are already in this folder; you can predict on them without adding anything.

If your XYZ live elsewhere, skip copying them here and set `molecules_path` in the JSON (Step 4).

### XYZ format

Standard [ASE](https://wiki.fysik.dtu.dk/ase/)-readable XYZ:

```
<number of atoms>
<comment line (any text)>
Symbol   x   y   z
Symbol   x   y   z
...
```

Real example from the repo (first lines):

```
29
../path/to/log	Energy: -504390.8418429
O   0.72423   3.30362   0.00038
C   0.94690   2.11919   0.00003
...
```

### Input rules

| Rule | Detail |
|------|--------|
| Geometry | **Ground-state S₀** 3D structure (same as training) |
| Units | Ångström |
| Elements | Standard symbols (`C`, `N`, `O`, `F`, …) |
| Filename | Any name ending in `.xyz`; shown in output as `xyz_file` |
| Comment line | Free text |

Molecules with elements **not seen in training** are skipped with a warning.

---

## Step 3 — Choose a property and config file

Each property has **one JSON file** in `tutorial/configs/new_molecules/`. Each file loads the matching checkpoint from `data_/predict/<same name>/verde_best.pt`.

| Config file | Checkpoint | Property | `data.target_key` | Units |
|-------------|------------|----------|-------------------|-------|
| `oxS0.json` | `data_/predict/oxS0/verde_best.pt` | Oxidation potential, S₀ | `oxidation_potential_S0 (eV)` | eV |
| `oxS1.json` | `data_/predict/oxS1/verde_best.pt` | Oxidation potential, S₁ | `oxidation_potential_S1 (eV)` | eV |
| `oxT1.json` | `data_/predict/oxT1/verde_best.pt` | Oxidation potential, T₁ | `oxidation_potential_T1 (eV)` | eV |
| `redS0.json` | `data_/predict/redS0/verde_best.pt` | Reduction potential, S₀ | `reduction_potential_S0 (eV)` | eV |
| `redS1.json` | `data_/predict/redS1/verde_best.pt` | Reduction potential, S₁ | `reduction_potential_S1 (eV)` | eV |
| `redT1.json` | `data_/predict/redT1/verde_best.pt` | Reduction potential, T₁ | `reduction_potential_T1 (eV)` | eV |
| `e00S1.json` | `data_/predict/e00S1/verde_best.pt` | 0–0 excitation energy, S₁ | `0-0_S1 (eV)` | eV |
| `e00T1.json` | `data_/predict/e00T1/verde_best.pt` | 0–0 excitation energy, T₁ | `0-0_T1 (eV)` | eV |

All eight models use **`structure_field`: `"xyz_S0"`** — input must be S₀ geometry even for S₁/T₁ properties.

**Do not** mix config and checkpoint (e.g. never point `oxS0.json` at `redS0/verde_best.pt`).

---

## Step 4 — Config JSON: format, what to change, what to keep

Open the JSON for your property in a text editor. For paper models you usually **do not need to edit anything** if your XYZ are in `tutorial/my_molecules/`.

### Structure of the file

```
{
  "mode": "predict",
  "pretrained_model_file": "...",
  "molecules_path": "...",
  "molecules_format": "xyz",
  "model": { ... },
  "train": { ... },
  "data": { ... }
}
```

### Full example (`oxS0.json`)

```json
{
  "mode": "predict",
  "pretrained_model_file": "data_/predict/oxS0/verde_best.pt",
  "molecules_path": "tutorial/my_molecules",
  "molecules_format": "xyz",
  "model": {
    "em_dim": 32,
    "irreps_in": "32x0e",
    "irreps_out": "1x0e",
    "layers": 5,
    "mul": 48,
    "lmax": 2,
    "max_radius": 5.0,
    "readout_type": "attention"
  },
  "train": {
    "num_epochs": 300,
    "batch_size": 8,
    "learning_rate": 0.00549
  },
  "data": {
    "dataset_path": "tutorial/data/verde_mini.json",
    "target_key": "oxidation_potential_S0 (eV)",
    "structure_field": "xyz_S0",
    "test_size": 0.1,
    "val_size": 0.1,
    "normalize_targets": true,
    "target_normalization": "auto",
    "seed": 42
  }
}
```

The `model` and `train` blocks are truncated above; **leave them unchanged** in the actual file. Architecture comes from the `.pt` checkpoint.

### Top-level fields

| Field | Change for predict? | Meaning |
|-------|---------------------|---------|
| `mode` | **Keep** `"predict"` | Inference mode |
| `pretrained_model_file` | **Keep** | Must match the property (see Step 3) |
| `molecules_path` | **Yes**, if needed | Folder with `.xyz`, single `.xyz` file, or JSON database |
| `molecules_format` | **Yes**, if needed | `"xyz"` (default) or `"json"` |

Examples for `molecules_path`:

```json
"molecules_path": "tutorial/my_molecules"
"molecules_path": "C:/my_structures/batch_01"
"molecules_path": "tutorial/my_molecules/my_molecule.xyz"
```

### `data` block

| Field | Change? | Meaning |
|-------|---------|---------|
| `dataset_path` | **Keep** `tutorial/data/verde_mini.json` | Internal reference for normalization — **not** your prediction set |
| `target_key` | **Keep** (Step 3 table) | Property name for the output |
| `structure_field` | **Keep** `"xyz_S0"` | Geometry field |
| `test_size`, `val_size`, `seed` | **Keep** | Reference dataset splits |
| `normalize_targets`, `target_normalization` | **Keep** | Must match training |

### `model` and `train` blocks

| Block | Change? | Why |
|-------|---------|-----|
| `model` | **No** | Loaded from checkpoint |
| `train` | **No** | Ignored at predict time |

### Why `dataset_path` appears in a predict config

Your molecules come from `molecules_path`. The program still reads `dataset_path` to set up target normalization the same way as during training. The bundled `tutorial/data/verde_mini.json` is enough — you do **not** need `verde+PCs.json` for prediction.

---

## Step 5 — Run predict

Single property:

```bash
python -m e3verde.run tutorial/configs/new_molecules/oxS0.json
```

Replace `oxS0.json` with any file from Step 3.

All eight properties:

```bash
python -m e3verde.run tutorial/configs/new_molecules/oxS0.json
python -m e3verde.run tutorial/configs/new_molecules/oxS1.json
python -m e3verde.run tutorial/configs/new_molecules/oxT1.json
python -m e3verde.run tutorial/configs/new_molecules/redS0.json
python -m e3verde.run tutorial/configs/new_molecules/redS1.json
python -m e3verde.run tutorial/configs/new_molecules/redT1.json
python -m e3verde.run tutorial/configs/new_molecules/e00S1.json
python -m e3verde.run tutorial/configs/new_molecules/e00T1.json
```

Each run takes about 10–15 s on CPU (often faster on GPU) for three molecules. Success = terminal table + `Completed successfully!`.

---

## Step 6 — Read the output

### CSV columns

| Column | Meaning |
|--------|---------|
| `molecule_id` | `molecule_1`, `molecule_2`, … (or InChI key if in input) |
| `xyz_file` | Source filename |
| `formula` | Chemical formula |
| `num_atoms` | Atom count |
| `predicted_value` | Predicted property in **eV** |

### How many rows to expect

Number of rows = number of valid `.xyz` files in `molecules_path`. With the default examples, expect **3 rows**.

---

## Your own database and custom models

This section applies if you want to **train a new model**, **fine-tune on different data**, or **predict with a checkpoint you trained yourself** — not when using the pre-trained paper models from `data_/predict/`.

### What you need (training / custom DB)

| Item | Required? | Notes |
|------|-----------|-------|
| JSON database | **Yes** | Array of molecules; see format below |
| `verde+PCs.json` (VERDE+PCs) | Optional | Full paper database — [paper / SI](../README.md#data-availability); place in project root |
| Your own JSON file | Optional | Same format as VERDE; any path you choose |
| Train config (JSON) | **Yes** | Copy from `configs/train.json`, `data_/train/<target>/config_train.json`, or `tutorial/configs/train_demo.json` |
| GPU | Recommended | CPU works; training is much slower on large sets |
| XYZ for predict | **Yes** (after training) | Same as Step 2 — folder of `.xyz` or JSON entries |

You do **not** need `verde+PCs.json` to learn the workflow — use `tutorial/data/verde_mini.json` (~80 molecules) first, then swap in your database.

### Database format (JSON)

`data.dataset_path` must point to a **JSON file** that is an **array** of objects. Each object is one molecule.

**Minimum requirements per entry**

| Field | Set via config | Example |
|-------|----------------|---------|
| 3D structure | `data.structure_field` | `"xyz_S0": "29\ncomment\nO 0.72 3.30 0.00\n..."` |
| Target property | `data.target_key` | `"oxidation_potential_S0 (eV)": 3.12` |

Optional but useful: `inchi_key`, `smiles`, `formula` (used as IDs in outputs when present).

**Example entry** (same layout as VERDE+PCs):

```json
{
  "inchi_key": "AAEBJCSCYHPBRF-FZSIALSZSA-N",
  "smiles": "O=C1C(C=NC(F)(F)F)=CC(=O)c2ncccc21",
  "oxidation_potential_S0 (eV)": 3.12,
  "xyz_S0": "23\nEnergy: ...\nO 0.409  -2.276  0.000\nC 0.749  -1.103  0.000\n..."
}
```

Rules:

- XYZ inside JSON uses the **same format** as Step 2 (atom count, comment, then `Symbol x y z` rows).
- Entries missing the structure or target field are **skipped**.
- Scalar targets: numbers or strings (`1.43`, `"-2.33"`). Vector targets: lists or comma-separated strings.
- Pick **one** column name for `target_key` and **one** structure field for `structure_field`; every kept entry must have both.

See [DOCUMENTATION.md](../DOCUMENTATION.md#6-dataset-format) for the full specification.

### Option A — VERDE+PCs (reproduce or extend paper training)

1. Obtain `verde+PCs.json` from the paper / Supporting Information.
2. Place it in the **project root** (same folder as `e3verde/`).
3. Use a paper config or generic template:

```bash
python -m e3verde.run data_/train/oxS0/config_train.json
```

Paper configs live in `data_/train/<target>/` (8 targets — same names as predict). Each expects:

```json
"data": {
  "dataset_path": "verde+PCs.json",
  "target_key": "oxidation_potential_S0 (eV)",
  "structure_field": "xyz_S0"
}
```

Target keys for all eight properties: [data_/README.md](../data_/README.md).

**Outputs:** checkpoint at `data_/predict/<target>/verde_best.pt`, plus metrics in `data/csv/` and plots in `data/figures/`.

### Option B — Your own database

1. Build a JSON array in the format above (one file, e.g. `my_dataset.json` in the project root or any path).
2. Copy a train config and edit the `data` block:

```json
{
  "mode": "train",
  "pretrained_model_file": "data/models/my_model.pt",
  "data": {
    "dataset_path": "my_dataset.json",
    "target_key": "my_property (eV)",
    "structure_field": "xyz_S0",
    "test_size": 0.1,
    "val_size": 0.1,
    "normalize_targets": true,
    "normalize_features": true,
    "target_normalization": "auto",
    "seed": 42
  },
  "train": { "num_epochs": 100, "batch_size": 8, "..." : "..." },
  "model": { "em_dim": 32, "layers": 3, "..." : "..." }
}
```

3. Train:

```bash
python -m e3verde.run my_train_config.json
```

4. Start from `configs/train.json` (generic) or `tutorial/configs/train_demo.json` (small quick test on `verde_mini.json`).

| Field | You must set |
|-------|----------------|
| `data.dataset_path` | Path to **your** JSON file |
| `data.target_key` | Exact column name in your JSON |
| `data.structure_field` | Field with XYZ text (e.g. `"xyz_S0"`) |
| `pretrained_model_file` | Where to save the new `.pt` |
| `model.*` | Architecture — **must match** when you predict later |
| `train.*` | Epochs, batch size, learning rate, etc. |

### Predict with a model you trained

After training, create a predict config (copy `tutorial/configs/predict_trained.json` or `data_/predict/oxS0/config_predict.json`) and set:

| Field | Value |
|-------|-------|
| `"mode"` | `"predict"` |
| `pretrained_model_file` | Your checkpoint, e.g. `data/models/my_model.pt` |
| `molecules_path` | Folder of `.xyz` (Step 2) |
| `molecules_format` | `"xyz"` |
| `data.dataset_path` | **Same JSON you trained on** (needed for normalization setup) |
| `data.target_key` | **Same** as in train config |
| `data.structure_field` | **Same** as in train config |
| `model` / `train` blocks | Same as train config (architecture placeholders) |

```bash
python -m e3verde.run my_predict_config.json
```

Results still go to `data/predictions/predictions_*.csv`.

### Hyperparameter search and learning curve (custom training)

Once `dataset_path` points to your database:

```bash
python -m e3verde.hpo configs/hpo.json --n_trials 50 --hpo_epochs 60
python -m e3verde.learning_curve configs/learning_curve.json
```

Paper baselines: `data_/hpo/<target>/config_hpo.json` and `data_/learning_curve/<target>/config_learning_curve.json`.  
Tutorial mini-demos: `tutorial/configs/hpo_demo.json` and `tutorial/configs/learning_curve_demo.json` (use `verde_mini.json`).

HPO writes `data/optuna/best_config_*.json` — you can train production models from that file.

### Quick path: learn training on mini data, then swap database

```bash
# 1. Train on bundled subset (~30 s CPU)
python -m e3verde.run tutorial/configs/train_demo.json

# 2. Predict with that checkpoint
python -m e3verde.run tutorial/configs/predict_trained.json

# 3. When ready, copy train_demo.json → my_train.json,
#    set dataset_path to verde+PCs.json or your my_dataset.json,
#    adjust target_key / structure_field, and run again
```

---

## Optional workflows (mini demos)

These use `tutorial/data/verde_mini.json` only — a **short practice run**, not full paper training. For VERDE+PCs or your own database, see [Your own database and custom models](#your-own-database-and-custom-models) above.

### Train a small demo model, then predict with it

**Step A — train** (~30 s on CPU with `verde_mini.json`):

```bash
python -m e3verde.run tutorial/configs/train_demo.json
```

Creates `data/models/tutorial_demo.pt` plus metrics in `data/csv/` and plots in `data/figures/`.

**Step B — predict** with that checkpoint (same three example molecules):

```bash
python -m e3verde.run tutorial/configs/predict_trained.json
```

`predict_trained.json` points to `data/models/tutorial_demo.pt`. **Run Step A before Step B** — otherwise you get `Model file not found`.

### Train config — fields you may change

```json
{
  "mode": "train",
  "pretrained_model_file": "data/models/tutorial_demo.pt",
  "data": {
    "dataset_path": "tutorial/data/verde_mini.json",
    "target_key": "oxidation_potential_S0 (eV)",
    "structure_field": "xyz_S0"
  },
  "train": { "num_epochs": 3, "batch_size": 8, "..." : "..." },
  "model": { "em_dim": 32, "layers": 3, "..." : "..." }
}
```

| Field | Change? | Notes |
|-------|---------|-------|
| `mode` | Must stay `"train"` | |
| `data.dataset_path` | Yes | e.g. `verde+PCs.json` for full training |
| `data.target_key` / `structure_field` | Yes | Must match your database |
| `pretrained_model_file` | Yes | Output checkpoint path |
| `train.*`, `model.*` | Yes | Hyperparameters and architecture |

Paper-scale configs: `data_/train/<target>/config_train.json` ([data_/README.md](../data_/README.md)).

### Hyperparameter optimization (HPO)

```bash
python -m e3verde.hpo tutorial/configs/hpo_demo.json
```

Uses `tutorial/data/verde_mini.json`. Outputs in `data/optuna/`. For a quick test, add `--n_trials 2 --hpo_epochs 2 --no_retrain`.

### Learning curve

```bash
python -m e3verde.learning_curve tutorial/configs/learning_curve_demo.json
```

Outputs in `data/learning_curve/`. For a quick test, add `--fractions 0.5 1.0 --repeats 1 --epochs 2 --patience 1`.

---

## Troubleshooting

| Problem | What to check |
|---------|---------------|
| `Model file not found` | `data_/predict/<target>/verde_best.pt` exists; for `predict_trained.json`, run `train_demo.json` first |
| `dataset_path` not found | Predict: keep `tutorial/data/verde_mini.json`. Train: path to your JSON or `verde+PCs.json` in project root |
| Custom model gives wrong values | `target_key`, `structure_field`, and `dataset_path` must match the train config |
| `Predictions: 0 molecules` | Valid `.xyz` in `molecules_path`; correct XYZ format |
| Fewer rows than expected | Some molecules skipped (unknown elements — see log) |
| Wrong values | Config, checkpoint, and `target_key` must match (Step 3) |
| PyG / torch import error | Reinstall PyG wheels for your PyTorch version |
| Expected GPU but see `Device: cpu` | Install CUDA-enabled PyTorch; check `python -c "import torch; print(torch.cuda.is_available())"` |
| Run from wrong folder | All paths are relative to the **project root** |

Full technical reference: [DOCUMENTATION.md](../DOCUMENTATION.md)
