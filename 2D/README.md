# COMPASS 2D (Segmentation → Scalar CP)

This folder contains the **2D** pipeline for COMPASS-style conformal prediction (CP) methods for **scalar targets derived from 2D segmentation masks**.

This release ships with a single scalar target:

- `mask_sum`: sum of mask pixels (area proxy)

2D entrypoints live under `2D/`, and the reusable implementation lives under `compass2d/`.

## What’s included in this release (2D)

- `2D/run_cp_L.py`: **logits-only** runner (recommended; no Jacobians, much cheaper)
- `2D/run_cp.py`: full runner (includes Jacobian-based COMPASS-J methods)
- `2D/run_jactime.py`: Jacobian timing utility
- `2D/datasets/<dataset>/`: dataset-specific preprocessing + training helpers
- `compass2d/`: reusable 2D code (loaders, caching, methods)
- `compass_paths.py`: results directory helper (writes under `COMPASS_RESULTS_DIR`)

Note: this release intentionally does **not** include figure-generation / plotting scripts.

## Install

1) Create a Python environment.
2) Install requirements:

```bash
pip install -r requirements-2d.txt
```

Notes:
- Install **PyTorch** according to your CUDA setup.
- Most runs assume a GPU.

## Results location

All experiment outputs are written under:

- `COMPASS_RESULTS_DIR` (default: `/scratch/yc130/compass_results`)

Set it explicitly if desired:

```bash
export COMPASS_RESULTS_DIR=/scratch/yc130/compass_results
```

## Dataset layout (required)

The 2D runners expect each dataset to be laid out as:

- `imagesTr/`, `labelsTr/` (train)
- `imagesVa/`, `labelsVa/` (calibration/validation)
- `imagesTs/`, `labelsTs/` (test)

and to include trained checkpoints at:

- `models/<MODEL_NAME>/best_model.pth` (baseline segmentation model)
- `models/<QR_MODEL_NAME>/best_model.pth` (3-head “QR/CQR” segmentation model, `out_channels=3`)

Default dataset roots in the runners are hardcoded under `/scratch/...` (author environment). Update `DATASET_ROOTS` in:

- `2D/run_cp.py`
- `2D/run_cp_L.py`

## Preprocessing / splitting datasets

The scripts under `2D/datasets/<dataset>/` include lightweight preprocessing helpers (e.g. `split_dataset.py`).

They are **dataset-specific** and usually assume a particular on-disk raw layout and a `/scratch/...` path.
Before running them, open the script and edit `base_path` / `root_dir` / raw folder names as needed.

Example:

```bash
python 2D/datasets/kvasir/split_dataset.py
```

## Training models (baseline “NN” + 3-head “CQR/QR”)

The CP runners require two trained segmentation models per (dataset, architecture):

1) A standard single-head segmentation model (`out_channels=1`).
2) A 3-head quantile segmentation model (`out_channels=3`) used by the `E2E-CQR` baseline.

### Train baseline segmentation model (“NN”)

Run the dataset-specific script:

```bash
python 2D/datasets/<dataset>/run_nn.py
```

You typically need to edit at least:

- `data_dir` / `root_dir` (dataset location)
- `device`
- `arch` (`UNet` or `SegResNet`)

The script saves checkpoints under:

- `<DATASET_ROOT>/models/<ARCH>_<TIMESTAMP>/best_model.pth`

### Train 3-head quantile model (“CQR/QR model”)

Run:

```bash
python 2D/datasets/<dataset>/run_qrnn.py
```

You typically need to edit:

- `data_dir` / `root_dir`
- `device`
- `arch` (`QRUNet` or `QRSegResNet`)

The script saves:

- `<DATASET_ROOT>/models/<QRARCH>_<TIMESTAMP>/best_model.pth`

### Point the CP runners to your trained checkpoints

The runners select checkpoints via dictionaries inside the scripts:

- `model_names[dataset][arch]`
- `qr_model_names[dataset][arch]`

After training, update those entries to match the directory names you produced (e.g. `UNet_20260101_120000`).

## Run COMPASS 2D

### Recommended: logits-only (`COMPASS-L` / `A-COMPASS-L`)

Edit `config = {...}` near the top of `2D/run_cp_L.py`:

- `dataset`, `arch`, `alpha`, `device`
- set `generate_vars=True` for the first run (to cache variables)

Run:

```bash
python 2D/run_cp_L.py
```

This runner:

- caches **no Jacobians**
- evaluates baselines (`SCP`, `CQR`, `Local`, `E2E-CQR`) and logits methods (`COMPASS-L`, `A-COMPASS-L`)

### Full run (includes Jacobians + COMPASS-J)

If you want Jacobian-based methods, use:

```bash
python 2D/run_cp.py
```

## Outputs (what gets saved)

Raw per-test-point intervals are saved as pickles under:

- `COMPASS_RESULTS_DIR/2D/raw_<alpha>/raw_<dataset>_<method>_<rep>.pkl`

Each pickle is typically `[arr, runtime_s]` where `arr` is shaped `(3, N)` as `(hi, lo, gt)`.

This release does **not** ship summarization/plotting scripts; you can load a file with:

```python
import pickle

arr, runtime_s = pickle.load(open(path, "rb"))
hi, lo, gt = arr
```

## Jacobian computation warnings (read this first)

Some methods use Jacobian-based features (saved under `J/` in cached variables) and require autograd.

Practical limitations:

- **GPU memory / runtime:** Jacobian extraction performs backprop per sample and can be much slower than logits-only. OOM is common unless you use batch size 1 and/or reduced resolution.
- **Model dependence:** Jacobians are checkpoint-dependent. Changing seeds/training/preprocessing/checkpoints can materially change Jacobians and downstream CP behavior.

Recommendation:

- Start with **logits-only** (`2D/run_cp_L.py`) first.

## Limitations and failure modes (important)

Current COMPASS formulations are **model-dependent**: performance is tightly coupled to the segmentation model’s calibration and confidence structure.

Common ways performance can deteriorate:

- **Overconfident models:** intervals can become too tight and under-cover; in such cases, **SCP can be much better**.
- **Checkpoint sensitivity:** results vary across architectures, seeds, recipes, checkpoints.
- **Scalarization loss:** scalar proxies (like `mask_sum`) discard shape information.

Methods to overcome these limitations are left for **future work**.

## Adding a new architecture

1) Add it to `2D/run_cp.py` and `2D/run_cp_L.py` (import, configs, checkpoint names).
2) Add forward hooks in `compass2d/model_fns.py` via `get_fns(arch, post_trans_diff)`.

## Adding a new metric (scalar target)

1) Implement `post_trans_<metric>` and `post_trans_<metric>_diff` in `compass2d/model_fns.py`.
2) Register it in `get_post_transforms` in the runner scripts.

## Adding datasets

Follow `docs/adding_datasets_2d.md`.
