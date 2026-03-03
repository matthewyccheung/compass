# COMPASS 2D (Segmentation → Scalar CP)

The 2D pipeline in this repository evaluates conformal prediction (CP) methods for **scalar targets derived from 2D segmentation masks**.

This repo currently ships with a single scalar target:

- `mask_sum`: sum of mask pixels (area proxy)

2D entrypoints live under `2D/` (e.g. `2D/run_cp.py`), and the reusable implementation lives under `compass2d/`.

## Run 2D experiments (requires dataset + trained models)

### 1) Prepare a dataset folder

Supported built-in dataset keys:

- `ebhi`
- `ham10k`
- `kvasir`
- `tn3k`

Dataset-specific preprocessing/training helpers (used to create the expected on-disk layout and checkpoints) live under `2D/datasets/<dataset>/`.

Most loaders in `compass2d/dataloaders.py` assume you have **images** and **labels** split into:

- `imagesTr/`, `labelsTr/` (train)
- `imagesVa/`, `labelsVa/` (calibration/validation)
- `imagesTs/`, `labelsTs/` (test)

The default scripts are written for the original author’s filesystem and may set `root_dir` to paths under `/scratch/...`.
When running locally, update the `root_dir` selection logic in `2D/run_cp.py` (and related scripts) to point to your dataset location.
In the cleaned repo layout, these scripts are under `2D/` (e.g. `2D/run_cp.py`).

### 2) Provide trained model weights

The experiment scripts expect a dataset root that contains:

- `models/<MODEL_NAME>/best_model.pth` (baseline segmentation model)
- `models/<QR_MODEL_NAME>/best_model.pth` (3-head “QR” model with `out_channels=3`)

Model names are currently selected from dictionaries in `2D/run_cp.py`:

- `model_names[dataset][arch]`
- `qr_model_names[dataset][arch]`

If you add your own models, update these mappings (or add a new dataset entry).

### 3) Extract cached variables (optional but typical)

`2D/run_cp.py` can cache intermediate variables (latents, predictions, jacobians, scalar targets) into:

`<DATASET_ROOT>/data/<MODEL_NAME>/<METRIC>/`

Set in `2D/run_cp.py`:

- `config['generate_vars'] = True` (first time)

### 4) Run evaluations and write raw interval outputs

`2D/run_cp.py` evaluates multiple CP methods and (when enabled) writes per-test-point raw outputs under `COMPASS_RESULTS_DIR/2D/`:

- `COMPASS_RESULTS_DIR/2D/raw_<alpha>/raw_<dataset>_<method>_<rep>.pkl`

These `.pkl` files store arrays of `(hi, lo, gt)` and can be summarized by:

```bash
python scripts/figures/summarize_2d_raw_results.py --results_root "$COMPASS_RESULTS_DIR/2D" --out_csv "$COMPASS_RESULTS_DIR/2D/summary_2d.csv"
```

## Common configuration knobs

In `2D/run_cp.py`, key settings are in the `config = {...}` dictionary near the top:

- `dataset`: dataset key used in loader/model mappings
- `arch`: `UNet` or `SegResNet`
- `metric`: scalar target (default: `mask_sum`)
- `alpha`: miscoverage rate
- `device` / `device2`: GPU devices for baseline vs QR model

## Jacobian computation warnings (read this first)

Some methods in this repo use **Jacobian-based features** (saved under `J/` in extracted variables) and require computing gradients via autograd.

Practical limitations:

- **GPU memory / runtime:** Jacobian extraction performs backprop through the model per sample and can be **~2–3× slower** than logits-only in typical runs. Memory spikes are common; expect to run with small batch sizes (often `1`) and/or reduced resolution.
- **Model dependence:** Jacobians are properties of a *specific trained checkpoint*. If you change the architecture, training seed, data preprocessing, or checkpoint, Jacobians (and interval behavior) can change materially. Comparisons should be made **within the same trained model family**.
- **Approximation details:** Jacobians are taken with respect to a *differentiable scalar target* (e.g., `post_trans_mask_sum_diff` uses a temperature-scaled sigmoid). Different “soft” approximations can change gradients.

Recommendation:

- Start with **logits-based** methods first (they are substantially cheaper and often competitive). The script `2D/run_cp_boostlogits.py` is a logits-only evaluation path that avoids Jacobian extraction (`jacobians=False` during variable caching).
- In practice, logits-only runs often provide **most of the efficiency gains** versus baselines and can be competitive with Jacobian-based methods. Always validate on your dataset/checkpoint.

## Limitations and failure modes (important)

Current COMPASS formulations (2D and 3D) are **model-dependent**: performance is tightly coupled to the underlying segmentation model’s calibration and confidence structure.

Common ways performance can deteriorate:

- **Overconfident models:** If the segmentation model is systematically overconfident (e.g., very sharp logits but wrong), COMPASS-style perturbations can produce intervals that are too tight and under-cover. In these cases, **SCP can be much better** (more robust coverage at the cost of larger intervals).
- **Checkpoint sensitivity:** Results can vary materially across architectures, seeds, training recipes, or checkpoints even on the same dataset.
- **Scalarization loss:** Using a scalar proxy like `mask_sum` discards shape information; two masks with the same area can have very different segmentation quality.
- **Hyperparameter/search fragility:** Beta search thresholds/ranges can affect both runtime and the final interval, especially when logits saturate or when the target is insensitive to perturbations.
- **Finite-sample calibration:** With limited calibration data, all methods can exhibit high variance; interval length/coverage estimates can be noisy.

Addressing these limitations (e.g., better calibration, robustness to overconfidence, and more stable search/targets) is left for **future work**.

## Adding a new architecture

To add a new 2D architecture (example: `MyNet`):

1) Add it to `2D/run_cp.py`:
   - import the class
   - add it to `model_architectures`
   - add per-dataset entries in `model_configs` and `qr_model_configs`
   - add checkpoint names to `model_names` and `qr_model_names` (or refactor to your own naming)
2) Add Jacobian support in `compass2d/model_fns.py`:
   - update `get_fns(arch, post_trans_diff)` to return `forward_x`, `forward_latent`, and `compute_jacobian` for your architecture.

## Adding a new metric (scalar target)

This repo currently ships with a single scalar target:

- `mask_sum`: sum of mask pixels (area proxy)

To add a new metric:

1) Implement two functions in `compass2d/model_fns.py`:
   - `post_trans_<metric>(logits_or_mask, channel, ...)` (discrete version)
   - `post_trans_<metric>_diff(logits, channel, ...)` (differentiable version for Jacobians)
2) Register it in `2D/run_cp.py:get_post_transforms`.
3) If you use QR caching, ensure `compass2d/data_utils.py:QRDataExtractionWrapperSave` computes the correct target for your metric.

## Next: adding datasets

Follow `docs/adding_datasets_2d.md`.
