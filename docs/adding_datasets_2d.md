# Adding a new 2D dataset

This repo’s 2D pipeline expects **(image, mask)** pairs and derives a scalar target (e.g., mask area) used by CP methods.

At a high level you need to:

1) Implement a loader (`torch.utils.data.Dataset`).
2) Register the loader in the experiment script(s).
3) Provide a trained baseline model and a trained 3-head “QR/CQR” model.
4) (Optional) cache intermediate variables to disk for repeatable evaluations.

## 1) Implement a loader

Add a new `Dataset` class to `compass2d/dataloaders.py`.

Your loader should:

- return `(image_tensor, mask_tensor)` where `mask_tensor` is binary (0/1)
- support a `size` argument (the existing loaders resize to `(size, size)`)
- optionally support augmentation when `transform=True` (train split)

Tip: copy the pattern used by `ebhi` or `ham10k` in `compass2d/dataloaders.py`.

## 2) Register the dataset

In `2D/run_cp.py` (and `2D/run_cp_L.py`), datasets are selected via:

- the `dataset_loaders` mapping
- the `DATASET_ROOTS` mapping (dataset key → on-disk root directory)

Add your dataset key (e.g. `"mydataset"`) to `dataset_loaders`, and update `DATASET_ROOTS` to point to your local data path.

Note: this release currently allows only `ebhi`, `ham10k`, `kvasir`, and `tn3k` by default.

## 3) Add model configs and checkpoint names

`2D/run_cp.py` / `2D/run_cp_L.py` define:

- `model_configs[dataset][arch]` (single-head model)
- `qr_model_configs[dataset][arch]` (3-head QR/CQR model)

Checkpoints are expected at:

- `<root_dir>/models/<model_name>/best_model.pth`
- `<root_dir>/models/<qr_model_name>/best_model.pth`

so you also need to add:

- `model_names[dataset][arch]`
- `qr_model_names[dataset][arch]`

## 4) Run variable extraction (optional but recommended)

Most methods operate on cached variables saved by `DataExtractionWrapperSave` / `QRDataExtractionWrapperSave` in `compass2d/data_utils.py`.

Set `config['generate_vars']=True` in the runner to generate cached variables under:

- `<root_dir>/data/<model_name>/<metric>/` (full runner)
- `<root_dir>/data/<model_name>_Logits/<metric>/` (logits-only runner)

### Jacobian warning

If you enable Jacobian caching (via `2D/run_cp.py`), it can be memory-heavy and slow.
If you want a cheaper starting point, use `2D/run_cp_L.py` (logits-only; no Jacobians).

## 5) Verify end-to-end on a tiny run

Start with:

- `evaluation_runs = 1`
- a small `size` (e.g. 128)
- `num_workers = 0` to simplify debugging

Then confirm raw interval pickles are written under:

- `COMPASS_RESULTS_DIR/2D/raw_<alpha>/`
