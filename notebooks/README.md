# Kaggle driver notebooks

Notebooks here are thin drivers. All logic lives in the `fcr` package; a
notebook only installs the repo at a pinned commit and runs a script, so every
Kaggle result can be traced back to exact code. A driver is committed when its
phase finishes.

## One-time Kaggle setup

1. Open the [competition page](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)
   and accept its rules; the data cannot be attached to a notebook until you do.
2. Phone-verify your Kaggle account (Settings), which is required for internet access.
3. New notebook, then in the right panel: **Session options → Internet: On**,
   and **Add Input → Competitions → Home Credit - Credit Risk Model Stability**.

## Driver cell (same pattern for every commit)

```python
REPO = "https://github.com/bahetiaditi/faithful_credit_reasons.git"
SHA = "main"

!rm -rf /tmp/fcr && git clone -q {REPO} /tmp/fcr && git -C /tmp/fcr checkout -q {SHA}
!pip install -q -r /tmp/fcr/requirements.txt
!pip install -q -e /tmp/fcr --no-deps
!cd /tmp/fcr && python -m pytest -q
!python /tmp/fcr/scripts/smoke_cNN.py
```

While iterating, `SHA = "main"` runs the latest push, and every smoke script
prints the commit it actually ran, so the output stays traceable. A notebook
committed to this folder at the end of a phase pins an exact SHA instead
(`git rev-parse --short HEAD`), so it re-runs identically later. Replace
`smoke_cNN.py` with the commit's own smoke script. Keep each notebook cell to
exactly these lines; pasting a new cell onto the end of an old one joins the
last line of one to the first of the next.

**Why install the pinned lock on Kaggle.** In C1 a loose install kept Kaggle's
own versions (polars 1.35 against 2.2 locally, pandas 2.3 against 3.1), so code
tested on the Mac would have run on different libraries on Kaggle. Installing
`requirements.txt` gives both machines identical versions, and running pytest
on Kaggle confirms it. The lock installs and passes on Kaggle's Python 3.13.

**Expected noise.** pip prints dependency-conflict errors for Kaggle packages
this project never imports (ydata-profiling, google-colab, torch, gradio and
others), and pandas may warn about Kaggle's older `bottleneck`. Neither affects
this project; what matters is that pytest passes and the smoke run ends
`SMOKE PASSED`.

**Why clone instead of `pip install git+...`.** A plain pip install copies
only the Python package, so `configs/` and `scripts/` would be missing. The
editable install also lets the config loader find `configs/config.yaml` from
the package location.

**Where the data is.** Kaggle mounts the competition at
`/kaggle/input/competitions/home-credit-credit-risk-model-stability`.
`fcr.data.paths.locate_competition_dir` also checks older layouts, and the
smoke output warns if the data turns up somewhere other than the configured
path.

Scripts run with `!python` start a fresh process, so the new install is
visible immediately. To `import fcr` inside the notebook's own kernel, add
`import sys; sys.path.insert(0, "/tmp/fcr/src")` first.

## Bringing results back

Scripts that produce results write them under `/kaggle/working/cNN/` and zip
them to `/kaggle/working/cNN_results.zip`. Download the zip from the
notebook's Output panel, then from the repo root:

```bash
mkdir -p results/cNN
unzip -o ~/Downloads/cNN_results.zip -d results/cNN
```

Only aggregates (counts, shares, column names, figures) are written there.
The one exception is the 50k-case sample from C7 (next section), which goes into
`data/` for local development and is never committed; `data/*` is gitignored.

## The feature dataset (C7 onwards)

From C8 on, notebooks read the feature table from a private Kaggle Dataset
instead of rebuilding it from the raw competition files (about 6 minutes and
16 GB each time). C7 builds it once:

1. End the driver cell with two lines instead of one:

   ```python
   !python /tmp/fcr/scripts/smoke_c05.py
   !python /tmp/fcr/scripts/smoke_c07.py --clean-working
   ```

   C5 runs in its own process first, so its memory (peak 18 GB) is released
   before screening starts (peak 16 GB) on Kaggle's 31 GB. `--clean-working`
   then deletes the C5 intermediates, so the output holds only
   `fcr_features/` and two zips.
2. **Save Version → Save & Run All (Commit).** An interactive session's files
   cannot become a Dataset; a saved version's output can. The run takes about
   10 minutes (install, tests, C5 rebuild, screening, writing).
3. Open the saved version, go to **Output**, and download `c07_results.zip`
   (aggregates) and `fcr_sample.zip` (the 50k sample).
4. On the same Output page choose **New Dataset**, name it `fcr-features`,
   keep it **Private**. A later rebuild becomes a new version of this Dataset.
5. In later notebooks: **Add Input → Your Datasets → fcr-features**.
   `fcr.data.dataset.locate_feature_dataset` finds it wherever Kaggle mounts it,
   by the name inside its `dataset_manifest.json`.

To run a model script in the same Save & Run All (so a rebuild and its first
use cost one run), point it at the fresh output instead of the attached Dataset:
`!FCR_FEATURES_DIR=/kaggle/working/fcr_features python /tmp/fcr/scripts/smoke_cNN.py`.

Every load re-checks the files' hashes against the manifest, the feature
columns against its feature list, and the split against
`configs/split_lock.json`, so a model cannot train on a table that changed.

On the Mac, install the sample and commit the lock:

```bash
unzip -o -j ~/Downloads/c07_results.zip split_lock.json -d configs
python scripts/smoke_c07.py --local --bundle ~/Downloads/fcr_sample.zip
python -m pytest -q
```
