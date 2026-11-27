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

Only aggregates (counts, shares, column names, figures) are written there;
applicant-level data never leaves Kaggle.
