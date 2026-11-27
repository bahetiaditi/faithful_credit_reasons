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
SHA = "PASTE_COMMIT_SHA"   # git rev-parse --short HEAD, after pushing

!rm -rf /tmp/fcr && git clone -q {REPO} /tmp/fcr && git -C /tmp/fcr checkout -q {SHA}
!pip install -q -e /tmp/fcr
!python /tmp/fcr/scripts/smoke_c01.py
```

Why clone and install in editable mode, rather than `pip install git+...`:
a plain pip install copies only the Python package, so `configs/` and
`scripts/` would be missing. The clone keeps the whole repo, and the editable
install lets the config loader find `configs/config.yaml` from the package
location.

Why `pyproject.toml` has only lower bounds: Kaggle preinstalls numpy, pandas,
scikit-learn, LightGBM and others. Exact pins would make pip replace them.
Lower bounds let pip keep Kaggle's versions and install only what is missing.
Each smoke run prints the versions it actually used.

Scripts run with `!python` start a fresh process, so the new install is
visible immediately. To `import fcr` inside the notebook's own kernel, add
`import sys; sys.path.insert(0, "/tmp/fcr/src")` first.
