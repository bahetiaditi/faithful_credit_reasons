"""C1 smoke run: is this environment ready for the project?

Runs the same way on the MacBook and on Kaggle:

    python scripts/smoke_c01.py

It checks, in order: interpreter and package versions, that every heavy
dependency imports (this is where a missing libomp breaks LightGBM on macOS),
that the config loads and validates, and that the synthetic dataset generates
and reads back. On Kaggle it also reports the machine (CPU, RAM, disk) and
whether the competition data is attached, which C3 needs to size the loader.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HEAVY_DEPENDENCIES = [
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "scipy",
    "sklearn",
    "statsmodels",
    "lightgbm",
    "optbinning",
    "shap",
    "optuna",
    "pydantic",
    "textstat",
    "matplotlib",
    "google.genai",
    "groq",
]


def _section(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 60 - len(title)))


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


def _total_ram_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


def _on_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").is_dir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=2000, help="synthetic cases to generate")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    _section("Environment")
    print(f"python      {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    print(f"repo commit {_git_sha(repo)}")
    import fcr

    print(f"fcr         {fcr.__version__} from {Path(fcr.__file__).parent}")
    ram = _total_ram_gb()
    print(f"cpu cores   {os.cpu_count()}")
    print(f"ram total   {ram:.1f} GB" if ram else "ram total   unknown")
    print(f"on kaggle   {_on_kaggle()}")

    _section("Dependencies")
    for name in HEAVY_DEPENDENCIES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "n/a")
            print(f"ok    {name:<12} {version}")
        except Exception as exc:  # report every failure, not just the first
            failures.append(f"import {name}: {exc}")
            print(f"FAIL  {name:<12} {type(exc).__name__}: {exc}")

    _section("Config")
    from fcr.config import find_config_path, load_config

    cfg = load_config()
    print(f"loaded      {find_config_path()}")
    for split in ("train", "validation", "test"):
        lo, hi = getattr(cfg.split, f"{split}_weeks")
        print(f"{split:<11} weeks {lo}-{hi} ({hi - lo + 1} weeks)")
    print(f"max reasons {cfg.explain.max_reasons}; approval-rate target {cfg.model.target_approval_rate}")

    _section("Synthetic dataset")
    import polars as pl

    from fcr.data.naming import parse_table_name
    from fcr.data.synthetic import SyntheticSpec, generate_synthetic_dataset

    workdir = Path(tempfile.mkdtemp(prefix="fcr_smoke_"))
    started = time.perf_counter()
    ds = generate_synthetic_dataset(workdir, SyntheticSpec(n_cases=args.cases, seed=cfg.project.seed))
    print(f"generated   {args.cases} cases in {time.perf_counter() - started:.2f}s at {workdir}")
    for path in sorted(ds.train_dir.glob("*.parquet")):
        t = parse_table_name(path.stem)
        df = pl.read_parquet(path)
        print(f"  {path.stem:<28} depth {t.depth}  rows {df.height:>6}  cols {df.width}")
    print(f"default rate {ds.manifest['observed_default_rate']:.3f} (spec {ds.manifest['spec']['default_rate']})")
    print(f"planted leakage rows {ds.manifest['leakage']['n_rows_after_decision']}")
    shutil.rmtree(workdir, ignore_errors=True)

    if _on_kaggle():
        _section("Kaggle")
        usage = shutil.disk_usage("/kaggle/working")
        print(f"/kaggle/working free {usage.free / 1024**3:.1f} GB of {usage.total / 1024**3:.1f} GB")
        comp = cfg.paths.kaggle_competition_dir
        if comp.is_dir():
            files = [p for p in comp.rglob("*") if p.is_file()]
            size_gb = sum(p.stat().st_size for p in files) / 1024**3
            parquet = sorted(p.stem for p in comp.rglob("parquet_files/train/*.parquet"))
            print(f"competition data attached: {len(files)} files, {size_gb:.1f} GB")
            print(f"train parquet tables: {len(parquet)}")
            for stem in parquet:
                print(f"  {stem}")
        else:
            print(f"competition data NOT attached (expected at {comp}); needed from C3")

    _section("Result")
    if failures:
        print("SMOKE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
