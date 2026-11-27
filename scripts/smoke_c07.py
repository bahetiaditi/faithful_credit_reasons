"""C7: freeze the out-of-time split and write the versioned feature dataset.

    python scripts/smoke_c07.py --clean-working      # on Kaggle (Save & Run All), see notebooks/README.md
    python scripts/smoke_c07.py --synthetic          # anywhere, against generated fixtures
    python scripts/smoke_c07.py --local [--bundle ~/Downloads/fcr_sample.zip]
                                                     # on the Mac: install and verify the 50k sample

Kaggle mode writes /kaggle/working/fcr_features/ (the directory that becomes the
private Kaggle Dataset `fcr-features`) and two zips next to it:
    c07_results.zip   aggregates only: split summary, split lock, sample strata, manifest, feature registry
    fcr_sample.zip    the 50k stratified sample for local work (applicant-level: data/ only, never git)

Pass condition (REFERENCE.md C7): the split matches Section 4.5 exactly, the
dataset reloads with every hash, column list and split check passing, and the
sample keeps each split's size and default rate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

from fcr.config import load_config, resolve_path
from fcr.data.build import ensure_features
from fcr.data.dataset import (
    LEVELS, MANIFEST, committed_lock, config_fingerprint, install_sample_bundle, load_feature_table,
    write_feature_dataset, write_sample_bundle,
)
from fcr.data.loader import discover_families, load_base
from fcr.data.paths import locate_competition_dir
from fcr.data.split import SPLITS, assign_split, build_lock, split_summary, write_lock
from fcr.data.synthetic import SyntheticSpec, generate_synthetic_dataset
from fcr.features.matrix import build_model_matrix
from fcr.runinfo import git_commit, peak_rss_gb

# Section 4.5, measured in C4. The real split must reproduce these exactly.
EXPECTED = {"train": (1_282_082, 42_524), "validation": (92_870, 2_255), "test": (151_707, 3_215)}


def _print_split(summary: pl.DataFrame, title: str) -> None:
    print(f"\n== {title}")
    print(f"  {'split':<11}{'weeks':>8}{'decisions':>26}{'cases':>12}{'defaults':>10}{'rate':>8}")
    for r in summary.iter_rows(named=True):
        print(f"  {r['split']:<11}{r['first_week']:>3}-{r['last_week']:<4}"
              f"{r['first_decision'][:10]:>13} {r['last_decision'][:10]:>12}"
              f"{r['cases']:>12,}{r['defaults']:>10,}{r['default_rate']:>8.2%}")


def _sample_check(table, manifest: dict, failures: list[str]) -> None:
    full = manifest["split_lock"]["splits"]
    summary = split_summary(table.matrix)
    _print_split(summary, f"Sample ({table.matrix.height:,} cases, stratified by split x target)")
    for r in summary.iter_rows(named=True):
        f = full[r["split"]]
        gap = abs(r["default_rate"] - f["defaults"] / f["cases"])
        # largest-remainder allocation is off by at most one case per stratum
        if gap > 1.0 / r["cases"] + 1e-12:
            failures.append(f"sample {r['split']} default rate off by {gap:.3%}")
    expected = {(s["split"], s["target"]): s["cases"] for s in manifest["sample"]["strata"]}
    got = {(r["split"], r["target"]): r["len"] for r in table.matrix.group_by("split", "target").len().iter_rows(named=True)}
    if got != expected:
        failures.append("sample strata differ from the manifest")
    if table.slices.height != table.matrix.height:
        failures.append(f"sample has {table.matrix.height:,} cases but {table.slices.height:,} slice rows")


def run_local(cfg, bundle: Path | None) -> int:
    data_dir = resolve_path(cfg.paths.local_data_dir)
    if bundle:
        install_sample_bundle(bundle.expanduser(), data_dir)
        print(f"installed {bundle} into {data_dir}")
    if not (data_dir / MANIFEST).is_file():
        print(f"no sample in {data_dir}; pass --bundle with the fcr_sample.zip downloaded from Kaggle")
        return 1
    started = time.perf_counter()
    table = load_feature_table(data_dir, cfg, sample=True)
    m = table.manifest
    print(f"mode: LOCAL   commit {git_commit()}   sample built at {m['commit']} on {m['created_utc']}")
    print(f"hashes, feature list and split lock verified ({m['n_features']} features)")
    failures: list[str] = []
    if m["config_fingerprint"] != config_fingerprint():
        print("NOTE: configs/ changed since the dataset was built; rebuild it before modelling on changed features")
    _sample_check(table, m, failures)
    levels = json.loads((data_dir / LEVELS).read_text())
    for f, keep in levels.items():
        extra = set(table.matrix[f].drop_nulls().unique().to_list()) - set(keep) - {"__other__"}
        if extra:
            failures.append(f"{f} has values outside its training levels: {sorted(extra)[:3]}")
    print(f"\nloaded and checked in {time.perf_counter() - started:.1f}s")
    return _finish(failures)


def _finish(failures: list[str]) -> int:
    print("\n== Result")
    if failures:
        print("SMOKE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--local", action="store_true")
    parser.add_argument("--bundle", type=Path, help="with --local: the fcr_sample.zip to install into data/")
    parser.add_argument("--clean-working", action="store_true",
                        help="Kaggle: delete the C5 intermediates so the saved output holds only the dataset")
    args = parser.parse_args()
    cfg = load_config()
    if args.local:
        return run_local(cfg, args.bundle)

    started = time.perf_counter()
    if args.synthetic:
        root = Path(tempfile.mkdtemp(prefix="fcr_smoke_c07_"))
        generate_synthetic_dataset(root, SyntheticSpec(seed=cfg.project.seed))
        work, sample_cases = root, 500
        cfg = cfg.model_copy(update={"features": cfg.features.model_copy(
            update={"corr_sample_cases": 1500, "min_minority_rows": 20})})
    else:
        root = locate_competition_dir(cfg)
        if root is None:
            print(f"competition data not found (configured: {cfg.paths.kaggle_competition_dir})")
            return 1
        work = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(tempfile.mkdtemp())
        sample_cases = cfg.features.local_sample_cases
    commit = git_commit()
    print(f"mode: {'SYNTHETIC' if args.synthetic else 'REAL'}   commit {commit}   data {root}")
    failures: list[str] = []

    families = discover_families(root / "parquet_files" / "train")
    base = assign_split(load_base(families), cfg.split)
    summary = split_summary(base)
    _print_split(summary, "Out-of-time split (frozen here)")
    lock = build_lock(base, cfg.split)
    if not args.synthetic:
        for name, (cases, defaults) in EXPECTED.items():
            s = lock["splits"][name]
            if (s["cases"], s["defaults"]) != (cases, defaults):
                failures.append(f"{name}: {s['cases']:,} cases / {s['defaults']:,} defaults, Section 4.5 says {cases:,} / {defaults:,}")
        committed = committed_lock()
        if committed is None:
            print("\nno configs/split_lock.json yet: this run's lock is in c07_results.zip; commit it")
        elif committed != lock:
            failures.append("the split differs from the committed configs/split_lock.json")
        else:
            print("\nsplit matches the committed configs/split_lock.json")

    manifest_rows = ensure_features(families, base, cfg, work / "c05" / "features", work / "c05" / "feature_manifest.csv")
    result = build_model_matrix(work / "c05" / "features", manifest_rows, base, cfg)
    registry = pl.DataFrame(result.registry)
    matrix = result.matrix.join(base.select("case_id", "split"), on="case_id", how="left")
    print(f"\nmodel matrix: {matrix.height:,} cases, kept {(registry['status'] == 'kept').sum()} "
          f"of {registry.height} candidate features")

    dataset_dir = work / "fcr_features"
    shutil.rmtree(dataset_dir, ignore_errors=True)
    manifest = write_feature_dataset(dataset_dir, matrix, result.slices, registry, result.levels, lock, cfg,
                                     commit=commit, sample_cases=sample_cases)

    print("\n== Dataset files")
    for name, f in manifest["files"].items():
        rows = f"{f['rows']:>11,} rows" if f["rows"] is not None else " " * 16
        print(f"  {name:<24}{f['bytes'] / 1024**2:>9.1f} MB {rows}   sha256 {f['sha256'][:12]}")

    # reload exactly as later commits will
    expected_lock = None if args.synthetic else "committed"
    full = load_feature_table(dataset_dir, cfg, sample=False, expected_lock=expected_lock)
    sample = load_feature_table(dataset_dir, cfg, sample=True, expected_lock=expected_lock)
    print("\nreloaded: hashes, feature list and split lock verified for the full table and the sample")
    if full.matrix.height != base.height:
        failures.append(f"dataset has {full.matrix.height:,} cases, base has {base.height:,}")
    _sample_check(sample, manifest, failures)

    results = work / "c07_results"
    shutil.rmtree(results, ignore_errors=True)
    results.mkdir()
    summary.write_csv(results / "split_summary.csv")
    write_lock(lock, results / "split_lock.json")
    sample.matrix.group_by("split", "target").len().sort("split", "target").write_csv(results / "sample_strata.csv")
    shutil.copy(dataset_dir / MANIFEST, results / MANIFEST)
    registry.write_csv(results / "feature_registry.csv")
    (results / "c07_summary.json").write_text(json.dumps({
        "commit": commit, "n_features": manifest["n_features"], "cases": full.matrix.height,
        "sample_cases": sample.matrix.height, "status_counts": registry.group_by("status").len()
        .sort("status").to_dicts(), "elapsed_s": round(time.perf_counter() - started, 1),
        "peak_memory_gb": round(peak_rss_gb(), 2), "failures": failures}, indent=1))
    bundle = shutil.make_archive(str(results), "zip", root_dir=results)
    shutil.rmtree(results)
    sample_bundle = write_sample_bundle(dataset_dir, work / "fcr_sample.zip")
    if args.clean_working:
        shutil.rmtree(work / "c05", ignore_errors=True)
        shutil.rmtree(work / "c06", ignore_errors=True)
        for stale in ("c05_results.zip", "c06_results.zip"):
            (work / stale).unlink(missing_ok=True)
        print("\ncleaned /kaggle/working: only fcr_features/ and the two zips remain")

    print(f"\nbundles: {bundle} (aggregates)   {sample_bundle} "
          f"({Path(sample_bundle).stat().st_size / 1024**2:.1f} MB, applicant-level: data/ only)")
    print(f"elapsed {time.perf_counter() - started:.1f}s   peak memory {peak_rss_gb():.2f} GB")
    return _finish(failures)


if __name__ == "__main__":
    sys.exit(main())
