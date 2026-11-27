"""C6: screen features, exclude protected attributes, assemble the model matrix.

    python scripts/smoke_c06.py               # on Kaggle; reuses /kaggle/working/c05 or rebuilds it
    python scripts/smoke_c06.py --synthetic   # anywhere, against generated fixtures

Writes to <out> (/kaggle/working/c06 on Kaggle):
    model_matrix.parquet   applicant-level: stays on Kaggle (C7 versions it as a Dataset)
    slices.parquet         protected attributes for slice analysis only; stays on Kaggle
and, zipped as c06_results.zip (aggregates only):
    feature_registry.csv   every candidate feature: family, source, suffix, stat, status and evidence
    category_levels.json   the training levels each category feature keeps
    c06_summary.json       counts by status and family, matrix shape, checks

Pass condition (REFERENCE.md C6): every matrix feature is traced to a table and
source in the registry, no protected or proxy source reaches the matrix, and the
matrix has exactly one row per base case.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import polars as pl

from fcr.config import load_config
from fcr.data.build import ensure_features
from fcr.data.loader import discover_families, load_base
from fcr.data.paths import locate_competition_dir
from fcr.data.synthetic import SyntheticSpec, generate_synthetic_dataset
from fcr.features.matrix import KEYS, build_model_matrix
from fcr.runinfo import git_commit, peak_rss_gb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    started = time.perf_counter()

    if args.synthetic:
        root = Path(tempfile.mkdtemp(prefix="fcr_smoke_c06_"))
        generate_synthetic_dataset(root, SyntheticSpec(seed=cfg.project.seed))
        work = root
        cfg = cfg.model_copy(update={"features": cfg.features.model_copy(update={"corr_sample_cases": 1500, "min_minority_rows": 20})})
    else:
        root = locate_competition_dir(cfg)
        if root is None:
            print(f"competition data not found (configured: {cfg.paths.kaggle_competition_dir})")
            return 1
        work = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(tempfile.mkdtemp())
    print(f"mode: {'SYNTHETIC' if args.synthetic else 'REAL'}   commit {git_commit()}   data {root}")

    families = discover_families(root / "parquet_files" / "train")
    base = load_base(families)
    manifest = ensure_features(families, base, cfg, work / "c05" / "features", work / "c05" / "feature_manifest.csv")

    out = work / "c06"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    result = build_model_matrix(work / "c05" / "features", manifest, base, cfg)
    reg = pl.DataFrame(result.registry)
    failures: list[str] = []

    print("\n== Screening (statistics on training weeks only)")
    for status, n in Counter(reg["status"].to_list()).most_common():
        print(f"  {status:<22} {n:>5}")
    print(f"  {'candidates':<22} {reg.height:>5}")

    print("\n== Kept by family")
    by_family = reg.group_by("family").agg(pl.len().alias("candidates"), (pl.col("status") == "kept").sum().alias("kept")).sort("family")
    for r in by_family.iter_rows(named=True):
        print(f"  {r['family']:<20} {r['kept']:>4} of {r['candidates']:>4}")

    print("\n== Exclusions")
    for r in reg.filter(pl.col("status").str.starts_with("excluded")).sort("status", "feature").iter_rows(named=True):
        print(f"  {r['status']:<20} {r['feature']}")

    print("\n== Largest correlated drops (sample)")
    for r in reg.filter(pl.col("status") == "dropped:correlated").sort("rho", descending=True).head(8).iter_rows(named=True):
        print(f"  {r['feature']:<44} ~ {r['correlated_with']:<44} rho {r['rho']:+.3f}")

    print("\n== Ratios and categories")
    for r in reg.filter(pl.col("stat") == "ratio").iter_rows(named=True):
        print(f"  {r['feature']:<30} {r['status']:<20} train missing {r['train_missing_share']:.1%}")
    levels = result.levels
    print(f"  {len(levels)} category features kept; levels per feature: "
          f"max {max((len(v) for v in levels.values()), default=0)}, "
          f"capped at {cfg.features.max_category_levels}: {sum(len(v) == cfg.features.max_category_levels for v in levels.values())}")

    # checks
    m = result.matrix
    features = [c for c in m.columns if c not in KEYS]
    kept = set(reg.filter(pl.col("status") == "kept")["feature"])
    if set(features) != kept:
        failures.append(f"matrix columns and kept registry rows differ by {len(set(features) ^ kept)}")
    excluded_sources = cfg.protected.excluded()
    leaked = reg.filter(pl.col("source").is_in(list(excluded_sources)) & pl.col("feature").is_in(features))
    if leaked.height:
        failures.append(f"protected or proxy features in the matrix: {leaked['feature'].to_list()}")
    if m.height != base.height or m["case_id"].n_unique() != m.height:
        failures.append(f"matrix has {m.height:,} rows for {base.height:,} cases")
    dangling = reg.filter((pl.col("status") == "dropped:correlated") & ~pl.col("correlated_with").is_in(list(kept)))
    if dangling.height:
        failures.append(f"{dangling.height} correlated drops point at a feature not in the matrix")
    untraced = reg.filter(pl.col("status") == "kept").filter(pl.col("family").is_null())
    if untraced.height:
        failures.append(f"{untraced.height} kept features without a family")

    m.write_parquet(out / "model_matrix.parquet")
    result.slices.write_parquet(out / "slices.parquet")
    reg.write_csv(out / "feature_registry.csv")
    (out / "category_levels.json").write_text(json.dumps(levels, indent=1))
    summary = {
        "commit": git_commit(), "matrix_rows": m.height, "matrix_features": len(features),
        "status_counts": dict(Counter(reg["status"].to_list())),
        "kept_by_family": {r["family"]: r["kept"] for r in by_family.iter_rows(named=True)},
        "slices_columns": result.slices.columns, "elapsed_s": round(time.perf_counter() - started, 1),
        "peak_memory_gb": round(peak_rss_gb(), 2), "failures": failures,
    }
    (out / "c06_summary.json").write_text(json.dumps(summary, indent=1))
    staging = work / "c06_results"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    for name in ("feature_registry.csv", "category_levels.json", "c06_summary.json"):
        shutil.copy(out / name, staging / name)
    bundle = shutil.make_archive(str(staging), "zip", root_dir=staging)
    shutil.rmtree(staging)

    size = (out / "model_matrix.parquet").stat().st_size / 1024**3
    print(f"\nmodel matrix: {m.height:,} cases x {len(features)} features ({size:.2f} GB parquet)   bundle: {bundle}")
    print(f"elapsed {summary['elapsed_s']}s   peak memory {summary['peak_memory_gb']} GB")
    print("\n== Result")
    if failures:
        print("SMOKE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
