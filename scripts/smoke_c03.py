"""C3 smoke run: load the real competition tables and describe them.

    python scripts/smoke_c03.py               # on Kaggle, against the attached competition data
    python scripts/smoke_c03.py --synthetic   # anywhere, against generated fixtures

Pass condition (REFERENCE.md C3): the base table reproduces 1,526,659 rows and
47,994 defaults, and every family scans. Everything else is reported as a
finding for C4 (EDA) and C5 (aggregation): duplicate keys, case_ids missing
from base, chunks that disagree on a column's type, columns that break the
suffix convention, and columns without a definition.

Writes the full schema registry as JSON (to /kaggle/working on Kaggle). It
holds column names and types only, no data, and is committed in C4.
"""

from __future__ import annotations

import argparse
import resource
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import polars as pl

from fcr.config import load_config, resolve_path
from fcr.data.loader import BASE_FAMILY, build_schema_registry, discover_families, load_base, write_registry
from fcr.data.paths import locate_competition_dir
from fcr.data.synthetic import SyntheticSpec, generate_synthetic_dataset
from fcr.runinfo import git_commit

# From the public ETL write-up cited in REFERENCE.md Section 4; C3 re-derives them.
EXPECTED_BASE_ROWS = 1_526_659
EXPECTED_DEFAULTS = 47_994


def _peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak / (1024**3 if sys.platform == "darwin" else 1024**2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true", help="run on generated fixtures instead of real data")
    args = parser.parse_args()
    cfg = load_config()
    failures: list[str] = []
    warnings: list[str] = []

    if args.synthetic:
        root = Path(tempfile.mkdtemp(prefix="fcr_smoke_c03_"))
        generate_synthetic_dataset(root, SyntheticSpec(seed=cfg.project.seed))
        out_path = root / "schema_registry.json"
        print(f"mode: SYNTHETIC at {root}")
    else:
        root = locate_competition_dir(cfg)
        if root is None:
            print(f"competition data not found (configured: {cfg.paths.kaggle_competition_dir})")
            return 1
        kaggle_out = Path("/kaggle/working")
        out_path = (kaggle_out if kaggle_out.is_dir() else resolve_path(cfg.paths.results_dir)) / "schema_registry.json"
        print(f"mode: REAL at {root}")
    print(f"commit {git_commit()}")
    table_dir = root / "parquet_files" / "train"

    print("\n== Families")
    families = discover_families(table_dir)
    for key, fam in families.items():
        print(f"  {key:<22} depth {fam.depth}  chunks {len(fam.files)}")

    print("\n== Base table")
    started = time.perf_counter()
    base = load_base(families)
    defaults = int(base["target"].sum())
    dates = base.select(pl.col("date_decision").cast(pl.String).min().alias("lo"),
                        pl.col("date_decision").cast(pl.String).max().alias("hi")).row(0)
    print(f"rows {base.height:,}   defaults {defaults:,}   rate {defaults / base.height:.4f}")
    print(f"WEEK_NUM {base['WEEK_NUM'].min()}-{base['WEEK_NUM'].max()}   date_decision {dates[0]} to {dates[1]}")
    print(f"dtypes: " + ", ".join(f"{c}={t}" for c, t in base.schema.items()))
    if not args.synthetic:
        if base.height != EXPECTED_BASE_ROWS:
            failures.append(f"base rows {base.height:,} != expected {EXPECTED_BASE_ROWS:,}")
        if defaults != EXPECTED_DEFAULTS:
            failures.append(f"defaults {defaults:,} != expected {EXPECTED_DEFAULTS:,}")
    del base

    print("\n== Schema registry (reads key columns and parquet metadata only)")
    registry = build_schema_registry(table_dir, root / "feature_definitions.csv")
    elapsed = time.perf_counter() - started
    header = f"  {'family':<22}{'rows':>12}{'cases':>11}{'cols':>6}{'dupkeys':>9}{'orphans':>9}{'typeconf':>9}{'badname':>8}{'nodef':>6}"
    print(header)
    for key, e in registry["families"].items():
        orphans = "-" if e["orphan_cases"] is None else f"{e['orphan_cases']:,}"
        print(f"  {key:<22}{e['rows']:>12,}{e['cases']:>11,}{len(e['columns']):>6}{e['duplicate_keys']:>9,}"
              f"{orphans:>9}{len(e['dtype_conflicts']):>9}{len(e['suffix_errors']):>8}{len(e['undescribed']):>6}")
        if e["duplicate_keys"]:
            warnings.append(f"{key}: {e['duplicate_keys']:,} duplicate key rows")
        if e["orphan_cases"]:
            warnings.append(f"{key}: {e['orphan_cases']:,} case_ids not in base")
        if e["suffix_errors"]:
            warnings.append(f"{key}: columns breaking the suffix convention {e['suffix_errors'][:5]}")
        if e["dtype_conflicts"]:
            warnings.append(f"{key}: chunk type disagreements in {e['dtype_conflicts'][:5]}")

    print("\n== Columns by suffix and type (all families; drives C5's casting rules)")
    by_suffix: dict[str, Counter] = {}
    for e in registry["families"].values():
        for c in e["columns"]:
            if c["suffix"]:
                by_suffix.setdefault(c["suffix"], Counter())[c["dtype"]] += 1
    for suffix in sorted(by_suffix):
        total = sum(by_suffix[suffix].values())
        print(f"  {suffix}: {total:>4} columns  " + ", ".join(f"{t} {n}" for t, n in by_suffix[suffix].most_common()))
    n_features = sum(by_suffix[s].total() for s in by_suffix)
    n_undescribed = sum(len(e["undescribed"]) for e in registry["families"].values())
    print(f"  total feature columns {n_features:,}; without a definition {n_undescribed}")

    path = write_registry(registry, out_path)
    print(f"\nregistry written to {path} ({path.stat().st_size / 1024:.0f} KB)")
    print(f"elapsed {elapsed:.1f}s   peak memory {_peak_rss_gb():.2f} GB")

    print("\n== Findings for C4/C5")
    for w in warnings or ["none"]:
        print(f"  - {w}")
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
