"""C4: exploratory analysis of the real training data.

    python scripts/smoke_c04.py               # on Kaggle, against the attached competition data
    python scripts/smoke_c04.py --synthetic   # anywhere, against generated fixtures

Produces, in one folder (/kaggle/working/c04 on Kaggle):
    eda_summary.json            headline facts, placeholder hashes, protected attributes, chunk containment
    schema_registry.json        C3's registry, rebuilt at this commit
    weekly_default_rate.csv     one row per WEEK_NUM
    missingness.csv             one row per feature column
    coverage_by_month.csv       one row per family x decision month
    date_checks.csv             one row per date column
    masked_profiles.json        top values of every masked (M) column
    figures/*.png               default rate, coverage, missingness
and zips the folder to c04_results.zip for download. Everything here is
column names, counts and shares; no applicant-level data leaves Kaggle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import polars as pl

from fcr.config import load_config
from fcr.data import eda
from fcr.data.loader import (
    BASE_FAMILY,
    build_schema_registry,
    discover_families,
    load_base,
    read_feature_definitions,
    write_registry,
)
from fcr.data.paths import locate_competition_dir
from fcr.data.synthetic import SyntheticSpec, generate_synthetic_dataset
from fcr.eval.figures import plot_coverage_heatmap, plot_missingness_histogram, plot_weekly_default_rate
from fcr.runinfo import git_commit, peak_rss_gb

PANDEMIC_DECLARED = date(2020, 3, 11)  # WHO declares COVID-19 a pandemic


def _section(title: str) -> None:
    print(f"\n== {title}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    started = time.perf_counter()

    if args.synthetic:
        root = Path(tempfile.mkdtemp(prefix="fcr_smoke_c04_"))
        generate_synthetic_dataset(root, SyntheticSpec(seed=cfg.project.seed))
        out = root / "c04"
    else:
        root = locate_competition_dir(cfg)
        if root is None:
            print(f"competition data not found (configured: {cfg.paths.kaggle_competition_dir})")
            return 1
        work = Path("/kaggle/working")
        out = (work if work.is_dir() else Path(tempfile.mkdtemp())) / "c04"
    shutil.rmtree(out, ignore_errors=True)
    (out / "figures").mkdir(parents=True)
    table_dir = root / "parquet_files" / "train"
    print(f"mode: {'SYNTHETIC' if args.synthetic else 'REAL'}   commit {git_commit()}   data {root}")

    families = discover_families(table_dir)
    definitions = read_feature_definitions(root / "feature_definitions.csv")
    base = load_base(families)
    split = cfg.split
    summary: dict = {"commit": git_commit(), "mode": "synthetic" if args.synthetic else "real"}

    # 1 ---------------------------------------------------------------------
    _section("1. Default rate by split and week")
    weekly = eda.weekly_default_rate(base)
    weekly.write_csv(out / "weekly_default_rate.csv")
    by_split = {}
    for name in ("train", "validation", "test"):
        lo, hi = getattr(split, f"{name}_weeks")
        part = weekly.filter(pl.col("WEEK_NUM").is_between(lo, hi))
        apps, defs = int(part["applications"].sum()), int(part["defaults"].sum())
        by_split[name] = {"weeks": [lo, hi], "applications": apps, "defaults": defs, "default_rate": defs / apps}
        print(f"  {name:<11} weeks {lo:>2}-{hi:<2}  applications {apps:>9,}  defaults {defs:>7,}  rate {defs / apps:.4f}")
    busiest, quietest = weekly.sort("applications").row(-1, named=True), weekly.sort("applications").row(0, named=True)
    print(f"  applications per week: min {quietest['applications']:,} (week {quietest['WEEK_NUM']}), "
          f"max {busiest['applications']:,} (week {busiest['WEEK_NUM']})")
    covid_week = eda.first_week_on_or_after(weekly, PANDEMIC_DECLARED)
    print(f"  first week on or after {PANDEMIC_DECLARED}: {covid_week}")
    summary["splits"] = by_split
    summary["pandemic_week"] = covid_week
    markers = {"validation": split.validation_weeks[0], "test": split.test_weeks[0]}
    if covid_week is not None:
        markers["pandemic declared"] = covid_week
    plot_weekly_default_rate(weekly, markers, out / "figures" / "weekly_default_rate.png")

    # 2 ---------------------------------------------------------------------
    _section("2. Missing values (row level, nulls plus empty strings)")
    miss = pl.concat([eda.column_missingness(f) for k, f in families.items() if k != BASE_FAMILY])
    miss.write_csv(out / "missingness.csv")
    threshold = cfg.features.max_missing_share
    bands = {
        "under 10%": int((miss["missing_share"] < 0.10).sum()),
        "10-50%": int(miss["missing_share"].is_between(0.10, 0.50, closed="left").sum()),
        "50-95%": int(miss["missing_share"].is_between(0.50, threshold, closed="left").sum()),
        f"over {threshold:.0%} (screened out)": int((miss["missing_share"] > threshold).sum()),
    }
    for band, n in bands.items():
        print(f"  {band:<28} {n:>4} columns")
    empties = miss.filter(pl.col("empty_strings") > 0).sort("empty_strings", descending=True)
    print(f"  columns using empty strings as missing: {empties.height}"
          + (f" (largest: {empties['family'][0]}.{empties['column'][0]}, {empties['empty_strings'][0]:,})" if empties.height else ""))
    summary["missingness_bands"] = bands
    summary["columns_with_empty_strings"] = empties.height
    plot_missingness_histogram(miss, threshold, out / "figures" / "missingness.png")

    # 3 ---------------------------------------------------------------------
    _section("3. Masked (M) columns and placeholder candidates")
    profiles = [p for f in families.values() for p in eda.masked_value_profiles(f)]
    (out / "masked_profiles.json").write_text(json.dumps(profiles, indent=1))
    placeholders = eda.placeholder_candidates(profiles)
    print(f"  {len(profiles)} masked columns; distinct values median "
          f"{sorted(p['distinct'] for p in profiles)[len(profiles) // 2] if profiles else 0}")
    for c in placeholders[:8]:
        print(f"  {c['value']!r:<16} most frequent value in {c['columns']:>3} columns "
              f"(median share {c['median_share']:.0%}), e.g. {', '.join(c['examples'][:3])}")
    summary["placeholder_candidates"] = placeholders

    # 4 ---------------------------------------------------------------------
    _section("4. Protected and sensitive attributes (name, definition or value match)")
    protected = eda.protected_inventory(families, definitions)
    for p in protected:
        values = ", ".join(str(v["value"]) for v in (p["values"] or [])[:8])
        print(f"  {p['family']}.{p['column']:<32} [{p['matched']}] {p['description'][:60]!r}"
              + (f"\n      values: {values}" if values else ""))
    summary["protected_inventory"] = protected
    summary["sex_or_gender_columns"] = [f"{p['family']}.{p['column']}" for p in protected
                                        if p["matched"] in ("sex", "gender")]
    print(f"  sex/gender columns: {summary['sex_or_gender_columns'] or 'none found'}")

    # 5 ---------------------------------------------------------------------
    _section("5. Coverage over time (share of applications with any record)")
    coverage = eda.coverage_by_month(families, base)
    coverage.write_csv(out / "coverage_by_month.csv")
    cov_stats = (
        coverage.group_by("family").agg(
            pl.col("coverage").mean().alias("mean"), pl.col("coverage").min().alias("min"),
            pl.col("coverage").max().alias("max"),
            pl.col("coverage").sort_by("MONTH").first().alias("first_month"),
            pl.col("coverage").sort_by("MONTH").last().alias("last_month"),
        ).sort("family")
    )
    print(f"  {'family':<22}{'mean':>7}{'min':>7}{'max':>7}{'first':>8}{'last':>7}")
    for r in cov_stats.iter_rows(named=True):
        print(f"  {r['family']:<22}{r['mean']:>7.2f}{r['min']:>7.2f}{r['max']:>7.2f}{r['first_month']:>8.2f}{r['last_month']:>7.2f}")
    summary["coverage"] = cov_stats.to_dicts()
    plot_coverage_heatmap(coverage, out / "figures" / "coverage_by_month.png")

    # 6 ---------------------------------------------------------------------
    _section("6. Chunk containment (can each chunk be aggregated on its own?)")
    containment = [eda.chunk_containment(f) for f in families.values() if len(f.files) > 1]
    for c in containment:
        verdict = "per-chunk aggregation is exact" if c["cases_spanning_chunks"] == 0 else "cases SPAN chunks"
        print(f"  {c['family']:<22} {c['chunks']:>2} chunks, {c['cases_spanning_chunks']:,} cases spanning -> {verdict}")
    summary["chunk_containment"] = containment

    # 7 ---------------------------------------------------------------------
    _section("7. Date columns")
    dates = pl.concat([eda.date_parse_checks(f) for f in families.values()])
    dates.write_csv(out / "date_checks.csv")
    bad = dates.filter(pl.col("unparseable") > 0)
    print(f"  {dates.height} date columns; {bad.height} with unparseable values "
          f"({int(bad['unparseable'].sum()) if bad.height else 0:,} values)")
    print(f"  earliest {dates['min'].min()}, latest {dates['max'].max()} "
          f"(decisions run {base['date_decision'].min()} to {base['date_decision'].max()})")
    late = dates.filter(pl.col("max") > pl.lit(base["date_decision"].max()).str.to_date())
    print(f"  date columns with values after the last decision date: {late.height}"
          + (f" (e.g. {', '.join(late['column'].head(4).to_list())})" if late.height else ""))
    summary["date_columns"] = {"count": dates.height, "with_unparseable": bad.height,
                               "after_last_decision": late["column"].to_list()}

    # registry, summary, bundle -------------------------------------------
    _section("Outputs")
    write_registry(build_schema_registry(table_dir, root / "feature_definitions.csv"), out / "schema_registry.json")
    summary["elapsed_s"] = round(time.perf_counter() - started, 1)
    summary["peak_memory_gb"] = round(peak_rss_gb(), 2)
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    bundle = shutil.make_archive(str(out.parent / "c04_results"), "zip", root_dir=out)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  ({p.stat().st_size / 1024:.0f} KB)")
    print(f"  bundle: {bundle}")
    print(f"\nelapsed {summary['elapsed_s']}s   peak memory {summary['peak_memory_gb']} GB")
    print("\n== Result\nSMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
