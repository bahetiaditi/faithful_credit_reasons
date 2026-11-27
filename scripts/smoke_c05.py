"""C5: build case-level features for every table family.

    python scripts/smoke_c05.py               # on Kaggle, against the attached competition data
    python scripts/smoke_c05.py --synthetic   # anywhere, against generated fixtures

Writes one parquet per family to <out>/features/ (applicant-level data: stays on
Kaggle and becomes the versioned feature Dataset in C7), and aggregate-only
results to <out>/ zipped as c05_results.zip:
    feature_manifest.csv     every feature: family, source column, stat, kind, null share
    date_diagnostics.csv     per family x date column: sentinels nulled, future events nulled,
                             future scheduled values kept, records dropped by anchors
    date_lag_profile.csv     per source table x date column: role, lag-behind-decision buckets,
                             share constant within a case (C5.1: separates retrieval timestamps
                             from event histories)
    c05_summary.json         per family: cases, features, rows in/out, seconds

Pass condition (REFERENCE.md C5): every family builds with one row per case, and
no event-date feature in the real output is negative (looks past the decision).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

from fcr.config import load_config, resolve_path
from fcr.data.build import build_all, write_family
from fcr.data.dates import LAG_BUCKETS, load_date_roles
from fcr.data.loader import discover_families, load_base
from fcr.data.paths import locate_competition_dir
from fcr.data.synthetic import SyntheticSpec, generate_synthetic_dataset
from fcr.runinfo import git_commit, peak_rss_gb

EVENT_STATS = ("mean", "min", "max", "applicant", "value")
ANCHOR_DROP_WARN = 0.20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    roles = load_date_roles(resolve_path(cfg.features.date_roles_file))
    started = time.perf_counter()

    if args.synthetic:
        root = Path(tempfile.mkdtemp(prefix="fcr_smoke_c05_"))
        generate_synthetic_dataset(root, SyntheticSpec(seed=cfg.project.seed))
        out = root / "c05"
    else:
        root = locate_competition_dir(cfg)
        if root is None:
            print(f"competition data not found (configured: {cfg.paths.kaggle_competition_dir})")
            return 1
        work = Path("/kaggle/working")
        out = (work if work.is_dir() else Path(tempfile.mkdtemp())) / "c05"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    print(f"mode: {'SYNTHETIC' if args.synthetic else 'REAL'}   commit {git_commit()}   data {root}")

    families = discover_families(root / "parquet_files" / "train")
    base = load_base(families)
    failures: list[str] = []
    manifest, date_rows, summary = [], [], {}
    lag = defaultdict(lambda: defaultdict(int))
    sources = defaultdict(lambda: defaultdict(int))
    warnings: list[str] = []

    print(f"\n  {'family':<20}{'cases':>11}{'feats':>7}{'rows in':>13}{'rows out':>13}{'anchor drop':>12}"
          f"{'fut.null':>10}{'sentinel':>10}{'masked':>11}{'sec':>7}{'peakGB':>8}")
    t_family = time.perf_counter()
    for result in build_all(families, base, cfg, roles):
        f = result.features
        rows_in = sum(d["rows_in"] for d in result.diagnostics)
        rows_out = sum(d["rows_out"] for d in result.diagnostics)
        dropped = sum(d["rows_dropped_by_anchor"] for d in result.diagnostics)
        totals = defaultdict(lambda: defaultdict(int))
        for d in result.diagnostics:
            for kind in ("sentinels_nulled", "future_events_nulled", "scheduled_future_kept", "masked_nulled"):
                for col, n in d.get(kind, {}).items():
                    totals[kind][col] += n
        for d in result.diagnostics:
            src = sources[d["family"]]
            src["rows_in"] += d["rows_in"]
            src["rows_dropped_by_anchor"] += d["rows_dropped_by_anchor"]
            for col, counts in d.get("lag_profile", {}).items():
                for k, v in counts.items():
                    lag[(d["family"], col)][k] += v
        for col in sorted(set(totals["sentinels_nulled"]) | set(totals["future_events_nulled"]) | set(totals["scheduled_future_kept"])):
            date_rows.append({
                "family": result.key, "column": col,
                "role": "scheduled" if roles.is_scheduled(col) else "event",
                "anchor": col in roles.anchors.get(result.key, ()),
                "sentinels_nulled": totals["sentinels_nulled"].get(col, 0),
                "future_events_nulled": totals["future_events_nulled"].get(col, 0),
                "scheduled_future_kept": totals["scheduled_future_kept"].get(col, 0),
            })

        # leakage invariant on the real output
        leaks = []
        null_counts = f.null_count().row(0, named=True)
        for spec in result.specs:
            manifest.append({**spec.as_dict(), "null_share": null_counts[spec.feature] / f.height if f.height else 1.0})
            if spec.source.endswith("D") and not roles.is_scheduled(spec.source) and spec.stat in EVENT_STATS:
                negative = int((f[spec.feature].drop_nulls() < 0).sum())
                if negative:
                    leaks.append(f"{spec.feature} ({negative:,})")
        if leaks:
            failures.append(f"{result.key}: event-date features after the decision: {leaks[:5]}")

        write_family(result, out / "features")
        seconds = time.perf_counter() - t_family
        t_family = time.perf_counter()
        summary[result.key] = {
            "cases": f.height, "features": len(result.specs), "rows_in": rows_in, "rows_out": rows_out,
            "rows_dropped_by_anchor": dropped, "future_events_nulled": sum(totals["future_events_nulled"].values()),
            "sentinels_nulled": sum(totals["sentinels_nulled"].values()),
            "masked_nulled": sum(totals["masked_nulled"].values()), "seconds": round(seconds, 1),
        }
        s = summary[result.key]
        print(f"  {result.key:<20}{s['cases']:>11,}{s['features']:>7}{rows_in:>13,}{rows_out:>13,}{dropped:>12,}"
              f"{s['future_events_nulled']:>10,}{s['sentinels_nulled']:>10,}{s['masked_nulled']:>11,}"
              f"{seconds:>7.0f}{peak_rss_gb():>8.1f}")
        del result, f

    pl.DataFrame(manifest).write_csv(out / "feature_manifest.csv")

    lag_rows = []
    for (family, col), c in sorted(lag.items()):
        after = c["non_null"] - c["before"]
        lag_rows.append({"family": family, "column": col, "role": roles.role(col),
                         "anchor": col in roles.anchors.get(family, ()), "non_null": c["non_null"],
                         "share_after": after / c["non_null"] if c["non_null"] else 0.0,
                         **{name: c[name] for name, *_ in LAG_BUCKETS},
                         "cases_multi": c.get("cases_multi"), "cases_constant": c.get("cases_constant")})
    lag_df = pl.DataFrame(lag_rows)
    lag_df.write_csv(out / "date_lag_profile.csv")
    print("\n== Date lag profile (columns with >5% of values after the decision)")
    print(f"  {'table.column':<46}{'role':<10}{'non-null':>11}{'after':>7}  {'1d':>6}{'2-7d':>6}{'8-30d':>6}"
          f"{'31-90':>6}{'91-365':>7}{'>365':>6}  constant/case")
    for r in lag_df.filter(pl.col("share_after") > 0.05).sort("share_after", descending=True).iter_rows(named=True):
        after = r["non_null"] - r["before"]
        mix = [r[k] / after if after else 0 for k in ("1d", "2-7d", "8-30d", "31-90d", "91-365d", ">365d")]
        const = f"{r['cases_constant'] / r['cases_multi']:.0%}" if r["cases_multi"] else "-"
        print(f"  {r['family'] + '.' + r['column']:<46}{r['role']:<10}{r['non_null']:>11,}{r['share_after']:>7.0%}  "
              + "".join(f"{m:>6.0%}" if i != 4 else f"{m:>7.0%}" for i, m in enumerate(mix)) + f"  {const:>8}")

    print("\n== Anchor drops by source table")
    for family, src in sorted(sources.items()):
        anchors = roles.anchors.get(family, ())
        if not anchors:
            continue
        share = src["rows_dropped_by_anchor"] / src["rows_in"] if src["rows_in"] else 0.0
        limit = roles.anchor_drop_limit(family, ANCHOR_DROP_WARN)
        flag = "  WARN" if share > limit else ("  (expected)" if family in roles.anchor_drop_expected else "")
        print(f"  {family:<20} {', '.join(anchors):<48} dropped {src['rows_dropped_by_anchor']:>10,} of {src['rows_in']:>11,} ({share:.1%}){flag}")
        if share > limit:
            warnings.append(f"{family}: anchor dropped {share:.0%} of rows (limit {limit:.0%})")
    dates = pl.DataFrame(date_rows) if date_rows else pl.DataFrame()
    dates.write_csv(out / "date_diagnostics.csv")
    total_features = sum(s["features"] for s in summary.values())

    print("\n== Date rules (largest effects)")
    if dates.height:
        for r in dates.sort("future_events_nulled", descending=True).head(8).iter_rows(named=True):
            if r["future_events_nulled"]:
                print(f"  event nulled   {r['family']}.{r['column']}: {r['future_events_nulled']:,}")
        for r in dates.filter(pl.col("scheduled_future_kept") > 0).iter_rows(named=True):
            print(f"  scheduled kept {r['family']}.{r['column']}: {r['scheduled_future_kept']:,}")
        for r in dates.filter(pl.col("sentinels_nulled") > 0).sort("sentinels_nulled", descending=True).iter_rows(named=True):
            print(f"  sentinel       {r['family']}.{r['column']}: {r['sentinels_nulled']:,}")

    manifest_df = pl.DataFrame(manifest)
    print(f"\n== Features: {total_features:,} across {len(summary)} families; "
          f"{int((manifest_df['null_share'] > cfg.features.max_missing_share).sum())} over "
          f"{cfg.features.max_missing_share:.0%} null at case level (C6 screens them)")

    elapsed = round(time.perf_counter() - started, 1)
    (out / "c05_summary.json").write_text(json.dumps(
        {"commit": git_commit(), "families": summary, "total_features": total_features,
         "elapsed_s": elapsed, "peak_memory_gb": round(peak_rss_gb(), 2), "failures": failures,
         "warnings": warnings}, indent=1))
    # Zip only the aggregate files; the feature parquet stays on Kaggle.
    staging = out.parent / "c05_results"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    for name in ("feature_manifest.csv", "date_diagnostics.csv", "date_lag_profile.csv", "c05_summary.json"):
        shutil.copy(out / name, staging / name)
    bundle = shutil.make_archive(str(staging), "zip", root_dir=staging)
    shutil.rmtree(staging)
    size_gb = sum(p.stat().st_size for p in (out / "features").glob("*.parquet")) / 1024**3
    print(f"\nfeature parquet: {out / 'features'} ({size_gb:.2f} GB)   results bundle: {bundle}")
    print(f"elapsed {elapsed}s   peak memory {peak_rss_gb():.2f} GB")

    print("\n== Result")
    for w in warnings:
        print(f"WARNING: {w}")
    if failures:
        print("SMOKE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
