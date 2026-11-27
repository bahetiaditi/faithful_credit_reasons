"""C8: the S0 scorecard (WoE logistic regression) and its points-below-maximum reason codes.

    python scripts/smoke_c08.py                 # on Kaggle, with the fcr-features Dataset attached
                                                # (or FCR_FEATURES_DIR=/kaggle/working/fcr_features)
    python scripts/smoke_c08.py --local         # on the Mac, on the 50k sample in data/
    python scripts/smoke_c08.py --synthetic     # anywhere, against generated fixtures

Uses training weeks to fit and validation weeks to report; the test rows are
removed right after the dataset is verified and never reach this script's model.

Writes c08_results.zip (aggregates only):
    s0_selection.csv       every candidate: IV, bins, selected or why not
    s0_points_table.csv    the scorecard: feature, bin, count, event rate, WoE, points
    s0_metrics.json        ROC-AUC, Gini, PR-AUC, KS on train and validation; score scaling
    s0_reason_frequency.csv  how often each feature is the first, or any, of the four reasons
                           among validation applicants below the provisional cutoff
    c08_summary.json

Pass condition (REFERENCE.md C8): validation AUC recorded; reason codes checked
by hand on 5 applicants (printed below, and re-derived automatically).
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

import numpy as np
import polars as pl

from fcr.config import load_config, resolve_path
from fcr.data.dataset import REGISTRY, load_feature_table, locate_feature_dataset
from fcr.eval.metrics import discrimination
from fcr.models.guard import development_table, train_validation
from fcr.models.scorecard import build_s0, to_pandas
from fcr.runinfo import git_commit, peak_rss_gb

AUC_SANE = (0.65, 0.90)   # below: something is broken; above: suspect leakage


def _synthetic_dataset(cfg) -> Path:
    """The C7 pipeline on fixtures, into a temp dir."""
    from fcr.data.build import ensure_features
    from fcr.data.dataset import write_feature_dataset
    from fcr.data.loader import discover_families, load_base
    from fcr.data.split import assign_split, build_lock
    from fcr.data.synthetic import SyntheticSpec, generate_synthetic_dataset
    from fcr.features.matrix import build_model_matrix

    root = Path(tempfile.mkdtemp(prefix="fcr_smoke_c08_"))
    generate_synthetic_dataset(root, SyntheticSpec(seed=cfg.project.seed))
    families = discover_families(root / "parquet_files" / "train")
    base = assign_split(load_base(families), cfg.split)
    manifest = ensure_features(families, base, cfg, root / "c05" / "features", root / "c05" / "manifest.csv",
                               log=lambda *_: None)
    result = build_model_matrix(root / "c05" / "features", manifest, base, cfg)
    matrix = result.matrix.join(base.select("case_id", "split"), on="case_id")
    out = root / "fcr_features"
    write_feature_dataset(out, matrix, result.slices, pl.DataFrame(result.registry), result.levels,
                          build_lock(base, cfg.split), cfg, commit=git_commit(), sample_cases=500)
    return out


def _bucket(reason: str) -> str:
    """Group selection reasons for printing ("woe correlation 0.71 with x" -> "woe correlation above limit")."""
    if not reason:
        return "selected"
    for prefix, label in (("woe correlation", "woe correlation above limit with a chosen feature"),
                          ("adding it flips", "adding it flips a coefficient sign (collinear)"),
                          ("coefficient sign flipped", "sign flipped in the full-data fit"),
                          ("limit of", "feature limit reached"), ("iv below", "iv below minimum"),
                          ("iv above", "iv above maximum (suspiciously strong)")):
        if reason.startswith(prefix):
            return label
    return reason


def _definitions(cfg) -> dict[str, str]:
    """Column descriptions from the competition, when it is attached (Kaggle only)."""
    try:
        from fcr.data.loader import read_feature_definitions
        from fcr.data.paths import locate_competition_dir
        root = locate_competition_dir(cfg)
        return read_feature_definitions(root / "feature_definitions.csv") if root else {}
    except Exception:  # noqa: BLE001 - descriptions are a convenience for the printout only
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--local", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    started = time.perf_counter()

    if args.synthetic:
        cfg = cfg.model_copy(update={"features": cfg.features.model_copy(
            update={"corr_sample_cases": 1500, "min_minority_rows": 20})})
        directory, sample, expected_lock = _synthetic_dataset(cfg), False, None
        work = directory.parent
    elif args.local:
        directory, sample, expected_lock = resolve_path(cfg.paths.local_data_dir), True, "committed"
        work = Path(tempfile.mkdtemp(prefix="fcr_c08_local_"))
    else:
        directory = locate_feature_dataset(cfg)
        if directory is None:
            print(f"feature dataset '{cfg.paths.feature_dataset_name}' not found: attach it, or set FCR_FEATURES_DIR")
            return 1
        sample, expected_lock = False, "committed"
        work = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(tempfile.mkdtemp())
    mode_name = "SYNTHETIC" if args.synthetic else "LOCAL (50k sample)" if args.local else "REAL"
    print(f"mode: {mode_name}   commit {git_commit()}   features {directory}")

    table = development_table(load_feature_table(directory, cfg, sample=sample, expected_lock=expected_lock), cfg)
    train, validation = train_validation(table, cfg)
    print(f"dataset built at {table.manifest['commit']}; test rows removed after verification; "
          f"train {train.height:,} / validation {validation.height:,}")
    failures: list[str] = []

    registry = pl.read_csv(directory / REGISTRY, infer_schema_length=10000).filter(pl.col("status") == "kept")
    info = {r["feature"]: r for r in registry.iter_rows(named=True)}
    scfg = cfg.model.scorecard
    candidates = [f for f in table.features if not (scfg.exclude_masked and info[f]["source_suffix"] == "M")]
    categorical = {f for f in candidates if info[f]["kind"] == "category"}
    print(f"candidates: {len(candidates)} of {len(table.features)} features "
          f"({len(table.features) - len(candidates)} masked left out; {len(categorical)} categorical)")

    t = time.perf_counter()
    s0, records = build_s0(train, candidates, categorical, scfg, cfg.project.seed)
    print(f"scorecard built in {time.perf_counter() - t:.0f}s: {len(s0.features)} features")

    reasons_count = Counter(_bucket(r["reason"]) for r in records)
    print("\n== Selection (training weeks)")
    for why, n in reasons_count.most_common():
        print(f"  {n:>5}  {why}")

    print(f"\n== Scorecard: {len(s0.features)} features, base {scfg.base_points:.0f} points at "
          f"{scfg.base_odds:.0f}:1 good:bad, {scfg.pdo:.0f} points to double the odds")
    iv = {r["feature"]: r["iv"] for r in records}
    coef = dict(zip(s0.features, s0.scorecard.estimator_.coef_[0]))
    spread = s0.table.filter(pl.col("count") > 0).group_by("feature").agg(
        (pl.col("points").max() - pl.col("points").min()).alias("range"), pl.len().alias("bins"))
    spread = dict(zip(spread["feature"], zip(spread["range"], spread["bins"])))
    print(f"  {'feature':<46}{'IV':>7}{'coef':>8}{'bins':>6}{'points range':>14}")
    for f in sorted(s0.features, key=lambda f: -spread[f][0]):
        print(f"  {f:<46}{iv[f]:>7.3f}{coef[f]:>8.3f}{spread[f][1]:>6}{spread[f][0]:>14.1f}")

    metrics = {}
    print("\n== Discrimination (higher score = lower risk)")
    for name, d in (("train", train), ("validation", validation)):
        metrics[name] = discrimination(d["target"].to_numpy(), -s0.score(d))
        m = metrics[name]
        print(f"  {name:<11} cases {m['cases']:>9,}  defaults {m['defaults']:>6,}   ROC-AUC {m['roc_auc']:.4f}  "
              f"Gini {m['gini']:.4f}  PR-AUC {m['pr_auc']:.4f}  KS {m['ks']:.4f}")

    # checks
    vs = s0.score(validation)
    native = np.asarray(s0.scorecard.score(to_pandas(validation, s0.features, s0.categorical)))
    if np.abs(vs - native).max() > 1e-6:
        failures.append(f"points do not sum to optbinning's score (max diff {np.abs(vs - native).max():.2e})")
    if (s0.scorecard.estimator_.coef_[0] >= 0).any():
        failures.append("a coefficient is not negative")
    if len(s0.features) > scfg.max_features:
        failures.append(f"{len(s0.features)} features, limit {scfg.max_features}")
    if any(info[f]["source_suffix"] == "M" for f in s0.features) and scfg.exclude_masked:
        failures.append("a masked feature is in the scorecard")
    raw_calendar = [f for f in s0.features if info[f]["source_suffix"] == "T" and ("year" in f or "month" in f)]
    if raw_calendar:
        failures.append(f"raw calendar year/month features in the scorecard: {raw_calendar} (C5.2 should have composed them)")
    lo, hi = AUC_SANE
    if not args.synthetic and not lo <= metrics["validation"]["roc_auc"] <= hi:
        failures.append(f"validation AUC {metrics['validation']['roc_auc']:.3f} outside the sane range {AUC_SANE}")

    # provisional cutoff for reason codes: the validation approval rate the config targets.
    # The real threshold tau is set on the calibrated production model in C11.
    cutoff = float(np.quantile(vs, 1 - cfg.model.target_approval_rate))
    declined = validation.filter(pl.Series(vs < cutoff))
    reasons = s0.reasons(declined, cfg.explain.max_reasons)
    first = Counter(r[0][0] for r in reasons if r)
    anyof = Counter(f for r in reasons for f, _ in r)
    freq = pl.DataFrame({"feature": s0.features,
                         "first_reason_share": [first[f] / len(reasons) for f in s0.features],
                         "any_reason_share": [anyof[f] / len(reasons) for f in s0.features]}
                        ).sort("first_reason_share", descending=True)
    print(f"\n== Reasons for the {declined.height:,} validation applicants below the provisional cutoff "
          f"({cutoff:.1f} points, {cfg.model.target_approval_rate:.0%} approved)")
    print(f"  {'feature':<46}{'first reason':>14}{'any of top 4':>14}")
    for r in freq.head(10).iter_rows(named=True):
        print(f"  {r['feature']:<46}{r['first_reason_share']:>14.1%}{r['any_reason_share']:>14.1%}")
    if any(len(r) == 0 for r in reasons):
        failures.append("a declined applicant has no reason")

    print("\n== Hand check: 5 declined validation applicants (bins only; no raw values)")
    defs = _definitions(cfg)
    rng = np.random.default_rng(cfg.project.seed)
    rows = np.sort(rng.choice(declined.height, size=min(5, declined.height), replace=False)).tolist()
    pick = declined.with_row_index("__i").filter(pl.col("__i").is_in(rows)).drop("__i")
    pts, bins = s0.points(pick), s0.bins(pick)
    for i, rs in enumerate(s0.reasons(pick, cfg.explain.max_reasons)):
        print(f"\n  applicant {i + 1}: score {pts[i].sum():.1f}   PD {s0.pd(pick.slice(i, 1))[0]:.1%}")
        gaps = []
        for rank, (f, gap) in enumerate(rs, 1):
            j = s0.features.index(f)
            check = s0.max_points[f] - pts[i, j]
            if abs(check - gap) > 1e-9:
                failures.append(f"applicant {i + 1} reason {f}: gap {gap:.3f} but table says {check:.3f}")
            gaps.append(gap)
            desc = defs.get(info[f]["source"], "")
            print(f"    {rank}. {f:<44} bin {bins[i][j]:<22} {pts[i, j]:6.1f} of max {s0.max_points[f]:6.1f}  (-{gap:.1f})")
            if desc:
                print(f"       {desc[:110]}")
        if gaps != sorted(gaps, reverse=True) or any(g <= 0 for g in gaps):
            failures.append(f"applicant {i + 1}: reasons are not strictly positive and in descending order")

    results = work / "c08_results"
    shutil.rmtree(results, ignore_errors=True)
    results.mkdir(parents=True)
    pl.DataFrame(records).write_csv(results / "s0_selection.csv")
    s0.table.write_csv(results / "s0_points_table.csv")
    freq.write_csv(results / "s0_reason_frequency.csv")
    (results / "s0_metrics.json").write_text(json.dumps({
        **metrics, "features": s0.features, "max_points": s0.max_points,
        "scaling": {"pdo": scfg.pdo, "base_points": scfg.base_points, "base_odds": scfg.base_odds},
        "provisional_cutoff": cutoff, "declined_validation": declined.height}, indent=1))
    summary = {"commit": git_commit(), "dataset_commit": table.manifest["commit"], "mode": mode_name,
               "n_features": len(s0.features), "validation_roc_auc": metrics["validation"]["roc_auc"],
               "train_roc_auc": metrics["train"]["roc_auc"], "elapsed_s": round(time.perf_counter() - started, 1),
               "peak_memory_gb": round(peak_rss_gb(), 2), "failures": failures}
    (results / "c08_summary.json").write_text(json.dumps(summary, indent=1))
    bundle = shutil.make_archive(str(results), "zip", root_dir=results)
    shutil.rmtree(results)

    print(f"\nbundle: {bundle}   elapsed {summary['elapsed_s']}s   peak memory {summary['peak_memory_gb']} GB")
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
