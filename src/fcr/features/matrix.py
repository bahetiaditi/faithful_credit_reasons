"""Assemble the model matrix from C5's per-family features (C6).

Inputs: one parquet per family plus the C5 feature manifest. Outputs:
    model matrix   case_id, WEEK_NUM, target, date_decision + every kept feature
    slices         case_id + protected attributes, for fairness slices only (Section 5.5)
    registry       one record per candidate feature: where it came from and why it
                   was kept or dropped (the "every feature traced" check)
    levels         the training levels each category feature keeps

Protected and proxy sources are removed before any statistic is computed and
never enter the matrix. Income type is recoded in the matrix only; the slices
keep the original. Screening statistics use training weeks only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from fcr.config import Config
from fcr.data.naming import feature_suffix
from fcr.features.screen import (
    ScreenRecord,
    add_ratios,
    cap_categories,
    category_levels,
    correlation_screen,
    exclusion_status,
    missing_and_constant,
    recode_values,
)

KEYS = ("case_id", "WEEK_NUM", "target", "date_decision")
DERIVED_FAMILY = "derived"
# Which of two perfectly correlated features to keep: the more explainable one.
STAT_PRIORITY = {"rows": 0, "groups": 1, "related_persons": 1, "ratio": 2, "value": 3, "applicant": 3,
                 "max": 4, "mean": 4, "min": 4, "count_positive": 5, "share_true": 5, "n_unique": 6, "count": 7}


@dataclass
class MatrixResult:
    matrix: pl.DataFrame
    slices: pl.DataFrame
    registry: list[dict]
    levels: dict[str, list[str]]


def _source_suffix(source: str) -> str | None:
    try:
        return feature_suffix(source) if source else None
    except ValueError:
        return None


def _slices(person: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Protected attributes for slice analysis. Never joined into the model matrix."""
    def first(*names):
        present = [pl.col(n) for n in names if n in person.columns]
        return pl.coalesce(present) if present else pl.lit(None)

    disability = cfg.protected.disability_income_values
    income = f"{cfg.protected.disability_recode_column}__applicant"
    return person.select(
        "case_id",
        (first("birth_259D__applicant", "birthdate_87D__applicant") / 365.25).cast(pl.Float32).alias("age_years"),
        first("sex_738L__applicant", "gender_992L__applicant").alias("sex"),
        first("familystate_447L__applicant", "maritalst_703L__applicant").alias("family_status"),
        (pl.col(income).is_in(list(disability)) if income in person.columns else pl.lit(None)).alias("disability_income_type"),
    )


def build_model_matrix(feature_dir: Path, manifest: pl.DataFrame, base: pl.DataFrame, cfg: Config) -> MatrixResult:
    fcfg, pcfg = cfg.features, cfg.protected
    lo, hi = cfg.split.train_weeks
    keys = base.select(*KEYS)
    train_ids = keys.filter(pl.col("WEEK_NUM").is_between(lo, hi)).select("case_id")
    protected, proxies = set(pcfg.columns), set(pcfg.proxy_columns)

    specs = {r["feature"]: r for r in manifest.iter_rows(named=True)}
    families = sorted(manifest["family"].unique().to_list())
    frames = {fam: pl.read_parquet(feature_dir / f"{fam}.parquet") for fam in families}

    # derived ratios, screened like any other feature
    needed = {c for r in fcfg.ratios for c in (r.numerator, r.denominator)}
    derived = keys.select("case_id")
    for fam, frame in frames.items():
        cols = [c for c in frame.columns if c in needed]
        if cols:
            derived = derived.join(frame.select("case_id", *cols), on="case_id", how="left")
    derived, ratio_names = add_ratios(derived, fcfg.ratios)
    if ratio_names:
        frames[DERIVED_FAMILY] = derived.select("case_id", *ratio_names)
        for name in ratio_names:
            specs[name] = {"feature": name, "family": DERIVED_FAMILY, "source": "", "stat": "ratio", "kind": "numeric"}

    # 1. exclusion, then 2-3. missing and constant, per family on training rows
    records: dict[str, ScreenRecord] = {}
    for fam, frame in frames.items():
        candidates = []
        for f in (c for c in frame.columns if c != "case_id"):
            status = exclusion_status(specs[f]["source"], protected, proxies)
            if status:
                records[f] = ScreenRecord(f, status=status)
            else:
                candidates.append(f)
        train = frame.join(train_ids, on="case_id", how="right").select(candidates)
        records |= missing_and_constant(train, candidates, fcfg.max_missing_share, fcfg.min_minority_rows)

    # 4. correlation, across families, on a seeded sample of training cases
    numeric_kept = [f for f, r in records.items() if r.status == "kept" and specs[f]["kind"] != "category"]
    sample_ids = train_ids.sample(n=min(fcfg.corr_sample_cases, train_ids.height), seed=cfg.project.seed)
    sample = sample_ids
    for fam, frame in frames.items():
        cols = [f for f in frame.columns if f in numeric_kept]
        if cols:
            sample = sample.join(frame.select("case_id", *cols), on="case_id", how="left")
    priority = {f: STAT_PRIORITY.get(specs[f]["stat"], 9) for f in numeric_kept}
    corr = correlation_screen(sample, numeric_kept, {f: records[f].train_missing_share or 0.0 for f in numeric_kept},
                              fcfg.max_abs_spearman, priority,
                              preferred={f for f in numeric_kept if specs[f]["stat"] == "ratio"})
    for f, rec in corr.items():
        if rec.status != "kept":
            records[f].status, records[f].correlated_with, records[f].rho = rec.status, rec.correlated_with, rec.rho

    # assemble kept features for every case
    kept = {f for f, r in records.items() if r.status == "kept"}
    matrix = keys
    for fam, frame in frames.items():
        cols = [f for f in frame.columns if f in kept]
        if cols:
            matrix = matrix.join(frame.select("case_id", *cols), on="case_id", how="left")

    income = f"{pcfg.disability_recode_column}__applicant"
    matrix = recode_values(matrix, income, pcfg.disability_income_values, pcfg.disability_recode_value)
    train_rows = matrix.join(train_ids, on="case_id", how="semi")
    cats = [f for f in kept if specs[f]["kind"] == "category"]
    levels = {f: category_levels(train_rows, f, fcfg.max_category_levels) for f in sorted(cats)}
    matrix = cap_categories(matrix, levels)

    slices = _slices(frames["person_1"], cfg) if "person_1" in frames else keys.select("case_id")

    registry = []
    for f, rec in sorted(records.items()):
        spec = specs[f]
        registry.append({
            "feature": f, "family": spec["family"], "source": spec["source"], "stat": spec["stat"],
            "kind": spec["kind"], "source_suffix": _source_suffix(spec["source"]), "status": rec.status,
            "train_missing_share": rec.train_missing_share, "top_share": rec.top_share, "minority_rows": rec.minority_rows,
            "correlated_with": rec.correlated_with, "rho": rec.rho,
            "category_levels": len(levels[f]) if f in levels else None,
        })
    return MatrixResult(matrix, slices, registry, levels)
