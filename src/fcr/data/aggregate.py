"""Turn one cleaned chunk into one row per case (REFERENCE.md 5.1 step 5).

Feature names are "<source column>__<stat>". The returned manifest records the
source family, column and statistic for every feature explicitly, so later
steps (screening, reason groups) never parse names to learn where a feature
came from.

Statistics
    numeric (A, P, numeric L/T, and D after conversion to days): the configured
        stats (mean, min, max, count); P columns also get count_positive.
    boolean: share true, count.
    text categories (M, text L/T): mode and number of distinct values.
    every family: rows per case; depth-2 families also distinct num_group1.
Depth 0 tables pass through unchanged (one row per case already).
The applicant-row family (person_1) keeps the num_group1 == 0 row as-is and
adds the number of related persons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from fcr.data.dates import DECISION
from fcr.data.naming import KEY_COLUMNS

NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


@dataclass(frozen=True)
class FeatureSpec:
    feature: str
    family: str
    source: str        # source column, or "" for row-count features
    stat: str
    kind: str          # numeric | boolean | category

    def as_dict(self) -> dict:
        return asdict(self)


def _value_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in KEY_COLUMNS and c != DECISION]


def _kind(dtype: pl.DataType) -> str:
    if dtype == pl.Boolean:
        return "boolean"
    if dtype in NUMERIC:
        return "numeric"
    return "category"


def _numeric_exprs(c: str, stats: tuple[str, ...]) -> list[tuple[str, pl.Expr]]:
    fns = {
        "mean": pl.col(c).mean(), "min": pl.col(c).min(), "max": pl.col(c).max(),
        "count": pl.col(c).count(), "std": pl.col(c).std(), "sum": pl.col(c).sum(),
    }
    out = [(s, fns[s]) for s in stats]
    if c.endswith("P"):
        out.append(("count_positive", (pl.col(c) > 0).sum()))
    return out


def _mode_frame(df: pl.DataFrame, c: str) -> pl.DataFrame:
    """Most frequent non-null value per case; ties broken by value so the result is deterministic."""
    return (
        df.filter(pl.col(c).is_not_null())
        .group_by("case_id", c).agg(pl.len().alias("__n"))
        .sort(["case_id", "__n", c], descending=[False, True, False])
        .group_by("case_id", maintain_order=True).first()
        .select("case_id", pl.col(c).alias(f"{c}__mode"))
    )


def aggregate_chunk(
    df: pl.DataFrame,
    family_key: str,
    depth: int,
    numeric_stats: tuple[str, ...],
    applicant_row: bool = False,
) -> tuple[pl.DataFrame, list[FeatureSpec]]:
    """One row per case_id for this chunk, plus the manifest of the features produced."""
    values = _value_columns(df)
    specs: list[FeatureSpec] = []

    if depth == 0:
        out = df.select("case_id", *values)
        specs = [FeatureSpec(c, family_key, c, "value", _kind(df.schema[c])) for c in values]
        return out, specs

    if applicant_row:
        applicant = df.filter(pl.col("num_group1") == 0).select("case_id", *values)
        renamed = {c: f"{c}__applicant" for c in values}
        counts = df.group_by("case_id").agg((pl.len() - 1).alias(f"{family_key}__related_persons"))
        out = applicant.rename(renamed).join(counts, on="case_id", how="full", coalesce=True)
        specs = [FeatureSpec(renamed[c], family_key, c, "applicant", _kind(df.schema[c])) for c in values]
        specs.append(FeatureSpec(f"{family_key}__related_persons", family_key, "", "related_persons", "numeric"))
        return out, specs

    exprs: list[pl.Expr] = [pl.len().alias(f"{family_key}__rows")]
    specs.append(FeatureSpec(f"{family_key}__rows", family_key, "", "rows", "numeric"))
    if depth == 2:
        exprs.append(pl.col("num_group1").n_unique().alias(f"{family_key}__groups"))
        specs.append(FeatureSpec(f"{family_key}__groups", family_key, "", "groups", "numeric"))
    categories = []
    for c in values:
        kind = _kind(df.schema[c])
        if kind == "numeric":
            for stat, e in _numeric_exprs(c, numeric_stats):
                exprs.append(e.alias(f"{c}__{stat}"))
                specs.append(FeatureSpec(f"{c}__{stat}", family_key, c, stat, "numeric"))
        elif kind == "boolean":
            exprs += [pl.col(c).cast(pl.Float32).mean().alias(f"{c}__share_true"), pl.col(c).count().alias(f"{c}__count")]
            specs += [FeatureSpec(f"{c}__share_true", family_key, c, "share_true", "numeric"),
                      FeatureSpec(f"{c}__count", family_key, c, "count", "numeric")]
        else:
            exprs.append(pl.col(c).n_unique().alias(f"{c}__n_unique"))
            specs.append(FeatureSpec(f"{c}__n_unique", family_key, c, "n_unique", "numeric"))
            categories.append(c)
    out = df.group_by("case_id").agg(exprs)
    for c in categories:
        out = out.join(_mode_frame(df, c), on="case_id", how="left")
        specs.append(FeatureSpec(f"{c}__mode", family_key, c, "mode", "category"))
    # n_unique counts null as a value in polars; recount distinct non-null values.
    if categories:
        fixed = df.group_by("case_id").agg([pl.col(c).drop_nulls().n_unique().alias(f"{c}__n_unique") for c in categories])
        out = out.drop([f"{c}__n_unique" for c in categories]).join(fixed, on="case_id", how="left")
    float_cols = [s.feature for s in specs if s.kind == "numeric" and out.schema[s.feature] in (pl.Float64,)]
    out = out.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])
    return out.select("case_id", *[s.feature for s in specs]), specs
