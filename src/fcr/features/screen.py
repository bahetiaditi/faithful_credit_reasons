"""Feature screening (REFERENCE.md 5.1 step 6), protected exclusion and recoding (4.4).

Every rule is measured on training cases only, so validation and test weeks
never influence which features exist or how categories are capped. Each rule
returns, for every feature it looks at, a status and the evidence behind it;
the C6 feature registry is the union of those records.

Order:
    1. exclusion: protected and proxy sources never reach the model matrix
    2. missing: more than max_missing_share null on training cases
    3. constant: one value (or none) covers more than max_top_share of non-nulls
    4. correlated: one of each numeric pair with |Spearman rho| above the limit
Category features that survive keep their most frequent training levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

OTHER_LEVEL = "__other__"


@dataclass
class ScreenRecord:
    feature: str
    status: str = "kept"           # kept | excluded:protected | excluded:proxy | dropped:missing | dropped:constant | dropped:correlated
    train_missing_share: float | None = None
    top_share: float | None = None
    correlated_with: str | None = None
    rho: float | None = None
    detail: dict = field(default_factory=dict)


def exclusion_status(source: str, protected: set[str], proxies: set[str]) -> str | None:
    if source in protected:
        return "excluded:protected"
    if source in proxies:
        return "excluded:proxy"
    return None


def missing_and_constant(train: pl.DataFrame, features: list[str], max_missing: float, max_top: float) -> dict[str, ScreenRecord]:
    """Missing share and most-frequent-value share per feature, on training rows."""
    n = train.height
    records = {}
    nulls = train.select([pl.col(f).null_count() for f in features]).row(0, named=True) if features else {}
    for f in features:
        missing = nulls[f] / n if n else 1.0
        rec = ScreenRecord(f, train_missing_share=missing)
        if missing > max_missing:
            rec.status = "dropped:missing"
        else:
            col = train[f].drop_nulls()
            top = col.value_counts(sort=True)["count"][0] / col.len() if col.len() else 1.0
            rec.top_share = float(top)
            if top > max_top:
                rec.status = "dropped:constant"
        records[f] = rec
    return records


def _rank_matrix(sample: pl.DataFrame, features: list[str]) -> np.ndarray:
    """Average ranks per column, missing filled with the column's middle rank, standardised."""
    cols = []
    for f in features:
        r = sample[f].cast(pl.Float64).rank("average").to_numpy().astype(np.float64)
        mid = np.nanmedian(r) if np.isfinite(r).any() else 0.0
        r = np.where(np.isnan(r), mid, r)
        sd = r.std()
        cols.append((r - r.mean()) / sd if sd > 0 else np.zeros_like(r))
    return np.column_stack(cols) if cols else np.zeros((sample.height, 0))


def _exact_spearman(sample: pl.DataFrame, a: str, b: str) -> float | None:
    pair = sample.select(a, b).drop_nulls()
    if pair.height < 30:
        return None
    ra = pair[a].cast(pl.Float64).rank("average").to_numpy()
    rb = pair[b].cast(pl.Float64).rank("average").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def correlation_screen(sample: pl.DataFrame, features: list[str], missing: dict[str, float],
                       max_abs_rho: float, priority: dict[str, int] | None = None) -> dict[str, ScreenRecord]:
    """Drop one of each numeric pair whose |Spearman rho| exceeds max_abs_rho.

    A fast pass ranks every column (missing filled with the middle rank) and
    multiplies the standardised ranks; candidate pairs are then confirmed with
    an exact pairwise-complete Spearman, because filling can manufacture
    correlation between two mostly-missing columns. Of a confirmed pair the
    feature with more missing values is dropped; ties go to `priority` (lower
    is kept, e.g. a table's row count over a column's non-null count, because
    "number of previous applications" can be explained to a customer), then
    to the earlier name. A feature already dropped cannot cause another drop,
    and a feature that has already caused a drop is not dropped later (the
    other member of the pair goes instead; if both have, both stay), so every
    drop's `correlated_with` names a feature that is in the matrix.
    """
    records = {f: ScreenRecord(f) for f in features}
    if len(features) < 2:
        return records
    z = _rank_matrix(sample, features)
    corr = (z.T @ z) / z.shape[0]
    candidates = [
        (abs(corr[i, j]), features[i], features[j])
        for i in range(len(features)) for j in range(i + 1, len(features))
        if abs(corr[i, j]) > max_abs_rho - 0.02
    ]
    dropped: set[str] = set()
    keepers: set[str] = set()
    pri = priority or {}
    # Strongest pairs first; among equally strong ones, pairs involving the most
    # explainable feature first, so a drop always names a feature that is kept.
    order = sorted(candidates, key=lambda t: (-round(t[0], 6), min(pri.get(t[1], 9), pri.get(t[2], 9)), t[1], t[2]))
    for _, a, b in order:
        if a in dropped or b in dropped:
            continue
        rho = _exact_spearman(sample, a, b)
        if rho is None or abs(rho) <= max_abs_rho:
            continue
        if a in keepers and b in keepers:
            continue
        keep, drop = sorted((a, b), key=lambda f: (f not in keepers, missing.get(f, 0.0), pri.get(f, 9), f))
        dropped.add(drop)
        keepers.add(keep)
        records[drop].status = "dropped:correlated"
        records[drop].correlated_with = keep
        records[drop].rho = rho
    return records


def category_levels(train: pl.DataFrame, feature: str, max_levels: int) -> list[str]:
    """The most frequent training levels (ties by value), kept as-is; everything else becomes OTHER_LEVEL."""
    counts = (
        train.select(pl.col(feature).cast(pl.String)).drop_nulls()
        .group_by(feature).agg(pl.len().alias("n"))
        .sort(["n", feature], descending=[True, False])
    )
    return counts[feature].head(max_levels).to_list()


def cap_categories(df: pl.DataFrame, levels: dict[str, list[str]]) -> pl.DataFrame:
    return df.with_columns([
        pl.when(pl.col(f).is_null()).then(None)
        .when(pl.col(f).cast(pl.String).is_in(keep)).then(pl.col(f).cast(pl.String))
        .otherwise(pl.lit(OTHER_LEVEL)).alias(f)
        for f, keep in levels.items()
    ])


def recode_values(df: pl.DataFrame, feature: str, values: tuple[str, ...], to: str) -> pl.DataFrame:
    if feature not in df.columns:
        return df
    return df.with_columns(
        pl.when(pl.col(feature).is_in(list(values))).then(pl.lit(to)).otherwise(pl.col(feature)).alias(feature)
    )


def add_ratios(df: pl.DataFrame, ratios) -> tuple[pl.DataFrame, list[str]]:
    """numerator / denominator; null when either side is missing or the denominator is not positive."""
    added = []
    for r in ratios:
        if r.numerator not in df.columns or r.denominator not in df.columns:
            continue
        df = df.with_columns(
            pl.when(pl.col(r.denominator) > 0)
            .then(pl.col(r.numerator).cast(pl.Float64) / pl.col(r.denominator).cast(pl.Float64))
            .otherwise(None).cast(pl.Float32).alias(r.name)
        )
        added.append(r.name)
    return df, added
