"""Build case-level features for each table family (C5).

Per chunk: read -> null masked placeholders -> attach decision dates -> date
rules (sentinels, anchors, future events, days-before) -> [tax registries:
rename onto one schema] -> aggregate to one row per case. Chunks are processed
independently and concatenated, which is exact because C4 showed no case spans
two chunks; `build_family` re-checks that on its own output and raises if a
case appears twice.

Each family is written as its own parquet file rather than joined into one
wide table: the full join (about 1.5M cases x 2,000 columns) is only made in
C6, after screening has removed most columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from fcr.config import Config
from fcr.data.aggregate import FeatureSpec, aggregate_chunk
from fcr.data.dates import DATE_FORMAT, DateRoles, apply_date_rules, attach_decision_dates
from fcr.data.loader import TableFamily

TAX_FAMILY = "tax_registry_1"
# Harmonised tax-registry columns. Derived ids in the 9000xx range follow the
# suffix convention so the rest of the pipeline treats them like any column.
# No date: registry a's only date is a retrieval timestamp (C5.1), so a merged
# date would exist only for b and c, and its absence would reveal the period.
# Each provider's date is still used for leakage control before merging.
TAX_COLUMNS = {"amount": "taxregamount_900001A", "employer": "taxregemployer_900003M"}


@dataclass
class FamilyResult:
    key: str
    features: pl.DataFrame
    specs: list[FeatureSpec]
    diagnostics: list[dict] = field(default_factory=list)


def decision_dates(base: pl.DataFrame) -> pl.DataFrame:
    return base.select("case_id", pl.col("date_decision").str.to_date(DATE_FORMAT))


def null_masked_placeholders(df: pl.DataFrame, placeholders: tuple[str, ...]) -> tuple[pl.DataFrame, dict]:
    """Masked (M) values that encode missing (C4: a55475b1) become null."""
    cols = [c for c in df.columns if c.endswith("M") and df.schema[c] == pl.String]
    if not cols or not placeholders:
        return df, {}
    counts = df.select([pl.col(c).is_in(list(placeholders)).sum().alias(c) for c in cols]).row(0, named=True)
    cleaned = df.with_columns(
        [pl.when(pl.col(c).is_in(list(placeholders))).then(None).otherwise(pl.col(c)).alias(c) for c in cols]
    )
    return cleaned, {c: n for c, n in counts.items() if n}


def _clean_chunk(df: pl.DataFrame, key: str, cfg: Config, roles: DateRoles, decisions: pl.DataFrame):
    df, masked = null_masked_placeholders(df, cfg.features.masked_missing_values)
    df = attach_decision_dates(df, decisions)
    df, diag = apply_date_rules(df, key, roles)
    diag["masked_nulled"] = masked
    return df, diag


def _check_one_row_per_case(key: str, out: pl.DataFrame) -> None:
    dupes = out.height - out["case_id"].n_unique()
    if dupes:
        raise ValueError(f"{key}: {dupes} cases appear in more than one chunk; per-chunk aggregation is not exact")


def build_family(family: TableFamily, cfg: Config, roles: DateRoles, decisions: pl.DataFrame) -> FamilyResult:
    """All chunks of one family, cleaned and aggregated to one row per case."""
    applicant = family.key in cfg.features.applicant_row_families
    parts, diags, specs = [], [], None
    for path in family.files:
        df, diag = _clean_chunk(pl.read_parquet(path), family.key, cfg, roles, decisions)
        diag["chunk"] = path.name
        out, chunk_specs = aggregate_chunk(df, family.key, family.depth, cfg.features.numeric_stats, applicant)
        specs = specs or chunk_specs
        parts.append(out)
        diags.append(diag)
    features = pl.concat(parts, how="diagonal_relaxed")
    _check_one_row_per_case(family.key, features)
    return FamilyResult(family.key, features, specs or [], diags)


def build_tax_registry(families: dict[str, TableFamily], cfg: Config, roles: DateRoles,
                       decisions: pl.DataFrame) -> FamilyResult | None:
    """Clean each provider's table with its own anchor, rename onto one schema, aggregate once."""
    frames, diags = [], []
    for key, mapping in cfg.features.tax_registries.items():
        if key not in families:
            continue
        for path in families[key].files:
            df, diag = _clean_chunk(pl.read_parquet(path), key, cfg, roles, decisions)
            diag["chunk"] = path.name
            diags.append(diag)
            # The provider is deliberately not kept: it encodes the period, not the applicant.
            frames.append(df.select(
                "case_id", "num_group1",
                pl.col(mapping.amount).cast(pl.Float64).alias(TAX_COLUMNS["amount"]),
                pl.col(mapping.employer).alias(TAX_COLUMNS["employer"]),
            ))
    if not frames:
        return None
    combined = pl.concat(frames)
    out, specs = aggregate_chunk(combined, TAX_FAMILY, 1, cfg.features.numeric_stats)
    _check_one_row_per_case(TAX_FAMILY, out)
    return FamilyResult(TAX_FAMILY, out, specs, diags)


def build_all(families: dict[str, TableFamily], base: pl.DataFrame, cfg: Config, roles: DateRoles):
    """Yield one FamilyResult per family (tax registries merged), smallest memory first-come order."""
    decisions = decision_dates(base)
    tax_sources = set(cfg.features.tax_registries)
    for key, family in families.items():
        if key == "base" or key in tax_sources:
            continue
        yield build_family(family, cfg, roles, decisions)
    tax = build_tax_registry(families, cfg, roles, decisions)
    if tax is not None:
        yield tax


def write_family(result: FamilyResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.key}.parquet"
    result.features.write_parquet(path)
    return path
