"""C4 exploratory analysis: the facts the feature build and model card depend on.

Every function works one chunk file at a time and reads only the columns it
needs. C3 showed why: counting unique keys across credit_bureau_a_2's 188M rows
in one pass peaked at 16 GB. Per-file partial results are merged afterwards,
which is exact for sums and counts and for unions of id sets.

Missing values: a String column can encode "missing" as an empty string as well
as null (C1 saw this in employment fields), so both are counted.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import polars as pl

from fcr.data.loader import BASE_FAMILY, TableFamily
from fcr.data.naming import KEY_COLUMNS, feature_suffix

DATE_FORMAT = "%Y-%m-%d"

# Names and descriptions that could identify a protected or sensitive
# attribute. Word boundaries keep "age" from matching "mortgage" or "average".
PROTECTED_PATTERN = re.compile(
    r"\b(sex|gender|age|birth\w*|born|family|marital|married|spouse|handicap\w*|disab\w*|"
    r"religio\w*|ethnic\w*|nationality|citizen\w*|race)\b",
    re.IGNORECASE,
)


# Category VALUES that reveal a protected attribute even when the column's name
# and definition do not (income type carries HANDICAPPED_2 and _3).
# The inline (?i) flag travels with .pattern into polars, which ignores re flags.
PROTECTED_VALUE_PATTERN = re.compile(r"(?i)HANDICAP|DISAB|\bFEMALE\b|\bMALE\b|PREGNAN")


def _feature_columns(schema: dict[str, Any]) -> list[str]:
    return [c for c in schema if c not in KEY_COLUMNS]


# ---------------------------------------------------------------------------
# 1. Default rate and volume over time
# ---------------------------------------------------------------------------


def weekly_default_rate(base: pl.DataFrame) -> pl.DataFrame:
    """One row per WEEK_NUM: applications, defaults, default rate, first decision date."""
    return (
        base.group_by("WEEK_NUM")
        .agg(
            pl.len().alias("applications"),
            pl.col("target").sum().alias("defaults"),
            pl.col("date_decision").min().alias("first_date"),
        )
        .with_columns((pl.col("defaults") / pl.col("applications")).alias("default_rate"))
        .sort("WEEK_NUM")
    )


def first_week_on_or_after(weekly: pl.DataFrame, day: date) -> int | None:
    rows = weekly.filter(pl.col("first_date").str.to_date(DATE_FORMAT) >= day)
    return int(rows["WEEK_NUM"].min()) if rows.height else None


# ---------------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------------


def column_missingness(family: TableFamily) -> pl.DataFrame:
    """Per feature column: rows, nulls, empty strings, and missing share (row level)."""
    totals: dict[str, dict[str, Any]] = {}
    rows_total = 0
    for path in family.files:
        schema = pl.read_parquet_schema(path)
        cols = _feature_columns(schema)
        exprs = [pl.len().alias("__rows")]
        for c in cols:
            exprs.append(pl.col(c).null_count().alias(f"{c}__null"))
            if schema[c] == pl.String:
                exprs.append((pl.col(c) == "").sum().alias(f"{c}__empty"))
        stats = pl.scan_parquet(path).select(exprs).collect().row(0, named=True)
        rows_total += stats["__rows"]
        for c in cols:
            entry = totals.setdefault(c, {"nulls": 0, "empty_strings": 0, "dtypes": set()})
            entry["nulls"] += stats[f"{c}__null"]
            entry["empty_strings"] += stats.get(f"{c}__empty") or 0
            entry["dtypes"].add(str(schema[c]))
    out = []
    for c, e in totals.items():
        missing = e["nulls"] + e["empty_strings"]
        out.append({
            "family": family.key,
            "column": c,
            "suffix": feature_suffix(c),
            "dtype": "/".join(sorted(e["dtypes"])),
            "rows": rows_total,
            "nulls": e["nulls"],
            "empty_strings": e["empty_strings"],
            "missing_share": missing / rows_total if rows_total else 1.0,
        })
    return pl.DataFrame(out, schema_overrides={"rows": pl.Int64, "nulls": pl.Int64, "empty_strings": pl.Int64})


# ---------------------------------------------------------------------------
# 3. Masked categorical columns
# ---------------------------------------------------------------------------


def masked_value_profiles(family: TableFamily, top_k: int = 5) -> list[dict[str, Any]]:
    """For each M column: distinct values, and the most frequent values with their shares."""
    counts: dict[str, list[pl.DataFrame]] = {}
    for path in family.files:
        schema = pl.read_parquet_schema(path)
        for c in (c for c in _feature_columns(schema) if c.endswith("M")):
            vc = (
                pl.scan_parquet(path).select(pl.col(c).cast(pl.String).alias("value"))
                .group_by("value").agg(pl.len().alias("n")).collect()
            )
            counts.setdefault(c, []).append(vc)
    profiles = []
    for c, parts in counts.items():
        merged = pl.concat(parts).group_by("value").agg(pl.col("n").sum()).sort("n", descending=True)
        non_null = merged.filter(pl.col("value").is_not_null())
        total_non_null = int(non_null["n"].sum()) if non_null.height else 0
        top = [
            {"value": v, "share_of_non_null": n / total_non_null}
            for v, n in non_null.head(top_k).iter_rows()
        ] if total_non_null else []
        profiles.append({
            "family": family.key,
            "column": c,
            "distinct": non_null.height,
            "non_null": total_non_null,
            "top": top,
        })
    return profiles


def placeholder_candidates(profiles: list[dict[str, Any]], min_columns: int = 3) -> list[dict[str, Any]]:
    """Values that are the single most frequent value in many masked columns.

    A hash that tops many unrelated columns is far more likely to encode
    "missing" or "not applicable" than a real category (C1 saw a55475b1 cover
    52% of education). Candidates only: C6 decides how to treat each one.
    """
    tops: dict[str, list[tuple[str, float]]] = {}
    for p in profiles:
        if p["top"]:
            first = p["top"][0]
            tops.setdefault(first["value"], []).append((f"{p['family']}.{p['column']}", first["share_of_non_null"]))
    return sorted(
        (
            {"value": v, "columns": len(cols), "median_share": sorted(s for _, s in cols)[len(cols) // 2],
             "examples": [name for name, _ in cols[:5]]}
            for v, cols in tops.items() if len(cols) >= min_columns
        ),
        key=lambda d: -d["columns"],
    )


# ---------------------------------------------------------------------------
# 4. Protected and sensitive attributes
# ---------------------------------------------------------------------------


def protected_inventory(
    families: dict[str, TableFamily], definitions: dict[str, str], top_k: int = 12
) -> list[dict[str, Any]]:
    """Columns that may carry a protected attribute, with their category values.

    Two routes: the name or definition mentions one, or (for text category
    columns) one of the values does. The second route is what finds
    disability-coded income types.
    """
    found = []
    for key, family in families.items():
        schema = pl.read_parquet_schema(family.files[0])
        for c in _feature_columns(schema):
            text = f"{c.replace('_', ' ')} {definitions.get(c, '')}"
            match = PROTECTED_PATTERN.search(text)
            matched = match.group(0).lower() if match else None
            if matched is None and schema[c] == pl.String and not c.endswith("D"):
                flagged = (
                    pl.concat([pl.scan_parquet(p).select(pl.col(c).unique()) for p in family.files])
                    .unique().filter(pl.col(c).str.contains(PROTECTED_VALUE_PATTERN.pattern))
                    .collect()[c].to_list()
                )
                if flagged:
                    matched = "value: " + ", ".join(sorted(flagged)[:4])
            if matched is None:
                continue
            entry = {"family": key, "column": c, "matched": matched,
                     "description": definitions.get(c, ""), "dtype": str(schema[c]), "values": None}
            # Category values only: printing sample dates (birth dates) adds
            # nothing to the inventory and is closer to applicant-level data.
            if schema[c] in (pl.String, pl.Boolean) and not c.endswith("D"):
                vc = (
                    pl.concat([pl.scan_parquet(p).select(pl.col(c).cast(pl.String).alias("v")) for p in family.files])
                    .group_by("v").agg(pl.len().alias("n")).sort("n", descending=True).head(top_k).collect()
                )
                entry["values"] = [{"value": v, "rows": n} for v, n in vc.iter_rows()]
            found.append(entry)
    return found


# ---------------------------------------------------------------------------
# 5. Coverage over time
# ---------------------------------------------------------------------------


def family_case_ids(family: TableFamily) -> pl.Series:
    parts = [pl.scan_parquet(p).select("case_id").unique().collect()["case_id"] for p in family.files]
    return pl.concat(parts).unique()


def coverage_by_month(families: dict[str, TableFamily], base: pl.DataFrame) -> pl.DataFrame:
    """Long table: MONTH, family, share of that month's applications with any row in the family."""
    frames = []
    for key, family in families.items():
        if key == BASE_FAMILY:
            continue
        ids = family_case_ids(family)
        frames.append(
            base.select("case_id", "MONTH")
            .with_columns(pl.col("case_id").is_in(ids.implode()).alias("covered"))
            .group_by("MONTH").agg(pl.col("covered").mean().alias("coverage"), pl.len().alias("applications"))
            .with_columns(pl.lit(key).alias("family"))
        )
    return pl.concat(frames).select("family", "MONTH", "applications", "coverage").sort("family", "MONTH")


# ---------------------------------------------------------------------------
# 6. Chunk containment
# ---------------------------------------------------------------------------


def chunk_containment(family: TableFamily) -> dict[str, Any]:
    """Do a case's rows ever span two chunk files?

    If never, any per-case aggregate computed chunk by chunk is exact, which
    is what makes credit_bureau_a_2 tractable in memory.
    """
    per_chunk = []
    for i, path in enumerate(family.files):
        ids = pl.scan_parquet(path).select("case_id").unique().collect()
        per_chunk.append(ids.with_columns(pl.lit(i).alias("chunk")))
    stacked = pl.concat(per_chunk)
    spanning = stacked.group_by("case_id").agg(pl.len().alias("chunks")).filter(pl.col("chunks") > 1).height
    ranges = (
        stacked.group_by("chunk")
        .agg(pl.col("case_id").min().alias("min_case_id"), pl.col("case_id").max().alias("max_case_id"),
             pl.len().alias("cases"))
        .sort("chunk")
    )
    return {
        "family": family.key,
        "chunks": len(family.files),
        "cases_spanning_chunks": spanning,
        "chunk_ranges": ranges.to_dicts(),
    }


# ---------------------------------------------------------------------------
# 7. Date formats
# ---------------------------------------------------------------------------


def date_parse_checks(family: TableFamily, fmt: str = DATE_FORMAT) -> pl.DataFrame:
    """Per D column: non-null values, values that fail to parse, and the parsed range."""
    agg: dict[str, dict[str, Any]] = {}
    for path in family.files:
        schema = pl.read_parquet_schema(path)
        cols = [c for c in _feature_columns(schema) if c.endswith("D")]
        if not cols:
            continue
        exprs = []
        for c in cols:
            parsed = (pl.col(c).str.to_date(fmt, strict=False) if schema[c] == pl.String else pl.col(c).cast(pl.Date))
            exprs += [
                pl.col(c).is_not_null().sum().alias(f"{c}__nn"),
                (pl.col(c).is_not_null() & parsed.is_null()).sum().alias(f"{c}__bad"),
                parsed.min().alias(f"{c}__min"),
                parsed.max().alias(f"{c}__max"),
            ]
        row = pl.scan_parquet(path).select(exprs).collect().row(0, named=True)
        for c in cols:
            e = agg.setdefault(c, {"non_null": 0, "unparseable": 0, "min": None, "max": None})
            e["non_null"] += row[f"{c}__nn"]
            e["unparseable"] += row[f"{c}__bad"]
            for k, pick in (("min", min), ("max", max)):
                v = row[f"{c}__{k}"]
                if v is not None:
                    e[k] = v if e[k] is None else pick(e[k], v)
    return pl.DataFrame(
        [{"family": family.key, "column": c, **e} for c, e in agg.items()],
        schema={"family": pl.String, "column": pl.String, "non_null": pl.Int64, "unparseable": pl.Int64,
                "min": pl.Date, "max": pl.Date},
    )
