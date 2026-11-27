"""Discover, scan and describe the competition tables.

Everything here is lazy. A family's chunks are scanned and concatenated
without reading their contents, so the caller decides which columns and rows
ever reach memory. That matters on Kaggle (31 GB for about 1.3 GB of
compressed parquet that expands many-fold) and is the only way the same code
can also run on the 8 GB MacBook against the local sample.

Chunks of one family can disagree on a column's type: a column that is
entirely null in one chunk is stored as Null there and as String or Float64
in another. Chunks are therefore combined with diagonal_relaxed, which widens
each column to the common supertype, and every disagreement is recorded in the
schema registry so it is visible rather than silently absorbed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from fcr.data.naming import GROUP_COLUMNS_BY_DEPTH, KEY_COLUMNS, Split, feature_suffix, parse_table_name

BASE_FAMILY = "base"


@dataclass(frozen=True)
class TableFamily:
    """All chunk files of one table, for example credit_bureau_a at depth 2."""

    key: str            # "base", or "<family>_<depth>", e.g. "credit_bureau_a_2"
    family: str
    depth: int
    files: tuple[Path, ...]

    @property
    def group_columns(self) -> tuple[str, ...]:
        return GROUP_COLUMNS_BY_DEPTH[self.depth]

    @property
    def key_columns(self) -> tuple[str, ...]:
        return ("case_id", *self.group_columns)


def discover_families(table_dir: str | Path, split: Split = "train") -> dict[str, TableFamily]:
    """Group the parquet files of one split into families, chunks in numeric order."""
    table_dir = Path(table_dir)
    grouped: dict[str, list[tuple[int, Path]]] = {}
    meta: dict[str, tuple[str, int]] = {}
    for path in table_dir.glob(f"{split}_*.parquet"):
        name = parse_table_name(path.stem)
        key = BASE_FAMILY if name.is_base else f"{name.family}_{name.depth}"
        # Numeric chunk order: chunk 10 must follow 9, not 1.
        grouped.setdefault(key, []).append((name.chunk if name.chunk is not None else -1, path))
        meta[key] = (name.family, name.depth)
    if BASE_FAMILY not in grouped:
        raise FileNotFoundError(f"no {split}_base.parquet in {table_dir}")
    return {
        key: TableFamily(key=key, family=meta[key][0], depth=meta[key][1],
                         files=tuple(p for _, p in sorted(chunks)))
        for key, chunks in sorted(grouped.items())
    }


def scan_family(family: TableFamily) -> pl.LazyFrame:
    """One lazy frame over all chunks, with column types widened to agree."""
    frames = [pl.scan_parquet(path) for path in family.files]
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="diagonal_relaxed")


def load_base(families: dict[str, TableFamily]) -> pl.DataFrame:
    """The label table, with the invariants every later step relies on checked."""
    base = scan_family(families[BASE_FAMILY]).collect()
    expected = {"case_id", "date_decision", "MONTH", "WEEK_NUM", "target"}
    missing = expected - set(base.columns)
    if missing:
        raise ValueError(f"base table is missing columns {sorted(missing)}")
    if base["case_id"].is_duplicated().any():
        raise ValueError("case_id is not unique in the base table")
    if not set(base["target"].unique().to_list()) <= {0, 1}:
        raise ValueError("target must be binary")
    return base


def read_feature_definitions(path: str | Path) -> dict[str, str]:
    defs = pl.read_csv(path)
    if not {"Variable", "Description"} <= set(defs.columns):
        raise ValueError(f"unexpected feature_definitions columns: {defs.columns}")
    return dict(zip(defs["Variable"].to_list(), defs["Description"].to_list()))


def _column_types_by_chunk(family: TableFamily) -> dict[str, dict[str, str]]:
    """{column: {chunk file name: dtype}} from parquet metadata only (no data read)."""
    out: dict[str, dict[str, str]] = {}
    for path in family.files:
        for column, dtype in pl.read_parquet_schema(path).items():
            out.setdefault(column, {})[path.name] = str(dtype)
    return out


def describe_family(
    family: TableFamily,
    definitions: dict[str, str],
    base_case_ids: pl.LazyFrame | None = None,
) -> dict[str, Any]:
    """One registry entry: sizes, key integrity, column types and naming issues.

    Reads only the key columns, so it is cheap even for the largest family.
    """
    lf = scan_family(family)
    unified = lf.collect_schema()
    keys = list(family.key_columns)
    stats = lf.select(
        pl.len().alias("rows"),
        pl.col("case_id").n_unique().alias("cases"),
        pl.struct(keys).n_unique().alias("unique_keys"),
    ).collect().row(0, named=True)

    orphans = None
    if base_case_ids is not None and family.key != BASE_FAMILY:
        orphans = (
            lf.select("case_id").unique()
            .join(base_case_ids, on="case_id", how="anti")
            .select(pl.len()).collect().item()
        )

    chunk_types = _column_types_by_chunk(family)
    columns, suffix_errors = [], []
    for name, dtype in unified.items():
        try:
            suffix = feature_suffix(name)
        except ValueError:
            suffix = None
            suffix_errors.append(name)
        types_seen = sorted(set(chunk_types.get(name, {}).values()))
        columns.append({
            "name": name,
            "dtype": str(dtype),
            "suffix": suffix,
            "described": name in definitions or name in KEY_COLUMNS,
            "chunk_dtypes": types_seen if len(types_seen) > 1 else None,
        })

    return {
        "key": family.key,
        "family": family.family,
        "depth": family.depth,
        "files": [p.name for p in family.files],
        "rows": stats["rows"],
        "cases": stats["cases"],
        "duplicate_keys": stats["rows"] - stats["unique_keys"],
        "orphan_cases": orphans,
        "columns": columns,
        "suffix_errors": suffix_errors,
        "dtype_conflicts": [c["name"] for c in columns if c["chunk_dtypes"]],
        "undescribed": [c["name"] for c in columns if not c["described"]],
    }


def build_schema_registry(table_dir: str | Path, definitions_path: str | Path, split: Split = "train") -> dict[str, Any]:
    """Describe every family of a split. The input C4's EDA and C5's aggregation plan start from."""
    families = discover_families(table_dir, split)
    definitions = read_feature_definitions(definitions_path)
    base_ids = scan_family(families[BASE_FAMILY]).select("case_id")
    return {
        "split": split,
        "families": {key: describe_family(fam, definitions, base_ids) for key, fam in families.items()},
    }


def write_registry(registry: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=1))
    return path
