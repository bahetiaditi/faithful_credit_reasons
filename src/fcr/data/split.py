"""The out-of-time split (Section 4.5): assign, summarise, freeze and verify.

The split is frozen before any model is trained. C7 writes `configs/split_lock.json`
with, per split, its week range, case and default counts, first and last decision
date, and a fingerprint of its case ids. Every later step that reads the feature
table calls `verify_split` and refuses to run if the table no longer matches the
lock, so a changed config, a rebuilt table or a stray filter cannot silently move
cases between train, validation and test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from fcr.config import SplitConfig

SPLITS = ("train", "validation", "test")
SPLIT_DTYPE = pl.Enum(list(SPLITS))
LOCK_VERSION = 1


def assign_split(df: pl.DataFrame, split: SplitConfig) -> pl.DataFrame:
    """Add a `split` column from WEEK_NUM; every row must fall in exactly one split."""
    expr = pl.lit(None, dtype=pl.String)
    for name in reversed(SPLITS):
        lo, hi = getattr(split, f"{name}_weeks")
        expr = pl.when(pl.col("WEEK_NUM").is_between(lo, hi)).then(pl.lit(name)).otherwise(expr)
    out = df.with_columns(expr.cast(SPLIT_DTYPE).alias("split"))
    unassigned = out.filter(pl.col("split").is_null())
    if unassigned.height:
        weeks = sorted(unassigned["WEEK_NUM"].unique().to_list())
        raise ValueError(f"{unassigned.height} rows in weeks outside every split: {weeks[:10]}")
    return out


def case_fingerprint(case_ids: pl.Series) -> str:
    """Order-independent sha256 (first 16 hex) of a set of case ids."""
    ids = np.sort(case_ids.cast(pl.Int64).to_numpy())
    if len(np.unique(ids)) != len(ids):
        raise ValueError("case ids must be unique to fingerprint")
    return hashlib.sha256(ids.tobytes()).hexdigest()[:16]


def split_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Per split: weeks, decision dates, cases, defaults and default rate."""
    return (
        df.group_by("split")
        .agg(
            pl.col("WEEK_NUM").min().alias("first_week"), pl.col("WEEK_NUM").max().alias("last_week"),
            pl.col("date_decision").cast(pl.String).min().alias("first_decision"),
            pl.col("date_decision").cast(pl.String).max().alias("last_decision"),
            pl.len().alias("cases"), pl.col("target").sum().alias("defaults"),
        )
        .with_columns((pl.col("defaults") / pl.col("cases")).alias("default_rate"))
        .sort("split")
    )


def build_lock(df: pl.DataFrame, split: SplitConfig) -> dict:
    """The frozen description of the split, from a frame that already has a `split` column."""
    summary = {r["split"]: r for r in split_summary(df).iter_rows(named=True)}
    lock = {"version": LOCK_VERSION, "splits": {}}
    for name in SPLITS:
        lo, hi = getattr(split, f"{name}_weeks")
        s = summary[name]
        lock["splits"][name] = {
            "weeks": [lo, hi], "cases": s["cases"], "defaults": int(s["defaults"]),
            "first_decision": s["first_decision"], "last_decision": s["last_decision"],
            "case_fingerprint": case_fingerprint(df.filter(pl.col("split") == name)["case_id"]),
        }
    return lock


def write_lock(lock: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(lock, indent=2) + "\n")


def read_lock(path: str | Path) -> dict:
    lock = json.loads(Path(path).read_text())
    if lock.get("version") != LOCK_VERSION or set(lock.get("splits", {})) != set(SPLITS):
        raise ValueError(f"{path} is not a version-{LOCK_VERSION} split lock")
    return lock


def lock_mismatches(df: pl.DataFrame, split: SplitConfig, lock: dict, *, full: bool = True) -> list[str]:
    """Differences between a table (with `split`) and the lock; empty means it matches.

    `full=False` checks a sample: only that its cases sit in the locked weeks.
    """
    problems = []
    for name in SPLITS:
        locked = lock["splits"][name]
        weeks = list(getattr(split, f"{name}_weeks"))
        if weeks != locked["weeks"]:
            problems.append(f"{name}: config weeks {weeks} but lock {locked['weeks']}")
        part = df.filter(pl.col("split") == name)
        lo, hi = locked["weeks"]
        if part.height and not part["WEEK_NUM"].is_between(lo, hi).all():
            problems.append(f"{name}: rows outside locked weeks {lo}-{hi}")
        if not full:
            continue
        if part.height != locked["cases"]:
            problems.append(f"{name}: {part.height:,} cases but lock {locked['cases']:,}")
        elif int(part["target"].sum()) != locked["defaults"]:
            problems.append(f"{name}: {int(part['target'].sum()):,} defaults but lock {locked['defaults']:,}")
        elif case_fingerprint(part["case_id"]) != locked["case_fingerprint"]:
            problems.append(f"{name}: same counts but different cases (fingerprint mismatch)")
    return problems


def verify_split(df: pl.DataFrame, split: SplitConfig, lock: dict, *, full: bool = True) -> None:
    problems = lock_mismatches(df, split, lock, full=full)
    if problems:
        raise ValueError("feature table does not match the frozen split:\n  " + "\n  ".join(problems))


def stratified_sample(df: pl.DataFrame, n: int, seed: int, by: tuple[str, ...] = ("split", "target")) -> pl.DataFrame:
    """Proportional stratified sample of exactly n rows (largest-remainder allocation).

    Every stratum keeps its share of the population, so the sample's split sizes
    and default rates match the full table to within one case per stratum.
    """
    if n > df.height:
        raise ValueError(f"cannot sample {n:,} rows from {df.height:,}")
    sizes = df.group_by(by).len().sort(list(by))
    quota = sizes["len"].to_numpy() * n / df.height
    take = np.floor(quota).astype(int)
    for i in np.argsort(-(quota - take), kind="stable")[: n - take.sum()]:
        take[i] += 1
    alloc = sizes.with_columns(pl.Series("take", take))
    rng = np.random.default_rng(seed)
    return (
        df.with_columns(pl.Series("__u", rng.random(df.height)))
        .join(alloc.select(*by, "take"), on=list(by), how="left")
        .with_columns(pl.col("__u").rank("ordinal").over(list(by)).alias("__r"))
        .filter(pl.col("__r") <= pl.col("take"))
        .drop("__u", "__r", "take")
        .sort("case_id")
    )
