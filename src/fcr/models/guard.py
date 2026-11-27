"""Keep the test weeks out of model development (Section 7.5: the test set is used once).

Every model script up to the final evaluation calls `development_table`, which
loads and fully verifies the feature table (the split check needs every row),
then removes the test rows before anything else can see them. Fitting and
tuning functions call `assert_no_test` on what they are given, so a test row
reaching a model is an exception, not a silent leak.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl

from fcr.config import Config
from fcr.data.dataset import FeatureTable


class HeldOutWeeksTouched(RuntimeError):
    """Raised when a test-week row reaches model development."""


def assert_no_test(df: pl.DataFrame, cfg: Config) -> None:
    first_test_week = cfg.split.test_weeks[0]
    if df.height and df["WEEK_NUM"].max() >= first_test_week:
        raise HeldOutWeeksTouched(f"rows from week {first_test_week} or later reached model development")
    if "split" in df.columns and (df["split"] == "test").any():
        raise HeldOutWeeksTouched("rows labelled test reached model development")


def development_table(table: FeatureTable, cfg: Config) -> FeatureTable:
    """The same table with its test rows removed (after the loader has verified the split)."""
    dev = table.matrix.filter(pl.col("split") != "test")
    assert_no_test(dev, cfg)
    slices = table.slices.join(dev.select("case_id"), on="case_id", how="semi")
    return replace(table, matrix=dev, slices=slices)


def train_validation(table: FeatureTable, cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    assert_no_test(table.matrix, cfg)
    return table.split("train"), table.split("validation")
