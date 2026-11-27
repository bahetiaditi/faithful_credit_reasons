import json

import polars as pl
import pytest

from fcr.config import load_config
from fcr.data.split import (
    SPLITS, assign_split, build_lock, case_fingerprint, lock_mismatches, read_lock,
    split_summary, stratified_sample, verify_split, write_lock,
)


@pytest.fixture(scope="module")
def cfg(repo_root):
    return load_config(repo_root / "configs" / "config.yaml")


@pytest.fixture()
def base():
    weeks = list(range(92)) * 50
    return pl.DataFrame({
        "case_id": range(len(weeks)), "WEEK_NUM": weeks,
        "date_decision": [f"2019-01-{1 + w % 28:02d}" for w in weeks],
        "target": [int(i % 29 == 0) for i in range(len(weeks))],
    })


def test_boundary_weeks_land_in_the_right_split(cfg, base):
    out = assign_split(base, cfg.split)
    by_week = dict(out.select("WEEK_NUM", "split").unique().iter_rows())
    assert (by_week[63], by_week[64], by_week[77], by_week[78], by_week[91]) == (
        "train", "validation", "validation", "test", "test")
    assert out["split"].dtype == pl.Enum(list(SPLITS))


def test_a_week_outside_every_split_is_an_error(cfg, base):
    with pytest.raises(ValueError, match="outside every split"):
        assign_split(base.with_columns(pl.lit(92).alias("WEEK_NUM")), cfg.split)


def test_summary_counts(cfg, base):
    s = {r["split"]: r for r in split_summary(assign_split(base, cfg.split)).iter_rows(named=True)}
    assert s["train"]["cases"] == 64 * 50 and s["validation"]["cases"] == 14 * 50
    assert s["test"]["first_week"] == 78 and s["test"]["last_week"] == 91
    assert sum(r["cases"] for r in s.values()) == base.height


def test_fingerprint_ignores_order_and_sees_one_changed_id():
    ids = pl.Series(range(1000))
    assert case_fingerprint(ids) == case_fingerprint(ids.reverse())
    assert case_fingerprint(ids) != case_fingerprint(ids.set(pl.Series([False] * 999 + [True]), 5000))
    with pytest.raises(ValueError, match="unique"):
        case_fingerprint(pl.Series([1, 1, 2]))


def test_lock_round_trip_and_verification(cfg, base, tmp_path):
    df = assign_split(base, cfg.split)
    lock = build_lock(df, cfg.split)
    write_lock(lock, tmp_path / "lock.json")
    assert read_lock(tmp_path / "lock.json") == lock
    verify_split(df, cfg.split, lock)

    dropped = df.filter(pl.col("case_id") != 0)
    assert any("cases but lock" in p for p in lock_mismatches(dropped, cfg.split, lock))
    swapped = df.with_columns(pl.when(pl.col("case_id") == 0).then(10**9).otherwise(pl.col("case_id")).alias("case_id"))
    assert any("fingerprint" in p for p in lock_mismatches(swapped, cfg.split, lock))
    flipped = df.with_columns(pl.when(pl.col("case_id") == 1).then(1 - pl.col("target")).otherwise(pl.col("target")).alias("target"))
    assert any("defaults" in p for p in lock_mismatches(flipped, cfg.split, lock))


def test_a_moved_boundary_is_caught_even_on_a_sample(cfg, base):
    df = assign_split(base, cfg.split)
    lock = build_lock(df, cfg.split)
    moved = cfg.split.model_copy(update={"train_weeks": (0, 62), "validation_weeks": (63, 77)})
    problems = lock_mismatches(assign_split(base, moved).sample(200, seed=1), moved, lock, full=False)
    assert any("config weeks" in p for p in problems)


def test_bad_lock_files_are_rejected(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"version": 99, "splits": {}}))
    with pytest.raises(ValueError, match="split lock"):
        read_lock(tmp_path / "x.json")


def test_stratified_sample_is_exact_proportional_and_seeded(cfg, base):
    df = assign_split(base, cfg.split)
    s1 = stratified_sample(df, 1000, seed=3)
    assert s1.height == 1000 and s1["case_id"].is_unique().all()
    assert s1.equals(stratified_sample(df, 1000, seed=3))
    assert not s1.equals(stratified_sample(df, 1000, seed=4))
    full = df.group_by("split", "target").len().sort("split", "target")
    got = s1.group_by("split", "target").len().sort("split", "target")
    expected = full["len"] * 1000 / df.height
    assert ((got["len"] - expected).abs() < 1).all()
    with pytest.raises(ValueError, match="cannot sample"):
        stratified_sample(df, df.height + 1, seed=0)


def test_committed_lock_matches_the_config(repo_root, cfg):
    path = repo_root / "configs" / "split_lock.json"
    if not path.is_file():
        pytest.skip("split lock not committed yet (C7 writes it on Kaggle)")
    lock = read_lock(path)
    for name in SPLITS:
        assert lock["splits"][name]["weeks"] == list(getattr(cfg.split, f"{name}_weeks"))
    sizes = {n: (lock["splits"][n]["cases"], lock["splits"][n]["defaults"]) for n in SPLITS}
    assert sizes == {"train": (1_282_082, 42_524), "validation": (92_870, 2_255), "test": (151_707, 3_215)}


def test_stratified_sample_survives_counts_that_overflow_uint32():
    """C7 regression: 100k rows x n=60k overflows UInt32 (6e9 > 4.3e9) unless cast first."""
    df = pl.DataFrame({"case_id": range(100_000), "split": ["train"] * 100_000,
                       "target": [int(i % 33 == 0) for i in range(100_000)]})
    s = stratified_sample(df, 60_000, seed=1)
    assert s.height == 60_000
    assert abs(s["target"].mean() - df["target"].mean()) < 1 / 60_000 + 1e-9
