"""The synthetic fixture contract (see fcr/data/synthetic.py)."""

import json
from collections import defaultdict

import polars as pl
import pytest

from fcr.data.naming import GROUP_COLUMNS_BY_DEPTH, KEY_COLUMNS, feature_suffix, parse_table_name
from fcr.data.synthetic import SyntheticSpec, generate_synthetic_dataset


def _frames(ds):
    return {p.stem: pl.read_parquet(p) for p in sorted(ds.train_dir.glob("*.parquet"))}


def test_every_file_follows_the_naming_convention(synthetic):
    stems = [p.stem for p in synthetic.train_dir.glob("*.parquet")]
    assert "train_base" in stems
    parsed = [parse_table_name(s) for s in stems]
    assert {t.depth for t in parsed} == {0, 1, 2}, "fixture must exercise all three depths"
    assert any(t.chunk is not None for t in parsed), "fixture must exercise chunked families"


def test_chunks_of_a_family_share_identical_schemas(synthetic):
    by_family = defaultdict(list)
    for stem, df in _frames(synthetic).items():
        t = parse_table_name(stem)
        by_family[(t.family, t.depth)].append(df.schema)
    for family, schemas in by_family.items():
        assert all(s == schemas[0] for s in schemas), f"chunk schemas differ for {family}"


def test_group_columns_match_depth(synthetic):
    for stem, df in _frames(synthetic).items():
        t = parse_table_name(stem)
        if t.is_base:
            continue
        groups = [c for c in df.columns if c.startswith("num_group")]
        assert tuple(groups) == GROUP_COLUMNS_BY_DEPTH[t.depth], stem
        keys = ["case_id", *groups]
        assert df.select(keys).is_duplicated().sum() == 0, f"duplicate keys in {stem}"


def test_every_feature_column_has_a_suffix_and_a_definition(synthetic):
    definitions = pl.read_csv(synthetic.root / "feature_definitions.csv")
    assert definitions.columns == ["Variable", "Description"]
    defined = set(definitions["Variable"])
    for stem, df in _frames(synthetic).items():
        for col in df.columns:
            if col in KEY_COLUMNS:
                continue
            assert feature_suffix(col) is not None
            assert col in defined, f"{col} in {stem} has no definition"


def test_all_case_ids_belong_to_base(synthetic):
    frames = _frames(synthetic)
    base_ids = set(frames["train_base"]["case_id"])
    for stem, df in frames.items():
        assert set(df["case_id"]) <= base_ids, stem


def test_base_time_columns_are_consistent(synthetic):
    base = pl.read_parquet(synthetic.train_dir / "train_base.parquet")
    spec = SyntheticSpec(**synthetic.manifest["spec"])
    assert base["WEEK_NUM"].min() >= 0 and base["WEEK_NUM"].max() <= spec.n_weeks - 1
    parsed = base.with_columns(pl.col("date_decision").str.to_date())
    month = parsed.select(
        (pl.col("date_decision").dt.year() * 100 + pl.col("date_decision").dt.month()).alias("m")
    )["m"]
    assert (month == base["MONTH"]).all()
    # later weeks never have earlier decision dates
    weekly_min = parsed.group_by("WEEK_NUM").agg(pl.col("date_decision").min()).sort("WEEK_NUM")
    assert weekly_min["date_decision"].is_sorted()


def test_default_rate_is_close_to_spec(synthetic):
    observed = synthetic.manifest["observed_default_rate"]
    assert abs(observed - synthetic.manifest["spec"]["default_rate"]) < 0.02


def test_target_carries_signal(synthetic):
    # Applicants with any bureau days past due must default more often; later
    # model commits rely on there being something to learn.
    base = pl.read_parquet(synthetic.train_dir / "train_base.parquet")
    bureau = pl.concat([pl.read_parquet(p) for p in synthetic.train_dir.glob("train_credit_bureau_a_2_*.parquet")])
    dpd = bureau.group_by("case_id").agg((pl.col("syn_pmtsdpd_1031P").max() > 0).alias("any_dpd"))
    joined = base.join(dpd, on="case_id", how="inner")
    rates = joined.group_by("any_dpd").agg(pl.col("target").mean()).sort("any_dpd")["target"]
    assert rates[1] > 2 * rates[0]


def test_planted_leakage_matches_manifest(synthetic):
    base = pl.read_parquet(synthetic.train_dir / "train_base.parquet").select("case_id", "date_decision")
    prev = pl.concat([pl.read_parquet(p) for p in synthetic.train_dir.glob("train_applprev_1_*.parquet")])
    leaked = (
        prev.join(base, on="case_id")
        .filter(pl.col("approvaldate_319D").str.to_date() > pl.col("date_decision").str.to_date())
        .height
    )
    assert leaked > 0
    assert leaked == synthetic.manifest["leakage"]["n_rows_after_decision"]


def test_realistic_data_quirks_are_present(synthetic):
    person = pl.read_parquet(synthetic.train_dir / "train_person_1.parquet").filter(pl.col("num_group1") == 0)
    # dominant masked hash, nulls for missing categories, disability-coded income types
    assert person["education_927M"].value_counts(sort=True)["education_927M"][0] == "a55475b1"
    assert person["empl_employedtotal_800L"].null_count() / person.height > 0.5
    assert (person["empl_employedtotal_800L"] == "").sum() == 0
    assert set(person["sex_738L"].drop_nulls().unique()) == {"F", "M"}
    assert person["incometype_1044T"].str.starts_with("HANDICAPPED").any()
    cb = pl.read_parquet(synthetic.train_dir / "train_static_cb_0.parquet")
    assert cb["riskassesment_940T"].null_count() / cb.height > 0.9
    related = pl.read_parquet(synthetic.train_dir / "train_person_1.parquet").filter(pl.col("num_group1") > 0)
    assert related.height > 0 and related["birth_259D"].null_count() == related.height


def test_generation_is_deterministic(tmp_path):
    spec = SyntheticSpec(n_cases=300, seed=3)
    a = generate_synthetic_dataset(tmp_path / "a", spec)
    b = generate_synthetic_dataset(tmp_path / "b", spec)
    for path in a.train_dir.glob("*.parquet"):
        assert pl.read_parquet(path).equals(pl.read_parquet(b.train_dir / path.name)), path.name
    assert json.loads((a.root / "manifest.json").read_text()) == json.loads((b.root / "manifest.json").read_text())


def test_different_seeds_differ(tmp_path):
    a = generate_synthetic_dataset(tmp_path / "a", SyntheticSpec(n_cases=300, seed=1))
    b = generate_synthetic_dataset(tmp_path / "b", SyntheticSpec(n_cases=300, seed=2))
    base_a = pl.read_parquet(a.train_dir / "train_base.parquet")
    base_b = pl.read_parquet(b.train_dir / "train_base.parquet")
    assert not base_a.equals(base_b)


def test_invalid_default_rate_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate_synthetic_dataset(tmp_path, SyntheticSpec(n_cases=10, default_rate=1.0))


def test_no_case_spans_two_chunks(synthetic):
    """Real chunked families never split a case across files (C4); per-chunk
    aggregation in C5 depends on the fixtures sharing that property."""
    seen: dict[tuple[str, int], str] = {}
    for path in sorted(synthetic.train_dir.glob("*.parquet")):
        t = parse_table_name(path.stem)
        if t.chunk is None:
            continue
        for cid in pl.read_parquet(path)["case_id"].unique().to_list():
            owner = seen.setdefault((t.family, cid), path.stem)
            assert owner == path.stem, f"case {cid} in both {owner} and {path.stem}"
