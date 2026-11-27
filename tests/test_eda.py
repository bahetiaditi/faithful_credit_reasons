from datetime import date

import polars as pl
import pytest

from fcr.data import eda
from fcr.data.loader import BASE_FAMILY, discover_families, load_base, read_feature_definitions


@pytest.fixture(scope="module")
def families(synthetic):
    return discover_families(synthetic.train_dir)


@pytest.fixture(scope="module")
def base(families):
    return load_base(families)


def test_weekly_rates_add_up_to_the_base_table(base):
    weekly = eda.weekly_default_rate(base)
    assert weekly["applications"].sum() == base.height
    assert weekly["defaults"].sum() == base["target"].sum()
    assert weekly["WEEK_NUM"].is_sorted()
    assert eda.first_week_on_or_after(weekly, date(2020, 3, 11)) is not None
    assert eda.first_week_on_or_after(weekly, date(2030, 1, 1)) is None


def test_missingness_counts_nulls_and_empty_strings(families):
    person = eda.column_missingness(families["person_1"]).filter(pl.col("column") == "empl_employedtotal_800L").row(0, named=True)
    assert person["empty_strings"] > 0, "empty strings are a second form of missing"
    assert person["missing_share"] == pytest.approx((person["nulls"] + person["empty_strings"]) / person["rows"])
    cb = eda.column_missingness(families["static_cb_0"]).filter(pl.col("column") == "riskassesment_940T")
    assert cb["missing_share"][0] > 0.9


def test_missingness_merges_chunks(families):
    miss = eda.column_missingness(families["static_0"])
    assert set(miss["rows"].to_list()) == {2000}, "rows must be summed across both chunks"


def test_placeholder_hash_is_detected(families):
    profiles = [p for f in families.values() for p in eda.masked_value_profiles(f)]
    candidates = eda.placeholder_candidates(profiles)
    assert candidates[0]["value"] == "a55475b1"
    assert candidates[0]["columns"] >= 3


def test_protected_inventory_finds_names_and_values(families, synthetic):
    inventory = eda.protected_inventory(families, read_feature_definitions(synthetic.root / "feature_definitions.csv"))
    by_column = {p["column"]: p for p in inventory}
    assert {"birth_259D", "familystate_447L", "incometype_1044T"} <= set(by_column)
    # income type is found only through its values
    assert by_column["incometype_1044T"]["matched"].startswith("value:")
    assert "HANDICAPPED_2" in by_column["incometype_1044T"]["matched"]
    assert by_column["birth_259D"]["values"] is None, "no sample birth dates in the inventory"
    assert "mainoccupationinc_384A" not in by_column


def test_coverage_is_a_share_per_month(families, base):
    cov = eda.coverage_by_month(families, base)
    assert BASE_FAMILY not in cov["family"].unique().to_list()
    assert cov["coverage"].is_between(0, 1).all()
    person = cov.filter(pl.col("family") == "person_1")
    assert (person["coverage"] == 1.0).all()
    assert person["applications"].sum() == base.height


def _write_base(tmp_path, ids):
    pl.DataFrame({"case_id": ids, "date_decision": ["2019-01-01"] * len(ids), "MONTH": [201901] * len(ids),
                  "WEEK_NUM": [0] * len(ids), "target": [0] * len(ids)}).write_parquet(tmp_path / "train_base.parquet")


def test_chunk_containment_detects_cases_spanning_chunks(tmp_path):
    _write_base(tmp_path, [1, 2, 3])
    pl.DataFrame({"case_id": [1, 2], "num_group1": [0, 0], "x_1A": [1.0, 2.0]}).write_parquet(tmp_path / "train_t_1_0.parquet")
    pl.DataFrame({"case_id": [2, 3], "num_group1": [1, 0], "x_1A": [3.0, 4.0]}).write_parquet(tmp_path / "train_t_1_1.parquet")
    result = eda.chunk_containment(discover_families(tmp_path)["t_1"])
    assert result["cases_spanning_chunks"] == 1
    assert [r["cases"] for r in result["chunk_ranges"]] == [2, 2]


def test_chunk_containment_is_zero_when_chunks_split_by_case(families):
    assert eda.chunk_containment(families["static_0"])["cases_spanning_chunks"] == 0


def test_date_checks_count_unparseable_values_and_ranges(tmp_path):
    _write_base(tmp_path, [1, 2, 3])
    pl.DataFrame({"case_id": [1, 2, 3], "num_group1": [0, 0, 0],
                  "when_5D": ["2019-02-03", "2019-13-45", None]}).write_parquet(tmp_path / "train_t_1.parquet")
    row = eda.date_parse_checks(discover_families(tmp_path)["t_1"]).row(0, named=True)
    assert (row["non_null"], row["unparseable"]) == (2, 1)
    assert row["min"] == row["max"] == date(2019, 2, 3)


def test_date_checks_on_fixtures_find_the_planted_late_dates(families, base):
    checks = pl.concat([eda.date_parse_checks(f) for f in families.values()])
    assert checks["unparseable"].sum() == 0
    approval = checks.filter(pl.col("column") == "approvaldate_319D").row(0, named=True)
    assert approval["max"] > date.fromisoformat(base["date_decision"].max())
