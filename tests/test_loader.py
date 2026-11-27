import json

import polars as pl
import pytest

from fcr.data.loader import (
    BASE_FAMILY,
    build_schema_registry,
    describe_family,
    discover_families,
    load_base,
    read_feature_definitions,
    scan_family,
    write_registry,
)


@pytest.fixture(scope="module")
def families(synthetic):
    return discover_families(synthetic.train_dir)


def test_discovers_every_family_with_its_depth(families):
    assert {k: f.depth for k, f in families.items()} == {
        "applprev_1": 1,
        "base": 0,
        "credit_bureau_a_2": 2,
        "person_1": 1,
        "static_0": 0,
        "static_cb_0": 0,
        "tax_registry_a_1": 1,
    }
    assert families["credit_bureau_a_2"].key_columns == ("case_id", "num_group1", "num_group2")


def test_chunks_are_concatenated_without_losing_rows(families, synthetic):
    for key, family in families.items():
        expected = sum(synthetic.manifest["tables"][p.stem] for p in family.files)
        assert scan_family(family).select(pl.len()).collect().item() == expected, key


def test_chunks_sort_numerically_not_alphabetically(tmp_path):
    df = pl.DataFrame({"case_id": [1], "num_group1": [0], "x_1A": [1.0]})
    for chunk in (0, 1, 2, 10):
        df.write_parquet(tmp_path / f"train_debt_1_{chunk}.parquet")
    pl.DataFrame({"case_id": [1], "date_decision": ["2019-01-01"], "MONTH": [201901],
                  "WEEK_NUM": [0], "target": [0]}).write_parquet(tmp_path / "train_base.parquet")
    files = discover_families(tmp_path)["debt_1"].files
    assert [p.stem.rsplit("_", 1)[1] for p in files] == ["0", "1", "2", "10"]


def test_missing_base_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_families(tmp_path)


def test_load_base_checks_invariants(families, synthetic):
    base = load_base(families)
    assert base.height == synthetic.manifest["n_cases"]
    assert base["target"].mean() == pytest.approx(synthetic.manifest["observed_default_rate"])


@pytest.fixture
def awkward(tmp_path):
    """Chunks that disagree on a type, a badly named column, a duplicate key and an orphan."""
    pl.DataFrame({"case_id": [1, 2, 3], "date_decision": ["2019-01-01"] * 3, "MONTH": [201901] * 3,
                  "WEEK_NUM": [0] * 3, "target": [0, 1, 0]}).write_parquet(tmp_path / "train_base.parquet")
    pl.DataFrame({"case_id": [1, 1], "num_group1": [0, 0], "note_7M": [None, None], "oddname": [1, 2]},
                 schema_overrides={"note_7M": pl.Null}).write_parquet(tmp_path / "train_misc_1_0.parquet")
    pl.DataFrame({"case_id": [2, 99], "num_group1": [0, 0], "note_7M": ["a", "b"], "oddname": [3, 4]}
                 ).write_parquet(tmp_path / "train_misc_1_1.parquet")
    (tmp_path / "defs.csv").write_text("Variable,Description\nnote_7M,A note.\n")
    return tmp_path


def test_type_disagreements_are_widened_and_reported(awkward):
    family = discover_families(awkward)["misc_1"]
    assert scan_family(family).collect_schema()["note_7M"] == pl.String
    entry = describe_family(family, read_feature_definitions(awkward / "defs.csv"))
    assert entry["dtype_conflicts"] == ["note_7M"]


def test_registry_reports_problems_instead_of_crashing(awkward):
    entry = build_schema_registry(awkward, awkward / "defs.csv")["families"]["misc_1"]
    assert entry["suffix_errors"] == ["oddname"]
    assert entry["duplicate_keys"] == 1
    assert entry["orphan_cases"] == 1          # case 99 is not in base
    assert "oddname" in entry["undescribed"]
    assert "note_7M" not in entry["undescribed"]


def test_registry_on_clean_fixtures_has_no_problems(synthetic, tmp_path):
    registry = build_schema_registry(synthetic.train_dir, synthetic.root / "feature_definitions.csv")
    for key, entry in registry["families"].items():
        assert entry["suffix_errors"] == [], key
        assert entry["duplicate_keys"] == 0, key
        assert entry["undescribed"] == [], key
        assert entry["orphan_cases"] in (0, None), key
    assert registry["families"][BASE_FAMILY]["rows"] == synthetic.manifest["n_cases"]
    path = write_registry(registry, tmp_path / "out" / "registry.json")
    assert json.loads(path.read_text())["split"] == "train"
