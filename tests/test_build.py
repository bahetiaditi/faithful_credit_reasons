"""End to end on the synthetic fixtures: the C5 checks that need a whole dataset."""

import polars as pl
import pytest

from fcr.config import load_config, resolve_path
from fcr.data.build import TAX_COLUMNS, TAX_FAMILY, build_all, build_family, decision_dates
from fcr.data.dates import load_date_roles
from fcr.data.loader import discover_families, load_base


@pytest.fixture(scope="module")
def setup(synthetic, repo_root):
    cfg = load_config(repo_root / "configs" / "config.yaml")
    roles = load_date_roles(resolve_path(cfg.features.date_roles_file, repo_root))
    families = discover_families(synthetic.train_dir)
    base = load_base(families)
    results = {r.key: r for r in build_all(families, base, cfg, roles)}
    return cfg, roles, families, base, results


def test_every_family_has_one_row_per_case(setup):
    *_, base, results = setup
    base_ids = set(base["case_id"].to_list())
    for key, r in results.items():
        assert r.features["case_id"].is_unique().all(), key
        assert set(r.features["case_id"].to_list()) <= base_ids, key


def test_no_event_date_feature_looks_into_the_future(setup):
    """The C5 leakage check: after the rules, no event-date feature is negative."""
    _, roles, _, _, results = setup
    checked = 0
    for r in results.values():
        for spec in r.specs:
            if spec.source.endswith("D") and not roles.is_scheduled(spec.source) and spec.stat in (
                "mean", "min", "max", "applicant", "value"
            ):
                values = r.features[spec.feature].drop_nulls()
                assert (values >= 0).all(), f"{spec.feature} has values after the decision"
                checked += 1
    assert checked > 0


def test_leaked_previous_applications_are_dropped(setup, synthetic):
    _, _, families, _, results = setup
    raw_rows = sum(pl.read_parquet(p).height for p in families["applprev_1"].files)
    kept = int(results["applprev_1"].features["applprev_1__rows"].sum())
    assert kept == raw_rows - synthetic.manifest["leakage"]["n_rows_after_decision"]
    assert sum(d["rows_dropped_by_anchor"] for d in results["applprev_1"].diagnostics) == raw_rows - kept


def test_tax_registries_are_merged_without_the_provider(setup, synthetic):
    cfg, _, families, _, results = setup
    assert TAX_FAMILY in results and not set(cfg.features.tax_registries) & set(results)
    tax = results[TAX_FAMILY]
    raw = sum(pl.read_parquet(p).height for k in cfg.features.tax_registries for p in families[k].files)
    assert int(tax.features[f"{TAX_FAMILY}__rows"].sum()) == raw - synthetic.manifest["tax_rows_after_decision"]
    sources = {s.source for s in tax.specs if s.source}
    assert sources == set(TAX_COLUMNS.values()), "features come from the merged schema, not a provider's columns"


def test_tax_features_exist_in_every_period(setup):
    """The point of merging: early and late cases get the same features."""
    *_, base, results = setup
    tax = results[TAX_FAMILY].features.join(base.select("case_id", "date_decision"), on="case_id")
    early = tax.filter(pl.col("date_decision") < "2019-06-01")["taxregamount_900001A__mean"].drop_nulls()
    late = tax.filter(pl.col("date_decision") > "2020-06-01")["taxregamount_900001A__mean"].drop_nulls()
    assert early.len() > 0 and late.len() > 0


def test_masked_placeholder_never_survives(setup):
    *_, results = setup
    for r in results.values():
        for spec in r.specs:
            if spec.kind == "category" and spec.source.endswith("M"):
                assert not (r.features[spec.feature] == "a55475b1").any(), spec.feature


def test_applicant_attributes_come_from_the_applicant_row(setup, families_raw=None):
    _, _, families, _, results = setup
    raw = pl.read_parquet(families["person_1"].files[0]).filter(pl.col("num_group1") == 0)
    built = results["person_1"].features.join(raw.select("case_id", "mainoccupationinc_384A"), on="case_id")
    assert (built["mainoccupationinc_384A__applicant"] == built["mainoccupationinc_384A"]).all()


def test_a_case_split_across_chunks_is_refused(tmp_path, setup):
    cfg, roles, *_ = setup
    pl.DataFrame({"case_id": [1, 2], "date_decision": ["2020-01-10"] * 2, "MONTH": [202001] * 2,
                  "WEEK_NUM": [0] * 2, "target": [0, 1]}).write_parquet(tmp_path / "train_base.parquet")
    for i, ids in enumerate(([1, 2], [2])):
        pl.DataFrame({"case_id": ids, "num_group1": list(range(len(ids))), "x_1A": [1.0] * len(ids)}
                     ).write_parquet(tmp_path / f"train_t_1_{i}.parquet")
    families = discover_families(tmp_path)
    with pytest.raises(ValueError, match="more than one chunk"):
        build_family(families["t_1"], cfg, roles, decision_dates(load_base(families)))


def test_retrieval_timestamps_never_become_features(setup):
    _, roles, *_, results = setup
    sources = {s.source for r in results.values() for s in r.specs}
    assert not sources & set(roles.retrieval)


def test_registry_a_keeps_every_row_because_it_has_no_anchor(setup):
    cfg, roles, families, _, results = setup
    assert "tax_registry_a_1" not in roles.anchors
    diags = [d for d in results[TAX_FAMILY].diagnostics if d["family"] == "tax_registry_a_1"]
    assert diags and all(d["rows_dropped_by_anchor"] == 0 for d in diags)
    profile = diags[0]["lag_profile"]["recorddate_4527225D"]
    assert profile["before"] < profile["non_null"], "retrieval dates fall after the decision"
    assert profile["cases_constant"] == profile["cases_multi"], "one retrieval date per case"
