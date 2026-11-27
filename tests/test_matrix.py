"""C6 checks on the synthetic fixtures: exclusion, traceability, train-only screening."""

import polars as pl
import pytest

from fcr.config import load_config, resolve_path
from fcr.data.build import build_all, write_family
from fcr.data.dates import load_date_roles
from fcr.data.loader import discover_families, load_base
from fcr.features.matrix import KEYS, build_model_matrix
from fcr.features.screen import OTHER_LEVEL


@pytest.fixture(scope="module")
def built(synthetic, repo_root, tmp_path_factory):
    cfg = load_config(repo_root / "configs" / "config.yaml")
    cfg = cfg.model_copy(update={"features": cfg.features.model_copy(update={"corr_sample_cases": 1500, "min_minority_rows": 20})})
    roles = load_date_roles(resolve_path(cfg.features.date_roles_file, repo_root))
    families = discover_families(synthetic.train_dir)
    base = load_base(families)
    out = tmp_path_factory.mktemp("features")
    manifest = []
    for r in build_all(families, base, cfg, roles):
        write_family(r, out)
        manifest += [s.as_dict() for s in r.specs]
    manifest = pl.DataFrame(manifest)
    return cfg, base, manifest, build_model_matrix(out, manifest, base, cfg)


def test_protected_and_proxy_sources_never_reach_the_matrix(built):
    cfg, _, manifest, result = built
    excluded = cfg.protected.excluded()
    leaked = manifest.filter(pl.col("source").is_in(list(excluded)) & pl.col("feature").is_in(result.matrix.columns))
    assert leaked.height == 0
    assert "sex_738L__applicant" not in result.matrix.columns
    statuses = {r["feature"]: r["status"] for r in result.registry}
    assert statuses["sex_738L__applicant"] == "excluded:protected"


def test_slices_hold_the_protected_attributes_instead(built):
    *_, result = built
    s = result.slices
    assert {"age_years", "sex", "family_status", "disability_income_type"} <= set(s.columns)
    assert s["age_years"].drop_nulls().is_between(18, 90).all()
    assert s["disability_income_type"].any()


def test_income_type_is_recoded_in_the_matrix_only(built):
    cfg, *_, result = built
    col = f"{cfg.protected.disability_recode_column}__applicant"
    if col in result.matrix.columns:
        assert not result.matrix[col].is_in(list(cfg.protected.disability_income_values)).any()


def test_every_candidate_feature_is_traced(built):
    _, _, manifest, result = built
    registered = {r["feature"]: r for r in result.registry}
    assert set(manifest["feature"]) <= set(registered)
    features = [c for c in result.matrix.columns if c not in KEYS]
    for f in features:
        assert registered[f]["status"] == "kept", f
        assert registered[f]["family"], f
    kept = {f for f, r in registered.items() if r["status"] == "kept"}
    assert kept == set(features)


def test_screening_statistics_come_from_training_weeks(built):
    cfg, base, _, result = built
    lo, hi = cfg.split.train_weeks
    train_ids = base.filter(pl.col("WEEK_NUM").is_between(lo, hi)).select("case_id")
    rec = next(r for r in result.registry if r["status"] == "kept" and r["kind"] == "numeric"
               and r["train_missing_share"] and r["train_missing_share"] > 0)
    col = result.matrix.join(train_ids, on="case_id", how="semi")[rec["feature"]]
    assert rec["train_missing_share"] == pytest.approx(col.null_count() / col.len())


def test_ratios_are_added_and_screened(built):
    *_, result = built
    ratios = {r["feature"]: r for r in result.registry if r["stat"] == "ratio"}
    assert "annuity_to_credit__ratio" in ratios and "credit_to_income__ratio" in ratios
    assert "debt_to_income__ratio" not in ratios, "inputs absent in the fixtures, so it is skipped, not faked"
    assert all(r["status"] == "kept" for r in ratios.values()), "configured ratios are never screened out as duplicates"


def test_categories_are_capped_to_training_levels(built):
    cfg, *_, result = built
    for f, levels in result.levels.items():
        assert len(levels) <= cfg.features.max_category_levels
        values = set(result.matrix[f].drop_nulls().unique().to_list())
        assert values <= set(levels) | {OTHER_LEVEL}, f


def test_correlated_duplicates_are_dropped(built):
    *_, result = built
    dropped = [r for r in result.registry if r["status"] == "dropped:correlated"]
    assert dropped, "count features of fully observed columns duplicate the row count"
    statuses = {r["feature"]: r["status"] for r in result.registry}
    for r in dropped:
        assert abs(r["rho"]) > 0.95 and r["correlated_with"]
        assert statuses[r["correlated_with"]] == "kept", f"{r['feature']} points at a dropped feature"


def test_constant_drops_record_their_minority_count(built):
    *_, result = built
    for r in result.registry:
        if r["status"] == "dropped:constant":
            assert r["minority_rows"] is not None and r["minority_rows"] < 20
        elif r["status"] == "kept" and r["kind"] != "category":
            assert r["minority_rows"] >= 20, r["feature"]
