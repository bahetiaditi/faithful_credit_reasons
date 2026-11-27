"""C7: the feature dataset round-trips and refuses to load when anything changed."""

import json
import shutil

import polars as pl
import pytest

from fcr.config import load_config
from fcr.data.build import ensure_features
from fcr.data.dataset import (
    FULL, MANIFEST, SAMPLE, SAMPLE_FILES, install_sample_bundle, load_feature_table,
    locate_feature_dataset, write_feature_dataset, write_sample_bundle,
)
from fcr.data.loader import discover_families, load_base
from fcr.data.split import assign_split, build_lock
from fcr.features.matrix import build_model_matrix


@pytest.fixture(scope="module")
def dataset(synthetic, repo_root, tmp_path_factory):
    cfg = load_config(repo_root / "configs" / "config.yaml")
    cfg = cfg.model_copy(update={"features": cfg.features.model_copy(
        update={"corr_sample_cases": 1500, "min_minority_rows": 20})})
    work = tmp_path_factory.mktemp("c07")
    families = discover_families(synthetic.train_dir)
    base = assign_split(load_base(families), cfg.split)
    manifest = ensure_features(families, base, cfg, work / "features", work / "manifest.csv", log=lambda *_: None)
    result = build_model_matrix(work / "features", manifest, base, cfg)
    matrix = result.matrix.join(base.select("case_id", "split"), on="case_id")
    lock = build_lock(base, cfg.split)
    out = work / "fcr_features"
    m = write_feature_dataset(out, matrix, result.slices, pl.DataFrame(result.registry), result.levels, lock, cfg,
                              commit="test", sample_cases=400)
    return cfg, out, m, lock


def _copy(src, tmp_path):
    dest = tmp_path / "copy"
    shutil.copytree(src, dest)
    return dest


def test_full_and_sample_reload_with_every_check(dataset):
    cfg, out, m, _ = dataset
    full = load_feature_table(out, cfg, sample=False, expected_lock=None)
    sample = load_feature_table(out, cfg, sample=True, expected_lock=None)
    assert full.features == m["features"] and len(m["features"]) == m["n_features"]
    assert sample.matrix.height == 400 == m["sample"]["cases"]
    assert set(sample.matrix["case_id"]) <= set(full.matrix["case_id"])
    assert sample.slices["case_id"].to_list() == sample.matrix["case_id"].to_list()
    assert full.matrix.columns[:5] == ["case_id", "WEEK_NUM", "target", "date_decision", "split"]


def test_no_protected_column_is_a_feature_but_slices_hold_them(dataset):
    cfg, out, m, _ = dataset
    excluded = cfg.protected.excluded()
    assert not [f for f in m["features"] if f.split("__")[0] in excluded]
    assert {"sex", "age_years"} <= set(pl.read_parquet(out / "slices.parquet").columns)


def test_a_modified_file_is_refused(dataset, tmp_path):
    cfg, out, *_ = dataset
    d = _copy(out, tmp_path)
    t = pl.read_parquet(d / FULL)
    t.with_columns(pl.col("target").reverse()).write_parquet(d / FULL)
    with pytest.raises(ValueError, match="hash"):
        load_feature_table(d, cfg, sample=False, expected_lock=None)


def test_a_different_committed_lock_is_refused(dataset):
    cfg, out, _, lock = dataset
    other = json.loads(json.dumps(lock))
    other["splits"]["train"]["case_fingerprint"] = "0" * 16
    with pytest.raises(ValueError, match="different split"):
        load_feature_table(out, cfg, sample=True, expected_lock=other)


def test_a_wrong_dataset_name_is_refused(dataset, tmp_path):
    cfg, out, *_ = dataset
    d = _copy(out, tmp_path)
    m = json.loads((d / MANIFEST).read_text())
    m["name"] = "someone-elses-data"
    (d / MANIFEST).write_text(json.dumps(m))
    with pytest.raises(ValueError, match="holds"):
        load_feature_table(d, cfg, sample=True, expected_lock=None)


def test_sample_bundle_installs_and_loads_without_the_full_table(dataset, tmp_path):
    cfg, out, *_ = dataset
    bundle = write_sample_bundle(out, tmp_path / "fcr_sample.zip")
    data = install_sample_bundle(bundle, tmp_path / "data")
    assert sorted(p.name for p in data.iterdir()) == sorted(SAMPLE_FILES)
    table = load_feature_table(data, cfg, sample=True, expected_lock=None)
    assert table.is_sample and table.matrix.height == pl.read_parquet(out / SAMPLE).height


def test_locate_finds_the_dataset_by_manifest_name(dataset, tmp_path, monkeypatch):
    cfg, out, *_ = dataset
    monkeypatch.delenv("FCR_FEATURES_DIR", raising=False)
    mount = tmp_path / "input" / "datasets" / "someone" / "fcr-features-v3"
    shutil.copytree(out, mount)
    assert locate_feature_dataset(cfg, tmp_path / "input") == mount
    assert locate_feature_dataset(cfg, tmp_path / "nowhere") is None
    monkeypatch.setenv("FCR_FEATURES_DIR", str(out))
    assert locate_feature_dataset(cfg, tmp_path / "input") == out
