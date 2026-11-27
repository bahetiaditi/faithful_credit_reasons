import pytest

from fcr.config import load_config
from fcr.data.paths import locate_competition_dir

SLUG = "home-credit-credit-risk-model-stability"


@pytest.fixture
def cfg(repo_root, tmp_path):
    base = load_config(repo_root / "configs" / "config.yaml")
    paths = base.paths.model_copy(update={"kaggle_competition_dir": tmp_path / "configured" / SLUG})
    return base.model_copy(update={"paths": paths})


def _make(path, marker="parquet"):
    path.mkdir(parents=True)
    if marker == "parquet":
        (path / "parquet_files").mkdir()
    else:
        (path / "feature_definitions.csv").write_text("Variable,Description\n")
    return path


def test_configured_path_wins(cfg, tmp_path):
    configured = _make(cfg.paths.kaggle_competition_dir)
    _make(tmp_path / "input" / "competitions" / SLUG)
    assert locate_competition_dir(cfg, tmp_path / "input") == configured


@pytest.mark.parametrize("layout", [SLUG, f"competitions/{SLUG}", f"datasets/{SLUG}"])
def test_falls_back_to_known_kaggle_layouts(cfg, tmp_path, layout):
    expected = _make(tmp_path / "input" / layout, marker="csv")
    assert locate_competition_dir(cfg, tmp_path / "input") == expected


def test_empty_dir_with_the_slug_name_is_not_data(cfg, tmp_path):
    (tmp_path / "input" / SLUG).mkdir(parents=True)
    assert locate_competition_dir(cfg, tmp_path / "input") is None


def test_returns_none_when_nothing_is_attached(cfg, tmp_path):
    assert locate_competition_dir(cfg, tmp_path / "missing") is None
