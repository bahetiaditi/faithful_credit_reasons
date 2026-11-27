import copy

import pytest
import yaml
from pydantic import ValidationError

from fcr.config import CONFIG_ENV_VAR, Config, find_config_path, load_config


@pytest.fixture(scope="module")
def raw(repo_root):
    with (repo_root / "configs" / "config.yaml").open() as fh:
        return yaml.safe_load(fh)


def _with(raw, dotted: str, value):
    data = copy.deepcopy(raw)
    *parents, leaf = dotted.split(".")
    node = data
    for key in parents:
        node = node[key]
    node[leaf] = value
    return data


def test_repo_config_loads(repo_root):
    cfg = load_config(repo_root / "configs" / "config.yaml")
    assert cfg.explain.max_reasons == 4
    assert cfg.split.train_weeks == (0, 63)


def test_every_week_maps_to_exactly_one_split(repo_root):
    cfg = load_config(repo_root / "configs" / "config.yaml")
    seen = [cfg.split.split_of_week(w) for w in range(cfg.data.max_week_num + 1)]
    assert seen[0] == "train" and seen[-1] == "test"
    # time order is never violated: once a later split starts, earlier ones never reappear
    order = {"train": 0, "validation": 1, "test": 2}
    assert all(order[a] <= order[b] for a, b in zip(seen, seen[1:]))
    with pytest.raises(ValueError):
        cfg.split.split_of_week(cfg.data.max_week_num + 1)


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("split.validation_weeks", [60, 77]),   # overlaps train: leakage
        ("split.validation_weeks", [66, 77]),   # gap after train: weeks silently dropped
        ("split.test_weeks", [78, 90]),         # does not reach the last week
        ("split.train_weeks", [1, 63]),         # does not start at week 0
        ("split.train_weeks", [63, 0]),         # reversed
        ("model.target_approval_rate", 1.2),
        ("explain.max_reasons", 5),             # Reg B guidance: four at most
        ("recourse.bound_percentiles", [95, 5]),
        ("explain.reference_population", "everyone"),
        ("evaluation.stability_seeds", 1),
    ],
)
def test_invalid_values_are_rejected(raw, dotted, value):
    with pytest.raises(ValidationError):
        Config.model_validate(_with(raw, dotted, value))


def test_unknown_keys_are_rejected(raw):
    data = copy.deepcopy(raw)
    data["model"]["optuna_trails"] = 10  # typo of optuna_trials
    with pytest.raises(ValidationError):
        Config.model_validate(data)


def test_env_var_overrides_search(tmp_path, monkeypatch, raw):
    alt = tmp_path / "alt.yaml"
    alt.write_text(yaml.safe_dump(_with(raw, "project.seed", 7)))
    monkeypatch.setenv(CONFIG_ENV_VAR, str(alt))
    assert find_config_path() == alt
    assert load_config().project.seed == 7


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
