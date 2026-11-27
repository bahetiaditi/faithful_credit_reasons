import copy
import json
from pathlib import Path

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


# --- llm section (C2) ------------------------------------------------------


def _model(name, provider="groq"):
    return {"provider": provider, "model": name, "limits": {"rpm": 30, "tpd": 1000}}


@pytest.mark.parametrize(
    ("dotted", "value", "message"),
    [
        ("llm.roles", {"planner": [_model("m")]}, "unknown LLM roles"),
        ("llm.pin", {"generation": "not-in-chain"}, "not in the generation chain"),
        # The grader must never grade letters written by its own model.
        ("llm.roles", {"generation": [_model("same")], "grader": [_model("same", "gemini")]}, "shares models"),
        (
            "llm.roles",
            {"generation": [_model("m")], "extraction": [{**_model("m"), "limits": {"rpm": 1}}]},
            "two different limit sets",
        ),
    ],
)
def test_invalid_llm_setups_are_rejected(raw, dotted, value, message):
    with pytest.raises(ValidationError, match=message):
        Config.model_validate(_with(raw, dotted, value))


def test_repo_llm_config_has_every_role_and_an_independent_grader(repo_root):
    cfg = load_config(repo_root / "configs" / "config.yaml")
    assert set(cfg.llm.roles) == {"generation", "extraction", "grader"}
    generators = {m.model for m in cfg.llm.roles["generation"]}
    assert generators.isdisjoint(m.model for m in cfg.llm.roles["grader"])
    assert all(v is None for v in cfg.llm.pin.values()), "pins are set in C19, not before"


def test_relative_paths_resolve_against_the_repo_root(repo_root, tmp_path):
    from fcr.config import repo_root as find_root
    from fcr.config import resolve_path

    assert find_root(repo_root / "configs" / "config.yaml") == repo_root
    assert resolve_path(".cache/llm", tmp_path) == tmp_path / ".cache" / "llm"
    assert resolve_path(tmp_path / "abs", repo_root) == tmp_path / "abs"


# --- protected attributes against the real schema (C4.1) -------------------

C4_RESULTS = Path(__file__).resolve().parents[1] / "results" / "c04"


@pytest.fixture(scope="module")
def c04():
    if not (C4_RESULTS / "eda_summary.json").is_file():
        pytest.skip("results/c04 not present")
    summary = json.loads((C4_RESULTS / "eda_summary.json").read_text())
    registry = json.loads((C4_RESULTS / "schema_registry.json").read_text())
    columns = {c["name"] for fam in registry["families"].values() for c in fam["columns"]}
    return summary, columns


def test_every_protected_column_exists_in_the_real_schema(repo_root, c04):
    cfg = load_config(repo_root / "configs" / "config.yaml").protected
    _, columns = c04
    missing = {*cfg.columns, cfg.disability_recode_column} - columns
    assert not missing, f"configured protected columns not in the data: {sorted(missing)}"


def test_every_proxy_and_ratio_column_exists_in_the_real_schema(repo_root, c04):
    cfg = load_config(repo_root / "configs" / "config.yaml")
    _, columns = c04
    missing = set(cfg.protected.proxy_columns) - columns
    assert not missing, f"configured proxy columns not in the data: {sorted(missing)}"
    inputs = {c.removesuffix("__applicant") for r in cfg.features.ratios for c in (r.numerator, r.denominator)}
    assert not inputs - columns, f"ratio inputs not in the data: {sorted(inputs - columns)}"


def test_every_column_the_inventory_flagged_has_a_decision(repo_root, c04):
    """A new EDA run that flags a column nobody has triaged fails here."""
    cfg = load_config(repo_root / "configs" / "config.yaml").protected
    summary, _ = c04
    decided = {*cfg.columns, cfg.disability_recode_column, *cfg.reviewed_not_protected}
    flagged = {p["column"] for p in summary["protected_inventory"]}
    assert flagged <= decided, f"undecided: {sorted(flagged - decided)}"


def test_both_sex_columns_are_excluded(repo_root, c04):
    summary, _ = c04
    cfg = load_config(repo_root / "configs" / "config.yaml").protected
    for qualified in summary["sex_or_gender_columns"]:
        assert qualified.split(".", 1)[1] in cfg.columns


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("protected.columns", ["sex_738L", "sex_738L"]),
        ("protected.columns", ["incometype_1044T"]),          # the recode column is a feature
        ("protected.reviewed_not_protected", ["sex_738L"]),   # cannot be both
        ("protected.disability_recode_value", "HANDICAPPED"),
    ],
)
def test_inconsistent_protected_lists_are_rejected(raw, dotted, value):
    with pytest.raises(ValidationError):
        Config.model_validate(_with(raw, dotted, value))
