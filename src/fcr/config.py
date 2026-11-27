"""Typed, validated access to configs/config.yaml.

Every model forbids unknown keys, so a misspelt setting raises instead of being
silently ignored. Cross-field rules that protect the experiment's validity (for
example, that the train/validation/test weeks never overlap) live here too, so
they are checked every time anything loads the config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_ENV_VAR = "FCR_CONFIG"
CONFIG_RELATIVE_PATH = Path("configs") / "config.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


WeekRange = tuple[int, int]


class ProjectConfig(_Strict):
    seed: int


class PathsConfig(_Strict):
    local_data_dir: Path
    results_dir: Path
    kaggle_competition_dir: Path


class DataConfig(_Strict):
    competition_slug: str
    file_format: Literal["parquet", "csv"]
    max_week_num: int = Field(ge=0)


class SplitConfig(_Strict):
    train_weeks: WeekRange
    validation_weeks: WeekRange
    test_weeks: WeekRange

    @field_validator("train_weeks", "validation_weeks", "test_weeks")
    @classmethod
    def _ordered(cls, v: WeekRange) -> WeekRange:
        lo, hi = v
        if lo < 0 or hi < lo:
            raise ValueError(f"week range must satisfy 0 <= start <= end, got {v}")
        return v

    @model_validator(mode="after")
    def _contiguous_and_in_time_order(self) -> SplitConfig:
        # Out-of-time validity needs train < validation < test with no gaps or
        # overlaps; a gap would silently drop weeks, an overlap would leak.
        if self.validation_weeks[0] != self.train_weeks[1] + 1:
            raise ValueError("validation_weeks must start the week after train_weeks ends")
        if self.test_weeks[0] != self.validation_weeks[1] + 1:
            raise ValueError("test_weeks must start the week after validation_weeks ends")
        return self

    def split_of_week(self, week: int) -> Literal["train", "validation", "test"]:
        for name, (lo, hi) in (
            ("train", self.train_weeks),
            ("validation", self.validation_weeks),
            ("test", self.test_weeks),
        ):
            if lo <= week <= hi:
                return name  # type: ignore[return-value]
        raise ValueError(f"week {week} is outside every split")


class TaxRegistryMap(_Strict):
    amount: str
    date: str
    employer: str


class FeaturesConfig(_Strict):
    max_missing_share: float = Field(gt=0, le=1)
    max_abs_spearman: float = Field(gt=0, le=1)
    local_sample_cases: int = Field(gt=0)
    masked_missing_values: tuple[str, ...] = ()
    date_roles_file: Path
    applicant_row_families: tuple[str, ...] = ()
    numeric_stats: tuple[Literal["mean", "min", "max", "count", "std", "sum"], ...]
    tax_registries: dict[str, TaxRegistryMap] = Field(default_factory=dict)

    @field_validator("tax_registries")
    @classmethod
    def _date_suffix(cls, v: dict[str, TaxRegistryMap]) -> dict[str, TaxRegistryMap]:
        for family, m in v.items():
            if not (m.amount.endswith("A") and m.date.endswith("D") and m.employer.endswith("M")):
                raise ValueError(f"tax registry {family}: expected amount *A, date *D, employer *M columns")
        return v


class ProtectedConfig(_Strict):
    columns: tuple[str, ...]
    disability_recode_column: str
    disability_income_values: tuple[str, ...]
    disability_recode_value: str
    reviewed_not_protected: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _lists_are_consistent(self) -> ProtectedConfig:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("protected.columns has duplicates")
        if self.disability_recode_column in self.columns:
            raise ValueError("the recode column is kept as a feature, so it cannot also be excluded")
        overlap = set(self.columns) & set(self.reviewed_not_protected)
        if overlap:
            raise ValueError(f"columns both protected and reviewed-not-protected: {sorted(overlap)}")
        if self.disability_recode_value in self.disability_income_values:
            raise ValueError("disability codes cannot be recoded to one of themselves")
        return self


class ModelConfig(_Strict):
    target_approval_rate: float = Field(gt=0, lt=1)
    calibration_method: Literal["isotonic", "sigmoid"]
    optuna_trials: int = Field(gt=0)


class ExplainConfig(_Strict):
    # Capped at 4: the Reg B commentary says more than four reasons is unlikely
    # to help the applicant, and the faithfulness contract scores top-k <= 4.
    max_reasons: int = Field(ge=1, le=4)
    shap_background_rows: int = Field(gt=0)
    reference_population: Literal["all", "approved_near_threshold"]


class RecourseConfig(_Strict):
    bound_percentiles: tuple[float, float]
    max_groups_changed: int = Field(ge=1)

    @field_validator("bound_percentiles")
    @classmethod
    def _valid_percentiles(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0 <= lo < hi <= 100):
            raise ValueError(f"bound_percentiles must satisfy 0 <= low < high <= 100, got {v}")
        return v


class GenerateConfig(_Strict):
    max_regenerations: int = Field(ge=0)
    temperature: float = Field(ge=0, le=2)


class EvaluationConfig(_Strict):
    deterministic_explain_max: int = Field(gt=0)
    prompt_dev_n: int = Field(gt=0)
    llm_eval_n: int = Field(gt=0)
    annotate_per_condition: int = Field(gt=0)
    bootstrap_resamples: int = Field(gt=0)
    stability_seeds: int = Field(ge=2)
    perturbation_share: float = Field(gt=0, lt=1)


# ---------------------------------------------------------------------------
# LLM layer (ported from Project 2 in C2). Roles, not a single model: each
# role is an ordered chain, and a role can be pinned to one model for evals.
# ---------------------------------------------------------------------------

# generation: writes the letters under test (conditions E0-E3).
# extraction: turns free-text letters into claim JSON; mechanical work.
# grader:     the LLM-grader baseline that claim 1b is tested against.
ROLE_NAMES: tuple[str, ...] = ("generation", "extraction", "grader")


class LimitsConfig(_Strict):
    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None
    reset_timezone: str = "UTC"


class LLMModelConfig(_Strict):
    provider: Literal["groq", "gemini"]
    model: str
    max_output_tokens: int = Field(default=2048, gt=0)
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    thinking_budget: int | None = None
    supports_json_mode: bool = True
    supports_system_instruction: bool = True
    limits: LimitsConfig = LimitsConfig()


class RetryConfig(_Strict):
    max_attempts: int = Field(ge=1)
    initial_backoff_s: float = Field(ge=0)
    backoff_multiplier: float = Field(ge=1)
    retry_on_status: tuple[int, ...]
    max_quota_wait_s: float = Field(ge=0)
    max_pinned_wait_s: float = Field(default=600.0, ge=0)


class LLMCacheConfig(_Strict):
    enabled: bool
    dir: Path


class LLMConfig(_Strict):
    default_temperature: float = Field(ge=0, le=2)
    quota_ledger_path: Path
    retry: RetryConfig
    cache: LLMCacheConfig
    roles: dict[str, tuple[LLMModelConfig, ...]]
    pin: dict[str, str | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _roles_pins_and_independence(self) -> LLMConfig:
        unknown = (set(self.roles) | set(self.pin)) - set(ROLE_NAMES)
        if unknown:
            raise ValueError(f"unknown LLM roles {sorted(unknown)}; allowed: {list(ROLE_NAMES)}")
        for role, model in self.pin.items():
            chain = [m.model for m in self.roles.get(role, ())]
            if model is not None and model not in chain:
                raise ValueError(f"pin.{role}={model!r} is not in the {role} chain {chain}")
        # The grader must never share a model with the generator: a model
        # grading its own letters invites self-preference bias, which would
        # contaminate claim 1b (grader vs verifier).
        generators = {m.model for m in self.roles.get("generation", ())}
        shared = generators & {m.model for m in self.roles.get("grader", ())}
        if shared:
            raise ValueError(f"grader chain shares models with generation: {sorted(shared)}")
        # One model id must mean one set of limits, or the quota ledger
        # would account for it inconsistently.
        seen: dict[str, LimitsConfig] = {}
        for chain in self.roles.values():
            for m in chain:
                if m.model in seen and seen[m.model] != m.limits:
                    raise ValueError(f"model {m.model!r} is configured with two different limit sets")
                seen[m.model] = m.limits
        return self


class Config(_Strict):
    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    features: FeaturesConfig
    protected: ProtectedConfig
    model: ModelConfig
    explain: ExplainConfig
    recourse: RecourseConfig
    generate: GenerateConfig
    evaluation: EvaluationConfig
    llm: LLMConfig

    @model_validator(mode="after")
    def _splits_cover_all_weeks(self) -> Config:
        if self.split.train_weeks[0] != 0:
            raise ValueError("train_weeks must start at week 0")
        if self.split.test_weeks[1] != self.data.max_week_num:
            raise ValueError("test_weeks must end at data.max_week_num")
        return self


def find_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the config file: explicit path, then $FCR_CONFIG, then search upward.

    The upward search starts at the working directory and then at this file's
    location, so it works from the repo root, from a subfolder, and from an
    editable install on Kaggle.
    """
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise FileNotFoundError(f"config not found: {candidate}")
        return candidate
    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        candidate = Path(env_value)
        if not candidate.is_file():
            raise FileNotFoundError(f"${CONFIG_ENV_VAR} points to a missing file: {candidate}")
        return candidate
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (start, *start.parents):
            candidate = directory / CONFIG_RELATIVE_PATH
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"could not find {CONFIG_RELATIVE_PATH}; pass a path or set ${CONFIG_ENV_VAR}"
    )


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate the project config."""
    resolved = find_config_path(path)
    with resolved.open() as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)


def repo_root(path: str | os.PathLike[str] | None = None) -> Path:
    """The directory that holds configs/. Relative paths in the config resolve against it."""
    return find_config_path(path).resolve().parent.parent


def resolve_path(value: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Resolve a config path value: absolute stays absolute, relative joins the repo root."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (root or repo_root()) / candidate


# ---------------------------------------------------------------------------
# Secrets. API keys live in .env at the repo root (gitignored) locally, and in
# Kaggle Secrets on Kaggle. Names match Project 2 so one .env serves both.
# ---------------------------------------------------------------------------

_dotenv_loaded = False


def _load_dotenv_once() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a dependency
        return
    try:
        load_dotenv(repo_root() / ".env")
    except FileNotFoundError:
        pass


def has_env(name: str) -> bool:
    _load_dotenv_once()
    return bool(os.environ.get(name, "").strip())


def require_env(name: str) -> str:
    """Fetch a required environment variable, with an error that says how to fix it."""
    _load_dotenv_once()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"environment variable {name} is not set. Copy .env.example to .env at the repo root and fill it in."
        )
    return value
