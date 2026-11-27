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


class FeaturesConfig(_Strict):
    max_missing_share: float = Field(gt=0, le=1)
    max_abs_spearman: float = Field(gt=0, le=1)
    local_sample_cases: int = Field(gt=0)


class ProtectedConfig(_Strict):
    columns: tuple[str, ...]
    disability_income_values: tuple[str, ...]
    disability_recode_value: str


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
