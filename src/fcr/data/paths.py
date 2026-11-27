"""Find the competition data wherever Kaggle mounts it.

Kaggle has mounted competition inputs at /kaggle/input/<slug> and, as of
November 2026, at /kaggle/input/competitions/<slug>. The configured path is
tried first; the known layouts are a fallback, so a future mount change shows
up as a warning in the smoke output rather than as a missing-data failure.
"""

from __future__ import annotations

from pathlib import Path

from fcr.config import Config

KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def _looks_like_competition_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "parquet_files").is_dir() or (path / "feature_definitions.csv").is_file())


def locate_competition_dir(cfg: Config, search_root: Path = KAGGLE_INPUT_ROOT) -> Path | None:
    """Return the competition data directory, or None if it is not attached."""
    configured = cfg.paths.kaggle_competition_dir
    if _looks_like_competition_dir(configured):
        return configured
    slug = cfg.data.competition_slug
    candidates = [search_root / slug, search_root / "competitions" / slug]
    if search_root.is_dir():
        candidates += sorted(search_root.glob(f"*/{slug}"))
    for candidate in candidates:
        if _looks_like_competition_dir(candidate):
            return candidate
    return None
