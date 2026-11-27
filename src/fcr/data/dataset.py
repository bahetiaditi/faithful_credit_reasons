"""The versioned feature table: written once on Kaggle (C7), read by every later step.

Layout of a feature dataset directory (a private Kaggle Dataset on Kaggle, the
unzipped sample bundle in `data/` locally):

    dataset_manifest.json   name, commit, config fingerprint, feature list, file hashes, split lock
    features.parquet        every case: keys, `split`, the model features        (Kaggle only)
    slices.parquet          protected attributes per case, for slice analysis    (Kaggle only)
    sample.parquet          stratified sample of features.parquet (split x target)
    sample_slices.parquet   the same cases' slices
    feature_registry.csv    C6 registry: every candidate feature and why it was kept or dropped
    category_levels.json    training levels per category feature
    split_lock.json         the frozen split (also committed as configs/split_lock.json)

Loading checks file hashes against the manifest, the columns against its feature
list, and the split against the lock, so a model is never trained on a table
that differs from the one the manifest describes.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from fcr.config import Config, repo_root
from fcr.data.split import SPLIT_DTYPE, read_lock, stratified_sample, verify_split, write_lock

MANIFEST = "dataset_manifest.json"
FULL, SLICES = "features.parquet", "slices.parquet"
SAMPLE, SAMPLE_SLICES = "sample.parquet", "sample_slices.parquet"
REGISTRY, LEVELS, LOCK = "feature_registry.csv", "category_levels.json", "split_lock.json"
SAMPLE_FILES = (SAMPLE, SAMPLE_SLICES, REGISTRY, LEVELS, LOCK, MANIFEST)
KEY_COLUMNS = ("case_id", "WEEK_NUM", "target", "date_decision", "split")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_fingerprint(root: Path | None = None) -> str:
    """sha256 (first 16 hex) of the two config files that decide the feature table."""
    root = root or repo_root()
    h = hashlib.sha256()
    for rel in ("configs/config.yaml", "configs/date_roles.yaml"):
        h.update((root / rel).read_bytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class FeatureTable:
    matrix: pl.DataFrame
    slices: pl.DataFrame
    manifest: dict
    path: Path
    is_sample: bool

    @property
    def features(self) -> list[str]:
        return list(self.manifest["features"])

    def split(self, name: str) -> pl.DataFrame:
        return self.matrix.filter(pl.col("split") == name)


def write_feature_dataset(out_dir: Path, matrix: pl.DataFrame, slices: pl.DataFrame, registry: pl.DataFrame,
                          levels: dict, lock: dict, cfg: Config, *, commit: str, sample_cases: int) -> dict:
    """Write the dataset directory and return its manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if "split" not in matrix.columns:
        raise ValueError("matrix needs a split column (fcr.data.split.assign_split)")
    features = [c for c in matrix.columns if c not in KEY_COLUMNS]
    matrix = matrix.select(*KEY_COLUMNS, *features)
    sample = stratified_sample(matrix, min(sample_cases, matrix.height), cfg.project.seed)
    sample_slices = slices.join(sample.select("case_id"), on="case_id", how="semi").sort("case_id")

    matrix.write_parquet(out_dir / FULL)
    slices.write_parquet(out_dir / SLICES)
    sample.write_parquet(out_dir / SAMPLE)
    sample_slices.write_parquet(out_dir / SAMPLE_SLICES)
    registry.write_csv(out_dir / REGISTRY)
    (out_dir / LEVELS).write_text(json.dumps(levels, indent=1))
    write_lock(lock, out_dir / LOCK)

    rows = {FULL: matrix.height, SLICES: slices.height, SAMPLE: sample.height, SAMPLE_SLICES: sample_slices.height}
    files = {name: {"sha256": sha256_file(out_dir / name), "bytes": (out_dir / name).stat().st_size,
                    "rows": rows.get(name)}
             for name in (FULL, SLICES, SAMPLE, SAMPLE_SLICES, REGISTRY, LEVELS, LOCK)}
    strata = sample.group_by("split", "target").len().sort("split", "target")
    manifest = {
        "name": cfg.paths.feature_dataset_name,
        "commit": commit,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_fingerprint": config_fingerprint(),
        "n_features": len(features),
        "features": features,
        "files": files,
        "split_lock": lock,
        "sample": {"cases": sample.height, "seed": cfg.project.seed, "stratified_by": ["split", "target"],
                   "strata": [dict(split=r["split"], target=r["target"], cases=r["len"])
                              for r in strata.iter_rows(named=True)]},
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def write_sample_bundle(dataset_dir: Path, bundle: Path) -> Path:
    """Zip the files a local machine needs (sample, registry, manifest); unzip into data/."""
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for name in SAMPLE_FILES:
            z.write(dataset_dir / name, arcname=name)
    return bundle


def locate_feature_dataset(cfg: Config, search_root: Path = KAGGLE_INPUT_ROOT) -> Path | None:
    """$FCR_FEATURES_DIR, else an attached Kaggle Dataset whose manifest carries the configured name."""
    env = os.environ.get("FCR_FEATURES_DIR")
    if env:
        return Path(env)
    if not search_root.is_dir():
        return None
    for depth in range(1, 5):
        for manifest in sorted(search_root.glob("/".join(["*"] * depth) + f"/{MANIFEST}")):
            try:
                if json.loads(manifest.read_text()).get("name") == cfg.paths.feature_dataset_name:
                    return manifest.parent
            except (OSError, json.JSONDecodeError):
                continue
    return None


COMMITTED_LOCK = "configs/split_lock.json"


def committed_lock() -> dict | None:
    """The split lock committed to git in C7, or None before it exists."""
    path = repo_root() / COMMITTED_LOCK
    return read_lock(path) if path.is_file() else None


def load_feature_table(directory: str | Path, cfg: Config, *, sample: bool, verify_hashes: bool = True,
                       expected_lock: dict | None | str = "committed") -> FeatureTable:
    """Load and verify a feature dataset.

    `expected_lock="committed"` (the default) also requires the dataset's lock to
    equal configs/split_lock.json when that file exists; pass None to skip that
    check (synthetic data, which has its own split).
    """
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest.get("name") != cfg.paths.feature_dataset_name:
        raise ValueError(f"{directory} holds {manifest.get('name')!r}, not {cfg.paths.feature_dataset_name!r}")
    names = (SAMPLE, SAMPLE_SLICES) if sample else (FULL, SLICES)
    if verify_hashes:
        for name in (*names, REGISTRY, LEVELS, LOCK):
            if sha256_file(directory / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"{name} does not match its hash in {MANIFEST}; the dataset was modified")
    matrix = pl.read_parquet(directory / names[0]).with_columns(pl.col("split").cast(SPLIT_DTYPE))
    slices = pl.read_parquet(directory / names[1])
    if [c for c in matrix.columns if c not in KEY_COLUMNS] != manifest["features"]:
        raise ValueError("feature columns differ from the manifest's feature list")
    lock = read_lock(directory / LOCK)
    if lock != manifest["split_lock"]:
        raise ValueError("split_lock.json differs from the lock recorded in the manifest")
    expected = committed_lock() if expected_lock == "committed" else expected_lock
    if expected is not None and expected != lock:
        raise ValueError(f"this dataset was built on a different split than {COMMITTED_LOCK}")
    verify_split(matrix, cfg.split, lock, full=not sample)
    return FeatureTable(matrix, slices, manifest, directory, sample)


def install_sample_bundle(bundle: str | Path, data_dir: Path) -> Path:
    """Unzip a sample bundle into data_dir (replacing the previous one)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in SAMPLE_FILES:
        (data_dir / name).unlink(missing_ok=True)
    with zipfile.ZipFile(bundle) as z:
        z.extractall(data_dir)
    return data_dir

