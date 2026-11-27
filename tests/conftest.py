from pathlib import Path

import pytest

from fcr.data.synthetic import SyntheticDataset, SyntheticSpec, generate_synthetic_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticDataset:
    """One synthetic dataset shared by the whole test session (generation is the slow part)."""
    return generate_synthetic_dataset(tmp_path_factory.mktemp("synthetic"), SyntheticSpec(seed=7))
