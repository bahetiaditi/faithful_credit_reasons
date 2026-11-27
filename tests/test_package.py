import importlib

import pytest

import fcr

SUBPACKAGES = ["data", "features", "models", "monitoring", "explain", "generate", "verify", "llm", "eval"]


def test_version_is_set():
    assert fcr.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    module = importlib.import_module(f"fcr.{name}")
    assert module.__doc__, f"fcr.{name} should state its purpose in a docstring"
