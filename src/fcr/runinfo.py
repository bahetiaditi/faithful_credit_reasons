"""Facts about the current run that every smoke script prints: commit and peak memory."""

from __future__ import annotations

import resource
import subprocess
import sys
from pathlib import Path


def git_commit(repo: Path | None = None) -> str:
    """Short SHA of the checked-out commit, with '+dirty' when there are uncommitted changes."""
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


def peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak / (1024**3 if sys.platform == "darwin" else 1024**2)
