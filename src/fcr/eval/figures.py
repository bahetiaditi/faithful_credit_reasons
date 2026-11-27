"""Static figures for results/ (PNG, matplotlib).

Style follows the project's data-viz rules: one axis per panel (no dual axes),
one hue for a single series, a light-to-dark single-hue ramp for magnitude,
recessive grid and axes, text in text inks rather than series colour, and
titles that say what the panel shows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#e6e6e6"
SERIES_1 = "#257de0"
MARKER = "#9c9c9c"
# Reference sequential blue ramp, steps 100 -> 700.
BLUE_RAMP = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _markers(ax, markers: dict[str, float], y_text: float, label: bool) -> None:
    # Labels are staggered downwards in x order, so markers a week apart
    # (validation start and the pandemic declaration) do not overprint.
    for i, (name, x) in enumerate(sorted(markers.items(), key=lambda kv: kv[1])):
        ax.axvline(x, color=MARKER, linewidth=1, linestyle="--", zorder=1)
        if label:
            ax.text(x + 0.6, y_text * (1 - 0.08 * i), name, color=TEXT_2, fontsize=8, va="top",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1, "alpha": 0.85})


def plot_weekly_default_rate(weekly: pl.DataFrame, markers: dict[str, float], path: Path) -> Path:
    """Two stacked panels on one shared x: default rate (top), applications (bottom)."""
    weeks = weekly["WEEK_NUM"].to_numpy()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1.4])
    top.plot(weeks, weekly["default_rate"].to_numpy() * 100, color=SERIES_1, linewidth=2)
    top.set_ylabel("Default rate (%)", color=TEXT_2, fontsize=9)
    top.set_ylim(bottom=0)
    top.set_title("Weekly default rate and application volume", color=TEXT, fontsize=12, loc="left")
    _markers(top, markers, top.get_ylim()[1] * 0.98, label=True)
    bottom.bar(weeks, weekly["applications"].to_numpy(), color=BLUE_RAMP[1], width=0.8)
    bottom.set_ylabel("Applications", color=TEXT_2, fontsize=9)
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    bottom.set_xlabel("WEEK_NUM", color=TEXT_2, fontsize=9)
    _markers(bottom, markers, 0, label=False)
    for ax in (top, bottom):
        _style(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_coverage_heatmap(coverage: pl.DataFrame, path: Path) -> Path:
    """Families x months, cell = share of that month's applications with any row in the family."""
    wide = coverage.pivot(on="MONTH", index="family", values="coverage", sort_columns=True).sort("family")
    families = wide["family"].to_list()
    months = [c for c in wide.columns if c != "family"]
    grid = wide.select(months).to_numpy().astype(float)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(11, 0.38 * len(families) + 1.6))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(families)), families, fontsize=9, color=TEXT)
    step = max(1, len(months) // 11)
    ax.set_xticks(range(0, len(months), step), [str(m) for m in months[::step]], fontsize=8, color=TEXT_2, rotation=45)
    ax.set_title("Share of applications with any record in each table, by decision month",
                 color=TEXT, fontsize=12, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    bar.ax.tick_params(labelsize=8, colors=TEXT_2)
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    bar.outline.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_missingness_histogram(missingness: pl.DataFrame, threshold: float, path: Path) -> Path:
    """How many feature columns fall in each band of missing share, with the screening threshold."""
    shares = missingness["missing_share"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(shares * 100, bins=np.linspace(0, 100, 21), color=SERIES_1, rwidth=0.92)
    ax.axvline(threshold * 100, color=MARKER, linestyle="--", linewidth=1)
    dropped = int((shares > threshold).sum())
    noun = "column" if dropped == 1 else "columns"
    ax.text(threshold * 100 - 1, ax.get_ylim()[1] * 0.95, f"screen at {threshold:.0%}: {dropped} {noun} above",
            color=TEXT_2, fontsize=8, ha="right", va="top")
    ax.set_xlabel("Missing share of rows (%), nulls plus empty strings", color=TEXT_2, fontsize=9)
    ax.set_ylabel("Feature columns", color=TEXT_2, fontsize=9)
    ax.set_title(f"Missing values across {len(shares)} feature columns", color=TEXT, fontsize=12, loc="left")
    _style(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
