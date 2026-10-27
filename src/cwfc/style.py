"""Shared chart styling for the analysis and report figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SECONDARY_INK = "#52514e"


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def save_figure(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    fig.patch.set_facecolor(SURFACE)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
