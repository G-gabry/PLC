"""
Shared plot styling, so every figure in this research suite uses the same
colors for the same thing -- these match the colors already used for "this
LPC" / "this NN" throughout the rest of the project (test_irmas.py,
plc_challenge_test.py), kept consistent here rather than picked arbitrarily.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research import safe_io

LPC_COLOR = "#E8871E"     # orange -- AR-only
NN_COLOR = "#1BAF7A"      # green  -- AR+NN
TIE_COLOR = "#9CA3AF"     # neutral gray -- reference lines (e.g. 50% coin flip)
GRID_COLOR = "#E5E7EB"


def new_axes(figsize=(8, 4.5), ncols=1, nrows=1, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1]), **kwargs)
    return fig, axes


def style_axis(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8, zorder=0)


def save_fig(fig, path, **kwargs):
    """fig.savefig via safe_io's atomic write -- see that module's docstring
    for why (Windows has repeatedly locked these exact output files)."""
    safe_io.save_fig(fig, path, **kwargs)
