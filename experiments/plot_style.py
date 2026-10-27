"""Shared matplotlib style for the thesis figures: one fixed categorical order (never
cycled), thin marks, recessive grid, text in ink colours, one axis per panel."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fixed categorical order (validated colour-blind-safe adjacent pairs); slot i always means the same entity
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8984"
GRID = "#e5e5e2"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": INK3, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6,
        "lines.markersize": 4.5, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight", "axes.titleweight": "medium", "axes.titlelocation": "left",
    })


def color(i):
    return PALETTE[i % len(PALETTE)]


def finish(ax, xlabel=None, ylabel=None, title=None, legend=True):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
