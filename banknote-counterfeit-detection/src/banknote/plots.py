"""Static figures for the README. Every figure has a CSV twin in reports/metrics.

Palette: blue #2a78d6 / orange #eb6834 (validated for color-vision deficiency),
a single-hue blue ramp for magnitudes, hairline grids and muted axes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.colors import LinearSegmentedColormap

# Figures are built from matplotlib.figure.Figure directly (not pyplot), so the
# package never touches global backend state: it renders headless in the CLI
# and displays inline in Jupyter.
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL = "#f0efec"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
}


def _new_figure(width: float = 8.0, height: float = 4.6, **subplot_kw) -> tuple[Figure, object]:
    with rc_context(_RC):
        fig = Figure(figsize=(width, height))
        ax = fig.subplots(**subplot_kw)
    return fig, ax


def _titles(fig: Figure, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=13.5, color=INK, weight="bold")
    fig.text(0.02, 0.905, subtitle, ha="left", va="top", fontsize=10, color=INK_2)


def _style_value_axis(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if axis == "x":
        ax.spines["left"].set_color(BASELINE)
        ax.spines["bottom"].set_visible(False)
    else:
        ax.spines["left"].set_visible(False)


def _save(fig: Figure, path: Path | None) -> Figure:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
    return fig


def plot_model_comparison(results: pd.DataFrame, path: Path | None = None) -> Figure:
    """Dot plot of mean CV accuracy with +/-1 sd whiskers (values are close, so
    dots on a zoomed axis read better than bars that must start at zero)."""
    df = results.sort_values("cv_accuracy_mean")
    fig, ax = _new_figure(8, 4.4)
    with rc_context(_RC):
        y = np.arange(len(df))
        low = (df["cv_accuracy_mean"] - df["cv_accuracy_std"]).clip(lower=0.0)
        high = (df["cv_accuracy_mean"] + df["cv_accuracy_std"]).clip(upper=1.0)  # accuracy caps at 100%
        ax.hlines(y, low, high, color=INK_2, linewidth=1.5)
        ax.plot(df["cv_accuracy_mean"], y, "o", color=BLUE, markersize=10, markeredgecolor=SURFACE, markeredgewidth=2)
        for yi, (mean, hi) in enumerate(zip(df["cv_accuracy_mean"], high, strict=True)):
            ax.text(hi + 0.0015, yi, f"{mean:.1%}", va="center", ha="left", fontsize=10, color=INK)
        ax.set_yticks(y, df["display_name"])
        start = np.floor((low.min() - 0.005) * 100) / 100
        ax.set_xlim(start, 1.009)
        ax.set_xticks(np.arange(start, 1.0001, 0.01))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Accuracy")
        _style_value_axis(ax, "x")
        _titles(
            fig,
            "Cross-validated accuracy by model",
            "Training set only · mean of 5 stratified folds · whiskers show ±1 standard deviation",
        )
        fig.subplots_adjust(left=0.2, right=0.96, top=0.8, bottom=0.14)
    return _save(fig, path)


def plot_confusion_matrices(results: pd.DataFrame, path: Path | None = None) -> Figure:
    """Small multiples: one hold-out confusion matrix per model, shared color scale."""
    n = len(results)
    with rc_context(_RC):
        fig = Figure(figsize=(2.3 * n + 0.8, 3.5))
        axes = fig.subplots(1, n)
    axes = np.atleast_1d(axes)
    cmap = LinearSegmentedColormap.from_list("blue", [NEUTRAL, *BLUE_RAMP])
    vmax = int(results[["genuine_accepted", "fakes_caught"]].to_numpy().max())
    with rc_context(_RC):
        for i, (ax, (_, row)) in enumerate(zip(axes, results.iterrows(), strict=True)):
            m = np.array(
                [[row["genuine_accepted"], row["genuine_rejected"]], [row["fakes_accepted"], row["fakes_caught"]]]
            )
            ax.pcolormesh(m, cmap=cmap, vmin=0, vmax=vmax, edgecolors=SURFACE, linewidth=3)
            for (r, c), v in np.ndenumerate(m):
                dark = v > vmax * 0.45
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    f"{v}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#ffffff" if dark else INK,
                    weight="bold" if r != c and v else "normal",
                )
            ax.set_xlim(0, 2)
            ax.set_ylim(2, 0)
            ax.set_aspect("equal")
            ax.set_title(row["display_name"], fontsize=10.5, color=INK, pad=8)
            ax.set_xticks([0.5, 1.5], ["genuine", "fake"])
            ax.set_yticks([0.5, 1.5], ["genuine", "fake"] if i == 0 else ["", ""])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0].set_ylabel("Actual")
        fig.supxlabel("Predicted", fontsize=10.5, color=INK_2, y=0.04)
        first = results.iloc[0]
        n_genuine = int(first["genuine_accepted"] + first["genuine_rejected"])
        n_fake = int(first["fakes_accepted"] + first["fakes_caught"])
        _titles(
            fig,
            "Hold-out confusion matrices",
            f"{n_genuine + n_fake} unseen bills ({n_genuine} genuine, {n_fake} fake) · "
            "bottom-left cell = counterfeits accepted by the ATM",
        )
        fig.subplots_adjust(left=0.085, right=0.99, top=0.72, bottom=0.16, wspace=0.25)
    return _save(fig, path)


def plot_depth_tradeoff(depth_df: pd.DataFrame, atm_limit: int, path: Path | None = None) -> Figure:
    df = depth_df.dropna(subset=["max_depth"])
    fig, ax = _new_figure(8, 4.6)
    with rc_context(_RC):
        x = df["max_depth"].astype(int)
        ax.axvspan(atm_limit + 0.5, x.max() + 0.6, color=NEUTRAL, zorder=0, linewidth=0)
        ax.text(atm_limit + 0.7, 0.805, "Too deep for the ATM chip", color=INK_2, fontsize=9.5, va="bottom")
        ax.axvline(atm_limit, color=INK_2, linewidth=1, zorder=1)
        at_limit = df[df["max_depth"] == atm_limit].iloc[0]
        ax.text(
            atm_limit - 0.12,
            0.805,
            f"ATM limit: {atm_limit} checks\n{at_limit['cv_accuracy_mean']:.1%} CV · "
            f"{at_limit['test_accuracy']:.1%} test\n{int(at_limit['fakes_accepted'])} fakes accepted",
            color=INK,
            fontsize=9.5,
            ha="right",
            va="bottom",
            linespacing=1.4,
        )
        series = [("cv_accuracy_mean", "Cross-validation", BLUE), ("test_accuracy", "Hold-out test", ORANGE)]
        for col, label, color in series:
            ax.plot(x, df[col], color=color, linewidth=2, label=label, zorder=3)
            ax.plot(x, df[col], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.set_xticks(x)
        ax.set_xlim(0.6, x.max() + 0.6)
        ax.set_ylim(0.8, 1.0)
        ax.set_yticks(np.arange(0.80, 1.001, 0.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Maximum tree depth (consecutive IF/THEN checks)")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
        _style_value_axis(ax, "y")
        ax.spines["bottom"].set_color(BASELINE)
        unconstrained_depth = int(depth_df["actual_depth"].max())
        _titles(
            fig,
            f"What the {atm_limit}-check limit costs",
            "Decision tree accuracy as the maximum depth grows · an unconstrained tree stops "
            f"at depth {unconstrained_depth}",
        )
        fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.14)
    return _save(fig, path)


def plot_scaling(scaling_df: pd.DataFrame, majority_rate: float, path: Path | None = None) -> Figure:
    fig, ax = _new_figure(8, 4.6)
    with rc_context(_RC):
        groups = list(dict.fromkeys(scaling_df["feature_set"]))
        width = 0.36
        gap = 0.02  # surface gap between adjacent bars
        for j, (scaling, color, label) in enumerate(
            [("unscaled", ORANGE, "Unscaled"), ("scaled", BLUE, "Standardized")]
        ):
            rows = scaling_df[scaling_df["scaling"] == scaling].set_index("feature_set")
            vals = [rows.loc[g, "test_accuracy"] for g in groups]
            xs = np.arange(len(groups)) + (j - 0.5) * (width + gap)
            ax.bar(xs, vals, width=width, color=color, label=label, zorder=3)
            for xi, v in zip(xs, vals, strict=True):
                ax.text(xi, v + 0.015, f"{v:.1%}", ha="center", va="bottom", fontsize=10, color=INK)
        last_bar_edge = len(groups) - 1 + width + gap / 2
        ax.axhline(majority_rate, color=INK_2, linewidth=1, zorder=2)
        ax.text(
            last_bar_edge + 0.06,
            majority_rate,
            f"Always say\n'genuine': {majority_rate:.1%}",
            ha="left",
            va="center",
            fontsize=9.5,
            color=INK_2,
            linespacing=1.3,
        )
        ax.set_xticks(np.arange(len(groups)), groups)
        ax.set_xlim(-0.6, last_bar_edge + 0.62)
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("Hold-out accuracy")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=2, borderaxespad=0)
        _style_value_axis(ax, "y")
        ax.spines["bottom"].set_color(BASELINE)
        _titles(
            fig,
            "k-NN collapses when one feature dwarfs the others",
            "Adding ATM transaction volume (10,000–80,000 per day) next to sensor features that stay within ±20",
        )
        fig.subplots_adjust(left=0.1, right=0.97, top=0.74, bottom=0.1)
    return _save(fig, path)


def plot_decision_values(decisions: pd.DataFrame, margin: float, path: Path | None = None) -> Figure:
    fig, ax = _new_figure(8, 4.6)
    with rc_context(_RC):
        values = decisions["decision_value"]
        bins = np.arange(np.floor(values.min() * 2) / 2, np.ceil(values.max() * 2) / 2 + 0.5, 0.5)
        n_inside = int((values.abs() < margin).sum())
        ax.axvspan(-margin, margin, color=NEUTRAL, zorder=0, linewidth=0)
        genuine = values[decisions["actual"] == 0]
        fake = values[decisions["actual"] == 1]
        ax.hist(
            [genuine, fake],
            bins=bins,
            stacked=True,
            color=[BLUE, ORANGE],
            label=["Actually genuine", "Actually counterfeit"],
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        ax.axvline(0, color=INK_2, linewidth=1, zorder=4)
        ymax = ax.get_ylim()[1]
        ax.text(
            0,
            ymax * 1.02,
            f"Human review zone · {n_inside} of {len(values)} bills",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
        )
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xlabel("Signed distance from the linear SVM boundary  (← genuine side · counterfeit side →)")
        ax.set_ylabel("Number of test bills")
        ax.legend(loc="upper left")
        _style_value_axis(ax, "y")
        ax.spines["bottom"].set_color(BASELINE)
        _titles(
            fig,
            "Only the bills near the boundary need a second look",
            f"Hold-out set · shaded band = inside the SVM margin (|distance| < {margin:g})",
        )
        fig.subplots_adjust(left=0.09, right=0.97, top=0.8, bottom=0.14)
    return _save(fig, path)


def plot_logistic_coefficients(coef_df: pd.DataFrame, path: Path | None = None) -> Figure:
    df = coef_df.iloc[::-1]
    fig, ax = _new_figure(8, 3.8)
    with rc_context(_RC):
        colors = [ORANGE if c > 0 else BLUE for c in df["coefficient"]]
        ax.barh(df["feature"], df["coefficient"], color=colors, height=0.55, zorder=3)
        ax.axvline(0, color=BASELINE, linewidth=1, zorder=4)
        span = df["coefficient"].abs().max()
        for yi, c in enumerate(df["coefficient"]):
            ax.text(
                c + (0.08 if c > 0 else -0.08) * span / 4,
                yi,
                f"{c:+.2f}".replace("-", "−"),
                va="center",
                ha="left" if c > 0 else "right",
                fontsize=10,
                color=INK,
            )
        ax.set_xlim(-span * 1.25, span * 0.45)
        ax.set_xlabel("Change in log-odds of 'counterfeit' per +1 standard deviation")
        ax.legend(
            handles=[
                Patch(color=BLUE, label="Pushes towards genuine"),
                Patch(color=ORANGE, label="Pushes towards counterfeit"),
            ],
            loc="lower left",
        )
        _style_value_axis(ax, "x")
        ax.spines["left"].set_visible(False)
        _titles(
            fig,
            "What the logistic regression learned",
            "Standardized coefficients · larger magnitude = stronger influence on the decision",
        )
        fig.subplots_adjust(left=0.13, right=0.97, top=0.76, bottom=0.17)
    return _save(fig, path)
