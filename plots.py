"""Paper-style comparison charts for the CRM-ML hybrid study.

Every function returns a ``matplotlib.figure.Figure`` (no ``plt.show()``), so the
same code serves scripts, notebooks and the GUI.  Figure types reproduced:

    plot_crm_fit               ~ Fig. 9 / Fig. 18  observed vs CRM (history blue, forecast red)
    plot_all_approaches        ~ Fig. 10, 11, 14, 19  observed vs all 9 approaches + MAE table
    plot_mae_by_evaluation     ~ Fig. 12, 15, 20, 21  MAE per evaluation, per producer
    plot_average_mae_bars      ~ Fig. 13, 16, 22  grouped bars of average MAE / RMSE
    plot_ranking               ~ Fig. 23  ranking ladder (best at the top)
    plot_crm_parameters        ~ Fig. 8 / Fig. 17  heat-maps of lambda_ij and tau_ij
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # safe default; GUIs/notebooks can override before import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from crm_ml_hybrid import CRMParameters, rank_approaches, r_squared, mean_absolute_error, \
    root_mean_squared_error

# Colours follow the paper's legend as closely as possible.
APPROACH_ORDER: List[str] = ["CRM", "CRM-NuSVM", "CRM-XGB", "CRM-ELM", "CRM-MLP",
                             "NuSVM", "XGB", "ELM", "MLP"]
APPROACH_COLORS: Dict[str, str] = {
    "CRM": "#595959", "CRM-NuSVM": "#1f77d4", "CRM-XGB": "#e41a1c", "CRM-ELM": "#17a673",
    "CRM-MLP": "#f2b705", "NuSVM": "#2a4a9b", "XGB": "#b30000", "ELM": "#3d7d3a", "MLP": "#b8860b",
}


def _ordered(approaches: Sequence[str]) -> List[str]:
    """Orders approach names as in the paper, unknown names last."""
    known = [a for a in APPROACH_ORDER if a in approaches]
    return known + [a for a in approaches if a not in known]


def _grid(n: int, ncols: int = 2, cell=(6.0, 3.2)):
    """Creates an n-panel grid of axes and returns (fig, flat axes list)."""
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows),
                             squeeze=False)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


# --------------------------------------------------------------------------- #
def plot_crm_fit(time: np.ndarray, observed: np.ndarray, q_crm: np.ndarray,
                 n_forecast: int, producer_names: Sequence[str],
                 rate_label: str = "q", time_label: str = "Time") -> Figure:
    """Observed vs CRM estimate: history (blue) and forecast (red), with metrics.

    Args:
        time: Time stamps aligned with the rate arrays, shape (M,).
        observed: Observed rates, shape (M, K).
        q_crm: CRM rates, shape (M, K).
        n_forecast: Number of trailing samples that form the forecast period.
        producer_names: Producer labels.
        rate_label: y-axis label.
        time_label: x-axis label.

    Returns:
        Matplotlib figure with one panel per producer (paper Fig. 9 / 18).
    """
    k = observed.shape[1]
    fig, axes = _grid(k)
    split = len(time) - n_forecast
    for j, ax in enumerate(axes):
        ax.plot(time, observed[:, j], color="black", lw=1.2, label="Observed")
        ax.plot(time[:split], q_crm[:split, j], color="#1f77d4", lw=1.6, label="CRM (history)")
        ax.plot(time[split - 1:], q_crm[split - 1:, j], color="#e41a1c", lw=1.8, label="CRM (forecast)")
        h = slice(0, split), slice(split, None)
        txt = []
        for tag, sl in zip(("history", "forecast"), h):
            o, e = observed[sl, j], q_crm[sl, j]
            txt.append(f"{tag}: MAE={mean_absolute_error(o, e):,.0f}  "
                       f"RMSE={root_mean_squared_error(o, e):,.0f}  R²={r_squared(o, e):.3f}")
        ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.axvline(time[split], color="0.6", ls=":", lw=1)
        ax.set_title(producer_names[j], fontweight="bold")
        ax.set_xlabel(time_label)
        ax.set_ylabel(rate_label)
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.72))
    fig.tight_layout()
    return fig


def plot_all_approaches(time: np.ndarray, observed: np.ndarray, predictions: Dict[str, np.ndarray],
                        producer_names: Sequence[str], n_forecast: Optional[int] = None,
                        only_forecast: bool = False, rate_label: str = "q",
                        time_label: str = "Time") -> Figure:
    """Observed rates and all approaches' predictions, plus a MAE table.

    Args:
        time: Time stamps, shape (M,).
        observed: Observed rates, shape (M, K).
        predictions: ``{approach: array (M, K)}`` (NaN allowed before a well starts).
        producer_names: Producer labels.
        n_forecast: Forecast length; with ``only_forecast=True`` only that window is
            drawn and scored (paper Fig. 11 / 14), otherwise the whole record is.
        only_forecast: Restrict plot and MAE table to the forecast window.
        rate_label: y-axis label.
        time_label: x-axis label.

    Returns:
        Figure with one panel per producer and a MAE table at the bottom.
    """
    k = observed.shape[1]
    start = len(time) - n_forecast if (only_forecast and n_forecast) else 0
    sl = slice(start, None)
    approaches = _ordered(list(predictions))
    fig = plt.figure(figsize=(12, 3.2 * int(np.ceil(k / 2)) + 3.2))
    gs = fig.add_gridspec(int(np.ceil(k / 2)) + 1, 2,
                          height_ratios=[1] * int(np.ceil(k / 2)) + [0.9])
    mae = np.full((k, len(approaches)), np.nan)
    for j in range(k):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        ax.plot(time[sl], observed[sl, j], "k-", lw=1.4, label="Observed", zorder=5)
        for a_idx, a in enumerate(approaches):
            y = predictions[a][sl, j]
            ax.plot(time[sl], y, color=APPROACH_COLORS.get(a, None), lw=1.0, alpha=0.9, label=a)
            ok = ~np.isnan(y)
            if ok.any():
                mae[j, a_idx] = mean_absolute_error(observed[sl, j][ok], y[ok])
        ax.set_title(producer_names[j], fontweight="bold")
        ax.set_xlabel(time_label)
        ax.set_ylabel(rate_label)
        if j == 0:
            ax.legend(ncol=2, fontsize=6, loc="upper left")
    ax_t = fig.add_subplot(gs[-1, :])
    ax_t.axis("off")
    cell = [[("" if np.isnan(v) else f"{v:,.0f}") for v in row] for row in mae]
    table = ax_t.table(cellText=cell, rowLabels=list(producer_names), colLabels=approaches,
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for j in range(k):  # bold the best (smallest) MAE per producer
        if np.isfinite(mae[j]).any():
            table[(j + 1, int(np.nanargmin(mae[j])))].set_text_props(fontweight="bold",
                                                                   color="#0a7d33")
    ax_t.set_title("MAE per producer (best in bold green)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_mae_by_evaluation(results: pd.DataFrame, producer_names: Sequence[str],
                           case: str = "forecast", metric: str = "MAE",
                           log_scale: bool = False) -> Figure:
    """Metric vs evaluation number for every approach (paper Figs. 12, 15, 20, 21).

    Deterministic approaches appear as horizontal lines; ANN-based ones fluctuate.

    Args:
        results: Output of ``run_forecast_study`` / ``run_precomputed_crm_study``.
        producer_names: Producer labels (index = producer id in ``results``).
        case: ``"entire"`` or ``"forecast"``.
        metric: ``"MAE"``, ``"RMSE"`` or ``"R2"``.
        log_scale: Use a log y-axis (the paper does this for 5-month forecasts).

    Returns:
        Figure with one panel per producer.
    """
    sub = results[results["case"] == case]
    prods = sorted(sub["producer"].unique())
    fig, axes = _grid(len(prods))
    for ax, j in zip(axes, prods):
        for a in _ordered(sub["approach"].unique()):
            d = sub[(sub["producer"] == j) & (sub["approach"] == a)].sort_values("evaluation")
            ax.plot(d["evaluation"] + 1, d[metric], color=APPROACH_COLORS.get(a), lw=1.3, label=a)
        ax.set_title(producer_names[j], fontweight="bold")
        ax.set_xlabel("Evaluation number")
        ax.set_ylabel(metric)
        if log_scale:
            ax.set_yscale("log")
    axes[0].legend(fontsize=6, ncol=3)
    fig.suptitle(f"{metric} per evaluation ({case} data)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_average_mae_bars(results: pd.DataFrame, producer_names: Sequence[str],
                          case: str = "forecast", metric: str = "MAE") -> Figure:
    """Grouped bars of the average metric per producer and for all producers (Figs. 13/16/22).

    Args:
        results: Study results.
        producer_names: Producer labels.
        case: ``"entire"`` or ``"forecast"``.
        metric: ``"MAE"`` or ``"RMSE"``.

    Returns:
        Figure with the grouped bar chart; bar labels show the values.
    """
    sub = results[results["case"] == case]
    approaches = _ordered(sub["approach"].unique())
    per = sub.groupby(["producer", "approach"])[metric].mean().unstack("approach")
    per = per.reindex(columns=approaches)
    per.loc[-1] = sub.groupby("approach")[metric].mean().reindex(approaches)  # all producers
    per = per.sort_index(key=lambda idx: [(10**6 if i == -1 else i) for i in idx])
    labels = [producer_names[i] if i >= 0 else "ALL PRODUCERS" for i in per.index]

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(labels)), 4.6))
    width = 0.9 / len(approaches)
    x = np.arange(len(labels))
    for a_idx, a in enumerate(approaches):
        vals = per[a].to_numpy()
        bars = ax.bar(x + (a_idx - len(approaches) / 2 + 0.5) * width, vals, width,
                      color=APPROACH_COLORS.get(a), label=a)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", rotation=90, ha="center",
                        va="bottom", fontsize=5)
    ax.set_xticks(x, labels)
    ax.set_ylabel(f"Average {metric}")
    ax.set_title(f"Average {metric} over evaluations ({case} data)", fontweight="bold")
    ax.legend(ncol=5, fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def plot_ranking(ranking_by_case: Dict[str, pd.DataFrame]) -> Figure:
    """Ranking ladder: best approach on top, one column per case (paper Fig. 23).

    Args:
        ranking_by_case: ``{"Entire history": rank_df, "Forecast": rank_df, ...}`` where each
            ``rank_df`` is the output of :func:`crm_ml_hybrid.rank_approaches`.

    Returns:
        Figure with one coloured ladder per case.
    """
    n = len(ranking_by_case)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n + 1, 5), squeeze=False)
    for ax, (title, rk) in zip(axes[0], ranking_by_case.items()):
        order = list(rk.index)
        for pos, a in enumerate(order):
            ax.text(0.5, -pos, a, ha="center", va="center", fontsize=10, fontweight="bold",
                    color=APPROACH_COLORS.get(a, "k"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"))
            ax.text(1.02, -pos, f"{rk.loc[a, 'mean_rank']:.2f}", ha="left", va="center",
                    fontsize=7, color="0.4")
        ax.set_xlim(0, 1.3)
        ax.set_ylim(-len(order) + 0.4, 0.6)
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold")
    fig.text(0.01, 0.5, "Increasing accuracy in production forecasts  ↑", rotation=90,
             va="center", fontsize=9)
    fig.text(0.99, 0.01, "grey number = mean rank", ha="right", fontsize=7, color="0.4")
    fig.tight_layout(rect=(0.03, 0.02, 1, 1))
    return fig


def plot_crm_parameters(params: CRMParameters, injector_names: Sequence[str],
                        producer_names: Sequence[str]) -> Figure:
    """Heat-maps of lambda_ij and tau_ij (paper Figs. 8 and 17).

    Args:
        params: Calibrated CRM parameters.
        injector_names: Row labels.
        producer_names: Column labels.

    Returns:
        Figure with the connectivity and time-constant heat-maps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6 + 0.3 * len(injector_names)))
    for ax, mat, title, cmap in (
        (axes[0], params.lambda_ij, "Interwell connectivity indices λij [-]", "RdYlGn"),
        (axes[1], params.tau_ij, "Interwell time constants τij", "RdYlGn"),
    ):
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(producer_names)), producer_names, rotation=30, ha="right")
        ax.set_yticks(range(len(injector_names)), injector_names)
        for (r, c), v in np.ndenumerate(mat):
            ax.text(c, r, f"{v:.3g}", ha="center", va="center", fontsize=8)
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def build_all_figures(results: pd.DataFrame, producer_names: Sequence[str]) -> Dict[str, Figure]:
    """Convenience: bar charts, per-evaluation charts and the ranking ladder.

    Args:
        results: Study results.
        producer_names: Producer labels.

    Returns:
        ``{figure_name: Figure}`` ready to display or save.
    """
    figs: Dict[str, Figure] = {}
    for case in ("entire", "forecast"):
        if (results["case"] == case).any():
            figs[f"mae_by_evaluation_{case}"] = plot_mae_by_evaluation(results, producer_names, case)
            figs[f"average_mae_{case}"] = plot_average_mae_bars(results, producer_names, case)
    ranks = {("Entire history" if c == "entire" else "Forecast"): rank_approaches(results, c)
             for c in ("entire", "forecast") if (results["case"] == c).any()}
    figs["ranking"] = plot_ranking(ranks)
    return figs
