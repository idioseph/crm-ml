"""High-level pipeline shared by the command line (run_study.py) and the GUI (app.py).

``run_pipeline`` takes validated :class:`data_io.FieldData`, picks the right study
(full CRM calibration or pre-computed CRM), and returns a :class:`StudyOutput`
from which every paper-style chart and table can be produced or exported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import plots
from crm_ml_hybrid import (ML_MODELS, CRMParameters, rank_approaches, run_forecast_study,
                           run_precomputed_crm_study)
from data_io import FieldData


@dataclass
class StudyOutput:
    """Everything produced by a study run.

    Attributes:
        results: Long-format metrics (evaluation, producer, approach, case, MAE, RMSE, R2).
        predictions: ``{approach: (M, K) array}`` aligned with ``time_rows``/``observed_rows``.
        time_rows: Time stamps of the prediction rows, shape (M,).
        observed_rows: Observed rates aligned with the predictions, shape (M, K).
        producer_names: Producer labels.
        n_forecast: Forecast length (samples).
        mode: ``"full"`` or ``"precomputed"``.
        crm_params: Calibrated parameters (full mode only).
        injector_names: Injector labels (full mode only).
        time_label: Axis label for time, e.g. ``"Time [mth]"``.
        rate_label: Axis label for rates, e.g. ``"q [bbl/mth]"``.
    """

    results: pd.DataFrame
    predictions: Dict[str, np.ndarray]
    time_rows: np.ndarray
    observed_rows: np.ndarray
    producer_names: List[str]
    n_forecast: int
    mode: str
    crm_params: Optional[CRMParameters] = None
    injector_names: List[str] = field(default_factory=list)
    time_label: str = "Time"
    rate_label: str = "q"

    def ranking(self, case: str = "forecast", metric: str = "MAE") -> pd.DataFrame:
        """Ranking table (mean rank, % first, mean metric) for one case."""
        return rank_approaches(self.results, case=case, metric=metric)

    def summary_table(self, case: str = "forecast") -> pd.DataFrame:
        """Mean MAE / RMSE / R2 per approach for one case, sorted by MAE."""
        sub = self.results[self.results["case"] == case]
        return (sub.groupby("approach")[["MAE", "RMSE", "R2"]].mean().sort_values("MAE"))

    def predictions_long(self) -> pd.DataFrame:
        """Observed and predicted rates in tidy format for CSV export."""
        rows = []
        for j, name in enumerate(self.producer_names):
            frame = pd.DataFrame({"time": self.time_rows, "producer": name,
                                  "observed": self.observed_rows[:, j]})
            for a, arr in self.predictions.items():
                frame[a] = arr[:, j]
            rows.append(frame)
        return pd.concat(rows, ignore_index=True)


def run_pipeline(
    data: FieldData,
    n_forecast: int,
    n_evaluations: int = 20,
    models: Sequence[str] = ML_MODELS,
    trim_inactive: bool = True,
    crm_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> StudyOutput:
    """Runs the appropriate study for the loaded data.

    Args:
        data: Validated field data.
        n_forecast: Number of trailing samples used as forecast period.
        n_evaluations: Evaluations for ELM/MLP (paper: 20).
        models: Subset of ``("NuSVM", "XGB", "ELM", "MLP")`` to combine with the CRM.
        trim_inactive: Train/score each producer only from its first non-zero rate
            (recommended when wells start at different times).
        crm_kwargs: Extra keyword arguments for the CRM calibrator (full mode).
        verbose: Print progress.

    Returns:
        A :class:`StudyOutput`.
    """
    starts = data.first_active_index() if trim_inactive else None
    t_unit = f" [{data.time_unit}]" if data.time_unit else ""
    r_unit = f" [{data.rate_unit}]" if data.rate_unit else ""
    common = dict(producer_names=list(data.producer_names), n_forecast=n_forecast,
                  time_label=f"Time{t_unit}", rate_label=f"q{r_unit}")
    if data.mode == "full":
        results, crm, preds = run_forecast_study(
            data.time, data.injection, data.production, data.distances,
            n_forecast=n_forecast, n_evaluations=n_evaluations, bhp=data.bhp,
            crm_kwargs=crm_kwargs, models=models, verbose=verbose, active_start=starts)
        return StudyOutput(results, preds, data.time[1:], data.production[1:], mode="full",
                           crm_params=crm.params_, injector_names=list(data.injector_names),
                           **common)
    results, preds = run_precomputed_crm_study(
        data.time, data.production, data.q_crm, n_forecast=n_forecast,
        n_evaluations=n_evaluations, active_start=starts, models=models, verbose=verbose)
    return StudyOutput(results, preds, data.time, data.production, mode="precomputed", **common)


def make_figures(out: StudyOutput) -> Dict[str, Figure]:
    """Builds every paper-style figure available for this study.

    Args:
        out: Study output.

    Returns:
        Ordered ``{name: Figure}`` mapping (CRM fit, parameter heat-maps if available,
        all-approach comparisons, MAE per evaluation, average bars, ranking).
    """
    figs: Dict[str, Figure] = {}
    names, nf = out.producer_names, out.n_forecast
    figs["crm_fit"] = plots.plot_crm_fit(out.time_rows, out.observed_rows, out.predictions["CRM"],
                                         nf, names, out.rate_label, out.time_label)
    if out.crm_params is not None:
        figs["crm_parameters"] = plots.plot_crm_parameters(out.crm_params, out.injector_names, names)
    figs["approaches_forecast"] = plots.plot_all_approaches(
        out.time_rows, out.observed_rows, out.predictions, names, nf, True,
        out.rate_label, out.time_label)
    figs["approaches_entire"] = plots.plot_all_approaches(
        out.time_rows, out.observed_rows, out.predictions, names, nf, False,
        out.rate_label, out.time_label)
    figs.update(plots.build_all_figures(out.results, names))
    return figs


def export_outputs(out: StudyOutput, folder: str | Path, dpi: int = 150) -> List[Path]:
    """Writes all charts (PNG) and tables (CSV) to a folder.

    Args:
        out: Study output.
        folder: Destination directory (created if needed).
        dpi: PNG resolution.

    Returns:
        List of written paths.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for name, fig in make_figures(out).items():
        p = folder / f"{name}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
    tables = {
        "metrics_all_evaluations.csv": out.results,
        "ranking_forecast.csv": out.ranking("forecast").reset_index(),
        "ranking_entire.csv": out.ranking("entire").reset_index(),
        "summary_forecast.csv": out.summary_table("forecast").reset_index(),
        "summary_entire.csv": out.summary_table("entire").reset_index(),
        "predictions.csv": out.predictions_long(),
    }
    for fname, df in tables.items():
        p = folder / fname
        df.to_csv(p, index=False)
        written.append(p)
    if out.crm_params is not None:
        p = folder / "crm_parameters.xlsx"
        with pd.ExcelWriter(p) as xw:
            pd.DataFrame(out.crm_params.lambda_ij, index=out.injector_names,
                         columns=out.producer_names).to_excel(xw, sheet_name="lambda_ij")
            pd.DataFrame(out.crm_params.tau_ij, index=out.injector_names,
                         columns=out.producer_names).to_excel(xw, sheet_name="tau_ij")
            pd.DataFrame({"tau_j": out.crm_params.tau_j}, index=out.producer_names).to_excel(
                xw, sheet_name="tau_j")
        written.append(p)
    return written
