#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "scipy>=1.10",
#     "pandas>=2.0",
#     "scikit-learn>=1.3",
#     "matplotlib>=3.7",
#     "openpyxl>=3.1",
#     "xgboost>=2.0",
#     "streamlit>=1.30",
#     "pymupdf>=1.24",
# ]
# ///

"""Paper-faithful CRM-ML production forecasting implementation.

Reference
---------
Ogali, O.I.O. and Orodu, O.D. (2025).
"Concatenating data-driven and reduced-physics models for smart production
forecasting."
DOI: 10.1007/s12145-025-01745-9

Important replication rule
--------------------------
This program separates:
    1. settings explicitly supported by the article, and
    2. numerical/software settings that the article does not specify.

It therefore does NOT silently attribute solver settings, ML hyperparameters,
scaling, or the demonstration dataset to the paper.

The built-in demonstration field is clearly labelled as a demonstration.
For exact numerical reproduction of the published curves, the original
synfield/Buffalo input data are required.

Paper protocol implemented here
--------------------------------
- 9 approaches: CRM, four CRM-ML hybrids, four standalone ML models.
- CRM-ML feature vector: 4I + 2 inputs.
- Standalone ML input: historical injection rates.
- MLP: one hidden layer with 10 neurons.
- ANN-based models: randomized 75:25 historical train/validation split.
- 20 evaluations.
- CRM minimum tau: at least one sampling-time unit.
- Buffalo CRM: BHP term disabled.
- Overall ranking: combines the five paper cases and their 20 evaluations (560 producer-evaluation groups).

The plotting code uses descriptive engineering labels instead of unexplained
symbol-only labels. The published article's exact numerical figures cannot be
reconstructed from plotting code alone when the underlying data are absent.
The program therefore also provides a PDF-reference extraction utility for
the actual published figures/pages.
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # file/GUI-safe backend; figures are saved or handed to Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import fitz  # PyMuPDF: used only for exact published-figure/page extraction
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "XGBoost is required for article-faithful runs. "
        "Install it with: uv add xgboost"
    ) from exc

FloatArray = np.ndarray
PathLike = Union[str, Path, BinaryIO]

DEFAULT_RATE_UNIT = "MSTB/day"  # unit used in the article
DEFAULT_TIME_UNIT = "days"

# ---------------------------------------------------------------------------
# PAPER-SUPPORTED SETTINGS
# ---------------------------------------------------------------------------
PAPER_N_EVALUATIONS = 20
PAPER_MLP_HIDDEN_NEURONS = 10
PAPER_TRAIN_FRACTION = 0.75
PAPER_VALIDATION_FRACTION = 0.25
PAPER_MIN_TAU_SAMPLING_INTERVAL = True

PAPER_APPROACHES = (
    "CRM",
    "CRM-NuSVM",
    "CRM-XGB",
    "CRM-ELM",
    "CRM-MLP",
    "NuSVM",
    "XGB",
    "ELM",
    "MLP",
)

# Article case names. The actual sample counts are derived from the supplied
# time arrays rather than guessed from row counts.
PAPER_CASES = (
    "Synfield - 8-year production history",
    "Synfield - 1-year forecast period",
    "Synfield - 5-month forecast period",
    "Buffalo - 189-month production history",
    "Buffalo - 12-month forecast period",
)


# 1. ERROR METRICS (paper Eqs. 3-5)


def mean_absolute_error(q_obs: FloatArray, q_est: FloatArray) -> float:
    """Eq. 3: MAE = (1/N) * sum |q_est - q_obs| (same unit as the rates)."""
    return float(np.mean(np.abs(q_est - q_obs)))


def root_mean_squared_error(q_obs: FloatArray, q_est: FloatArray) -> float:
    """Eq. 4: RMSE = sqrt((1/N) * sum (q_est - q_obs)^2) (same unit as the rates)."""
    return float(np.sqrt(np.mean((q_est - q_obs) ** 2)))


def r_squared(q_obs: FloatArray, q_est: FloatArray) -> float:
    """Eq. 5: R2 as the squared correlation between observed and estimated rates.

    Returns:
        A value in [0, 1], or NaN when either series has zero variance.
    """
    d_obs, d_est = q_obs - q_obs.mean(), q_est - q_est.mean()
    denom = np.sum(d_obs**2) * np.sum(d_est**2)
    return float("nan") if denom <= 0.0 else float(np.sum(d_obs * d_est) ** 2 / denom)


def _num(v: float) -> str:
    """Compact number formatting that works for both MSTB/day (~0.01) and bbl/mth (~1e4)."""
    if not np.isfinite(v):
        return ""
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


# 2. DATA IMPORT / EXPORT
# Layout A ("full"):        sheets Time, q_Observed, w_Injection, [Distances], [BHP]
# Layout B ("precomputed"): sheets Time, q_Observed, q_CRM  (no injection data)
# Row 1 of each sheet = names; following non-numeric rows (fluid, [unit]) are notes.

SHEET_ALIASES: Dict[str, List[str]] = {
    "time": ["time", "t", "date", "days", "months"],
    "production": ["q_observed", "q_obs", "production", "q_production", "observed", "q"],
    "injection": ["w_injection", "injection", "w_inj", "w", "inj"],
    "distances": ["distances", "distance", "x_ij", "dist"],
    "bhp": ["bhp", "pwf", "p_wf"],
    "crm": ["q_crm", "crm"],
}


@dataclass
class FieldData:
    """Validated field data ready for the forecasting study.

    Attributes:
        time: Time stamps, shape (N,).
        production: Observed producer rates, shape (N, K).
        producer_names: Names of the K producers.
        injection: Injection rates, shape (N, I), or None (layout B).
        injector_names: Names of the I injectors.
        distances: Injector-producer distances, shape (I, K), or None.
        bhp: Producer BHPs, shape (N, K), or None.
        q_crm: Pre-computed CRM rates, shape (N, K), or None (layout A).
        time_unit: Time unit label (default "days").
        rate_unit: Rate unit label (default "MSTB/day").
        notes: Human-readable remarks collected while loading.
    """

    time: np.ndarray
    production: np.ndarray
    producer_names: List[str]
    injection: Optional[np.ndarray] = None
    injector_names: List[str] = field(default_factory=list)
    distances: Optional[np.ndarray] = None
    bhp: Optional[np.ndarray] = None
    q_crm: Optional[np.ndarray] = None
    time_unit: str = DEFAULT_TIME_UNIT
    rate_unit: str = DEFAULT_RATE_UNIT
    notes: List[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        """``"full"`` if injection data exist (CRM is calibrated), else ``"precomputed"``."""
        return "full" if self.injection is not None else "precomputed"

    def first_active_index(self) -> np.ndarray:
        """Index of the first non-zero observed rate of every producer, shape (K,)."""
        nz = self.production > 0
        return np.where(nz.any(axis=0), nz.argmax(axis=0), 0)

    def summary(self) -> pd.DataFrame:
        """Per-producer table: active window, number of active steps, peak rate."""
        first, rows = self.first_active_index(), []
        for j, name in enumerate(self.producer_names):
            nz = np.nonzero(self.production[:, j])[0]
            rows.append({"producer": name, "first_active_step": int(first[j]),
                         "last_active_step": int(nz.max()) if nz.size else -1,
                         "active_steps": int(nz.size),
                         f"peak_rate [{self.rate_unit}]": float(self.production[:, j].max())})
        return pd.DataFrame(rows)


def _find_sheet(names: List[str], key: str) -> Optional[str]:
    """Finds the sheet whose case-insensitive name matches an alias of ``key``."""
    lowered = {n.strip().lower(): n for n in names}
    for alias in SHEET_ALIASES[key]:
        if alias in lowered:
            return lowered[alias]
    return None


def _parse_block(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Splits a raw sheet into numeric data, column names (row 0) and header notes."""
    raw = raw.dropna(how="all").reset_index(drop=True)
    names = [str(c).strip() for c in raw.iloc[0].tolist()]
    body = raw.iloc[1:].reset_index(drop=True)
    numeric = body.apply(pd.to_numeric, errors="coerce")
    is_data = numeric.notna().all(axis=1)
    if not is_data.any():
        raise ValueError("No numeric rows found below the header.")
    first_data = int(is_data.values.argmax())
    notes = [" | ".join(str(v) for v in body.iloc[r].tolist()) for r in range(first_data)]
    data = numeric.iloc[first_data:].reset_index(drop=True)
    data.columns = names
    return data, names, notes


def _extract_unit(notes: List[str]) -> str:
    """Pulls a bracketed unit such as ``[MSTB/day]`` out of header notes."""
    for note in notes:
        for part in note.split("|"):
            part = part.strip()
            if part.startswith("[") and part.endswith("]"):
                return part[1:-1]
    return ""


def load_field_data(source: PathLike, sheet_map: Optional[Dict[str, str]] = None) -> FieldData:
    """Loads an Excel workbook into :class:`FieldData`.

    Args:
        source: Path or file-like object of an ``.xlsx`` workbook.
        sheet_map: Optional ``{"time": "MySheet", "production": ..., "injection": ...,
            "distances": ..., "bhp": ..., "crm": ...}`` for non-standard sheet names.

    Raises:
        ValueError: If mandatory sheets are missing or shapes are inconsistent.
    """
    if hasattr(source, "seek"):
        source.seek(0)
    xls = pd.ExcelFile(source)
    names = xls.sheet_names
    sm = {k: (sheet_map or {}).get(k) or _find_sheet(names, k) for k in SHEET_ALIASES}
    for mandatory in ("time", "production"):
        if sm[mandatory] is None:
            raise ValueError(f"Could not find a '{mandatory}' sheet. Sheets present: {names}. "
                             f"Pass sheet_map={{'{mandatory}': '<sheet name>'}}.")

    t_df, _, t_notes = _parse_block(xls.parse(sm["time"], header=None))
    q_df, prod_names, q_notes = _parse_block(xls.parse(sm["production"], header=None))
    data = FieldData(time=t_df.iloc[:, 0].to_numpy(float), production=q_df.to_numpy(float),
                     producer_names=prod_names,
                     time_unit=_extract_unit(t_notes) or DEFAULT_TIME_UNIT,
                     rate_unit=_extract_unit(q_notes) or DEFAULT_RATE_UNIT)
    if sm["injection"] is not None:
        w_df, inj_names, _ = _parse_block(xls.parse(sm["injection"], header=None))
        data.injection, data.injector_names = w_df.to_numpy(float), inj_names
    if sm["distances"] is not None:
        data.distances = _parse_block(xls.parse(sm["distances"], header=None))[0].to_numpy(float)
    if sm["bhp"] is not None:
        data.bhp = _parse_block(xls.parse(sm["bhp"], header=None))[0].to_numpy(float)
    if sm["crm"] is not None:
        data.q_crm = _parse_block(xls.parse(sm["crm"], header=None))[0].to_numpy(float)
    validate(data)
    if data.injection is None and data.q_crm is not None:
        data.notes.append("No injection sheet found: running in 'precomputed CRM' mode "
                          "(the supplied q_CRM series is used as the physics model).")
    return data


def load_from_bytes(content: bytes, **kwargs) -> FieldData:
    """Convenience wrapper for GUI uploads (bytes -> :class:`FieldData`)."""
    return load_field_data(io.BytesIO(content), **kwargs)


def validate(data: FieldData) -> None:
    """Checks shapes, ordering and finite values; fills sensible defaults (in place).

    Raises:
        ValueError: On any inconsistency that would break the model.
    """
    n = data.time.shape[0]
    if data.production.shape[0] != n:
        raise ValueError(f"q_Observed has {data.production.shape[0]} rows but Time has {n}.")
    if not np.all(np.diff(data.time) > 0):
        raise ValueError("Time must be strictly increasing.")
    if data.injection is None and data.q_crm is None:
        raise ValueError("Provide either an injection sheet (full mode) or a q_CRM sheet "
                         "(precomputed mode).")
    for label, arr in (("production", data.production), ("injection", data.injection),
                       ("bhp", data.bhp), ("q_crm", data.q_crm)):
        if arr is None:
            continue
        if arr.shape[0] != n:
            raise ValueError(f"'{label}' has {arr.shape[0]} rows, expected {n}.")
        if not np.isfinite(arr).all():
            raise ValueError(f"'{label}' contains NaN/inf - fill or remove them first.")
    k = data.production.shape[1]
    if data.q_crm is not None and data.q_crm.shape[1] != k:
        raise ValueError("q_CRM must have the same number of columns as q_Observed.")
    if data.bhp is not None and data.bhp.shape[1] != k:
        raise ValueError("BHP must have one column per producer.")
    if data.injection is not None:
        i = data.injection.shape[1]
        if not data.injector_names:
            data.injector_names = [f"I-{x + 1:02d}" for x in range(i)]
        if data.distances is None:
            data.distances = np.ones((i, k))  # uninformative feature
            data.notes.append("No distances provided: injector-producer distances set to 1.")
        if data.distances.shape != (i, k):
            raise ValueError(f"Distances must be {i} x {k} (injectors x producers), "
                             f"got {data.distances.shape}.")


def from_arrays(time, production, injection=None, distances=None, bhp=None, q_crm=None,
                producer_names: Optional[Sequence[str]] = None,
                injector_names: Optional[Sequence[str]] = None,
                time_unit: str = DEFAULT_TIME_UNIT, rate_unit: str = DEFAULT_RATE_UNIT) -> FieldData:
    """Builds :class:`FieldData` from NumPy arrays / DataFrames (e.g. CSV or a database).

    DataFrame column names are used as well names when explicit names are not given.
    """
    def arr(x):
        return None if x is None else np.asarray(x, dtype=float)

    if producer_names is None and isinstance(production, pd.DataFrame):
        producer_names = [str(c) for c in production.columns]
    if injector_names is None and isinstance(injection, pd.DataFrame):
        injector_names = [str(c) for c in injection.columns]
    prod = arr(production)
    prod = prod.reshape(-1, 1) if prod.ndim == 1 else prod
    data = FieldData(
        time=np.asarray(time, dtype=float), production=prod,
        producer_names=list(producer_names) if producer_names is not None
        else [f"P-{j + 1:02d}" for j in range(prod.shape[1])],
        injection=arr(injection), injector_names=list(injector_names or []),
        distances=arr(distances), bhp=arr(bhp), q_crm=arr(q_crm),
        time_unit=time_unit, rate_unit=rate_unit)
    validate(data)
    return data


def _sheet(cols: List[str], values: np.ndarray, unit: str, fluid: str = "Oil + Water") -> pd.DataFrame:
    """One workbook sheet: names row, fluid/description row, [unit] row, then numbers."""
    head = pd.DataFrame([cols, [fluid] * len(cols), [unit] * len(cols)])
    head.columns = range(len(cols))
    body = pd.DataFrame(np.asarray(values))
    body.columns = range(len(cols))
    return pd.concat([head, body], ignore_index=True)


def _write_workbook(path: Path, time: np.ndarray, prod: np.ndarray, inj: np.ndarray,
                    dist: np.ndarray, time_unit: str = "day", rate_unit: str = "MSTB/day") -> Path:
    """Writes a Layout-A workbook (Time, q_Observed, w_Injection, Distances)."""
    prods = [f"P-{j + 1:02d}" for j in range(prod.shape[1])]
    injs = [f"I-{i + 1:02d}" for i in range(inj.shape[1])]
    time_df = pd.concat([pd.DataFrame([["Time"], [""], [f"[{time_unit}]"]]),
                         pd.DataFrame(time)], ignore_index=True)
    with pd.ExcelWriter(path) as xw:
        time_df.to_excel(xw, sheet_name="Time", header=False, index=False)
        _sheet(prods, prod, f"[{rate_unit}]").to_excel(xw, sheet_name="q_Observed", header=False, index=False)
        _sheet(injs, inj, f"[{rate_unit}]", "Water").to_excel(xw, sheet_name="w_Injection", header=False, index=False)
        _sheet(prods, dist, "[ft]", "").to_excel(xw, sheet_name="Distances", header=False, index=False)
    return path


def write_template(path: Union[str, Path], n_steps: int = 36, n_injectors: int = 3,
                   n_producers: int = 2) -> Path:
    """Writes a filled-with-random-numbers Layout-A workbook to overwrite with your data."""
    rng = np.random.default_rng(0)
    return _write_workbook(Path(path), np.arange(1, n_steps + 1, dtype=float),
                           rng.uniform(1.0, 2.0, (n_steps, n_producers)),
                           rng.uniform(1.0, 2.0, (n_steps, n_injectors)),
                           rng.uniform(500, 3000, (n_injectors, n_producers)))


# 3. CAPACITANCE-RESISTANCE MODEL (Eq. 1) AND ITS CALIBRATION (Eq. 2)


@dataclass
class CRMParameters:
    """CRM parameters: tau_j (K,), lambda_ij (I,K), tau_ij (I,K); optional BHP-term v_kj, tau_kj (K,K)."""

    tau_j: FloatArray
    lambda_ij: FloatArray
    tau_ij: FloatArray
    v_kj: Optional[FloatArray] = None
    tau_kj: Optional[FloatArray] = None


def _exp_filter(signal: FloatArray, dt: FloatArray, tau: float) -> FloatArray:
    """CRM convolution sum S_n = sum_m [e^{(t_m-t_n)/tau} - e^{(t_{m-1}-t_n)/tau}] * s_m.

    Exactly equals the first-order recursion S_n = a_n S_{n-1} + (1-a_n) s_n with
    a_n = exp(-dt_n/tau): O(M) instead of O(M^2), vectorised with ``lfilter`` for
    uniform sampling.
    """
    if np.allclose(dt, dt[0]):
        alpha = np.exp(-dt[0] / tau)
        return lfilter([1.0 - alpha], [1.0, -alpha], signal)
    alpha_n, out, prev = np.exp(-dt / tau), np.empty_like(signal, dtype=float), 0.0
    for n in range(signal.shape[0]):
        prev = alpha_n[n] * prev + (1.0 - alpha_n[n]) * signal[n]
        out[n] = prev
    return out


def crm_simulate(params: CRMParameters, time: FloatArray, injection: FloatArray,
                 q0: FloatArray, bhp: Optional[FloatArray] = None) -> FloatArray:
    """Evaluates Eq. 1: q_hat = Production term + Injection term (+ BHP term).

    Args:
        params: CRM parameters.
        time: Time stamps t_0..t_{N-1}, shape (N,).
        injection: Injection rates w_i(t), shape (N, I).
        q0: Initial producer rates q_j(t0), shape (K,).
        bhp: Producer BHPs, shape (N, K), or None (constant-BHP assumption).

    Returns:
        Estimated rates for n = 1..N-1, shape (N-1, K).
    """
    t_rel, dt = time[1:] - time[0], np.diff(time)
    n_inj, n_prod = params.lambda_ij.shape
    q_hat = q0[None, :] * np.exp(-t_rel[:, None] / params.tau_j[None, :])  # production term
    for i in range(n_inj):  # injection term
        for j in range(n_prod):
            q_hat[:, j] += params.lambda_ij[i, j] * _exp_filter(injection[1:, i], dt, params.tau_ij[i, j])
    if bhp is not None and params.v_kj is not None and params.tau_kj is not None:  # BHP term
        for k in range(n_prod):
            p_k = bhp[1:, k]
            for j in range(n_prod):
                tau = params.tau_kj[k, j]
                q_hat[:, j] += params.v_kj[k, j] * (bhp[0, j] * np.exp(-t_rel / tau) - p_k
                                                    + _exp_filter(p_k, dt, tau))
    return q_hat


class CapacitanceResistanceModel:
    """CRM calibrated by constrained optimisation (SLSQP) of Eq. 2.

    Constraints (paper): 0 <= lambda_ij <= 1; sum_j lambda_ij <= 1 per injector;
    tau >= tau_min (default: the sampling interval). All producers are fitted
    concurrently; time constants are optimised in log-space and rates are
    normalised internally for numerical conditioning.

    Attributes:
        params_: Fitted :class:`CRMParameters` after ``fit``.
        objective_: Final (normalised) value of the objective function.
    """

    def __init__(self, tau_min: Optional[float] = None, tau_max: Optional[float] = None,
                 n_starts: int = 3, max_iter: int = 300, random_state: Optional[int] = 0) -> None:
        """Configure CRM calibration.

        ``tau_max`` is deliberately optional and has NO automatic paper-unjustified
        upper bound. If supplied, it is an implementation constraint and should be
        reported as such. The article explicitly defines the lower tau constraint,
        not a universal upper bound.
        """
        self.tau_min, self.tau_max = tau_min, tau_max
        self.n_starts, self.max_iter, self.random_state = n_starts, max_iter, random_state
        self.params_: Optional[CRMParameters] = None
        self.objective_: float = float("nan")
        self._flow_scale, self._bhp_scale, self._use_bhp = 1.0, 1.0, False

    def _unpack(self, x: FloatArray, n_inj: int, n_prod: int) -> CRMParameters:
        """Optimiser vector -> :class:`CRMParameters`."""
        k, i, pos = n_prod, n_inj, 0
        tau_j = np.exp(x[pos:pos + k]); pos += k
        lam = x[pos:pos + i * k].reshape(i, k); pos += i * k
        tau_ij = np.exp(x[pos:pos + i * k].reshape(i, k)); pos += i * k
        v_kj = tau_kj = None
        if self._use_bhp:
            v_kj = x[pos:pos + k * k].reshape(k, k); pos += k * k
            tau_kj = np.exp(x[pos:pos + k * k].reshape(k, k))
        return CRMParameters(tau_j, lam, tau_ij, v_kj, tau_kj)

    def fit(self, time: FloatArray, injection: FloatArray, production: FloatArray,
            bhp: Optional[FloatArray] = None) -> "CapacitanceResistanceModel":
        """Calibrates the CRM on historical data (shapes (N,), (N,I), (N,K), optional (N,K))."""
        time, injection, production = (np.asarray(a, float) for a in (time, injection, production))
        n_steps, n_inj = injection.shape
        n_prod = production.shape[1]
        if time.shape[0] != n_steps or production.shape[0] != n_steps:
            raise ValueError("time, injection and production must have equal length.")
        self._use_bhp = bhp is not None
        self._flow_scale = max(float(np.mean(np.abs(production))), 1e-12)
        q_s, w_s, bhp_s = production / self._flow_scale, injection / self._flow_scale, None
        if bhp is not None:
            self._bhp_scale = max(float(np.mean(np.abs(bhp))), 1e-12)
            bhp_s = np.asarray(bhp, float) / self._bhp_scale

        tau_min = self.tau_min if self.tau_min is not None else float(np.min(np.diff(time)))
        if tau_min <= 0:
            raise ValueError("tau_min must be positive.")

        # The article gives a lower bound for tau but does not specify a universal
        # upper bound. Therefore None means an unbounded upper side.
        tau_max = self.tau_max
        if tau_max is not None and tau_max <= tau_min:
            raise ValueError("tau_max must be greater than tau_min when supplied.")

        lo = np.log(tau_min)
        hi = None if tau_max is None else np.log(tau_max)
        bounds: List[Tuple[Optional[float], Optional[float]]] = (
            [(lo, hi)] * n_prod
            + [(0.0, 1.0)] * (n_inj * n_prod)
            + [(lo, hi)] * (n_inj * n_prod)
        )
        if self._use_bhp:
            bounds += [(None, None)] * (n_prod**2) + [(lo, hi)] * (n_prod**2)
        n_par, lam_off = len(bounds), n_prod
        A = np.zeros((n_inj, n_par))  # sum_j lambda_ij <= 1
        for i in range(n_inj):
            A[i, lam_off + i * n_prod:lam_off + (i + 1) * n_prod] = 1.0
        constraints = [{"type": "ineq", "fun": lambda x: 1.0 - A @ x, "jac": lambda x: -A}]
        q0 = q_s[0]

        def objective(x: FloatArray) -> float:  # Eq. 2
            p = self._unpack(x, n_inj, n_prod)
            return float(np.mean((q_s[1:] - crm_simulate(p, time, w_s, q0, bhp_s)) ** 2))

        rng = np.random.default_rng(self.random_state)
        best_x, best_f = None, np.inf
        for start in range(max(1, self.n_starts)):
            x0 = np.empty(n_par)
            if start == 0:
                x0[:n_prod] = np.log(10 * tau_min)
            else:
                # No paper-defined upper bound exists. Random starts therefore use
                # a broad, documented initialization scale only; this is not a CRM constraint.
                random_tau_scale = max(10.0, float(time[-1] - time[0]))
                x0[:n_prod] = rng.uniform(lo, np.log(tau_min * random_tau_scale), n_prod)
            lam0 = (0.8 / n_prod) * np.ones((n_inj, n_prod)) if start == 0 else \
                rng.dirichlet(np.ones(n_prod), size=n_inj) * 0.9
            x0[lam_off:lam_off + n_inj * n_prod] = lam0.ravel()
            tsl = slice(lam_off + n_inj * n_prod, lam_off + 2 * n_inj * n_prod)
            if start == 0:
                x0[tsl] = np.log(10 * tau_min)
            else:
                random_tau_scale = max(10.0, float(time[-1] - time[0]))
                x0[tsl] = rng.uniform(lo, np.log(tau_min * random_tau_scale), n_inj * n_prod)
            if self._use_bhp:
                x0[tsl.stop:tsl.stop + n_prod**2] = 0.0
                x0[tsl.stop + n_prod**2:] = np.log(10 * tau_min)
            res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                           options={"maxiter": self.max_iter, "ftol": 1e-12})
            if res.fun < best_f:
                best_x, best_f = res.x, float(res.fun)
        assert best_x is not None
        self.params_, self.objective_ = self._unpack(best_x, n_inj, n_prod), best_f
        return self

    def predict(self, time: FloatArray, injection: FloatArray, q0: FloatArray,
                bhp: Optional[FloatArray] = None) -> FloatArray:
        """Estimates/forecasts rates for n = 1..N-1 (physical units), shape (N-1, K)."""
        if self.params_ is None:
            raise RuntimeError("Call fit() before predict().")
        bhp_s = None if (bhp is None or not self._use_bhp) else bhp / self._bhp_scale
        q_hat = crm_simulate(self.params_, np.asarray(time, float),
                             np.asarray(injection, float) / self._flow_scale,
                             np.asarray(q0, float) / self._flow_scale, bhp_s)
        return q_hat * self._flow_scale



def validate_bhp_policy(case_name: str, include_bhp_term: bool) -> None:
    """Enforce the article's BHP policy for named paper cases.

    The supplied paper states that producer BHP remained constant in the
    synfield and therefore BHP was not used for CRM calibration. It also
    explicitly excludes the BHP term for the Buffalo implementation.
    """
    normalized = case_name.lower()
    if "buffalo" in normalized and include_bhp_term:
        raise ValueError(
            "Paper-faithful Buffalo runs must disable the CRM BHP term."
        )
    if "synfield" in normalized and include_bhp_term:
        raise ValueError(
            "Paper-faithful Synfield runs must disable the CRM BHP term."
        )




# 4. MACHINE-LEARNING MODELS AND FEATURES

STOCHASTIC_MODELS = ("ELM", "MLP")
ML_MODELS = ("NuSVM", "XGB", "ELM", "MLP")

# These are implementation settings, not values stated by the paper.
IMPLEMENTATION_ML_SETTINGS = {
    "NuSVM": {"kernel": "rbf", "nu": 0.5, "C": 10.0, "gamma": "scale"},
    "XGB": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_jobs": 1,
        "verbosity": 0,
    },
    "ELM": {"ridge": 1e-3},
    "MLP": {"activation": "tanh", "solver": "adam", "max_iter": 2000},
}


class ExtremeLearningMachine(BaseEstimator, RegressorMixin):
    """Single-hidden-layer ELM (Liang et al., 2006).

    Hidden weights/biases ~ U(-1, 1) are never trained; output weights come from a
    (lightly ridge-regularised) least-squares / Moore-Penrose solution.
    Set ``ridge=0`` for the pure pseudo-inverse.
    """

    def __init__(self, n_hidden: int = 10, random_state: Optional[int] = None, ridge: float = 1e-3) -> None:
        """Args: number of hidden neurons, seed, ridge strength."""
        self.n_hidden, self.random_state, self.ridge = n_hidden, random_state, ridge

    @staticmethod
    def _sigmoid(z: FloatArray) -> FloatArray:
        return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable logistic

    def fit(self, X: FloatArray, y: FloatArray) -> "ExtremeLearningMachine":
        """Draws the random hidden layer and solves for the output weights."""
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(len(X), -1)
        self.weights_ = rng.uniform(-1.0, 1.0, (X.shape[1], self.n_hidden))
        self.bias_ = rng.uniform(-1.0, 1.0, self.n_hidden)
        h = self._sigmoid(X @ self.weights_ + self.bias_)
        if self.ridge > 0:
            self.beta_ = np.linalg.solve(h.T @ h + self.ridge * np.eye(self.n_hidden), h.T @ y)
        else:
            self.beta_ = np.linalg.pinv(h) @ y
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        """Predicts targets, shape (n_samples,)."""
        out = self._sigmoid(np.asarray(X, float) @ self.weights_ + self.bias_) @ self.beta_
        return out.ravel() if out.shape[1] == 1 else out


def make_regressor(name: str, random_state: Optional[int] = 0, n_hidden: int = 10):
    """Build one of the four paper ML families.

    Paper-defined:
        MLP has one hidden layer with 10 neurons.

    Implementation-defined:
        scaling, activation, optimizer, NuSVM settings, XGB settings and ELM ridge.
    """
    if name == "NuSVM":
        settings = IMPLEMENTATION_ML_SETTINGS["NuSVM"]
        base = NuSVR(**settings)
    elif name == "XGB":
        settings = dict(IMPLEMENTATION_ML_SETTINGS["XGB"])
        settings["random_state"] = random_state
        base = XGBRegressor(**settings)
    elif name == "ELM":
        base = ExtremeLearningMachine(
            n_hidden=n_hidden,
            random_state=random_state,
            ridge=IMPLEMENTATION_ML_SETTINGS["ELM"]["ridge"],
        )
    elif name == "MLP":
        settings = IMPLEMENTATION_ML_SETTINGS["MLP"]
        base = MLPRegressor(
            hidden_layer_sizes=(PAPER_MLP_HIDDEN_NEURONS,),
            random_state=random_state,
            **settings,
        )
    else:
        raise ValueError(f"Unknown ML model '{name}'. Choose from {ML_MODELS}.")
    return TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("model", base)]),
        transformer=StandardScaler())


def build_crm_ml_features(time: FloatArray, injection: FloatArray, params: CRMParameters,
                          distances: FloatArray, producer: int) -> FloatArray:
    """The 4I+2 CRM-ML inputs for one producer: [t | X_ij | lambda_ij | tau_ij | w_i(t) | tau_j]."""
    m = time.shape[0]
    const = np.concatenate([distances[:, producer], params.lambda_ij[:, producer],
                            params.tau_ij[:, producer]])
    return np.column_stack([time, np.tile(const, (m, 1)), injection,
                            np.full(m, params.tau_j[producer])])


def _fit_predict_ml(name: str, features: FloatArray, target: FloatArray,
                    n_train_rows: int, seed: int) -> Tuple[FloatArray, float]:
    """Trains on the historical rows, predicts all rows, returns (predictions, validation MAE).

    NuSVM/XGB are deterministic and use every historical row. ELM/MLP use a random
    75:25 train/validation split (the paper's "erratic ANN" behaviour).
    """
    hist, val_mae = np.arange(n_train_rows), float("nan")
    if name in STOCHASTIC_MODELS:
        perm = np.random.default_rng(seed).permutation(hist)
        n_tr = int(round(PAPER_TRAIN_FRACTION * n_train_rows))
        train_idx, val_idx = perm[:n_tr], perm[n_tr:]
    else:
        train_idx, val_idx = hist, np.array([], dtype=int)
    model = make_regressor(name, random_state=seed)
    model.fit(features[train_idx], target[train_idx])
    if val_idx.size:
        val_mae = mean_absolute_error(target[val_idx], model.predict(features[val_idx]))
    return np.asarray(model.predict(features), float), val_mae


# 5. STUDIES, RANKING AND THE PIPELINE


def run_forecast_study(time, injection, production, distances, n_forecast: int,
                       n_evaluations: int = PAPER_N_EVALUATIONS, bhp=None,
                       include_bhp_term: bool = False, paper_case_name: str = "",
                       crm_kwargs: Optional[Dict] = None,
                       models: Sequence[str] = ML_MODELS, verbose: bool = True,
                       active_start: Optional[Sequence[int]] = None):
    """Full paper workflow (CRM calibrated from injection data).

    1. Calibrate the CRM on all but the last ``n_forecast`` samples; 2. forecast the
    whole record; 3. per producer/model/evaluation train a CRM-ML hybrid (4I+2
    inputs) and a stand-alone ML model (I inputs); 4. score on the "entire" record
    and on the "forecast" period.

    Returns:
        ``(results, crm, predictions)``: long-format DataFrame [evaluation, producer,
        approach, case, MAE, RMSE, R2, val_MAE]; the calibrated CRM; and
        ``{approach: (N-1, K) predictions of the last evaluation}``.
    """
    time, injection, production = (np.asarray(a, float) for a in (time, injection, production))
    n_total, _ = injection.shape
    n_prod = production.shape[1]
    n_hist = n_total - n_forecast
    if n_hist < 10:
        raise ValueError("Not enough historical data (need >= 10 samples before the forecast).")

    validate_bhp_policy(paper_case_name, include_bhp_term)

    if n_evaluations != PAPER_N_EVALUATIONS:
        warnings.warn(
            f"The article uses {PAPER_N_EVALUATIONS} evaluations; "
            f"this run requested {n_evaluations}.",
            RuntimeWarning,
        )

    crm = CapacitanceResistanceModel(**(crm_kwargs or {}))
    calibration_bhp = (
        bhp[:n_hist]
        if include_bhp_term and bhp is not None
        else None
    )
    prediction_bhp = bhp if include_bhp_term and bhp is not None else None
    crm.fit(time[:n_hist], injection[:n_hist], production[:n_hist], calibration_bhp)
    q_crm = crm.predict(time, injection, production[0], prediction_bhp)
    if verbose:
        print(f"CRM calibrated (normalised objective = {crm.objective_:.3e}).")

    q_obs, t_rows, w_rows = production[1:], time[1:], injection[1:]
    n_hist_rows = n_hist - 1
    cases = {"entire": 0, "forecast": n_hist_rows}
    start_rows = (np.zeros(n_prod, dtype=int) if active_start is None
                  else np.maximum(np.asarray(active_start, dtype=int) - 1, 0))
    records: List[Dict] = []
    preds: Dict[str, FloatArray] = {"CRM": q_crm.copy()}

    def score(approach, ev, j, q_hat, val):
        for case, first in cases.items():
            first = max(first, int(start_rows[j]))
            if q_obs.shape[0] - first < 2:
                continue
            o, e = q_obs[first:, j], q_hat[first:]
            records.append(dict(evaluation=ev, producer=j, approach=approach, case=case,
                                MAE=mean_absolute_error(o, e), RMSE=root_mean_squared_error(o, e),
                                R2=r_squared(o, e), val_MAE=val))

    for ev in range(n_evaluations):
        for j in range(n_prod):
            score("CRM", ev, j, q_crm[:, j], float("nan"))

    for name in models:
        n_runs = n_evaluations if name in STOCHASTIC_MODELS else 1
        hyb, mlp = np.zeros_like(q_obs), np.zeros_like(q_obs)
        for e in range(n_runs):
            for j in range(n_prod):
                s0, seed = int(start_rows[j]), 1000 * e + j
                if n_hist_rows - s0 < 8:
                    if verbose and e == 0:
                        print(f"  skipping {name} for producer {j}: <8 active historical samples")
                    continue
                x_h = build_crm_ml_features(t_rows, w_rows, crm.params_, distances, j)[s0:]
                p_h, v_h = _fit_predict_ml(name, x_h, q_obs[s0:, j], n_hist_rows - s0, seed)
                p_m, v_m = _fit_predict_ml(name, w_rows[s0:], q_obs[s0:, j], n_hist_rows - s0, seed)
                p_h = np.concatenate([np.full(s0, np.nan), p_h])
                p_m = np.concatenate([np.full(s0, np.nan), p_m])
                hyb[:, j], mlp[:, j] = p_h, p_m
                for ee in (range(n_evaluations) if n_runs == 1 else [e]):  # replicate deterministic runs
                    score(f"CRM-{name}", ee, j, p_h, v_h)
                    score(name, ee, j, p_m, v_m)
            if verbose:
                print(f"  {name}: evaluation {e + 1}/{n_runs} done")
        preds[f"CRM-{name}"], preds[name] = hyb.copy(), mlp.copy()
    return pd.DataFrame.from_records(records), crm, preds


def run_precomputed_crm_study(time, production, q_crm, n_forecast: int,
                              n_evaluations: int = PAPER_N_EVALUATIONS,
                              active_start: Optional[Sequence[int]] = None,
                              models: Sequence[str] = ML_MODELS, verbose: bool = True):
    """Explicit adaptation for data without injection rates.

    This is NOT the article's CRM-ML formulation. It exists only as a
    compatibility path for workbooks that already contain q_CRM.

    Returns:
        ``(results, predictions)`` with predictions of shape (N, K) (NaN before a well starts).
    """
    time, production, q_crm = (np.asarray(a, float) for a in (time, production, q_crm))
    n_total, n_prod = production.shape
    n_hist = n_total - n_forecast
    starts = np.zeros(n_prod, int) if active_start is None else np.asarray(active_start, int)
    cases = {"entire": 0, "forecast": n_hist}
    records: List[Dict] = []

    def score(approach, ev, j, q_hat, val):
        for case, first in cases.items():
            first = max(first, int(starts[j]))
            if n_total - first < 2:
                continue
            o, e = production[first:, j], q_hat[first:]
            records.append(dict(evaluation=ev, producer=j, approach=approach, case=case,
                                MAE=mean_absolute_error(o, e), RMSE=root_mean_squared_error(o, e),
                                R2=r_squared(o, e), val_MAE=val))

    preds: Dict[str, FloatArray] = {"CRM": q_crm.copy()}
    for ev in range(n_evaluations):
        for j in range(n_prod):
            score("CRM", ev, j, q_crm[:, j], float("nan"))
    for name in models:
        n_runs = n_evaluations if name in STOCHASTIC_MODELS else 1
        hyb, mlp = np.full((n_total, n_prod), np.nan), np.full((n_total, n_prod), np.nan)
        for e in range(n_runs):
            for j in range(n_prod):
                s0 = int(starts[j])
                if n_hist - s0 < 8:
                    if verbose and e == 0:
                        print(f"  skipping {name} for producer {j}: <8 active historical samples")
                    continue
                seed = 1000 * e + j
                p_h, v_h = _fit_predict_ml(name, np.column_stack([time, q_crm[:, j]])[s0:],
                                           production[s0:, j], n_hist - s0, seed)
                p_m, v_m = _fit_predict_ml(name, time[s0:, None], production[s0:, j], n_hist - s0, seed)
                hyb[s0:, j], mlp[s0:, j] = p_h, p_m
                for ee in (range(n_evaluations) if n_runs == 1 else [e]):
                    score(f"CRM-{name}", ee, j, hyb[:, j], v_h)
                    score(name, ee, j, mlp[:, j], v_m)
            if verbose:
                print(f"  {name}: evaluation {e + 1}/{n_runs} done")
        preds[f"CRM-{name}"], preds[name] = hyb, mlp
    return pd.DataFrame.from_records(records), preds


def rank_approaches(results: pd.DataFrame, case: str = "forecast", metric: str = "MAE") -> pd.DataFrame:
    """Ranks approaches per (evaluation, producer) as in paper Fig. 23 (1 = best).

    Ranks are not derived from averaged errors, so one bad evaluation cannot skew them.

    Returns:
        DataFrame indexed by approach: ``mean_rank``, ``pct_first`` (% of rankings
        where it was best), ``mean_<metric>``; sorted best to worst.
    """
    sub = results[results["case"] == case].copy()
    sub["rank"] = sub.groupby(["evaluation", "producer"])[metric].rank(method="min")
    out = sub.groupby("approach").agg(
        mean_rank=("rank", "mean"),
        pct_first=("rank", lambda r: 100.0 * float(np.mean(r == 1.0))),
        **{f"mean_{metric}": (metric, "mean")})
    return out.sort_values("mean_rank")


@dataclass
class StudyOutput:
    """Everything produced by a study run (results, predictions, labels, CRM parameters)."""

    results: pd.DataFrame
    predictions: Dict[str, np.ndarray]
    time_rows: np.ndarray
    observed_rows: np.ndarray
    producer_names: List[str]
    n_forecast: int
    mode: str
    n_evaluations: int
    crm_params: Optional[CRMParameters] = None
    injector_names: List[str] = field(default_factory=list)
    time_unit: str = DEFAULT_TIME_UNIT
    rate_unit: str = DEFAULT_RATE_UNIT

    def ranking(self, case: str = "forecast", metric: str = "MAE") -> pd.DataFrame:
        """Ranking table for one case."""
        return rank_approaches(self.results, case=case, metric=metric)

    def summary_table(self, case: str = "forecast") -> pd.DataFrame:
        """Mean MAE / RMSE / R2 per approach for one case, sorted by MAE."""
        sub = self.results[self.results["case"] == case]
        return sub.groupby("approach")[["MAE", "RMSE", "R2"]].mean().sort_values("MAE")

    def predictions_long(self) -> pd.DataFrame:
        """Observed and predicted rates in tidy format (for CSV export)."""
        frames = []
        for j, name in enumerate(self.producer_names):
            f = pd.DataFrame({"time": self.time_rows, "producer": name,
                              f"observed [{self.rate_unit}]": self.observed_rows[:, j]})
            for a, arr in self.predictions.items():
                f[a] = arr[:, j]
            frames.append(f)
        return pd.concat(frames, ignore_index=True)


def run_pipeline(data: FieldData, n_forecast: int,
                 n_evaluations: int = PAPER_N_EVALUATIONS,
                 models: Sequence[str] = ML_MODELS, trim_inactive: bool = False,
                 paper_case_name: str = "",
                 include_bhp_term: bool = False,
                 crm_kwargs: Optional[dict] = None, verbose: bool = False) -> StudyOutput:
    """Picks the right study for ``data`` (full CRM calibration or pre-computed CRM) and runs it.

    Args:
        data: Validated field data.
        n_forecast: Trailing samples used as the forecast period.
        n_evaluations: Evaluations for ELM/MLP (paper: 20).
        models: Subset of ``ML_MODELS`` to combine with the CRM.
        trim_inactive: Skip each producer's leading zero-rate period.
        crm_kwargs: Extra arguments for the CRM calibrator (full mode).
        verbose: Print progress.
    """
    starts = data.first_active_index() if trim_inactive else None
    common = dict(producer_names=list(data.producer_names), n_forecast=n_forecast,
                  n_evaluations=n_evaluations, time_unit=data.time_unit, rate_unit=data.rate_unit)
    if data.mode == "full":
        results, crm, preds = run_forecast_study(
            data.time, data.injection, data.production, data.distances, n_forecast=n_forecast,
            n_evaluations=n_evaluations, bhp=data.bhp,
            include_bhp_term=include_bhp_term, paper_case_name=paper_case_name,
            crm_kwargs=crm_kwargs, models=models,
            verbose=verbose, active_start=starts)
        return StudyOutput(results, preds, data.time[1:], data.production[1:], mode="full",
                           crm_params=crm.params_, injector_names=list(data.injector_names), **common)
    results, preds = run_precomputed_crm_study(
        data.time, data.production, data.q_crm, n_forecast=n_forecast, n_evaluations=n_evaluations,
        active_start=starts, models=models, verbose=verbose)
    return StudyOutput(results, preds, data.time, data.production, mode="precomputed", **common)


# 6. FIGURES - ARTICLE-MAPPED, DESCRIPTIVE AND DATA-DRIVEN

# The article's published figures use compact mathematical notation in places.
# This implementation intentionally uses descriptive labels so a reader can
# understand the figure without decoding qObs/qCRM/λij/τij first.

APPROACH_LABELS = {
    "CRM": "Capacitance-Resistance Model (CRM)",
    "CRM-NuSVM": "CRM + NuSVM hybrid",
    "CRM-XGB": "CRM + XGBoost hybrid",
    "CRM-ELM": "CRM + Extreme Learning Machine hybrid",
    "CRM-MLP": "CRM + Multilayer Perceptron hybrid",
    "NuSVM": "NuSVM",
    "XGB": "XGBoost",
    "ELM": "Extreme Learning Machine",
    "MLP": "Multilayer Perceptron",
}

APPROACH_ORDER = (
    "CRM",
    "CRM-NuSVM",
    "CRM-XGB",
    "CRM-ELM",
    "CRM-MLP",
    "NuSVM",
    "XGB",
    "ELM",
    "MLP",
)

APPROACH_SHORT_LABELS = {
    "CRM": "CRM",
    "CRM-NuSVM": "CRM + NuSVM",
    "CRM-XGB": "CRM + XGBoost",
    "CRM-ELM": "CRM + ELM",
    "CRM-MLP": "CRM + MLP",
    "NuSVM": "NuSVM",
    "XGB": "XGBoost",
    "ELM": "ELM",
    "MLP": "MLP",
}

CASE_TEXT = {
    "entire": "entire production record (historical period plus forecast period)",
    "forecast": "forecast period only",
}


def _ordered(approaches: Sequence[str]) -> List[str]:
    known = [a for a in APPROACH_ORDER if a in approaches]
    return known + [a for a in approaches if a not in known]


def _labels(rate_unit: str, time_unit: str) -> Tuple[str, str]:
    return f"Liquid production rate [{rate_unit}]", f"Time [{time_unit}]"


def _grid(n: int, ncols: int = 2, cell=(6.0, 3.4)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(cell[0] * ncols, cell[1] * nrows),
        squeeze=False,
    )
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _style_axis(ax):
    """Use a restrained article-like layout while keeping labels explanatory."""
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_crm_fit(out: StudyOutput) -> Figure:
    """Article Fig. 9/18 concept: observed rates, calibrated CRM and forecast CRM."""
    t = out.time_rows
    obs = out.observed_rows
    q_crm = out.predictions["CRM"]
    nf = out.n_forecast
    rate_label, time_label = _labels(out.rate_unit, out.time_unit)

    fig, axes = _grid(obs.shape[1])
    split = len(t) - nf

    for j, ax in enumerate(axes):
        ax.plot(t, obs[:, j], color="black", lw=1.15,
                label="Observed liquid production")
        ax.plot(
            t[:split], q_crm[:split, j],
            color="#1f77b4", lw=1.55,
            label="CRM estimate during historical calibration",
        )
        ax.plot(
            t[split - 1:], q_crm[split - 1:, j],
            color="#d62728", lw=1.55,
            label="CRM forecast during held-out period",
        )
        ax.axvline(
            t[split], color="0.45", ls=":", lw=1.0,
            label="Forecast boundary",
        )
        _style_axis(ax)
        ax.set_title(f"Producer {out.producer_names[j]}", fontweight="bold")
        ax.set_xlabel(time_label)
        ax.set_ylabel(rate_label)

        history_slice = slice(0, split)
        forecast_slice = slice(split, None)
        text = (
            f"Historical calibration: MAE {mean_absolute_error(obs[history_slice, j], q_crm[history_slice, j]):.3f} "
            f"{out.rate_unit}; RMSE {root_mean_squared_error(obs[history_slice, j], q_crm[history_slice, j]):.3f} "
            f"{out.rate_unit}; R² {r_squared(obs[history_slice, j], q_crm[history_slice, j]):.3f}\n"
            f"Forecast: MAE {mean_absolute_error(obs[forecast_slice, j], q_crm[forecast_slice, j]):.3f} "
            f"{out.rate_unit}; RMSE {root_mean_squared_error(obs[forecast_slice, j], q_crm[forecast_slice, j]):.3f} "
            f"{out.rate_unit}; R² {r_squared(obs[forecast_slice, j], q_crm[forecast_slice, j]):.3f}"
        )
        ax.text(
            0.02, 0.03, text, transform=ax.transAxes,
            va="bottom", fontsize=6.6,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )

    axes[0].legend(fontsize=6.7, loc="upper right")
    fig.suptitle(
        "Observed liquid production compared with CRM estimates and forecasts\n"
        "Historical data are used for CRM calibration; the final held-out period is the forecast",
        fontweight="bold", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_all_approaches(out: StudyOutput, only_forecast: bool) -> Figure:
    """Article Fig. 10/11/14/19 concept with readable model names."""
    t = out.time_rows
    obs = out.observed_rows
    preds = out.predictions
    rate_label, time_label = _labels(out.rate_unit, out.time_unit)

    k = obs.shape[1]
    start = len(t) - out.n_forecast if only_forecast else 0
    sl = slice(start, None)
    approaches = _ordered(list(preds))

    nr = int(np.ceil(k / 2))
    fig = plt.figure(figsize=(12, 3.4 * nr + 3.5))
    gs = fig.add_gridspec(nr + 1, 2, height_ratios=[1] * nr + [0.9])
    mae = np.full((k, len(approaches)), np.nan)

    for j in range(k):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        ax.plot(
            t[sl], obs[sl, j], "k-", lw=1.35,
            label="Observed liquid production", zorder=5,
        )

        for ai, approach in enumerate(approaches):
            y = preds[approach][sl, j]
            ax.plot(
                t[sl], y, lw=0.95, alpha=0.92,
                label=APPROACH_SHORT_LABELS.get(approach, approach),
            )
            valid = np.isfinite(y)
            if valid.any():
                mae[j, ai] = mean_absolute_error(obs[sl, j][valid], y[valid])

        _style_axis(ax)
        ax.set_title(f"Producer {out.producer_names[j]}", fontweight="bold")
        ax.set_xlabel(time_label)
        ax.set_ylabel(rate_label)
        if j == 0:
            ax.legend(ncol=2, fontsize=5.9, loc="best")

    ax_table = fig.add_subplot(gs[-1, :])
    ax_table.axis("off")
    table_values = [[_num(v) for v in row] for row in mae]
    table = ax_table.table(
        cellText=table_values,
        rowLabels=list(out.producer_names),
        colLabels=[APPROACH_SHORT_LABELS.get(a, a) for a in approaches],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    ax_table.set_title(
        f"Mean absolute error for the displayed period [{out.rate_unit}] - smaller values indicate more accurate production forecasts",
        fontsize=8.5,
    )

    scope = (
        f"forecast period ({out.n_forecast} samples)"
        if only_forecast
        else "entire production record (historical + forecast)"
    )
    fig.suptitle(
        "Observed production versus CRM, CRM-ML hybrids and standalone ML forecasts\n"
        f"Evaluation scope: {scope}",
        fontweight="bold", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_mae_by_evaluation(out: StudyOutput, case: str, metric: str = "MAE") -> Figure:
    """Article Fig. 12/15/20/21 concept: variability across the 20 evaluations."""
    sub = out.results[out.results["case"] == case]
    producers = sorted(sub["producer"].unique())
    fig, axes = _grid(len(producers))

    for ax, j in zip(axes, producers):
        for approach in _ordered(sub["approach"].unique()):
            d = sub[
                (sub["producer"] == j) &
                (sub["approach"] == approach)
            ].sort_values("evaluation")

            ax.plot(
                d["evaluation"] + 1,
                d[metric],
                lw=1.2,
                label=APPROACH_SHORT_LABELS.get(approach, approach),
            )

        _style_axis(ax)
        ax.set_title(f"Producer {out.producer_names[j]}", fontweight="bold")
        ax.set_xlabel("Evaluation number (1–20)")
        if metric == "R2":
            ax.set_ylabel("Coefficient of determination, R²")
        else:
            ax.set_ylabel(f"{metric} [{out.rate_unit}]")

    axes[0].legend(fontsize=5.8, ncol=3)
    fig.suptitle(
        f"{metric} across the 20 repeated evaluations\n"
        f"Scope: {CASE_TEXT[case]}. Fluctuating lines show models affected by the random ANN train/validation split.",
        fontweight="bold", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_average_mae_bars(out: StudyOutput, case: str, metric: str = "MAE") -> Figure:
    """Article Fig. 13/16/22 concept: average model error per producer and all producers."""
    sub = out.results[out.results["case"] == case]
    approaches = _ordered(sub["approach"].unique())

    per = (
        sub.groupby(["producer", "approach"])[metric]
        .mean()
        .unstack("approach")
        .reindex(columns=approaches)
    )
    all_row = sub.groupby("approach")[metric].mean().reindex(approaches)
    per.loc["All producers"] = all_row

    labels = [
        out.producer_names[i] if isinstance(i, (int, np.integer)) else str(i)
        for i in per.index
    ]

    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(labels)), 5.2))
    width = 0.9 / len(approaches)
    x = np.arange(len(labels))

    for ai, approach in enumerate(approaches):
        vals = per[approach].to_numpy()
        bars = ax.bar(
            x + (ai - len(approaches) / 2 + 0.5) * width,
            vals,
            width,
            label=APPROACH_SHORT_LABELS.get(approach, approach),
        )
        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    _num(value),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=5.2,
                )

    _style_axis(ax)
    ax.set_xticks(x, labels)
    ax.set_ylabel(f"Average {metric} [{out.rate_unit}]")
    ax.set_title(
        f"Average {metric} for each forecasting approach\n"
        f"Mean over 20 evaluations - {CASE_TEXT[case]}",
        fontweight="bold",
    )
    ax.legend(ncol=3, fontsize=6.8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_ranking(out: StudyOutput) -> Figure:
    """Current-study ranking, explicitly labelled as a local ranking."""
    ranks = {
        "Entire production record": rank_approaches(out.results, "entire"),
        f"Forecast period ({out.n_forecast} samples)": rank_approaches(out.results, "forecast"),
    }

    fig, axes = plt.subplots(
        1, len(ranks),
        figsize=(4.4 * len(ranks), 6.0),
        squeeze=False,
    )

    for ax, (title, ranking) in zip(axes[0], ranks.items()):
        for position, approach in enumerate(ranking.index):
            ax.text(
                0.05, -position,
                APPROACH_LABELS.get(approach, approach),
                ha="left", va="center",
                fontsize=8.5, fontweight="bold",
            )
            ax.text(
                1.02, -position,
                f"Mean rank {ranking.loc[approach, 'mean_rank']:.2f}",
                ha="left", va="center", fontsize=7,
            )

        ax.set_xlim(0, 1.45)
        ax.set_ylim(-len(ranking) + 0.4, 0.6)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        "Ranking of production-forecasting approaches by mean rank\n"
        "Rank 1 is best; rankings are calculated separately for each evaluation and producer",
        fontweight="bold", fontsize=11,
    )
    fig.text(
        0.01, 0.5,
        "Higher position = better average ranking",
        rotation=90, va="center", fontsize=8.5,
    )
    fig.tight_layout(rect=(0.03, 0.0, 1, 0.92))
    return fig


def plot_crm_parameters(out: StudyOutput) -> Figure:
    """Article Fig. 8/17 concept: calibrated injector-to-producer CRM parameters."""
    if out.crm_params is None:
        raise ValueError("CRM parameters are unavailable.")

    params = out.crm_params
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4.4 + 0.25 * len(out.injector_names)),
    )

    for ax, matrix, title, colorbar_label in (
        (
            axes[0],
            params.lambda_ij,
            "Injector-to-producer connectivity strength",
            "Connectivity index",
        ),
        (
            axes[1],
            params.tau_ij,
            "Injector-to-producer response time",
            f"Response time [{out.time_unit}]",
        ),
    ):
        image = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(
            range(len(out.producer_names)),
            out.producer_names,
            rotation=30,
            ha="right",
        )
        ax.set_yticks(
            range(len(out.injector_names)),
            out.injector_names,
        )
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(
                col, row,
                f"{value:.3g}",
                ha="center", va="center", fontsize=8,
            )

        ax.set_xlabel("Producer wells")
        ax.set_ylabel("Injector wells")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(image, ax=ax, fraction=0.046, label=colorbar_label)

    fig.suptitle(
        "Calibrated CRM interwell parameters\n"
        "Rows identify injectors; columns identify the producers receiving their modeled influence",
        fontweight="bold", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ---------------------------------------------------------------------------
# Exact article figure reference support
# ---------------------------------------------------------------------------

ARTICLE_FIGURE_PAGES = {
    8: 19, 9: 20, 10: 21, 11: 22, 12: 23, 13: 24,
    14: 25, 15: 26, 16: 27, 17: 28, 18: 29, 19: 31,
    20: 32, 21: 33, 22: 34, 23: 35,
}


def extract_published_figure_pages(
    article_pdf: Union[str, Path],
    output_folder: Union[str, Path],
    dpi: int = 300,
) -> List[Path]:
    """Render the exact published article pages containing Figs. 8–23.

    This is the only honest way to provide an exact visual reproduction when
    the paper's original numerical arrays are not available. These PNGs are
    rendered directly from the supplied published PDF; they are not recreated
    from guessed data.
    """
    article_pdf = Path(article_pdf)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    document = fitz.open(article_pdf)
    written = []

    for figure_number, page_number in ARTICLE_FIGURE_PAGES.items():
        page = document[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        path = output_folder / f"article_figure_{figure_number:02d}_page_{page_number:02d}.png"
        pixmap.save(path)
        written.append(path)

    document.close()
    return written


FIGURE_CAPTIONS = {
    8: "Optimized interwell CRM parameters: connectivity indices and interwell time constants for the Synfield.",
    9: "Observed production and CRM-generated production during the historical calibration period and forecast period for the Synfield.",
    10: "Observed production compared with CRM, CRM-ML hybrids and standalone ML approaches using the full 8-year Synfield record.",
    11: "Observed production compared with CRM, CRM-ML hybrids and standalone ML approaches using the 1-year Synfield forecast period.",
    12: "MAE values across 20 evaluations for CRM, CRM-ML hybrids and standalone ML approaches for the Synfield.",
    13: "Average MAE by producer and across all producers for the full Synfield record and the 1-year forecast.",
    14: "Observed and predicted production rates for the 5-month short-term Synfield forecast.",
    15: "MAE values across 20 evaluations for the 5-month Synfield forecast.",
    16: "Average MAE by producer and across all producers for the 5-month Synfield forecast.",
    17: "Optimized interwell CRM parameters for the selected Buffalo field sector.",
    18: "Observed production and CRM-estimated production during the historical and forecast periods for the Buffalo field.",
    19: "Observed production compared with CRM, CRM-ML hybrids and standalone ML approaches for the Buffalo field.",
    20: "MAE values across 20 evaluations using the full 189-month Buffalo production record.",
    21: "MAE values across 20 evaluations using the 12-month Buffalo forecast period.",
    22: "Average RMSE by producer and across all producers for the full Buffalo record and the 12-month forecast.",
    23: "Overall ranking of CRM, CRM-ML and ML approaches across the Synfield and Buffalo cases.",
}

FIG_TITLES = {
    "figure_08_or_17_crm_parameters": "Optimized CRM parameters",
    "figure_09_or_18_crm_fit": "Observed production versus CRM estimate",
    "figure_10_or_11_or_14_or_19_model_comparison_entire":
        "Model comparison over the entire record",
    "figure_forecast_model_comparison":
        "Model comparison over the forecast period",
    "figure_12_or_15_or_20_or_21_evaluation_mae_forecast":
        "Mean absolute error across evaluations",
    "figure_13_or_16_or_22_average_mae_forecast":
        "Average model error by producer",
    "figure_23_local_ranking":
        "Overall model ranking",
    "figure_evaluation_mae_entire_record":
        "Mean absolute error across evaluations - entire record",
    "figure_average_mae_entire_record":
        "Average model error by producer - entire record",
}


def make_figures(out: StudyOutput) -> Dict[str, Figure]:
    """Build descriptive, article-mapped figures from the current study."""
    figures = {
        "figure_09_or_18_crm_fit": plot_crm_fit(out),
        "figure_10_or_11_or_14_or_19_model_comparison_entire": plot_all_approaches(out, False),
        "figure_forecast_model_comparison": plot_all_approaches(out, True),
        "figure_12_or_15_or_20_or_21_evaluation_mae_forecast": plot_mae_by_evaluation(out, "forecast"),
        "figure_13_or_16_or_22_average_mae_forecast": plot_average_mae_bars(out, "forecast"),
        "figure_23_local_ranking": plot_ranking(out),
    }

    if out.crm_params is not None:
        figures["figure_08_or_17_crm_parameters"] = plot_crm_parameters(out)

    if (out.results["case"] == "entire").any():
        figures["figure_evaluation_mae_entire_record"] = plot_mae_by_evaluation(out, "entire")
        figures["figure_average_mae_entire_record"] = plot_average_mae_bars(out, "entire")

    return figures


# ---------------------------------------------------------------------------
# Overall paper ranking across the five cases
# ---------------------------------------------------------------------------

def combine_paper_case_results(case_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine the five paper cases before calculating the overall ranking.

    Each DataFrame must contain:
        evaluation, producer, approach, case, MAE, RMSE, R2

    The article reports 20 evaluations per case. This function preserves
    evaluation identity within each case and adds a ``paper_case`` column.
    """
    required = {
        "evaluation", "producer", "approach", "case",
        "MAE", "RMSE", "R2",
    }

    frames = []
    for case_name, frame in case_results.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Case '{case_name}' is missing required columns: {sorted(missing)}"
            )
        local = frame.copy()
        local["paper_case"] = case_name
        frames.append(local)

    if not frames:
        raise ValueError("No paper cases were supplied.")

    combined = pd.concat(frames, ignore_index=True)

    # The article uses 20 evaluations for each case. Do not silently accept
    # incomplete evaluation sets when this function is used for replication.
    counts = combined.groupby("paper_case")["evaluation"].nunique()
    incomplete = counts[counts != PAPER_N_EVALUATIONS]
    if not incomplete.empty:
        raise ValueError(
            "Every paper case must contain exactly 20 evaluations. "
            f"Incomplete cases: {incomplete.to_dict()}"
        )

    return combined


PAPER_CASE_RANKING_SCOPE = {
    "Synfield - 8-year production history": "entire",
    "Synfield - 1-year forecast period": "forecast",
    "Synfield - 5-month forecast period": "forecast",
    "Buffalo - 189-month production history": "entire",
    "Buffalo - 12-month forecast period": "forecast",
}


def rank_overall_paper_cases(
    case_results: Dict[str, pd.DataFrame],
    metric: str = "MAE",
) -> pd.DataFrame:
    """Reproduce the article's overall five-case ranking.

    The paper's five cases are not all the same evaluation window. Therefore
    this function selects the appropriate result scope for each named case:

        Synfield 8-year          -> entire record
        Synfield 1-year forecast -> forecast
        Synfield 5-month forecast -> forecast
        Buffalo 189-month        -> entire record
        Buffalo 12-month         -> forecast

    With 4 Synfield producers and 8 Buffalo producers this produces:

        3 × 20 × 4 + 2 × 20 × 8 = 560 producer-evaluation groups.

    This is the correct level at which the article's overall ranking is formed.
    """
    combined = combine_paper_case_results(case_results)

    selected_frames = []
    for paper_case, scope in PAPER_CASE_RANKING_SCOPE.items():
        if paper_case not in case_results:
            raise ValueError(
                f"Missing required paper case: '{paper_case}'. "
                "All five article cases are required for the 560-evaluation ranking."
            )

        frame = combined[
            (combined["paper_case"] == paper_case) &
            (combined["case"] == scope)
        ].copy()
        selected_frames.append(frame)

    subset = pd.concat(selected_frames, ignore_index=True)
    subset["rank"] = subset.groupby(
        ["paper_case", "evaluation", "producer"]
    )[metric].rank(method="min")

    ranking = (
        subset.groupby("approach")
        .agg(
            mean_rank=("rank", "mean"),
            pct_first=("rank", lambda values: 100.0 * float(np.mean(values == 1.0))),
            mean_metric=(metric, "mean"),
        )
        .sort_values("mean_rank")
    )

    expected_groups = 3 * PAPER_N_EVALUATIONS * 4 + 2 * PAPER_N_EVALUATIONS * 8
    actual_groups = len(subset[["paper_case", "evaluation", "producer"]].drop_duplicates())

    if actual_groups != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} producer-evaluation groups for the "
            f"article's five-case ranking, but found {actual_groups}. "
            "Check that the Synfield has 4 producers and Buffalo has 8."
        )

    ranking.attrs["number_of_cases"] = 5
    ranking.attrs["evaluations_per_case"] = PAPER_N_EVALUATIONS
    ranking.attrs["producer_evaluation_groups"] = actual_groups
    ranking.attrs["ranking_scope"] = "all five article cases"
    return ranking


# ---------------------------------------------------------------------------
# Output export
# ---------------------------------------------------------------------------

def export_outputs(out: StudyOutput, folder: Union[str, Path], dpi: int = 200) -> List[Path]:
    """Export descriptive figures, metrics and calibrated CRM parameters."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for name, fig in make_figures(out).items():
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    tables = {
        "metrics_all_evaluations.csv": out.results,
        "ranking_forecast.csv": out.ranking("forecast").reset_index(),
        "ranking_entire.csv": out.ranking("entire").reset_index(),
        "summary_forecast.csv": out.summary_table("forecast").reset_index(),
        "summary_entire.csv": out.summary_table("entire").reset_index(),
        "predictions.csv": out.predictions_long(),
    }

    for filename, dataframe in tables.items():
        path = folder / filename
        dataframe.to_csv(path, index=False)
        written.append(path)

    if out.crm_params is not None:
        path = folder / "crm_parameters.xlsx"
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame(
                out.crm_params.lambda_ij,
                index=out.injector_names,
                columns=out.producer_names,
            ).to_excel(writer, sheet_name="connectivity_indices")

            pd.DataFrame(
                out.crm_params.tau_ij,
                index=out.injector_names,
                columns=out.producer_names,
            ).to_excel(writer, sheet_name="response_times")

            pd.DataFrame(
                {"producer_time_constant": out.crm_params.tau_j},
                index=out.producer_names,
            ).to_excel(writer, sheet_name="producer_time_constants")

        written.append(path)

    return written


# 7. DEMONSTRATION FIELD - NOT THE ARTICLE SYNFIELD


def make_mock_field(n_steps: int = 400, seed: int = 7, noise: float = 0.02):
    """Generate a self-contained demonstration field for software testing.

    This dataset is intentionally NOT described as the article's Synfield.
    The article's Synfield is a reservoir-simulation dataset on a 67 × 67 × 5
    grid with 2,922 days of production/injection history. The actual numerical
    synfield arrays are not supplied by the article as a reusable workbook.

    Returns:
        ``(time, injection, production, distances_ft, true_params)``.
    """
    rng = np.random.default_rng(seed)
    n_inj, n_prod = 5, 4
    time = np.arange(n_steps, dtype=float)
    inj_xy = np.array([[500, 2100], [2100, 2100], [1300, 1300], [500, 500], [2100, 500]], float)
    prod_xy = np.array([[1300, 2100], [500, 1300], [2100, 1300], [1300, 500]], float)
    distances = np.linalg.norm(inj_xy[:, None, :] - prod_xy[None, :, :], axis=2)
    means = np.array([2.7, 0.9, 1.8, 0.9, 2.7])
    blocks = means * rng.uniform(0.6, 1.4, size=(int(np.ceil(n_steps / 30)), n_inj))
    injection = np.repeat(blocks, 30, axis=0)[:n_steps]
    lam = 1.0 / distances
    lam /= lam.sum(axis=1, keepdims=True)
    true = CRMParameters(tau_j=rng.uniform(5, 10, n_prod), lambda_ij=lam, tau_ij=3.0 + distances / 150.0)
    q0 = (lam * injection[0][:, None]).sum(axis=0)
    production = np.vstack([q0, crm_simulate(true, time, injection, q0)])
    production *= 1.0 + noise * rng.standard_normal(production.shape)
    production[0] = q0
    return time, injection, production, distances, true


def make_mock_fielddata(n_steps: int = 400) -> Tuple[FieldData, CRMParameters]:
    """The synthetic field as validated :class:`FieldData` (+ the true CRM parameters)."""
    t, inj, prod, dist, true = make_mock_field(n_steps)
    return from_arrays(t, prod, inj, dist, time_unit="days", rate_unit="MSTB/day"), true


def write_example_workbook(path: Union[str, Path], n_steps: int = 400) -> Path:
    """Writes the synthetic field as a Layout-A workbook (shows the exact expected format)."""
    t, inj, prod, dist, _ = make_mock_field(n_steps)
    return _write_workbook(Path(path), t, prod, inj, dist)


# 8. COMMAND LINE AND STREAMLIT GUI


def _print_report(out: StudyOutput, true: Optional[CRMParameters]) -> None:
    """Prints calibrated parameters, rankings and mean metrics to the console."""
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option("display.width", 140)
    if out.crm_params is not None:
        p = out.crm_params
        print(f"\nCalibrated λij (rows = injectors, columns = producers):\n{p.lambda_ij}")
        if true is not None:
            print(f"True λij of the synthetic field:\n{true.lambda_ij}")
        print(f"Calibrated τj [{out.time_unit}]: {p.tau_j}")
        print(f"Σj λij per injector (must be <= 1): {p.lambda_ij.sum(axis=1)}")
    for case in ("entire", "forecast"):
        print(f"\n=== Ranking on the {CASE_TEXT[case]} (by MAE; mean rank 1 = best) ===")
        print(out.ranking(case).round(3))
    print(f"\n=== Mean metrics on the forecast period, all producers (MAE/RMSE in {out.rate_unit}) ===")
    print(out.summary_table("forecast").round(4))


def validate_paper_protocol(
    *,
    n_evaluations: int,
    include_bhp_term: bool,
    paper_case_name: str,
) -> None:
    """Validate settings before an article-faithful run."""
    if n_evaluations != PAPER_N_EVALUATIONS:
        raise ValueError(
            f"Paper-faithful runs require {PAPER_N_EVALUATIONS} evaluations; "
            f"received {n_evaluations}."
        )

    validate_bhp_policy(paper_case_name, include_bhp_term)



def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point: load data, run the study, export charts and tables."""
    ap = argparse.ArgumentParser(
        description="CRM-ML hybrid production forecasting (Ogali & Orodu, 2025). "
                    "Without --data a built-in synthetic field is used.")
    ap.add_argument("--data", help="Excel workbook (.xlsx); omit to use the synthetic example")
    ap.add_argument("--forecast", type=int, help="forecast length in samples (default 60 synthetic / 12 workbook)")
    ap.add_argument("--evals", type=int, default=PAPER_N_EVALUATIONS, help="evaluations for ELM/MLP (paper protocol: 20)")
    ap.add_argument("--models", nargs="+", default=list(ML_MODELS), choices=list(ML_MODELS))
    ap.add_argument("--starts", type=int, default=2, help="CRM optimiser restarts (default 2)")
    ap.add_argument("--trim-inactive", action="store_true", help="optionally skip each producer's leading zero-rate period; this is not required by the paper")
    ap.add_argument("--days", type=int, default=400, help="length of the synthetic field (default 400 days)")
    ap.add_argument("--out", default="results", help="output folder (default ./results)")
    ap.add_argument("--quiet", action="store_true", help="hide progress output")
    ap.add_argument("--gui", action="store_true", help="launch the Streamlit GUI instead")
    ap.add_argument("--paper-case", default="", choices=["", *PAPER_CASES], help="name the article case being reproduced")
    ap.add_argument("--write-template", metavar="FILE", help="write a Layout-A template workbook and exit")
    ap.add_argument("--write-example", metavar="FILE", help="write the synthetic field as a workbook and exit")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    if args.write_template:
        print("wrote", write_template(args.write_template)); return
    if args.write_example:
        print("wrote", write_example_workbook(args.write_example, args.days)); return
    if args.gui:
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]))

    true = None
    if args.data:
        data, n_fc = load_field_data(args.data), args.forecast or 12
    else:
        data, true = make_mock_fielddata(args.days)
        n_fc = args.forecast or 60
        print(f"No --data given: using the built-in synthetic field ({args.days} days, 5 injectors, 4 producers).")
    print(f"Loaded {len(data.time)} samples, {len(data.producer_names)} producers, mode = {data.mode}, "
          f"rates in {data.rate_unit}, time in {data.time_unit}")
    for note in data.notes:
        print("  note:", note)
    out = run_pipeline(data, n_fc, args.evals, args.models, trim_inactive=args.trim_inactive,
                       paper_case_name=args.paper_case,
                       include_bhp_term=False,
                       crm_kwargs=dict(n_starts=args.starts, max_iter=150), verbose=not args.quiet)
    _print_report(out, true)
    paths = export_outputs(out, args.out)
    print(f"\nWrote {len(paths)} files (charts: PNG, tables: CSV/XLSX) to {Path(args.out).resolve()}")


def run_gui() -> None:
    """Streamlit app: upload (or use the synthetic field) -> settings -> run -> charts -> ZIP."""
    import streamlit as st

    warnings.filterwarnings("ignore")
    st.set_page_config(page_title="CRM-ML Production Forecasting", layout="wide")

    @st.cache_data(show_spinner=False)
    def _load(content: bytes) -> FieldData:
        return load_from_bytes(content)

    def _zip(out: StudyOutput) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in export_outputs(out, tmp):
                    zf.write(f, arcname=f.name)
            return buf.getvalue()

    st.title("CRM-ML hybrid production forecasting")
    st.caption("Capacitance-Resistance Model + machine learning (Ogali & Orodu, 2025)")
    with st.sidebar:
        st.header("1. Data")
        upload = st.file_uploader("Excel workbook (.xlsx)", type=["xlsx"])
        use_demo = st.checkbox("Use built-in synthetic field", value=upload is None)
        with tempfile.TemporaryDirectory() as tmp:
            st.download_button("Download blank template", write_template(Path(tmp) / "t.xlsx").read_bytes(),
                               "crm_ml_template.xlsx", help="Layout A: Time, q_Observed, w_Injection, Distances")
        st.header("2. Settings")
        n_forecast = st.number_input("Forecast length (samples)", 1, 1000, 60 if (use_demo or upload is None) else 12)
        n_evals = st.slider("Evaluations (ELM / MLP)", 1, 20, PAPER_N_EVALUATIONS, help="Paper protocol: 20 evaluations.")
        models = st.multiselect("ML models to combine with CRM", list(ML_MODELS), default=list(ML_MODELS))
        trim = st.checkbox("Skip each producer's leading zero-rate period (optional preprocessing)", value=False)
        n_starts = st.slider("CRM optimiser restarts", 1, 8, 2)
        run = st.button("Run study", type="primary", use_container_width=True)

    try:
        data = _load(upload.getvalue()) if (upload is not None and not use_demo) else make_mock_fielddata()[0]
    except Exception as exc:
        st.error(f"Could not read the workbook: {exc}")
        st.stop()
    st.markdown("**Full mode** - the CRM is calibrated from injection and production data." if data.mode == "full"
                else "**Pre-computed CRM mode** - no injection data; the supplied `q_CRM` series is used.")
    for note in data.notes:
        st.caption(f"i {note}")

    tab_data, tab_results = st.tabs(["Data preview", "Results"])
    with tab_data:
        st.dataframe(data.summary(), use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 3.4))
        for j, name in enumerate(data.producer_names):
            ax.plot(data.time, data.production[:, j], label=name, lw=1.2)
        ax.set_xlabel(f"Time [{data.time_unit}]")
        ax.set_ylabel(f"q [{data.rate_unit}]")
        ax.set_title("Observed liquid production rates of all producers")
        ax.legend(fontsize=7, ncol=3)
        st.pyplot(fig)
        plt.close(fig)
        if n_forecast >= len(data.time) - 10:
            st.warning("Forecast length leaves fewer than 10 historical samples.")

    if run:
        if not models:
            st.error("Select at least one ML model.")
            st.stop()
        with st.spinner("Running (CRM calibration and model training)..."):
            try:
                st.session_state["out"] = run_pipeline(data, int(n_forecast), int(n_evals), models,
                                                       trim_inactive=trim, crm_kwargs=dict(n_starts=int(n_starts), max_iter=150))
            except Exception as exc:
                st.error(f"Study failed: {exc}")
                st.stop()

    with tab_results:
        out: Optional[StudyOutput] = st.session_state.get("out")
        if out is None:
            st.write("Press **Run study** in the sidebar.")
            return
        c1, c2 = st.columns(2)
        c1.subheader("Ranking - forecast period")
        c1.dataframe(out.ranking("forecast").round(3), use_container_width=True)
        c2.subheader("Ranking - entire record")
        c2.dataframe(out.ranking("entire").round(3), use_container_width=True)
        st.download_button("Download all charts + tables (ZIP)", _zip(out), "crm_ml_results.zip", mime="application/zip")
        figs = make_figures(out)
        for tab, (name, fig) in zip(st.tabs([FIG_TITLES.get(k, k) for k in figs]), figs.items()):
            with tab:
                st.pyplot(fig)
                plt.close(fig)


if "streamlit" in sys.modules and __name__ != "main_cli_import":
    try:  # launched by `streamlit run main.py`
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            run_gui()
        elif __name__ == "__main__":
            main()
    except ImportError:
        if __name__ == "__main__":
            main()
elif __name__ == "__main__":
    main()