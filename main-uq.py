#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "pandas>=2.0",
#     "scipy>=1.10",
#     "scikit-learn>=1.3",
#     "matplotlib>=3.7",
#     "openpyxl>=3.1",
#     "xgboost>=2.0",
#     "torch>=2.0",
#     "streamlit>=1.30",
# ]
# ///
"""Bayesian uncertainty-quantified CRM-ML production forecasting.

Research focus
--------------
This program is intentionally focused on ONE task:

    CRM + selected machine-learning model + Bayesian UQ

Only one CRM-ML method is trained per run. CRM-XGB is the recommended/default
option, while CRM-NuSVM, CRM-ELM and CRM-MLP remain selectable alternatives.

Reference model
---------------
Ogali, O.I.O. and Orodu, O.D. (2025). "Concatenating data-driven and reduced-
physics models for smart production forecasting." DOI: 10.1007/s12145-025-01745-9.

UQ methodology
--------------
1. Calibrate the CRM from injection and production history.
2. Build the paper's CRM-ML feature vector and add the time-varying CRM response
   as a Bayesian-extension physics feature.
3. Fit the selected ML model to positive production rates on the historical data.
4. Use a chronological hold-out block to obtain out-of-sample calibration errors.
5. Model positive production with a Lognormal likelihood:

       log(q_obs) ~ Normal(alpha + beta * log(q_base), sigma)

   which is equivalent to q_obs ~ LogNormal(alpha + beta*log(q_base), sigma).
6. Use PyTorch automatic differentiation and Hamiltonian Monte Carlo (HMC) for the
   nonlinear CRM parameters. A CRM-only residual variance is sampled for posterior
   diagnostics, but it is NOT reused for the final CRM-ML predictive distribution.
7. Estimate the final aleatoric scale from chronological out-of-sample residuals of
   the selected CRM-ML model, then propagate CRM posterior uncertainty through the
   CRM-ML feature vector and combine it with that ML residual scale.
8. Report P90/P50/P10, 95% prediction intervals, MAE/RMSE/R2 of the P50 forecast,
   PICP, MPIW and interval score.

Important
---------
- The Lognormal likelihood is only defined for q > 0. The program does NOT replace
  zero production silently. Leading inactive rows can be excluded explicitly with
  --trim-leading-inactive, while interior/non-positive historical values still need
  to be resolved as a data/preprocessing decision.
- MCMC is applied directly to CRM lambda/tau parameters. The deterministic CRM
  solution is an initial state only; broad transformed-space priors are used.
- The final predictive variance is decomposed into CRM-parameter uncertainty
  (epistemic) and out-of-sample CRM-ML residual uncertainty (aleatoric).
- The paper's CRM formulation and minimum tau constraint are retained; ML hyperparameters,
  Bayesian priors and HMC settings are implementation choices unless the paper explicitly specifies them.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    raise ImportError("XGBoost is required. Install it with: uv add xgboost") from exc

FloatArray = np.ndarray
PathLike = Union[str, Path, BinaryIO]

DEFAULT_RATE_UNIT = "MSTB/day"
DEFAULT_TIME_UNIT = "days"

# ---------------------------------------------------------------------------
# User-facing choices: CRM-XGB is intentionally first and is the default.
# ---------------------------------------------------------------------------
CRM_ML_MODELS = ("CRM-XGB", "CRM-NuSVM", "CRM-ELM", "CRM-MLP")
DEFAULT_CRM_ML_MODEL = "CRM-XGB"

# Paper-supported CRM/ML settings.
PAPER_MLP_HIDDEN_NEURONS = 10
PAPER_TRAIN_FRACTION = 0.75
PAPER_VALIDATION_FRACTION = 0.25
PAPER_MIN_TAU_SAMPLING_INTERVAL = True

# Implementation settings: not attributed to the paper unless stated above.
ML_SETTINGS = {
    "NuSVM": {"kernel": "rbf", "nu": 0.5, "C": 10.0, "gamma": "scale"},
    "XGB": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_jobs": 1,
        "verbosity": 0,
    },
    "ELM": {"ridge": 1e-3},
    "MLP": {"activation": "tanh", "solver": "lbfgs", "max_iter": 500},
}

# Bayesian UQ defaults. These are implementation choices.
MCMC_DEFAULT_CHAINS = 3
MCMC_DEFAULT_SAMPLES = 1500
MCMC_DEFAULT_BURN_IN = 1000
MCMC_DEFAULT_THIN = 2
MCMC_DEFAULT_SEED = 2025
MCMC_TARGET_ACCEPTANCE = 0.30
UQ_CI_LEVEL = 0.95
UQ_RHAT_WARNING = 1.10
UQ_CALIBRATION_FRACTION = 0.75
UQ_MAX_CRM_PREDICTIVE_DRAWS = 500
UQ_WEAK_PRIOR_SD = 5.0



# 1. DATA STRUCTURES AND METRICS



@dataclass
class FieldData:
    """Full field data required for CRM-ML forecasting."""

    time: np.ndarray
    production: np.ndarray
    injection: np.ndarray
    distances: np.ndarray
    producer_names: List[str]
    injector_names: List[str]
    bhp: Optional[np.ndarray] = None
    time_unit: str = DEFAULT_TIME_UNIT
    rate_unit: str = DEFAULT_RATE_UNIT
    notes: List[str] = field(default_factory=list)
    crm_baseline: Optional[np.ndarray] = None

    def first_active_index(self) -> np.ndarray:
        positive = self.production > 0
        return np.where(positive.any(axis=0), positive.argmax(axis=0), 0)

    def summary(self) -> pd.DataFrame:
        starts = self.first_active_index()
        rows = []
        for j, name in enumerate(self.producer_names):
            idx = np.flatnonzero(self.production[:, j] > 0)
            rows.append({
                "producer": name,
                "first_positive_step": int(starts[j]),
                "last_positive_step": int(idx[-1]) if idx.size else -1,
                "positive_steps": int(idx.size),
                f"peak_rate [{self.rate_unit}]": float(self.production[:, j].max()),
            })
        return pd.DataFrame(rows)


def mean_absolute_error(q_obs: np.ndarray, q_est: np.ndarray) -> float:
    q_obs, q_est = np.asarray(q_obs, float), np.asarray(q_est, float)
    if q_obs.shape != q_est.shape or q_obs.size == 0:
        raise ValueError("MAE inputs must have equal non-empty shapes.")
    return float(np.mean(np.abs(q_est - q_obs)))


def root_mean_squared_error(q_obs: np.ndarray, q_est: np.ndarray) -> float:
    q_obs, q_est = np.asarray(q_obs, float), np.asarray(q_est, float)
    if q_obs.shape != q_est.shape or q_obs.size == 0:
        raise ValueError("RMSE inputs must have equal non-empty shapes.")
    return float(np.sqrt(np.mean((q_est - q_obs) ** 2)))


def r_squared(q_obs: np.ndarray, q_est: np.ndarray) -> float:
    """Conventional coefficient of determination: 1 - SSE/SST."""
    q_obs, q_est = np.asarray(q_obs, float), np.asarray(q_est, float)
    if q_obs.shape != q_est.shape or q_obs.size == 0:
        raise ValueError("R2 inputs must have equal non-empty shapes.")
    sst = float(np.sum((q_obs - q_obs.mean()) ** 2))
    if sst <= 0:
        return float("nan")
    sse = float(np.sum((q_obs - q_est) ** 2))
    return float(1.0 - sse / sst)



# 2. EXCEL IMPORT / VALIDATION



SHEET_ALIASES: Dict[str, List[str]] = {
    "time": ["time", "t", "date", "days", "months"],
    "production": ["q_observed", "q_obs", "production", "q_production", "observed", "q"],
    "crm_baseline": ["q_crm", "crm", "crm_prediction", "q_crm_prediction"],
    "injection": ["w_injection", "injection", "w_inj", "w", "inj"],
    "distances": ["distances", "distance", "x_ij", "dist"],
    "bhp": ["bhp", "pwf", "p_wf"],
}


def _find_sheet(names: Sequence[str], key: str) -> Optional[str]:
    lookup = {name.strip().lower(): name for name in names}
    return next((lookup[a] for a in SHEET_ALIASES[key] if a in lookup), None)


def _parse_block(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty:
        raise ValueError("A required worksheet is empty.")
    names = [str(c).strip() for c in raw.iloc[0].tolist()]
    body = raw.iloc[1:].reset_index(drop=True)
    numeric = body.apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna().all(axis=1)
    if not valid.any():
        raise ValueError("No numeric rows were found below the worksheet header.")
    first = int(valid.to_numpy().argmax())
    notes = [" | ".join(str(v) for v in body.iloc[r].tolist()) for r in range(first)]
    data = numeric.iloc[first:].reset_index(drop=True)
    data.columns = names
    return data, names, notes


def _extract_unit(notes: Sequence[str]) -> str:
    for note in notes:
        for part in str(note).split("|"):
            part = part.strip()
            if part.startswith("[") and part.endswith("]"):
                return part[1:-1]
    return ""


def validate_field(data: FieldData) -> None:
    time, prod, inj, dist = map(lambda x: np.asarray(x, float), (data.time, data.production, data.injection, data.distances))
    if time.ndim != 1 or prod.ndim != 2 or inj.ndim != 2 or dist.ndim != 2:
        raise ValueError("Time must be 1D; production, injection and distances must be 2D.")
    n, k = prod.shape
    i = inj.shape[1]
    if len(time) != n or len(inj) != n:
        raise ValueError("Time, production and injection must have the same number of rows.")
    if dist.shape != (i, k):
        raise ValueError(f"Distances must have shape ({i}, {k}). Got {dist.shape}.")
    if len(data.producer_names) != k or len(data.injector_names) != i:
        raise ValueError("Producer/injector names do not match data dimensions.")
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
        raise ValueError("Time must be finite and strictly increasing.")
    for label, arr in (("production", prod), ("injection", inj), ("distances", dist), ("BHP", data.bhp)):
        if arr is not None and not np.all(np.isfinite(arr)):
            raise ValueError(f"{label} contains NaN or infinity.")
    if np.any(inj < 0):
        warnings.warn("Negative injection values were supplied; verify their physical meaning.", RuntimeWarning)


def load_field_data(source: PathLike, sheet_map: Optional[Dict[str, str]] = None) -> FieldData:
    """Load the full workbook layout required by the UQ CRM-ML method."""
    if hasattr(source, "seek"):
        source.seek(0)
    xls = pd.ExcelFile(source)
    names = xls.sheet_names
    sm = {k: (sheet_map or {}).get(k) or _find_sheet(names, k) for k in SHEET_ALIASES}
    missing = [k for k in ("time", "production") if sm[k] is None]
    if missing:
        raise ValueError(f"Missing required worksheet(s): {missing}. Present: {names}")
    t_df, _, t_notes = _parse_block(xls.parse(sm["time"], header=None))
    q_df, prod_names, q_notes = _parse_block(xls.parse(sm["production"], header=None))
    if sm["injection"] is None:
        # A field may legitimately contain no injectors. Represent that case as
        # an N x 0 matrix rather than inventing a dummy injector.
        inj_names = []
        w_df = pd.DataFrame(index=np.arange(len(q_df)))
        no_injectors = True
    else:
        w_df, inj_names, _ = _parse_block(xls.parse(sm["injection"], header=None))
        no_injectors = w_df.shape[1] == 0
    # Distances are commonly stored as a matrix with injector names in the first
    # column (e.g. I-01, I-02, ...). The generic block parser cannot handle that
    # layout because the label column is non-numeric, so parse it separately.
    dist_df = None
    n_injectors = len(inj_names)
    n_producers = len(prod_names)
    if n_injectors == 0:
        # With no injectors there are no injector-producer distances. Keep an
        # explicit empty (0 x K) matrix so every downstream CRM dimension
        # remains well-defined. A Distances sheet is not required.
        dist_df = pd.DataFrame(np.empty((0, n_producers), dtype=float))
    elif sm["distances"] is not None:
        raw_dist = xls.parse(sm["distances"], header=None).dropna(how="all").reset_index(drop=True)
        if raw_dist.empty:
            raise ValueError("The Distances sheet is empty.")
        numeric_dist = raw_dist.apply(pd.to_numeric, errors="coerce")
        valid_dist_rows = numeric_dist.notna().sum(axis=1) > 0
        if not valid_dist_rows.any():
            raise ValueError("The Distances sheet contains no numeric distance values.")
        first_dist = int(valid_dist_rows.to_numpy().argmax())
        dist_body = numeric_dist.iloc[first_dist:].reset_index(drop=True)
        # Drop a leading injector-label column when present.
        if dist_body.shape[1] == n_producers + 1:
            dist_body = dist_body.iloc[:, 1:]
        elif dist_body.shape[1] > n_producers:
            dist_body = dist_body.iloc[:, -n_producers:]
        dist_df = dist_body

    time = t_df.iloc[:, 0].to_numpy(float)
    production = q_df.to_numpy(float)
    injection = w_df.to_numpy(float)
    if dist_df is None:
        distances = np.empty((injection.shape[1], production.shape[1]), dtype=float)
    else:
        distances = dist_df.to_numpy(float)
    notes = []
    if injection.shape[1] == 0:
        notes.append("No injectors detected. CRM interwell injection-connectivity terms are disabled; production-only CRM terms remain active.")
    elif dist_df is None:
        distances = np.ones((injection.shape[1], production.shape[1]))
        notes.append("No Distances sheet supplied; distances were set to 1 for all injector-producer pairs.")
    elif distances.shape != (injection.shape[1], production.shape[1]):
        raise ValueError(
            "Distances sheet has the wrong shape. Expected "
            f"({injection.shape[1]} injectors, {production.shape[1]} producers), "
            f"but parsed {distances.shape}. Arrange distances with injectors as rows and producers as columns."
        )
    crm_baseline = None
    if sm.get("crm_baseline") is not None:
        crm_df, crm_names, _ = _parse_block(xls.parse(sm["crm_baseline"], header=None))
        crm_baseline = crm_df.to_numpy(float)
        if crm_baseline.shape != production.shape:
            raise ValueError(
                f"q_CRM sheet must have shape {production.shape}; parsed {crm_baseline.shape}."
            )
        if len(crm_names) != len(prod_names) or any(a != b for a, b in zip(crm_names, prod_names)):
            raise ValueError("q_CRM producer columns must match q_Observed producer columns in the same order.")
        notes.append("Pre-computed q_CRM baseline detected. Bayesian UQ will use this supplied CRM response instead of inventing injector/connectivity parameters.")
    bhp = None if sm["bhp"] is None else _parse_block(xls.parse(sm["bhp"], header=None))[0].to_numpy(float)
    data = FieldData(
        time=time,
        production=production,
        injection=injection,
        distances=distances,
        producer_names=prod_names,
        injector_names=inj_names,
        bhp=bhp,
        time_unit=_extract_unit(t_notes) or DEFAULT_TIME_UNIT,
        rate_unit=_extract_unit(q_notes) or DEFAULT_RATE_UNIT,
        notes=notes,
        crm_baseline=crm_baseline,
    )
    validate_field(data)
    return data


def load_from_bytes(content: bytes) -> FieldData:
    return load_field_data(io.BytesIO(content))


def write_template(path: Union[str, Path], n_steps: int = 36, n_injectors: int = 3, n_producers: int = 2) -> Path:
    """Create a workbook that exactly matches the importer layout.

    The workbook contains Time and q_Observed sheets. When n_injectors > 0,
    it also contains w_Injection and Distances. When n_injectors == 0, no
    dummy injector is created and the two injector-dependent sheets are omitted.
    """
    rng = np.random.default_rng(0)
    path = Path(path)
    time = np.arange(n_steps, dtype=float)
    production = rng.uniform(1.0, 2.0, (n_steps, n_producers))
    injection = rng.uniform(1.0, 2.0, (n_steps, n_injectors))
    distances = rng.uniform(500, 3000, (n_injectors, n_producers))
    producer_names = [f"P-{j+1:02d}" for j in range(n_producers)]
    injector_names = [f"I-{i+1:02d}" for i in range(n_injectors)]

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        # Time: header, unit row, then numeric values.
        time_rows = [["Time"], ["[days]"]] + time.reshape(-1, 1).tolist()
        pd.DataFrame(time_rows).to_excel(
            xw, sheet_name="Time", header=False, index=False
        )

        # Production: names, units, then data.
        prod_rows = [producer_names, ["[MSTB/day]"] * n_producers] + production.tolist()
        pd.DataFrame(prod_rows).to_excel(
            xw, sheet_name="q_Observed", header=False, index=False
        )

        if n_injectors > 0:
            # Injection: names, units, then data.
            inj_rows = [injector_names, ["[MSTB/day]"] * n_injectors] + injection.tolist()
            pd.DataFrame(inj_rows).to_excel(
                xw, sheet_name="w_Injection", header=False, index=False
            )

            # Distances: first column identifies the injector; remaining columns
            # are producer columns.
            distance_rows = [[""] + producer_names, [""] + ["[ft]"] * n_producers]
            distance_rows += [[injector_names[i]] + distances[i].tolist() for i in range(n_injectors)]
            pd.DataFrame(distance_rows).to_excel(
                xw, sheet_name="Distances", header=False, index=False
            )

    return path



# 3. CRM



@dataclass
class CRMParameters:
    tau_j: np.ndarray
    lambda_ij: np.ndarray
    tau_ij: np.ndarray


def _exp_filter(signal: np.ndarray, dt: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0 or np.any(dt <= 0):
        raise ValueError("tau and sampling intervals must be positive.")
    if np.allclose(dt, dt[0]):
        a = math.exp(-float(dt[0]) / float(tau))
        return lfilter([1.0 - a], [1.0, -a], signal)
    out = np.empty_like(signal, dtype=float)
    previous = 0.0
    for n, a in enumerate(np.exp(-dt / tau)):
        previous = a * previous + (1.0 - a) * signal[n]
        out[n] = previous
    return out


def crm_simulate(params: CRMParameters, time: np.ndarray, injection: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """CRM Eq. (1), returning predictions for t[1:] in physical units/scales supplied."""
    time, injection, q0 = np.asarray(time, float), np.asarray(injection, float), np.asarray(q0, float)
    if injection.ndim != 2 or q0.ndim != 1:
        raise ValueError("Injection must be 2D and q0 must be 1D.")
    n_inj, n_prod = params.lambda_ij.shape
    if injection.shape[1] != n_inj or q0.size != n_prod or time.size != injection.shape[0]:
        raise ValueError("CRM parameter/data dimensions are inconsistent.")
    dt = np.diff(time)
    t_rel = time[1:] - time[0]
    out = q0[None, :] * np.exp(-t_rel[:, None] / params.tau_j[None, :])
    for i in range(n_inj):
        response = _exp_filter(injection[1:, i], dt, params.tau_ij[i, 0])
        for j in range(n_prod):
            if j:
                response = _exp_filter(injection[1:, i], dt, params.tau_ij[i, j])
            out[:, j] += params.lambda_ij[i, j] * response
    return out


class CapacitanceResistanceModel:
    """Constrained CRM calibration using SLSQP."""

    def __init__(self, tau_min: Optional[float] = None, tau_max: Optional[float] = None,
                 n_starts: int = 3, max_iter: int = 300, random_state: int = 0):
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_starts = max(1, int(n_starts))
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.params_: Optional[CRMParameters] = None
        self.objective_: float = float("nan")
        self._scale = 1.0

    def _unpack(self, x: np.ndarray, n_inj: int, n_prod: int) -> CRMParameters:
        p = 0
        tau_j = np.exp(x[p:p+n_prod]); p += n_prod
        lam = x[p:p+n_inj*n_prod].reshape(n_inj, n_prod); p += n_inj*n_prod
        tau_ij = np.exp(x[p:p+n_inj*n_prod].reshape(n_inj, n_prod))
        return CRMParameters(tau_j, lam, tau_ij)

    def fit(self, time: np.ndarray, injection: np.ndarray, production: np.ndarray) -> "CapacitanceResistanceModel":
        time, injection, production = map(lambda x: np.asarray(x, float), (time, injection, production))
        n, n_inj = injection.shape
        if time.size != n or production.shape[0] != n:
            raise ValueError("Time, injection and production must have equal lengths.")
        if production.ndim != 2 or n < 10:
            raise ValueError("Production must be 2D and at least 10 time steps are required.")
        n_prod = production.shape[1]
        tau_min = float(np.min(np.diff(time)) if self.tau_min is None else self.tau_min)
        if tau_min <= 0:
            raise ValueError("tau_min must be positive.")
        if self.tau_max is not None and self.tau_max <= tau_min:
            raise ValueError("tau_max must exceed tau_min.")
        self._scale = max(float(np.mean(np.abs(production))), 1e-12)
        q = production / self._scale
        w = injection / self._scale
        lo = np.log(tau_min)
        hi = None if self.tau_max is None else np.log(self.tau_max)
        n_par = n_prod + 2 * n_inj * n_prod
        bounds = [(lo, hi)] * n_prod + [(0.0, 1.0)] * (n_inj * n_prod) + [(lo, hi)] * (n_inj * n_prod)
        lam_offset = n_prod
        A = np.zeros((n_inj, n_par))
        for i in range(n_inj):
            A[i, lam_offset + i*n_prod:lam_offset + (i+1)*n_prod] = 1.0
        constraints = {"type": "ineq", "fun": lambda x: 1.0 - A @ x, "jac": lambda x: -A}
        q0 = q[0]

        def objective(x: np.ndarray) -> float:
            pred = crm_simulate(self._unpack(x, n_inj, n_prod), time, w, q0)
            return float(np.mean((q[1:] - pred) ** 2))

        rng = np.random.default_rng(self.random_state)
        best_x, best_f = None, np.inf
        for start in range(self.n_starts):
            x0 = np.empty(n_par)
            if start == 0:
                initial = np.log(10.0 * tau_min)
                if hi is not None:
                    initial = min(initial, hi)
                x0[:n_prod] = initial
                x0[lam_offset:lam_offset+n_inj*n_prod] = (0.8 / n_prod)
                x0[lam_offset+n_inj*n_prod:] = initial
            else:
                upper_init = max(10.0, time[-1] - time[0]) * tau_min
                if hi is not None:
                    upper_init = min(upper_init, self.tau_max)
                x0[:n_prod] = rng.uniform(lo, np.log(max(tau_min*1.01, upper_init)), n_prod)
                x0[lam_offset:lam_offset+n_inj*n_prod] = (0.9 * rng.dirichlet(np.ones(n_prod), size=n_inj)).ravel()
                x0[lam_offset+n_inj*n_prod:] = rng.uniform(lo, np.log(max(tau_min*1.01, upper_init)), n_inj*n_prod)
            result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                              constraints=constraints, options={"maxiter": self.max_iter, "ftol": 1e-12})
            if np.isfinite(result.fun) and result.fun < best_f:
                best_x, best_f = result.x, float(result.fun)
        if best_x is None:
            raise RuntimeError("CRM calibration failed for every optimisation start.")
        self.params_, self.objective_ = self._unpack(best_x, n_inj, n_prod), best_f
        return self

    def predict(self, time: np.ndarray, injection: np.ndarray, q0: np.ndarray) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("Call fit() before predict().")
        scaled = crm_simulate(self.params_, time, np.asarray(injection, float) / self._scale,
                              np.asarray(q0, float) / self._scale)
        return scaled * self._scale



# 4. SELECTED CRM-ML MODEL



class ExtremeLearningMachine(BaseEstimator, RegressorMixin):
    """Small single-hidden-layer ELM."""

    def __init__(self, n_hidden: int = 10, random_state: Optional[int] = None, ridge: float = 1e-3):
        self.n_hidden, self.random_state, self.ridge = n_hidden, random_state, ridge

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(0.5 * z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtremeLearningMachine":
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1, 1)
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.uniform(-1, 1, (X.shape[1], self.n_hidden))
        self.bias_ = rng.uniform(-1, 1, self.n_hidden)
        H = self._sigmoid(X @ self.weights_ + self.bias_)
        self.beta_ = np.linalg.solve(H.T @ H + self.ridge*np.eye(self.n_hidden), H.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "beta_"):
            raise RuntimeError("Call fit() before predict().")
        H = self._sigmoid(np.asarray(X, float) @ self.weights_ + self.bias_)
        return (H @ self.beta_).ravel()


def make_ml_regressor(model_name: str, seed: int = 0):
    if model_name == "XGB":
        base = XGBRegressor(**ML_SETTINGS["XGB"], random_state=seed)
    elif model_name == "NuSVM":
        base = NuSVR(**ML_SETTINGS["NuSVM"])
    elif model_name == "ELM":
        base = ExtremeLearningMachine(n_hidden=10, random_state=seed, ridge=ML_SETTINGS["ELM"]["ridge"])
    elif model_name == "MLP":
        base = MLPRegressor(hidden_layer_sizes=(PAPER_MLP_HIDDEN_NEURONS,),
                            random_state=seed, **ML_SETTINGS["MLP"])
    else:
        raise ValueError(f"Unknown ML model: {model_name}")
    # Standardise inputs and target in sklearn; the target is log(q), so final output is log(q).
    return TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("model", base)]),
        transformer=StandardScaler(),
    )


def build_crm_ml_features(time: np.ndarray, injection: np.ndarray, params: CRMParameters,
                          distances: np.ndarray, producer: int,
                          crm_response: np.ndarray) -> np.ndarray:
    """Build CRM-ML features with a dynamic CRM response feature.

    The published CRM-ML feature vector contains time, distances, calibrated
    CRM parameters, injection rates and producer response time. In a
    producer-specific model, lambda/tau values are constant across all rows.
    That makes them zero-variance features and prevents posterior CRM draws
    from affecting the fitted ML model.

    The Bayesian extension therefore adds the time-varying CRM response for
    this producer. The raw paper features are retained; the CRM response is an
    additional physics feature whose value changes when lambda/tau are drawn
    from the Bayesian posterior.
    """
    t = np.asarray(time, float).ravel()
    inj = np.asarray(injection, float)
    response = np.asarray(crm_response, float).ravel()
    if inj.ndim != 2 or inj.shape[0] != len(t):
        raise ValueError("Injection and time dimensions are inconsistent in CRM-ML features.")
    if response.size != len(t):
        raise ValueError("CRM response and time dimensions are inconsistent in CRM-ML features.")
    const = np.concatenate([
        distances[:, producer],
        params.lambda_ij[:, producer],
        params.tau_ij[:, producer],
    ])
    return np.column_stack([
        t,
        np.tile(const, (len(t), 1)),
        inj,
        np.full(len(t), params.tau_j[producer]),
        response,
    ])


def fit_positive_crm_ml(model_name: str, X: np.ndarray, y: np.ndarray, seed: int):
    """Train the selected CRM-ML estimator on log(q), guaranteeing positive predictions."""
    y = np.asarray(y, float).ravel()
    if np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("CRM-ML log-target training requires strictly positive production rates.")
    model = make_ml_regressor(model_name, seed)
    model.fit(np.asarray(X, float), np.log(y))
    return model


def positive_prediction(model, X: np.ndarray) -> np.ndarray:
    log_q = np.asarray(model.predict(np.asarray(X, float)), float).ravel()
    if np.any(~np.isfinite(log_q)):
        raise RuntimeError("The CRM-ML model returned non-finite log predictions.")
    q = np.exp(np.clip(log_q, -40.0, 40.0))
    if np.any(~np.isfinite(q)) or np.any(q <= 0):
        raise RuntimeError("The CRM-ML model returned invalid positive predictions.")
    return q



# 5. DEEP BAYESIAN CRM + CRM-ML POSTERIOR PREDICTIVE UQ



@dataclass
class BayesianUQResult:
    model: str
    forecast: pd.DataFrame
    metrics: pd.DataFrame
    baseline_metrics: pd.DataFrame
    posterior_summary: pd.DataFrame
    chain_samples: np.ndarray  # [chains, draws, n_theta + n_producers]
    rhat: pd.DataFrame
    calibration_fraction: float
    mcmc_chains: int
    mcmc_samples: int
    burn_in: int
    thin: int
    seed: int
    acceptance_rates: pd.DataFrame
    posterior_mean_params: CRMParameters
    ml_sigma: np.ndarray
    calibration_predictions: pd.DataFrame
    historical_production: pd.DataFrame


def _require_positive(values: np.ndarray, label: str) -> None:
    values = np.asarray(values, float)
    if values.size == 0:
        raise ValueError(f"{label} is empty.")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{label} contains NaN/inf values.")
    if np.any(values <= 0):
        bad = int(np.sum(values <= 0))
        raise ValueError(
            f"{label} contains {bad} non-positive values. The Lognormal likelihood requires q > 0. "
            "Do not replace zeros silently; explicitly handle shut-in/inactive periods."
        )


def _positive_active_mask(y: np.ndarray, start: int = 0) -> np.ndarray:
    """Return rows eligible for the positive-production Lognormal model.

    Exactly zero production is a valid shut-in/inactive observation. It is
    retained in the original dataset and in point-forecast evaluation, but it
    is excluded from the continuous Lognormal likelihood/log-target because
    log(0) is undefined. Negative production is different: it is physically
    invalid for this production-rate input and therefore raises an error rather
    than being silently classified as inactive.
    """
    y = np.asarray(y, float).ravel()
    if np.any(~np.isfinite(y)):
        raise ValueError("Production contains NaN/inf values.")
    if np.any(y < 0):
        bad = int(np.sum(y < 0))
        raise ValueError(
            f"Production contains {bad} negative values. Negative production rates are not supported; "
            "zero values are valid shut-in/inactive observations and are handled separately."
        )
    mask = np.zeros(y.shape, dtype=bool)
    s = max(0, int(start))
    mask[s:] = y[s:] > 0
    return mask


def _logit_slack_from_lambda(lam: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, float)
    out = np.empty_like(lam)
    for i in range(lam.shape[0]):
        row = np.clip(lam[i], 1e-8, None)
        slack = max(1.0 - float(row.sum()), 1e-8)
        probs = np.r_[row, slack]
        probs /= probs.sum()
        out[i] = np.log(probs[:-1] / probs[-1])
    return out


def _lambda_from_logit_slack(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    aug = np.concatenate([z, np.zeros((z.shape[0], 1))], axis=1)
    aug = aug - np.max(aug, axis=1, keepdims=True)
    w = np.exp(aug)
    w /= w.sum(axis=1, keepdims=True)
    return w[:, :-1]


def _crm_theta_from_params(params: CRMParameters) -> np.ndarray:
    return np.concatenate([
        np.log(np.clip(params.tau_j, 1e-12, None)),
        _logit_slack_from_lambda(params.lambda_ij).ravel(),
        np.log(np.clip(params.tau_ij, 1e-12, None)).ravel(),
    ])


def _crm_params_from_theta(theta: np.ndarray, n_inj: int, n_prod: int) -> CRMParameters:
    p = 0
    tau_j = np.exp(theta[p:p+n_prod]); p += n_prod
    z = theta[p:p+n_inj*n_prod].reshape(n_inj, n_prod); p += n_inj*n_prod
    tau_ij = np.exp(theta[p:p+n_inj*n_prod]).reshape(n_inj, n_prod)
    return CRMParameters(tau_j=tau_j, lambda_ij=_lambda_from_logit_slack(z), tau_ij=tau_ij)


def _split_rhat(samples: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    a = np.asarray(samples, float)
    if a.ndim != 3:
        raise ValueError("R-hat input must have shape [chains, draws, parameters].")
    m, n, p = a.shape
    if m < 2 or n < 4:
        vals = np.full(p, np.nan)
    else:
        half = n // 2
        split = a[:, :2*half].reshape(2*m, half, p)
        means = split.mean(axis=1)
        within = split.var(axis=1, ddof=1).mean(axis=0)
        between = half * means.var(axis=0, ddof=1)
        var_hat = ((half - 1) / half) * within + between / half
        vals = np.sqrt(np.divide(var_hat, within, out=np.full(p, np.nan), where=within > 0))
    return pd.DataFrame({"parameter": list(names), "R_hat": vals})


def _crm_simulate_torch(theta: torch.Tensor, time: torch.Tensor, injection_scaled: torch.Tensor,
                         q0_scaled: torch.Tensor, n_inj: int, n_prod: int) -> torch.Tensor:
    """Differentiable CRM recurrence for HMC."""
    p = 0
    tau_j = torch.exp(theta[p:p+n_prod]); p += n_prod
    z = theta[p:p+n_inj*n_prod].reshape(n_inj, n_prod); p += n_inj*n_prod
    tau_ij = torch.exp(theta[p:p+n_inj*n_prod]).reshape(n_inj, n_prod)
    aug = torch.cat([z, torch.zeros((n_inj, 1), dtype=theta.dtype, device=theta.device)], dim=1)
    weights = torch.softmax(aug, dim=1)[:, :n_prod]
    dt = torch.diff(time)
    t_rel = time[1:] - time[0]
    out = q0_scaled.unsqueeze(0) * torch.exp(-t_rel.unsqueeze(1) / tau_j.unsqueeze(0))
    signal = injection_scaled[1:]
    a = torch.exp(-dt[:, None, None] / tau_ij[None, :, :])
    cp = torch.cumprod(a, dim=0)
    response = cp * torch.cumsum((1.0 - a) * signal[:, :, None] / torch.clamp(cp, min=1e-30), dim=0)
    out = out + torch.sum(response * weights.unsqueeze(0), dim=1)
    return out


def _sample_inverse_gamma(shape: float, scale: float, rng: np.random.Generator) -> float:
    return 1.0 / float(rng.gamma(shape, 1.0 / max(scale, 1e-12)))


def _hmc_chain(time: np.ndarray, injection: np.ndarray, production: np.ndarray,
               production_scale: float, tau_min: float, initial_params: CRMParameters,
               active_mask: np.ndarray, samples: int, burn_in: int, thin: int, seed: int,
               prior_sd: float = UQ_WEAK_PRIOR_SD,
               step_size: float = 0.05, leapfrog_steps: int = 5) -> Tuple[np.ndarray, float]:
    """HMC for nonlinear CRM parameters with weak priors in transformed space.

    The deterministic SLSQP estimate is used only as an initial state, not as the
    center of the Bayesian prior. Priors are deliberately broad so the posterior
    can move away from the deterministic CRM when supported by the likelihood.

    The CRM residual variance is sampled for CRM diagnostics, but it is NOT used in
    the final CRM-XGB predictive distribution. Final aleatoric uncertainty is
    estimated separately from chronological out-of-sample CRM-ML residuals.
    """
    if samples < 100 or burn_in < 100 or thin < 1:
        raise ValueError("Use at least 100 retained samples, 100 burn-in iterations and thin >= 1.")
    if leapfrog_steps < 1 or step_size <= 0:
        raise ValueError("HMC step size must be positive and leapfrog steps must be >= 1.")

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    dtype = torch.float64
    t = torch.as_tensor(time, dtype=dtype)
    w = torch.as_tensor(injection / production_scale, dtype=dtype)
    q0 = torch.as_tensor(production[0] / production_scale, dtype=dtype)
    active_mask = np.asarray(active_mask, bool)
    if active_mask.shape != production.shape:
        raise ValueError("active_mask must have the same shape as production.")
    y_np = production[1:]
    mask_np = active_mask[1:]
    y = torch.as_tensor(y_np, dtype=dtype)
    mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=dtype.device if hasattr(dtype, "device") else None)
    n_inj, n_prod = injection.shape[1], production.shape[1]
    n_obs_by_prod = np.sum(mask_np, axis=0).astype(int)
    if np.any(n_obs_by_prod < 2):
        raise ValueError("Each producer needs at least two positive historical observations for Bayesian CRM likelihood.")
    theta0 = _crm_theta_from_params(initial_params)
    theta_np = theta0 + rng.normal(0.0, 0.20, theta0.shape)
    tau_idx = list(range(n_prod)) + list(range(n_prod+n_inj*n_prod, theta_np.size))
    theta_np[tau_idx] = np.maximum(theta_np[tau_idx], np.log(tau_min) + 1e-6)
    theta = torch.as_tensor(theta_np, dtype=dtype)
    n_theta = theta.numel()
    a0 = 2.0
    b0 = a0 * (0.20 ** 2)

    def logpost(th: torch.Tensor) -> torch.Tensor:
        tau_j = torch.exp(th[:n_prod])
        tau_ij = torch.exp(th[n_prod+n_inj*n_prod:])
        if torch.any(tau_j < tau_min) or torch.any(tau_ij < tau_min):
            return torch.tensor(-float("inf"), dtype=dtype)
        qhat = _crm_simulate_torch(th, t, w, q0, n_inj, n_prod) * production_scale
        if torch.any(~torch.isfinite(qhat)):
            return torch.tensor(-float("inf"), dtype=dtype)
        # Lognormal likelihood is evaluated only at positive/active observations.
        # Interior shut-in periods remain in the dataset but do not enter log(q).
        residual_terms = []
        marginal_ll = torch.tensor(0.0, dtype=dtype)
        for j in range(n_prod):
            mj = mask_t[:, j]
            qj = qhat[:, j][mj]
            yj = y[:, j][mj]
            if torch.any(qj <= 0) or torch.any(yj <= 0):
                return torch.tensor(-float("inf"), dtype=dtype)
            resid_j = torch.log(yj) - torch.log(qj)
            sse_j = torch.sum(resid_j * resid_j)
            marginal_ll = marginal_ll - (a0 + 0.5 * float(n_obs_by_prod[j])) * torch.log(b0 + 0.5 * sse_j)
            residual_terms.append(resid_j)

        # Weak priors in the unconstrained parameterisation. The deterministic
        # SLSQP estimate is an initial state only; it is not a prior centre.
        tau_log = torch.log(torch.clamp(tau_j / tau_min, min=1e-30))
        tau_ij_log = torch.log(torch.clamp(tau_ij / tau_min, min=1e-30))
        z = th[n_prod:n_prod+n_inj*n_prod]
        prior = -0.5 * torch.sum((tau_log / prior_sd) ** 2)
        prior = prior - 0.5 * torch.sum((z / prior_sd) ** 2)
        prior = prior - 0.5 * torch.sum((tau_ij_log / prior_sd) ** 2)
        return marginal_ll + prior

    if not torch.isfinite(logpost(theta)):
        raise RuntimeError("Initial Bayesian CRM state has invalid posterior density.")

    total = burn_in + samples * thin
    draws = torch.empty((samples, n_theta + n_prod), dtype=dtype)
    accepted = 0
    recent = []
    saved = 0
    eps = float(step_size)

    for it in range(total):
        q = theta.detach().clone().requires_grad_(True)
        momentum = torch.randn_like(q)
        current_p = momentum.detach().clone()
        lp = logpost(q)
        grad = torch.autograd.grad(lp, q)[0]
        p_m = momentum + 0.5 * eps * grad
        q_new = q
        valid = True
        for lf in range(leapfrog_steps):
            q_new = (q_new + eps * p_m).detach().requires_grad_(True)
            lp_new = logpost(q_new)
            if not torch.isfinite(lp_new):
                valid = False
                break
            grad_new = torch.autograd.grad(lp_new, q_new)[0]
            if lf < leapfrog_steps - 1:
                p_m = p_m + eps * grad_new
        accepted_this = False
        if valid:
            p_m = -(p_m + 0.5 * eps * grad_new)
            current_H = -lp.detach() + 0.5 * torch.sum(current_p**2)
            proposed_H = -lp_new.detach() + 0.5 * torch.sum(p_m**2)
            log_acc = torch.clamp(current_H - proposed_H, max=0.0)
            if torch.log(torch.rand((), dtype=dtype)) < log_acc:
                theta = q_new.detach()
                accepted += 1
                accepted_this = True
        recent.append(1.0 if accepted_this else 0.0)

        if it < burn_in and (it + 1) % 50 == 0:
            rate = float(np.mean(recent[-50:]))
            if rate < 0.20:
                eps *= 0.70
            elif rate < 0.35:
                eps *= 0.85
            elif rate > 0.70:
                eps *= 1.40
            elif rate > 0.55:
                eps *= 1.20
            eps = float(np.clip(eps, 0.005, 0.50))

        if it >= burn_in and (it - burn_in) % thin == 0:
            with torch.no_grad():
                qhat = _crm_simulate_torch(theta, t, w, q0, n_inj, n_prod) * production_scale
                sig = []
                for j in range(n_prod):
                    mj = mask_t[:, j]
                    resid_j = torch.log(y[:, j][mj]) - torch.log(qhat[:, j][mj])
                    shape = a0 + 0.5 * float(n_obs_by_prod[j])
                    scale = b0 + 0.5 * float(torch.sum(resid_j**2))
                    sig.append(np.sqrt(_sample_inverse_gamma(shape, scale, rng)))
                draws[saved, :n_theta] = theta
                draws[saved, n_theta:] = torch.as_tensor(sig, dtype=dtype)
            saved += 1

    return draws.numpy(), accepted / max(total, 1)


def _posterior_mean_crm_params(chain: np.ndarray, n_inj: int, n_prod: int) -> CRMParameters:
    flat = chain.reshape(-1, chain.shape[-1])
    n_theta = flat.shape[1] - n_prod
    theta = flat[:, :n_theta]
    tau_j = np.exp(theta[:, :n_prod]).mean(axis=0)
    z = theta[:, n_prod:n_prod+n_inj*n_prod].reshape(flat.shape[0], n_inj, n_prod)
    if n_inj == 0:
        lam = np.empty((0, n_prod), dtype=float)
        tau_ij = np.empty((0, n_prod), dtype=float)
    else:
        lam = np.stack([_lambda_from_logit_slack(v) for v in z], axis=0).mean(axis=0)
        tau_ij = np.exp(theta[:, n_prod+n_inj*n_prod:]).reshape(flat.shape[0], n_inj, n_prod).mean(axis=0)
    return CRMParameters(tau_j=tau_j, lambda_ij=lam, tau_ij=tau_ij)


def _crm_posterior_summary(chain: np.ndarray, n_inj: int, n_prod: int,
                           producer_names: Sequence[str], injector_names: Sequence[str]) -> pd.DataFrame:
    flat = chain.reshape(-1, chain.shape[-1])
    n_theta = flat.shape[1] - n_prod
    theta = flat[:, :n_theta]
    sigma = flat[:, n_theta:]
    tau_j = np.exp(theta[:, :n_prod])
    z = theta[:, n_prod:n_prod+n_inj*n_prod].reshape(flat.shape[0], n_inj, n_prod)
    if n_inj == 0:
        lam = np.empty((flat.shape[0], 0, n_prod), dtype=float)
        tau_ij = np.empty((flat.shape[0], 0, n_prod), dtype=float)
    else:
        lam = np.stack([_lambda_from_logit_slack(v) for v in z], axis=0)
        tau_ij = np.exp(theta[:, n_prod+n_inj*n_prod:]).reshape(flat.shape[0], n_inj, n_prod)
    rows = []
    def add(name, vals):
        rows.append({"parameter": name, "mean": float(np.mean(vals)), "median": float(np.median(vals)),
                     "sd": float(np.std(vals, ddof=1)), "q2.5": float(np.quantile(vals,.025)),
                     "q97.5": float(np.quantile(vals,.975))})
    for j, pname in enumerate(producer_names):
        add(f"{pname}::tau_j", tau_j[:, j])
    for i, inj in enumerate(injector_names):
        for j, pname in enumerate(producer_names):
            add(f"{inj}->{pname}::lambda", lam[:, i, j])
            add(f"{inj}->{pname}::tau_ij", tau_ij[:, i, j])
    for j, pname in enumerate(producer_names):
        add(f"{pname}::sigma", sigma[:, j])
    return pd.DataFrame(rows)


def _theta_names(n_inj: int, n_prod: int, producer_names: Sequence[str],
                 injector_names: Sequence[str]) -> List[str]:
    names = [f"{p}::log_tau_j" for p in producer_names]
    names += [f"{inj}->{prod}::lambda_logit" for inj in injector_names for prod in producer_names]
    names += [f"{inj}->{prod}::log_tau_ij" for inj in injector_names for prod in producer_names]
    names += [f"{p}::sigma" for p in producer_names]
    return names


def _fit_final_ml_models(model_name: str, time_rows: np.ndarray, inj_rows: np.ndarray,
                         prod_rows: np.ndarray, distances: np.ndarray, params: CRMParameters,
                         crm_response_rows: np.ndarray, active_starts: np.ndarray, seed: int):
    models = []
    for j in range(prod_rows.shape[1]):
        s = int(active_starts[j])
        X = build_crm_ml_features(time_rows, inj_rows, params, distances, j, crm_response_rows[:, j])
        y = prod_rows[:, j]
        mask = _positive_active_mask(y, s)
        if int(mask.sum()) < 8:
            raise ValueError(f"Producer {j+1} has only {int(mask.sum())} positive historical observations; at least 8 are required for CRM-ML fitting.")
        models.append(fit_positive_crm_ml(model_name, X[mask], y[mask], seed + j))
    return models


def _predict_posterior_ml(models, forecast_time: np.ndarray, forecast_injection: np.ndarray,
                          distances: np.ndarray, crm_draws: Sequence[CRMParameters],
                          full_time: np.ndarray, full_injection: np.ndarray, q0: np.ndarray,
                          forecast_start: int) -> np.ndarray:
    n_draws = len(crm_draws)
    n_steps = len(forecast_time)
    n_prod = distances.shape[1]
    out = np.empty((n_draws, n_steps, n_prod), dtype=float)
    for d, params in enumerate(crm_draws):
        # Simulate the full history + forecast so the CRM state at the forecast
        # boundary is conditioned on the observed history. crm_simulate returns
        # rows corresponding to full_time[1:], hence forecast_start - 1.
        response_all = crm_simulate(params, full_time, full_injection, q0)
        response_forecast = response_all[forecast_start - 1:]
        for j in range(n_prod):
            X = build_crm_ml_features(
                forecast_time, forecast_injection, params, distances, j,
                response_forecast[:, j]
            )
            out[d, :, j] = positive_prediction(models[j], X)
    return out


def _fit_oos_ml_calibration(models_name: str, time_rows: np.ndarray,
                             injection_rows: np.ndarray, production_rows: np.ndarray,
                             distances: np.ndarray, crm_params: CRMParameters,
                             crm_response_rows: np.ndarray, active_starts: np.ndarray, seed: int,
                             producer_names: Sequence[str],
                             fraction: float = UQ_CALIBRATION_FRACTION
                             ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Estimate ML aleatoric scale from chronological out-of-sample residuals."""
    records = []
    sigma = []
    for j, pname in enumerate(producer_names):
        s = int(active_starts[j])
        y = production_rows[:, j]
        X = build_crm_ml_features(time_rows, injection_rows, crm_params, distances, j, crm_response_rows[:, j])
        mask = _positive_active_mask(y, s)
        active_idx = np.flatnonzero(mask)
        active_n = len(active_idx)
        if active_n < 16:
            raise ValueError(
                f"Producer {j+1} needs at least 16 positive historical rows for "
                f"out-of-sample UQ calibration; got {active_n}. "
                "Zero/shut-in rows are excluded from the Lognormal calibration."
            )
        n_train = max(8, int(np.floor(fraction * active_n)))
        if active_n - n_train < 8:
            n_train = active_n - 8
        train_idx = active_idx[:n_train]
        cal_idx = active_idx[n_train:]
        cal_model = fit_positive_crm_ml(models_name, X[train_idx], y[train_idx], seed + 10_000 + j)
        pred = positive_prediction(cal_model, X[cal_idx])
        obs = y[cal_idx]
        log_resid = np.log(obs) - np.log(pred)
        sig_j = float(np.std(log_resid, ddof=1))
        if not np.isfinite(sig_j) or sig_j <= 0:
            raise ValueError(
                f"Producer {j+1} produced a non-positive/non-finite ML residual sigma. "
                "The historical calibration block is insufficient for UQ."
            )
        sigma.append(sig_j)
        for idx, observed, predicted, lr in zip(cal_idx, obs, pred, log_resid):
            records.append({
                "producer": pname,
                "historical_row": int(idx),
                "observed": float(observed),
                "crm_ml_prediction": float(predicted),
                "log_residual": float(lr),
                "calibration_role": "chronological_out_of_sample",
            })
    return pd.DataFrame(records), np.asarray(sigma, dtype=float)


def _deterministic_forecast_metrics(observed: np.ndarray, predicted: np.ndarray,
                                    producer_names: Sequence[str], model: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "producer": pname, "model": model,
        "MAE_P50": mean_absolute_error(observed[:, j], predicted[:, j]),
        "RMSE_P50": root_mean_squared_error(observed[:, j], predicted[:, j]),
        "R2_P50": r_squared(observed[:, j], predicted[:, j]),
    } for j, pname in enumerate(producer_names)])



def _run_precomputed_crm_uq(data: FieldData, n_forecast: int, model: str,
                            mcmc_chains: int, mcmc_samples: int, burn_in: int, thin: int,
                            seed: int, verbose: bool) -> BayesianUQResult:
    """UQ path for workbooks that already contain a q_CRM sheet.

    This is deliberately separate from full CRM calibration. A supplied q_CRM
    series is treated as the deterministic reduced-physics baseline; the code
    must not fabricate injector/connectivity parameters when the workbook does
    not contain injection and distance data. CRM-XGB is retained by training
    the selected ML model on [time, q_CRM], then Bayesian calibration is applied
    to the positive out-of-sample CRM-XGB response in log space.
    """
    if data.crm_baseline is None:
        raise ValueError("Pre-computed CRM UQ requires data.crm_baseline.")
    n_hist = len(data.time) - int(n_forecast)
    if n_hist < 20:
        raise ValueError("At least 20 historical rows are required before the forecast window.")
    qcrm = np.asarray(data.crm_baseline, float)
    if np.any(~np.isfinite(qcrm)) or np.any(qcrm < 0):
        raise ValueError("q_CRM contains invalid negative, NaN, or infinite values.")
    hist_t, hist_qcrm, hist_y = data.time[:n_hist], qcrm[:n_hist], data.production[:n_hist]
    active = np.zeros(hist_y.shape, dtype=bool)
    for j in range(hist_y.shape[1]):
        active[:, j] = _positive_active_mask(hist_y[:, j]) & (hist_qcrm[:, j] > 0)
        if active[:, j].sum() < 16:
            raise ValueError(f"Producer {j+1} needs at least 16 positive observed/q_CRM historical rows.")
    if verbose:
        print("CRM mode: pre-computed q_CRM baseline from workbook.")
        print(f"Historical positive observations used with positive q_CRM: {int(active.sum())}")
        print(f"Historical zero-production/shut-in observations retained: {int((hist_y == 0).sum())}")

    # CRM-XGB in pre-computed mode: q_CRM is the supplied reduced-physics feature.
    def make_X(t, qb):
        return np.column_stack([np.asarray(t, float), np.asarray(qb, float)])

    ml_models = []
    for j in range(hist_y.shape[1]):
        ml_models.append(fit_positive_crm_ml(model.split("-",1)[1], make_X(hist_t, hist_qcrm[:,j])[active[:,j]], hist_y[:,j][active[:,j]], seed+j))

    forecast_t = data.time[n_hist:]
    forecast_qcrm = qcrm[n_hist:]
    y_fore = data.production[n_hist:]
    det_pred = np.column_stack([positive_prediction(ml_models[j], make_X(forecast_t, forecast_qcrm[:,j])) for j in range(hist_y.shape[1])])

    # Chronological OOS residual scale.
    sigmas=[]; cal_records=[]
    for j,pname in enumerate(data.producer_names):
        idx=np.flatnonzero(active[:,j]); ntrain=max(8,int(np.floor(UQ_CALIBRATION_FRACTION*len(idx))))
        if len(idx)-ntrain < 8: ntrain=len(idx)-8
        tr,ca=idx[:ntrain],idx[ntrain:]
        cm=fit_positive_crm_ml(model.split("-",1)[1], make_X(hist_t, hist_qcrm[:,j])[tr], hist_y[:,j][tr], seed+10000+j)
        pred=positive_prediction(cm, make_X(hist_t, hist_qcrm[:,j])[ca])
        lr=np.log(hist_y[:,j][ca])-np.log(pred); sigma=float(np.std(lr,ddof=1))
        if not np.isfinite(sigma) or sigma<=0: raise ValueError(f"Producer {j+1} has invalid OOS residual sigma.")
        sigmas.append(sigma)
        for ii,yy,pp,rr in zip(ca,hist_y[:,j][ca],pred,lr):
            cal_records.append({"producer":pname,"historical_row":int(ii),"observed":float(yy),"crm_ml_prediction":float(pp),"log_residual":float(rr),"calibration_role":"chronological_out_of_sample"})
    ml_sigma=np.asarray(sigmas)

    # Proper HMC posterior for log-scale calibration alpha, beta, log_sigma.
    # The supplied q_CRM is fixed; uncertainty is placed on the Bayesian
    # calibration of the CRM-XGB response, not on unavailable CRM parameters.
    x_by_prod = []
    y_by_prod = []
    for j in range(hist_y.shape[1]):
        m = active[:, j]
        pred_hist = positive_prediction(ml_models[j], make_X(hist_t, hist_qcrm[:, j]))
        x_by_prod.append(np.log(np.maximum(pred_hist[m], 1e-30)))
        y_by_prod.append(np.log(hist_y[:, j][m]))

    chains=[]; rates=[]
    prior_log_sigma = math.log(max(float(np.mean(ml_sigma)), 1e-3))
    for c in range(mcmc_chains):
        torch.manual_seed(seed + 100000*c)
        theta = torch.tensor([0.0, 1.0, prior_log_sigma], dtype=torch.float64, requires_grad=True)
        rng = np.random.default_rng(seed + 100000*c)
        total = burn_in + mcmc_samples * thin
        accepted = 0
        rows=[]
        eps = 0.001

        def logpost(v):
            alpha, beta, log_sigma = v[0], v[1], v[2]
            sigma = torch.exp(log_sigma)
            ll = torch.tensor(0.0, dtype=torch.float64)
            for xx, yy in zip(x_by_prod, y_by_prod):
                xt = torch.as_tensor(xx, dtype=torch.float64)
                yt = torch.as_tensor(yy, dtype=torch.float64)
                resid = yt - (alpha + beta * xt)
                ll = ll - 0.5 * torch.sum((resid / sigma) ** 2) - yt.numel() * log_sigma
            prior = -0.5 * alpha**2 - 0.5 * ((beta - 1.0) / 0.5)**2
            prior = prior - 0.5 * ((log_sigma - prior_log_sigma) / 2.0)**2
            return ll + prior

        for it in range(total):
            q = theta.detach().clone().requires_grad_(True)
            p0 = torch.randn_like(q)
            current_lp = logpost(q)
            current_grad = torch.autograd.grad(current_lp, q)[0]
            p = p0 + 0.5 * eps * current_grad
            q_new = q
            valid = True
            for lf in range(5):
                q_new = (q_new + eps * p).detach().requires_grad_(True)
                lp_new = logpost(q_new)
                if not torch.isfinite(lp_new):
                    valid = False
                    break
                grad_new = torch.autograd.grad(lp_new, q_new)[0]
                if lf < 4:
                    p = p + eps * grad_new
            if valid:
                p = p + 0.5 * eps * grad_new
                p = -p
                proposed_H = -float(lp_new.detach()) + 0.5 * float(torch.sum(p*p))
                current_H = -float(current_lp.detach()) + 0.5 * float(torch.sum(p0*p0))
                log_accept = min(0.0, current_H - proposed_H)
                if math.log(rng.random()) < log_accept:
                    theta = q_new.detach().requires_grad_(True)
                    accepted += 1
                else:
                    theta = q.detach().requires_grad_(True)
            else:
                theta = q.detach().requires_grad_(True)
            if it >= burn_in and ((it-burn_in) % thin == 0):
                rows.append(theta.detach().numpy().copy())
        chains.append(np.asarray(rows)); rates.append(accepted / total)
    chain=np.stack(chains); flat=chain.reshape(-1,3); mean_theta=np.mean(flat,axis=0)
    alpha,beta,logsig=mean_theta
    rng=np.random.default_rng(seed+10000000)
    pred_draws=[]
    for d in range(min(UQ_MAX_CRM_PREDICTIVE_DRAWS,len(flat))):
        a,b,ls=flat[d]; noise=rng.normal(size=det_pred.shape)*np.exp(ls)
        pred_draws.append(np.exp(a+b*np.log(np.maximum(det_pred,1e-30))+noise))
    predictive=np.stack(pred_draws)
    q10,q50,q90=np.quantile(predictive,[.10,.50,.90],axis=0); lo,hi=np.quantile(predictive,[.025,.975],axis=0)
    rows=[]; metrics=[]
    for j,pname in enumerate(data.producer_names):
        y=y_fore[:,j]; covered=(y>=lo[:,j])&(y<=hi[:,j]); score=np.mean((hi[:,j]-lo[:,j])+(2/.05)*(lo[:,j]-y)*(y<lo[:,j])+(2/.05)*(y-hi[:,j])*(y>hi[:,j]))
        for k,tv in enumerate(forecast_t): rows.append({"time":float(tv),"producer":pname,"observed":float(y[k]),"crm_ml_deterministic":float(det_pred[k,j]),"crm_ml_posterior_mean":float(q50[k,j]),"P90_low":float(q10[k,j]),"P50":float(q50[k,j]),"P10_high":float(q90[k,j]),"95pct_lower":float(lo[k,j]),"95pct_upper":float(hi[k,j]),"deterministic_error":float(det_pred[k,j]-y[k]),"deterministic_abs_error":float(abs(det_pred[k,j]-y[k])),"bayesian_error":float(q50[k,j]-y[k]),"bayesian_abs_error":float(abs(q50[k,j]-y[k])),"interval_width_95":float(hi[k,j]-lo[k,j]),"covered_95":bool(covered[k])})
        metrics.append({"producer":pname,"model":model,"MAE_P50":mean_absolute_error(y,q50[:,j]),"RMSE_P50":root_mean_squared_error(y,q50[:,j]),"R2_P50":r_squared(y,q50[:,j]),"PICP_95":float(np.mean(covered)),"MPIW_95":float(np.mean(hi[:,j]-lo[:,j])),"interval_score_95":float(score),"MAE_improvement_vs_deterministic":mean_absolute_error(y,det_pred[:,j])-mean_absolute_error(y,q50[:,j])})
    posterior_summary=pd.DataFrame([{"parameter":"alpha","mean":float(np.mean(flat[:,0])),"median":float(np.median(flat[:,0])),"sd":float(np.std(flat[:,0],ddof=1)),"q2.5":float(np.quantile(flat[:,0],.025)),"q97.5":float(np.quantile(flat[:,0],.975))},{"parameter":"beta","mean":float(np.mean(flat[:,1])),"median":float(np.median(flat[:,1])),"sd":float(np.std(flat[:,1],ddof=1)),"q2.5":float(np.quantile(flat[:,1],.025)),"q97.5":float(np.quantile(flat[:,1],.975))},{"parameter":"log_sigma","mean":float(np.mean(flat[:,2])),"median":float(np.median(flat[:,2])),"sd":float(np.std(flat[:,2],ddof=1)),"q2.5":float(np.quantile(flat[:,2],.025)),"q97.5":float(np.quantile(flat[:,2],.975))}])
    # Pseudo CRMParameters are intentionally empty: no injector parameters are claimed.
    pseudo=CRMParameters(np.ones(hist_y.shape[1]),np.empty((0,hist_y.shape[1])),np.empty((0,hist_y.shape[1])))
    return BayesianUQResult(model=model+" [precomputed CRM]",forecast=pd.DataFrame(rows),metrics=pd.DataFrame(metrics),baseline_metrics=_deterministic_forecast_metrics(y_fore,det_pred,data.producer_names,model+" deterministic"),posterior_summary=posterior_summary,chain_samples=chain,rhat=_split_rhat(chain,["alpha","beta","log_sigma"]),calibration_fraction=UQ_CALIBRATION_FRACTION,mcmc_chains=mcmc_chains,mcmc_samples=mcmc_samples,burn_in=burn_in,thin=thin,seed=seed,acceptance_rates=pd.DataFrame({"chain":np.arange(1,mcmc_chains+1),"HMC_acceptance":rates}),posterior_mean_params=pseudo,ml_sigma=ml_sigma,calibration_predictions=pd.DataFrame(cal_records),historical_production=pd.DataFrame({"time":data.time,**{n:data.production[:,j] for j,n in enumerate(data.producer_names)}}))

def run_bayesian_uq(data: FieldData, n_forecast: int, model: str = DEFAULT_CRM_ML_MODEL,
                    mcmc_chains: int = MCMC_DEFAULT_CHAINS,
                    mcmc_samples: int = MCMC_DEFAULT_SAMPLES,
                    burn_in: int = MCMC_DEFAULT_BURN_IN,
                    thin: int = MCMC_DEFAULT_THIN,
                    seed: int = MCMC_DEFAULT_SEED,
                    crm_starts: int = 3, crm_max_iter: int = 300,
                    trim_leading_inactive: bool = False, verbose: bool = True,
                    hmc_step_size: float = 0.05, hmc_leapfrog_steps: int = 5) -> BayesianUQResult:
    """Run deterministic CRM -> Bayesian CRM -> conditional CRM-ML -> UQ."""
    if data.crm_baseline is not None:
        return _run_precomputed_crm_uq(data, n_forecast, model, mcmc_chains, mcmc_samples, burn_in, thin, seed, verbose)
    if model not in CRM_ML_MODELS:
        raise ValueError(f"model must be one of {CRM_ML_MODELS}.")
    if not (0 < n_forecast < len(data.time)):
        raise ValueError("n_forecast must be between 1 and len(time)-1.")
    if mcmc_chains < 2 or mcmc_samples < 100 or burn_in < 100 or thin < 1:
        raise ValueError("MCMC requires >=2 chains, >=100 samples, >=100 burn-in and thin>=1.")

    n_hist = len(data.time) - int(n_forecast)
    if n_hist < 20:
        raise ValueError("At least 20 historical rows are required before the forecast window.")
    hist_time, hist_inj, hist_prod = data.time[:n_hist], data.injection[:n_hist], data.production[:n_hist]
    # Shut-in/inactive production is valid field data. It is retained for CRM
    # calibration and reporting, but excluded from the Lognormal likelihood.
    active_mask = np.zeros(hist_prod.shape, dtype=bool)
    for j in range(hist_prod.shape[1]):
        active_mask[:, j] = _positive_active_mask(hist_prod[:, j], 0)
        if int(active_mask[:, j].sum()) < 2:
            raise ValueError(f"Producer {j+1} has fewer than 2 positive historical observations; Bayesian Lognormal calibration is not identifiable.")
    zero_count = int((hist_prod == 0).sum())
    inactive_count = int((~active_mask).sum())
    if verbose:
        print(f"Historical positive production observations used in Lognormal likelihood: {int(active_mask.sum())}")
        print(f"Historical zero-production/shut-in observations retained but excluded from Lognormal likelihood: {zero_count}")

    active = data.first_active_index().astype(int)
    starts_full = active.copy() if trim_leading_inactive else np.zeros(data.production.shape[1], dtype=int)

    crm = CapacitanceResistanceModel(n_starts=crm_starts, max_iter=crm_max_iter, random_state=seed)
    crm.fit(hist_time, hist_inj, hist_prod)
    production_scale = float(crm._scale)
    tau_min = float(np.min(np.diff(hist_time)))
    if verbose:
        print(f"Deterministic CRM calibrated: objective={crm.objective_:.6g}")
        print(f"Bayesian CRM minimum tau constraint: {tau_min:g}")

    chains, rates = [], []
    for c in range(mcmc_chains):
        draws, rate = _hmc_chain(
            hist_time, hist_inj, hist_prod, production_scale, tau_min, crm.params_,
            active_mask, mcmc_samples, burn_in, thin, seed + 100_000*c,
            step_size=hmc_step_size, leapfrog_steps=hmc_leapfrog_steps
        )
        chains.append(draws)
        rates.append(rate)
        if verbose:
            print(f"Bayesian CRM HMC chain {c+1}/{mcmc_chains}: acceptance={rate:.2%}")
    chain = np.stack(chains, axis=0)

    n_prod = data.production.shape[1]
    posterior_mean_params = _posterior_mean_crm_params(chain, data.injection.shape[1], n_prod)
    model_name = model.split("-", 1)[1]
    time_rows, inj_rows, prod_rows = data.time[1:n_hist], data.injection[1:n_hist], data.production[1:n_hist]
    starts_rows = np.maximum(starts_full - 1, 0)

    # Build the dynamic CRM response for each parameter set. This is the key
    # Bayesian-extension feature: unlike the raw lambda/tau columns, q_CRM(t)
    # varies with time and changes when a posterior CRM draw changes.
    q0 = np.asarray(data.production[0], float)
    crm_hist_mean = crm_simulate(
        posterior_mean_params, data.time[:n_hist], data.injection[:n_hist], q0
    )
    crm_hist_det = crm_simulate(
        crm.params_, data.time[:n_hist], data.injection[:n_hist], q0
    )
    crm_response_rows_mean = crm_hist_mean
    crm_response_rows_det = crm_hist_det

    # Conditional ML fit uses posterior-mean CRM parameters, but now receives
    # the corresponding dynamic CRM response as a feature.
    bayes_models = _fit_final_ml_models(model_name, time_rows, inj_rows, prod_rows,
                                        data.distances, posterior_mean_params,
                                        crm_response_rows_mean, starts_rows, seed)
    baseline_models = _fit_final_ml_models(model_name, time_rows, inj_rows, prod_rows,
                                           data.distances, crm.params_,
                                           crm_response_rows_det, starts_rows, seed + 500_000)

    forecast_time = data.time[n_hist:]
    forecast_inj = data.injection[n_hist:]
    observed_forecast = data.production[n_hist:]
    crm_full_mean = crm_simulate(
        posterior_mean_params, data.time, data.injection, q0
    )
    crm_full_det = crm_simulate(
        crm.params_, data.time, data.injection, q0
    )
    crm_forecast_mean = crm_full_mean[n_hist - 1:]
    crm_forecast_det = crm_full_det[n_hist - 1:]
    baseline_pred = np.empty((n_forecast, n_prod))
    mean_pred = np.empty_like(baseline_pred)
    for j in range(n_prod):
        Xb = build_crm_ml_features(
            forecast_time, forecast_inj, crm.params_, data.distances, j,
            crm_forecast_det[:, j]
        )
        Xm = build_crm_ml_features(
            forecast_time, forecast_inj, posterior_mean_params, data.distances, j,
            crm_forecast_mean[:, j]
        )
        baseline_pred[:, j] = positive_prediction(baseline_models[j], Xb)
        mean_pred[:, j] = positive_prediction(bayes_models[j], Xm)

    baseline_metrics = _deterministic_forecast_metrics(observed_forecast, baseline_pred,
                                                       data.producer_names, f"{model} deterministic CRM")

    # Correct uncertainty decomposition:
    #   epistemic  = posterior uncertainty in CRM lambda/tau propagated through ML
    #   aleatoric = out-of-sample residual scale of the selected CRM-ML model
    # The CRM-only residual sigma from the HMC block is deliberately NOT reused.
    calibration_predictions, ml_sigma = _fit_oos_ml_calibration(
        model_name, time_rows, inj_rows, prod_rows, data.distances,
        posterior_mean_params, crm_response_rows_mean, starts_rows, seed + 700_000,
        producer_names=data.producer_names,
        fraction=UQ_CALIBRATION_FRACTION,
    )

    flat = chain.reshape(-1, chain.shape[-1])
    if len(flat) > UQ_MAX_CRM_PREDICTIVE_DRAWS:
        rng = np.random.default_rng(seed + 77)
        keep = np.sort(rng.choice(len(flat), size=UQ_MAX_CRM_PREDICTIVE_DRAWS, replace=False))
        flat = flat[keep]
    n_theta = flat.shape[1] - n_prod
    crm_draws = [_crm_params_from_theta(row[:n_theta], data.injection.shape[1], n_prod) for row in flat]
    ml_draws = _predict_posterior_ml(
        bayes_models, forecast_time, forecast_inj, data.distances, crm_draws,
        data.time, data.injection, q0, n_hist
    )

    # Use the selected CRM-ML model's historical out-of-sample residual scale.
    rng = np.random.default_rng(seed + 10_000_000)
    predictive = ml_draws * np.exp(
        rng.normal(size=ml_draws.shape) * ml_sigma[None, None, :]
    )
    if np.any(~np.isfinite(predictive)) or np.any(predictive <= 0):
        raise RuntimeError("Posterior predictive simulation produced invalid values.")

    q10, q50, q90 = [np.quantile(predictive, p, axis=0) for p in (0.10, 0.50, 0.90)]
    lower, upper = np.quantile(predictive, [0.025, 0.975], axis=0)
    rows, metric_rows = [], []
    for j, pname in enumerate(data.producer_names):
        y = observed_forecast[:, j]
        lo, med, hi = lower[:, j], q50[:, j], upper[:, j]
        covered = (y >= lo) & (y <= hi)
        alpha = 0.05
        interval_score = np.mean((hi-lo) + (2/alpha)*(lo-y)*(y < lo) + (2/alpha)*(y-hi)*(y > hi))
        for k, tval in enumerate(forecast_time):
            det = float(baseline_pred[k, j])
            bayes = float(med[k])
            rows.append({
                "time": float(tval), "producer": pname, "observed": float(y[k]),
                "crm_ml_deterministic": det,
                "crm_ml_posterior_mean": float(mean_pred[k,j]),
                "P90_low": float(q10[k,j]), "P50": bayes, "P10_high": float(q90[k,j]),
                "95pct_lower": float(lo[k]), "95pct_upper": float(hi[k]),
                "deterministic_error": det - float(y[k]),
                "deterministic_abs_error": abs(det - float(y[k])),
                "bayesian_error": bayes - float(y[k]),
                "bayesian_abs_error": abs(bayes - float(y[k])),
                "interval_width_95": float(hi[k] - lo[k]),
                "covered_95": bool(covered[k]),
            })
        metric_rows.append({
            "producer": pname, "model": model,
            "MAE_P50": mean_absolute_error(y, med),
            "RMSE_P50": root_mean_squared_error(y, med),
            "R2_P50": r_squared(y, med),
            "PICP_95": float(np.mean(covered)),
            "MPIW_95": float(np.mean(hi-lo)),
            "interval_score_95": float(interval_score),
            "MAE_improvement_vs_deterministic": mean_absolute_error(y, baseline_pred[:,j]) - mean_absolute_error(y, med),
        })

    names = _theta_names(data.injection.shape[1], n_prod, data.producer_names, data.injector_names)
    rhat = _split_rhat(chain, names)
    posterior_summary = _crm_posterior_summary(chain, data.injection.shape[1], n_prod,
                                               data.producer_names, data.injector_names)
    ml_sigma_rows = pd.DataFrame({
        "parameter": [f"{p}::sigma_ML_oos" for p in data.producer_names],
        "mean": ml_sigma,
        "median": ml_sigma,
        "sd": np.nan,
        "q2.5": np.nan,
        "q97.5": np.nan,
    })
    posterior_summary = pd.concat([posterior_summary, ml_sigma_rows], ignore_index=True)
    acceptance_df = pd.DataFrame({"chain": np.arange(1, mcmc_chains+1), "HMC_acceptance": rates})
    result = BayesianUQResult(
        model=model, forecast=pd.DataFrame(rows), metrics=pd.DataFrame(metric_rows),
        baseline_metrics=baseline_metrics, posterior_summary=posterior_summary,
        chain_samples=chain, rhat=rhat, calibration_fraction=1.0,
        mcmc_chains=mcmc_chains, mcmc_samples=mcmc_samples, burn_in=burn_in, thin=thin,
        seed=seed,
        acceptance_rates=acceptance_df,
        posterior_mean_params=posterior_mean_params,
        ml_sigma=ml_sigma,
        calibration_predictions=calibration_predictions,
        historical_production=pd.DataFrame({
            "time": data.time,
            **{name: data.production[:, j] for j, name in enumerate(data.producer_names)}
        }),
    )
    max_rhat = float(result.rhat["R_hat"].max(skipna=True))
    if np.isfinite(max_rhat) and max_rhat > UQ_RHAT_WARNING:
        warnings.warn(f"Maximum Bayesian CRM R-hat is {max_rhat:.3f}; increase MCMC iterations before treating intervals as final.", RuntimeWarning)
    return result



# 6. FIGURES AND EXPORTS



def plot_production_history(data: FieldData, n_hist: Optional[int] = None) -> plt.Figure:
    """Display the observed production record before any forecasting is run."""
    n_hist = len(data.time) if n_hist is None else int(n_hist)
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, pname in enumerate(data.producer_names):
        ax.plot(data.time, data.production[:, j], lw=1.5, label=f"Production - {pname}")
    if 0 < n_hist < len(data.time):
        ax.axvline(data.time[n_hist], linestyle="--", lw=1.4, label="Forecast start (t_f)")
    ax.set_xlabel(f"Time [{data.time_unit}]")
    ax.set_ylabel(f"Production rate [{data.rate_unit}]")
    ax.set_title("Observed production history and forecast boundary")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_producer_forecast(result: BayesianUQResult, producer: str, data: FieldData) -> plt.Figure:
    """One isolated production/forecast chart for a single producer.

    The chart deliberately keeps one well per figure so observed production,
    deterministic CRM-ML, Bayesian P50 and the 95% predictive interval cannot
    be confused across wells. The vertical line marks the held-out forecast
    boundary.
    """
    df = result.forecast[result.forecast["producer"] == producer].sort_values("time")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if df.empty:
        ax.set_title(f"{producer}: no forecast observations available")
        ax.set_xlabel(f"Time [{data.time_unit}]")
        ax.set_ylabel(f"Production rate [{data.rate_unit}]")
        fig.tight_layout()
        return fig

    ax.plot(df["time"], df["observed"], lw=2.0, label="Observed production")
    ax.plot(df["time"], df["crm_ml_deterministic"], lw=1.5, linestyle=":", label="Deterministic CRM-ML")
    ax.plot(df["time"], df["P50"], lw=2.0, linestyle="--", label="Bayesian P50")
    ax.fill_between(
        df["time"].to_numpy(float),
        df["95pct_lower"].to_numpy(float),
        df["95pct_upper"].to_numpy(float),
        alpha=0.20,
        label="95% predictive interval",
    )
    tf = float(df["time"].min())
    ax.axvline(tf, linestyle="--", lw=1.4, label="Forecast start ($t_f$)")
    ax.set_xlabel(f"Time [{data.time_unit}]")
    ax.set_ylabel(f"Production rate [{data.rate_unit}]")
    ax.set_title(f"{producer} — production and forecast")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_producer_history(data: FieldData, producer: str, n_hist: Optional[int] = None) -> plt.Figure:
    """One isolated observed-production history chart for a single producer."""
    j = data.producer_names.index(producer)
    n_hist = len(data.time) if n_hist is None else int(n_hist)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(data.time, data.production[:, j], lw=1.8, label="Observed production")
    if 0 < n_hist < len(data.time):
        ax.axvline(data.time[n_hist], linestyle="--", lw=1.4, label="Forecast start ($t_f$)")
    ax.set_xlabel(f"Time [{data.time_unit}]")
    ax.set_ylabel(f"Production rate [{data.rate_unit}]")
    ax.set_title(f"{producer} — observed production history")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_forecast_test(result: BayesianUQResult, data: FieldData) -> plt.Figure:
    """Show the held-out test block, Bayesian P50 and 95% interval from t_f."""
    df = result.forecast
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for pname in df["producer"].unique():
        d = df[df["producer"] == pname].sort_values("time")
        ax.plot(d["time"], d["observed"], lw=1.8, label=f"Observed test - {pname}")
        ax.plot(d["time"], d["crm_ml_deterministic"], lw=1.2, linestyle=":", label=f"Deterministic CRM-ML - {pname}")
        ax.plot(d["time"], d["P50"], lw=1.8, linestyle="--", label=f"Bayesian P50 - {pname}")
        ax.fill_between(
            d["time"].to_numpy(float),
            d["95pct_lower"].to_numpy(float),
            d["95pct_upper"].to_numpy(float),
            alpha=0.20,
            label=f"95% predictive interval - {pname}",
        )
    if len(df):
        tf = float(df["time"].min())
        ax.axvline(tf, linestyle="--", lw=1.5, label="Forecast start (t_f)")
    ax.set_xlabel(f"Time [{data.time_unit}]")
    ax.set_ylabel(f"Production rate [{data.rate_unit}]")
    ax.set_title("Held-out production test: deterministic CRM-ML vs Bayesian CRM-ML")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_ml_calibration(result: BayesianUQResult) -> plt.Figure:
    """Visualise the out-of-sample residuals used to estimate ML aleatoric sigma."""
    fig, ax = plt.subplots(figsize=(11, 5))
    cal = result.calibration_predictions
    for pname in cal["producer"].unique():
        d = cal[cal["producer"] == pname]
        ax.plot(d["historical_row"], d["log_residual"], marker=".", linestyle="-", label=pname)
    ax.axhline(0.0, linestyle="--", lw=1.0)
    ax.set_xlabel("Historical row")
    ax.set_ylabel("log(observed) - log(CRM-ML prediction)")
    ax.set_title("Chronological out-of-sample CRM-ML residuals used for aleatoric UQ")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_uq_forecast(result: BayesianUQResult, rate_unit: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    df = result.forecast
    for pname in df["producer"].unique():
        d = df[df["producer"] == pname]
        ax.plot(d["time"], d["observed"], lw=1.6, label=f"Observed - {pname}")
        ax.plot(d["time"], d["P50"], lw=1.8, linestyle="--", label=f"P50 - {pname}")
        ax.fill_between(d["time"].to_numpy(float), d["95pct_lower"].to_numpy(float),
                        d["95pct_upper"].to_numpy(float), alpha=0.20, label=f"95% interval - {pname}")
    ax.set_xlabel("Forecast time")
    ax.set_ylabel(f"Liquid production rate [{rate_unit}]")
    ax.set_title(f"Bayesian uncertainty-quantified production forecast - {result.model}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_mcmc_diagnostics(result: BayesianUQResult) -> plt.Figure:
    chain = result.chain_samples
    p = chain.shape[-1]
    if p == 3 and "alpha" in set(result.posterior_summary["parameter"]):
        labels = ["alpha", "beta", "log_sigma"]
        indices = [0, 1, 2]
        transforms = [lambda x: x, lambda x: x, lambda x: np.exp(x)]
        title = f"Bayesian calibration HMC trace diagnostics - {result.model}"
    else:
        n_prod = len(result.posterior_mean_params.tau_j)
        n_theta = p - n_prod
        indices = [0, n_prod, max(0, p-1)]
        labels = ["tau_j (first producer)", "lambda (first pair)", "sigma (last parameter)"]
        transforms = [np.exp, lambda x: 1.0/(1.0+np.exp(-x)), lambda x: np.exp(x)]
        title = f"Bayesian CRM HMC trace diagnostics - {result.model}"
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, idx, label, transform in zip(axes, indices, labels, transforms):
        for c in range(chain.shape[0]):
            ax.plot(transform(chain[c, :, idx]), lw=0.7, alpha=0.7, label=f"Chain {c+1}" if ax is axes[0] else None)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Retained MCMC draw")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def export_result(result: BayesianUQResult, folder: Union[str, Path], rate_unit: str) -> List[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    result.forecast.to_csv(folder / "uq_forecast.csv", index=False)
    result.metrics.to_csv(folder / "uq_metrics.csv", index=False)
    result.baseline_metrics.to_csv(folder / "deterministic_baseline_metrics.csv", index=False)
    result.posterior_summary.to_csv(folder / "uq_posterior_summary.csv", index=False)
    result.rhat.to_csv(folder / "uq_rhat.csv", index=False)
    result.acceptance_rates.to_csv(folder / "uq_acceptance_rates.csv", index=False)
    pd.DataFrame({"producer": result.forecast["producer"].unique(), "sigma_ML_oos": result.ml_sigma}).to_csv(
        folder / "uq_ml_aleatoric_sigma.csv", index=False
    )
    result.calibration_predictions.to_csv(folder / "uq_ml_calibration_predictions.csv", index=False)
    result.historical_production.to_csv(folder / "historical_production.csv", index=False)
    pd.DataFrame(result.posterior_mean_params.lambda_ij).to_csv(folder / "posterior_mean_lambda.csv", index=False)
    pd.DataFrame(result.posterior_mean_params.tau_ij).to_csv(folder / "posterior_mean_tau_ij.csv", index=False)
    pd.DataFrame({"tau_j": result.posterior_mean_params.tau_j}).to_csv(folder / "posterior_mean_tau_j.csv", index=False)
    np.savez_compressed(folder / "uq_mcmc_samples.npz", chain_samples=result.chain_samples)
    fig1 = plot_uq_forecast(result, rate_unit)
    fig2 = plot_mcmc_diagnostics(result)
    fig3 = plot_forecast_test(result, FieldData(
        time=result.historical_production["time"].to_numpy(float),
        production=result.historical_production.drop(columns=["time"]).to_numpy(float),
        injection=np.empty((len(result.historical_production), 0)),
        distances=np.empty((0, len(result.historical_production.columns)-1)),
        producer_names=list(result.historical_production.columns[1:]),
        injector_names=[],
    ))
    fig4 = plot_ml_calibration(result)
    fig1.savefig(folder / "uq_forecast.png", dpi=200, bbox_inches="tight")
    fig2.savefig(folder / "uq_mcmc_diagnostics.png", dpi=200, bbox_inches="tight")
    fig3.savefig(folder / "uq_forecast_test_with_interval.png", dpi=200, bbox_inches="tight")
    fig4.savefig(folder / "uq_ml_calibration_residuals.png", dpi=200, bbox_inches="tight")
    plt.close(fig1); plt.close(fig2); plt.close(fig3); plt.close(fig4)
    files.extend(sorted(folder.iterdir()))
    return files


def _print_report(result: BayesianUQResult) -> None:
    print("\n" + "="*72)
    print("BAYESIAN UQ CRM-ML PRODUCTION FORECAST")
    print("="*72)
    print(f"Selected method: {result.model}")
    print(f"MCMC: {result.mcmc_chains} chains x {result.mcmc_samples} retained draws, burn-in={result.burn_in}, thin={result.thin}")
    print(f"Historical calibration fraction: {result.calibration_fraction:.0%}")
    print("Bayesian layer: CRM-parameter posterior for full CRM inputs, or Bayesian log-scale calibration for supplied q_CRM inputs")
    print("\nDeterministic baseline metrics:")
    print(result.baseline_metrics.round(4).to_string(index=False))
    print("\nBayesian CRM posterior-predictive metrics:")
    print(result.metrics.round(4).to_string(index=False))
    print("\nHMC acceptance rates:")
    print(result.acceptance_rates.round(4).to_string(index=False))
    print("\nMCMC convergence (R-hat):")
    print(result.rhat.round(4).to_string(index=False))


# 7. CLI / STREAMLIT GUI


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Bayesian UQ CRM-ML production forecasting")
    parser.add_argument("--data", help="Excel workbook with Time, q_Observed, w_Injection and Distances")
    parser.add_argument("--forecast", type=int, default=12, help="Number of final time steps treated as forecast (default: 12)")
    parser.add_argument("--model", choices=CRM_ML_MODELS, default=DEFAULT_CRM_ML_MODEL,
                        help="CRM-ML hybrid for UQ (default: CRM-XGB)")
    parser.add_argument("--crm-starts", type=int, default=3)
    parser.add_argument("--crm-max-iter", type=int, default=300)
    parser.add_argument("--mcmc-chains", type=int, default=MCMC_DEFAULT_CHAINS)
    parser.add_argument("--mcmc-samples", type=int, default=MCMC_DEFAULT_SAMPLES)
    parser.add_argument("--mcmc-burnin", type=int, default=MCMC_DEFAULT_BURN_IN)
    parser.add_argument("--mcmc-thin", type=int, default=MCMC_DEFAULT_THIN)
    parser.add_argument("--hmc-step-size", type=float, default=0.05)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=MCMC_DEFAULT_SEED)
    parser.add_argument("--trim-leading-inactive", action="store_true",
                        help="Retained for compatibility; inactive rows are now handled explicitly throughout ML/Lognormal calibration.")
    parser.add_argument("--out", default="uq_results")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--write-template", metavar="FILE")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args(argv)

    if args.write_template:
        print(f"Wrote {write_template(args.write_template)}")
        return
    if args.gui:
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        raise SystemExit(stcli.main())
    if not args.data:
        raise ValueError("--data is required for the research implementation. Use --write-template to create a workbook template.")

    data = load_field_data(args.data)
    print(f"Loaded {len(data.time)} rows, {data.injection.shape[1]} injectors, {data.production.shape[1]} producers.")
    if data.injection.shape[1] == 0:
        print("CRM mode: no injectors; injection-connectivity terms are disabled.")
    for note in data.notes:
        print("Note:", note)
    result = run_bayesian_uq(
        data,
        n_forecast=args.forecast,
        model=args.model,
        mcmc_chains=args.mcmc_chains,
        mcmc_samples=args.mcmc_samples,
        burn_in=args.mcmc_burnin,
        thin=args.mcmc_thin,
        seed=args.seed,
        hmc_step_size=args.hmc_step_size,
        hmc_leapfrog_steps=args.hmc_leapfrog_steps,
        crm_starts=args.crm_starts,
        crm_max_iter=args.crm_max_iter,
        trim_leading_inactive=args.trim_leading_inactive,
        verbose=not args.quiet,
    )
    _print_report(result)
    paths = export_result(result, args.out, data.rate_unit)
    print(f"\nWrote {len(paths)} output files to {Path(args.out).resolve()}")


def run_gui() -> None:
    import streamlit as st

    st.set_page_config(page_title="Deep Bayesian CRM UQ Forecast", layout="wide")
    st.title("Deep Bayesian CRM-ML Production Forecasting")
    st.caption(
        "Bayesian posterior over CRM connectivity/response-time parameters → "
        "CRM-XGB → Lognormal posterior predictive uncertainty"
    )

    with st.sidebar:
        st.header("1. Data")
        upload = st.file_uploader("Upload Excel workbook (.xlsx)", type=["xlsx"])
        with tempfile.TemporaryDirectory() as tmp:
            template = write_template(Path(tmp) / "crm_ml_uq_template.xlsx").read_bytes()
            st.download_button(
                "Download workbook template",
                template,
                "crm_ml_uq_template.xlsx",
                use_container_width=True,
            )

        st.header("2. CRM-ML method")
        model = st.selectbox(
            "CRM-ML method used for UQ",
            CRM_ML_MODELS,
            index=0,
            help="CRM-XGB is the recommended primary configuration.",
        )
        forecast = st.number_input(
            "Forecast length (final samples)", min_value=1, value=12, step=1
        )
        crm_starts = st.slider("CRM optimisation restarts", 1, 8, 3)
        trim = st.checkbox("Exclude leading inactive producer rows", value=False)

        st.header("3. Bayesian UQ / HMC")
        chains = st.slider("MCMC chains", 2, 20, MCMC_DEFAULT_CHAINS)
        samples = st.number_input(
            "Retained samples per chain", 100, 10000, MCMC_DEFAULT_SAMPLES, 100
        )
        burnin = st.number_input(
            "Burn-in iterations", 100, 20000, MCMC_DEFAULT_BURN_IN, 100
        )
        thin = st.number_input("Thinning interval", 1, 20, MCMC_DEFAULT_THIN, 1)
        hmc_step = st.number_input(
            "Initial HMC step size", 0.005, 0.50, 0.05, 0.005, format="%.3f"
        )
        hmc_leapfrog = st.number_input(
            "HMC leapfrog steps", 1, 20, 5, 1
        )
        seed = st.number_input("Random seed", 0, 999999, MCMC_DEFAULT_SEED, 1)
        run = st.button("Run Bayesian UQ", type="primary", use_container_width=True)

    if upload is None:
        st.info("Upload an Excel workbook containing Time, q_Observed, w_Injection and Distances.")
        return

    try:
        data = load_from_bytes(upload.getvalue())
    except Exception as exc:
        st.error(f"Could not read workbook: {exc}")
        return

    n_hist_preview = max(1, len(data.time) - int(forecast))
    st.subheader("Production overview")
    st.pyplot(plot_production_history(data, n_hist_preview), use_container_width=True)

    with st.expander("Input data summary", expanded=True):
        st.dataframe(data.summary(), use_container_width=True)

    with st.expander("Input notes"):
        for note in data.notes:
            st.write(note)

    if run:
        try:
            with st.spinner(
                "Calibrating CRM, sampling the Bayesian posterior, fitting CRM-ML and "
                "propagating uncertainty..."
            ):
                result = run_bayesian_uq(
                    data,
                    int(forecast),
                    model,
                    int(chains),
                    int(samples),
                    int(burnin),
                    int(thin),
                    int(seed),
                    crm_starts=int(crm_starts),
                    trim_leading_inactive=bool(trim),
                    verbose=True,
                    hmc_step_size=float(hmc_step),
                    hmc_leapfrog_steps=int(hmc_leapfrog),
                )
                st.session_state["uq_result"] = result
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            return

    result = st.session_state.get("uq_result")
    if result is None:
        st.info("The production plot above is the input view. Run Bayesian UQ to display the result tabs.")
        return

    st.subheader(f"Results — {result.model}")

    max_rhat = float(result.rhat["R_hat"].max(skipna=True))
    mean_picp = float(result.metrics["PICP_95"].mean())
    mean_mpiw = float(result.metrics["MPIW_95"].mean())
    mean_mae = float(result.metrics["MAE_P50"].mean())
    mean_baseline_mae = float(result.baseline_metrics["MAE_P50"].mean())
    mean_improvement = float(result.metrics["MAE_improvement_vs_deterministic"].mean())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Max R-hat", f"{max_rhat:.3f}")
    c2.metric("Mean 95% PICP", f"{mean_picp:.1%}")
    c3.metric("Mean 95% MPIW", f"{mean_mpiw:.3f}")
    c4.metric("Bayesian P50 MAE", f"{mean_mae:.3f}")
    c5.metric("MAE improvement", f"{mean_improvement:+.3f}")

    if max_rhat > UQ_RHAT_WARNING:
        st.warning("At least one R-hat exceeds 1.10. Increase MCMC iterations before treating the UQ result as final.")
    else:
        st.success("MCMC convergence check: all reported R-hat values are at or below 1.10.")

    tabs = st.tabs([
        "Overview",
        "Production & Forecast",
        "Detailed Comparison",
        "UQ / Calibration",
        "MCMC Diagnostics",
        "Posterior Parameters",
        "Downloads",
    ])

    with tabs[0]:
        st.markdown("### Deterministic baseline vs Bayesian CRM-ML")
        comparison = result.baseline_metrics.merge(
            result.metrics[
                [
                    "producer", "MAE_P50", "RMSE_P50", "R2_P50",
                    "PICP_95", "MPIW_95", "interval_score_95",
                    "MAE_improvement_vs_deterministic",
                ]
            ],
            on="producer",
            suffixes=("_deterministic", "_bayesian"),
        )
        st.dataframe(comparison.round(4), use_container_width=True)
        st.markdown(
            "**Interpretation:** the deterministic model is the reference point forecast; "
            "the Bayesian P50 is the point forecast produced after propagating CRM parameter "
            "uncertainty. PICP, MPIW and interval score evaluate the predictive distribution."
        )
        st.markdown("### Model-level uncertainty scale")
        sigma_df = pd.DataFrame({
            "producer": data.producer_names,
            "sigma_ML_oos": result.ml_sigma,
            "approx_95pct_log_scale": 1.96 * result.ml_sigma,
        })
        st.dataframe(sigma_df.round(5), use_container_width=True)

    with tabs[1]:
        st.markdown("### Production and forecast by well")
        st.caption(
            "Each chart shows one producer only. Observed production is the measured series; "
            "Deterministic CRM-ML is the reference forecast; Bayesian P50 is the median Bayesian "
            "forecast; the shaded band is the 95% posterior predictive interval. The dashed vertical "
            "line marks the start of the held-out forecast period ($t_f$)."
        )

        # Isolate every well so curves from different producers cannot be mistaken for one another.
        for producer in data.producer_names:
            with st.container(border=True):
                st.pyplot(plot_producer_forecast(result, producer, data), use_container_width=True)
                pm = result.metrics[result.metrics["producer"] == producer]
                if not pm.empty:
                    row = pm.iloc[0]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("P50 MAE", f"{float(row['MAE_P50']):.3f}")
                    m2.metric("95% coverage", f"{float(row['PICP_95']):.1%}")
                    m3.metric("95% interval width", f"{float(row['MPIW_95']):.3f}")

        st.markdown("### Forecast table")
        st.dataframe(result.forecast.round(4), use_container_width=True)

    with tabs[2]:
        st.markdown("### Detailed real-vs-predicted comparison")
        detailed_cols = [
            "time", "producer", "observed",
            "crm_ml_deterministic", "deterministic_error", "deterministic_abs_error",
            "P50", "bayesian_error", "bayesian_abs_error",
            "P90_low", "P10_high", "95pct_lower", "95pct_upper",
            "interval_width_95", "covered_95",
        ]
        st.dataframe(result.forecast[detailed_cols].round(4), use_container_width=True)
        st.markdown("### Producer-level comparison metrics")
        st.dataframe(result.metrics.round(4), use_container_width=True)

    with tabs[3]:
        st.markdown("### Chronological out-of-sample ML calibration")
        st.pyplot(plot_ml_calibration(result), use_container_width=True)
        st.dataframe(result.calibration_predictions.round(5), use_container_width=True)
        st.markdown(
            "The aleatoric scale used in the final predictive distribution is estimated from "
            "these held-out historical CRM-ML residuals, not from CRM-only residuals."
        )

    with tabs[4]:
        st.markdown("### HMC trace diagnostics")
        st.pyplot(plot_mcmc_diagnostics(result), use_container_width=True)
        st.markdown("### R-hat")
        st.dataframe(result.rhat.round(4), use_container_width=True)
        st.markdown("### HMC acceptance")
        st.dataframe(result.acceptance_rates.round(4), use_container_width=True)

    with tabs[5]:
        st.markdown("### Bayesian CRM posterior summary")
        st.dataframe(result.posterior_summary.round(5), use_container_width=True)
        st.markdown("### Posterior-mean CRM connectivity")
        st.dataframe(
            pd.DataFrame(
                result.posterior_mean_params.lambda_ij,
                index=data.injector_names,
                columns=data.producer_names,
            ).round(5),
            use_container_width=True,
        )
        st.markdown("### Posterior-mean response times $\\tau_{ij}$")
        st.dataframe(
            pd.DataFrame(
                result.posterior_mean_params.tau_ij,
                index=data.injector_names,
                columns=data.producer_names,
            ).round(5),
            use_container_width=True,
        )
        st.markdown("### Posterior-mean producer response times $\\tau_j$")
        st.dataframe(
            pd.DataFrame(
                {"producer": data.producer_names, "tau_j": result.posterior_mean_params.tau_j}
            ).round(5),
            use_container_width=True,
        )

    with tabs[6]:
        st.markdown("### Exported research outputs")
        st.write(
            "The ZIP contains the forecast table, detailed comparison, metrics, posterior "
            "summaries, R-hat/acceptance diagnostics, CRM posterior parameters, ML aleatoric "
            "sigma, calibration residuals and the production/forecast figures."
        )
        with tempfile.TemporaryDirectory() as tmp:
            files = export_result(result, tmp, data.rate_unit)
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    zf.write(f, arcname=f.name)
            st.download_button(
                "Download complete Bayesian UQ results (ZIP)",
                buf.getvalue(),
                "deep_bayesian_crm_uq_results.zip",
                use_container_width=True,
            )




if "streamlit" in sys.modules:
    try:
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
