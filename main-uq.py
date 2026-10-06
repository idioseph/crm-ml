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

from __future__ import annotations

import argparse
import io
import math
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

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

CRM_ML_MODELS = ("CRM-XGB", "CRM-NuSVM", "CRM-ELM", "CRM-MLP")
DEFAULT_CRM_ML_MODEL = "CRM-XGB"

PAPER_MLP_HIDDEN_NEURONS = 10
PAPER_TRAIN_FRACTION = 0.75
PAPER_VALIDATION_FRACTION = 0.25
PAPER_MIN_TAU_SAMPLING_INTERVAL = True

# Implementation settings (not attributed to the paper).
ML_SETTINGS = {
    "NuSVM": {"kernel": "rbf", "nu": 0.5, "C": 10.0, "gamma": "scale"},
    "XGB": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
            "n_jobs": 1, "verbosity": 0},
    "ELM": {"ridge": 1e-3},
    "MLP": {"activation": "tanh", "solver": "lbfgs", "max_iter": 500},
}

MCMC_DEFAULT_CHAINS = 3
MCMC_DEFAULT_SAMPLES = 1500
MCMC_DEFAULT_BURN_IN = 1000
MCMC_DEFAULT_THIN = 2
MCMC_DEFAULT_SEED = 2025
MCMC_TARGET_ACCEPTANCE = 0.65
UQ_CI_LEVEL = 0.95
UQ_RHAT_WARNING = 1.10
UQ_MIN_ESS_WARNING = 100.0
UQ_CALIBRATION_FRACTION = 0.75
UQ_MAX_CRM_PREDICTIVE_DRAWS = 200
UQ_TAU_LOG_SD = 2.0          # prior sd of log(tau) around a data-scale reference
UQ_LAMBDA_LOGIT_SD = 1.5     # prior sd of the softmax-with-slack logits


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


def _truncate_field(data: FieldData, n: int) -> FieldData:
    return FieldData(
        time=data.time[:n], production=data.production[:n], injection=data.injection[:n],
        distances=data.distances, producer_names=list(data.producer_names),
        injector_names=list(data.injector_names),
        bhp=None if data.bhp is None else data.bhp[:n],
        time_unit=data.time_unit, rate_unit=data.rate_unit, notes=list(data.notes),
        crm_baseline=None if data.crm_baseline is None else data.crm_baseline[:n],
    )


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
    """Coefficient of determination: 1 - SSE/SST."""
    q_obs, q_est = np.asarray(q_obs, float), np.asarray(q_est, float)
    if q_obs.shape != q_est.shape or q_obs.size == 0:
        raise ValueError("R2 inputs must have equal non-empty shapes.")
    sst = float(np.sum((q_obs - q_obs.mean()) ** 2))
    if sst <= 0:
        return float("nan")
    return float(1.0 - float(np.sum((q_obs - q_est) ** 2)) / sst)


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
    time, prod, inj, dist = map(lambda x: np.asarray(x, float),
                                (data.time, data.production, data.injection, data.distances))
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
    """Load the workbook layout required by the UQ CRM-ML method."""
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
        inj_names = []
        w_df = pd.DataFrame(index=np.arange(len(q_df)))
    else:
        w_df, inj_names, _ = _parse_block(xls.parse(sm["injection"], header=None))
    dist_df = None
    n_injectors = len(inj_names)
    n_producers = len(prod_names)
    if n_injectors == 0:
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
            raise ValueError(f"q_CRM sheet must have shape {production.shape}; parsed {crm_baseline.shape}.")
        if len(crm_names) != len(prod_names) or any(a != b for a, b in zip(crm_names, prod_names)):
            raise ValueError("q_CRM producer columns must match q_Observed producer columns in the same order.")
        notes.append("Pre-computed q_CRM baseline detected. Bayesian UQ will use this supplied CRM response instead of inventing injector/connectivity parameters.")
    bhp = None if sm["bhp"] is None else _parse_block(xls.parse(sm["bhp"], header=None))[0].to_numpy(float)
    data = FieldData(
        time=time, production=production, injection=injection, distances=distances,
        producer_names=prod_names, injector_names=inj_names, bhp=bhp,
        time_unit=_extract_unit(t_notes) or DEFAULT_TIME_UNIT,
        rate_unit=_extract_unit(q_notes) or DEFAULT_RATE_UNIT,
        notes=notes, crm_baseline=crm_baseline,
    )
    validate_field(data)
    return data


def load_from_bytes(content: bytes) -> FieldData:
    return load_field_data(io.BytesIO(content))


def write_template(path: Union[str, Path], n_steps: int = 36, n_injectors: int = 3, n_producers: int = 2) -> Path:
    """Create a workbook that exactly matches the importer layout."""
    rng = np.random.default_rng(0)
    path = Path(path)
    time = np.arange(n_steps, dtype=float)
    production = rng.uniform(1.0, 2.0, (n_steps, n_producers))
    injection = rng.uniform(1.0, 2.0, (n_steps, n_injectors))
    distances = rng.uniform(500, 3000, (n_injectors, n_producers))
    producer_names = [f"P-{j+1:02d}" for j in range(n_producers)]
    injector_names = [f"I-{i+1:02d}" for i in range(n_injectors)]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame([["Time"], ["[days]"]] + time.reshape(-1, 1).tolist()).to_excel(
            xw, sheet_name="Time", header=False, index=False)
        pd.DataFrame([producer_names, ["[MSTB/day]"] * n_producers] + production.tolist()).to_excel(
            xw, sheet_name="q_Observed", header=False, index=False)
        if n_injectors > 0:
            pd.DataFrame([injector_names, ["[MSTB/day]"] * n_injectors] + injection.tolist()).to_excel(
                xw, sheet_name="w_Injection", header=False, index=False)
            distance_rows = [[""] + producer_names, [""] + ["[ft]"] * n_producers]
            distance_rows += [[injector_names[i]] + distances[i].tolist() for i in range(n_injectors)]
            pd.DataFrame(distance_rows).to_excel(xw, sheet_name="Distances", header=False, index=False)
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
    """CRM Eq. (1); returns predictions for time[1:]."""
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
        for j in range(n_prod):
            response = _exp_filter(injection[1:, i], dt, params.tau_ij[i, j])
            out[:, j] += params.lambda_ij[i, j] * response
    return out


class CapacitanceResistanceModel:
    """Constrained CRM calibration using SLSQP."""

    def __init__(self, tau_min: Optional[float] = None, tau_max: Optional[float] = None,
                 n_starts: int = 3, max_iter: int = 300, random_state: int = 0):
        self.tau_min, self.tau_max = tau_min, tau_max
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
        opt_kwargs = {}
        if n_inj > 0:  # an empty constraint set breaks SLSQP when there are no injectors
            opt_kwargs["constraints"] = {"type": "ineq", "fun": lambda x: 1.0 - A @ x, "jac": lambda x: -A}
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
                if n_inj > 0:
                    x0[lam_offset:lam_offset+n_inj*n_prod] = (0.9 * rng.dirichlet(np.ones(n_prod), size=n_inj)).ravel()
                x0[lam_offset+n_inj*n_prod:] = rng.uniform(lo, np.log(max(tau_min*1.01, upper_init)), n_inj*n_prod)
            result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                              options={"maxiter": self.max_iter, "ftol": 1e-12}, **opt_kwargs)
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
    return TransformedTargetRegressor(
        regressor=Pipeline([("scale", StandardScaler()), ("model", base)]),
        transformer=StandardScaler(),
    )


def build_crm_ml_features(time: np.ndarray, injection: np.ndarray, crm_response: np.ndarray,
                          use_time: bool = True) -> np.ndarray:
    """Features: [time (optional), injection rates, CRM response].

    Per-producer constants (distances, lambda_ij, tau_ij, tau_j) are deliberately
    omitted: they have zero variance inside a producer-specific model, so they
    carry no information, and substituting posterior draws into them produced
    spurious shifts for kernel/neural models. The time-varying CRM response
    carries the physics and changes with every posterior draw.
    """
    t = np.asarray(time, float).ravel()
    inj = np.asarray(injection, float)
    resp = np.asarray(crm_response, float).ravel()
    if inj.ndim != 2 or inj.shape[0] != len(t) or resp.size != len(t):
        raise ValueError("Time, injection and CRM response dimensions are inconsistent.")
    cols = ([t[:, None]] if use_time else []) + [inj, resp[:, None]]
    return np.hstack(cols)


def fit_positive_crm_ml(model_name: str, X: np.ndarray, y: np.ndarray, seed: int):
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


# 5. BAYESIAN CRM + CRM-ML POSTERIOR PREDICTIVE UQ

@dataclass
class BayesianUQResult:
    model: str
    forecast: pd.DataFrame
    metrics: pd.DataFrame
    baseline_metrics: pd.DataFrame
    posterior_summary: pd.DataFrame
    chain_samples: np.ndarray  # [chains, draws, n_params]
    rhat: pd.DataFrame         # columns: parameter, R_hat, ESS_bulk
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
    ml_bias: Optional[np.ndarray] = None
    settings: Dict[str, object] = field(default_factory=dict)


def _positive_active_mask(y: np.ndarray) -> np.ndarray:
    """Rows eligible for the Lognormal model: strictly positive production.

    Exact zeros are valid shut-in observations: retained in the data and in point
    metrics, excluded from log-space modelling. Negative values raise an error.
    """
    y = np.asarray(y, float).ravel()
    if np.any(~np.isfinite(y)):
        raise ValueError("Production contains NaN/inf values.")
    if np.any(y < 0):
        raise ValueError(f"Production contains {int(np.sum(y < 0))} negative values; these are not supported.")
    return y > 0


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
    lam = _lambda_from_logit_slack(z) if n_inj > 0 else np.empty((0, n_prod))
    return CRMParameters(tau_j=tau_j, lambda_ij=lam, tau_ij=tau_ij)


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


def _ess_bulk(samples: np.ndarray) -> np.ndarray:
    """Effective sample size (multi-chain, Geyer initial positive sequence)."""
    a = np.asarray(samples, float)
    m, n, p = a.shape
    out = np.full(p, np.nan)
    if n < 8:
        return out
    for k in range(p):
        x = a[:, :, k]
        cm = x.mean(axis=1)
        W = float(x.var(axis=1, ddof=1).mean())
        if not np.isfinite(W) or W <= 0:
            continue
        var_plus = (n - 1) / n * W + (float(cm.var(ddof=1)) if m > 1 else 0.0)
        f = np.fft.rfft(x - cm[:, None], n=2 * n, axis=1)
        acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
        rho = 1.0 - (W - acov.mean(axis=0)) / var_plus
        rho[0] = 1.0
        tau, t = -1.0, 0
        while t + 1 < n:
            pair = rho[t] + rho[t + 1]
            if pair < 0:
                break
            tau += 2.0 * pair
            t += 2
        out[k] = max(1.0, m * n / max(tau, 1e-12))
    return out


def _diagnostics(chain: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    df = _split_rhat(chain, names)
    df["ESS_bulk"] = _ess_bulk(chain)
    return df


def _lag1_autocorr(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    if r.size < 4:
        return 0.0
    r = r - r.mean()
    den = float(np.sum(r * r))
    return 0.0 if den <= 0 else float(np.sum(r[1:] * r[:-1]) / den)


def _crm_simulate_torch(theta: torch.Tensor, time: torch.Tensor, injection_scaled: torch.Tensor,
                        q0_scaled: torch.Tensor, n_inj: int, n_prod: int) -> torch.Tensor:
    """Differentiable CRM. The injection response is computed in log space:
    response_n = sum_{k<=n} (1-a_k) s_k exp(L_n - L_k), L = -cumsum(dt/tau),
    which never overflows/underflows (all exponents are <= 0)."""
    p = 0
    tau_j = torch.exp(theta[p:p+n_prod]); p += n_prod
    z = theta[p:p+n_inj*n_prod].reshape(n_inj, n_prod); p += n_inj*n_prod
    tau_ij = torch.exp(theta[p:p+n_inj*n_prod]).reshape(n_inj, n_prod)
    dt = torch.diff(time)
    t_rel = time[1:] - time[0]
    out = q0_scaled.unsqueeze(0) * torch.exp(-t_rel.unsqueeze(1) / tau_j.unsqueeze(0))
    if n_inj == 0:
        return out
    aug = torch.cat([z, torch.zeros((n_inj, 1), dtype=theta.dtype)], dim=1)
    weights = torch.softmax(aug, dim=1)[:, :n_prod]
    signal = injection_scaled[1:]
    x = dt[:, None, None] / tau_ij[None, :, :]
    a = torch.exp(-x)
    src = (1.0 - a) * signal[:, :, None]
    N = dt.shape[0]
    if N * N * n_inj * n_prod <= 3e7:
        L = -torch.cumsum(x, dim=0)
        diff = L[:, None] - L[None, :]
        upper = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        diff = diff.masked_fill(upper[:, :, None, None], float("-inf"))
        response = torch.einsum("nkip,kip->nip", torch.exp(diff), src)
    else:  # memory-safe fallback for very long histories
        prev = torch.zeros((n_inj, n_prod), dtype=theta.dtype)
        outs = []
        for n in range(N):
            prev = a[n] * prev + src[n]
            outs.append(prev)
        response = torch.stack(outs)
    return out + torch.sum(response * weights.unsqueeze(0), dim=1)


def _sample_inverse_gamma(shape: float, scale: float, rng: np.random.Generator) -> float:
    return 1.0 / float(rng.gamma(shape, 1.0 / max(scale, 1e-12)))


def _hmc_sample(logpost: Callable[[torch.Tensor], torch.Tensor], theta0: np.ndarray,
                n_samples: int, burn_in: int, thin: int, seed: int,
                step_size: float = 0.05, leapfrog_steps: int = 5,
                extra_fn: Optional[Callable[[torch.Tensor, np.random.Generator], np.ndarray]] = None,
                n_extra: int = 0) -> Tuple[np.ndarray, float, float]:
    """Generic HMC (identity mass matrix) with step-size adaptation during burn-in.

    Returns (draws [n_samples, d + n_extra], post-burn-in acceptance rate, final step size).
    """
    if n_samples < 100 or burn_in < 100 or thin < 1:
        raise ValueError("Use at least 100 retained samples, 100 burn-in iterations and thin >= 1.")
    if leapfrog_steps < 1 or step_size <= 0:
        raise ValueError("HMC step size must be positive and leapfrog steps must be >= 1.")
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    dtype = torch.float64
    theta = torch.as_tensor(np.asarray(theta0, float), dtype=dtype).detach().clone()
    if not torch.isfinite(logpost(theta)):
        raise RuntimeError("Initial state has an invalid posterior density.")
    d = theta.numel()
    total = burn_in + n_samples * thin
    draws = np.empty((n_samples, d + n_extra))
    eps = float(step_size)
    saved, acc_post = 0, 0
    recent: List[float] = []
    for it in range(total):
        q = theta.detach().clone().requires_grad_(True)
        p0 = torch.randn_like(q)
        lp = logpost(q)
        grad = torch.autograd.grad(lp, q)[0]
        p = p0 + 0.5 * eps * grad
        q_new, valid = q, True
        for lf in range(leapfrog_steps):
            q_new = (q_new + eps * p).detach().requires_grad_(True)
            lp_new = logpost(q_new)
            if not torch.isfinite(lp_new):
                valid = False
                break
            grad_new = torch.autograd.grad(lp_new, q_new)[0]
            if lf < leapfrog_steps - 1:
                p = p + eps * grad_new
        accepted = False
        if valid:
            p = -(p + 0.5 * eps * grad_new)
            cur_h = -lp.detach() + 0.5 * torch.sum(p0 ** 2)
            prop_h = -lp_new.detach() + 0.5 * torch.sum(p ** 2)
            log_acc = min(0.0, float(cur_h - prop_h))
            if math.log(max(rng.random(), 1e-300)) < log_acc:
                theta, accepted = q_new.detach(), True
        recent.append(1.0 if accepted else 0.0)
        if it < burn_in and (it + 1) % 50 == 0:
            rate = float(np.mean(recent[-50:]))
            eps = float(np.clip(eps * math.exp(2.0 * (rate - MCMC_TARGET_ACCEPTANCE)), 0.002, 0.5))
        if it >= burn_in:
            acc_post += int(accepted)
            if (it - burn_in) % thin == 0:
                draws[saved, :d] = theta.numpy()
                if extra_fn is not None:
                    draws[saved, d:] = extra_fn(theta, rng)
                saved += 1
    return draws, acc_post / max(n_samples * thin, 1), eps


def _make_crm_posterior(time: np.ndarray, injection: np.ndarray, production: np.ndarray,
                        production_scale: float, tau_min: float, active_mask: np.ndarray,
                        rho_weights: np.ndarray):
    """Build log-posterior and sigma sampler for the CRM parameters.

    Likelihood: log y ~ Normal(log q_CRM, sigma_j) on positive rows, sigma_j^2 ~
    InvGamma(a0, b0) marginalised analytically. The log-likelihood is tempered by
    w_j = (1-rho_j)/(1+rho_j) (lag-1 effective-sample-size correction) because CRM
    residuals are serially correlated. Priors: log(tau/tau_ref) ~ N(0, UQ_TAU_LOG_SD^2)
    with tau_ref = sqrt(tau_min * span) (truncated at tau_min), softmax logits
    ~ N(0, UQ_LAMBDA_LOGIT_SD^2).
    """
    dtype = torch.float64
    t = torch.as_tensor(time, dtype=dtype)
    w = torch.as_tensor(injection / production_scale, dtype=dtype)
    q0 = torch.as_tensor(production[0] / production_scale, dtype=dtype)
    y = torch.as_tensor(production[1:], dtype=dtype)
    mask_np = np.asarray(active_mask, bool)[1:]
    mask_t = torch.as_tensor(mask_np, dtype=torch.bool)
    n_inj, n_prod = injection.shape[1], production.shape[1]
    n_obs = mask_np.sum(axis=0).astype(int)
    span = max(float(time[-1] - time[0]), tau_min * 1.01)
    log_tau_ref = 0.5 * (math.log(tau_min) + math.log(span))
    lt_min = math.log(tau_min)
    a0 = 2.0
    b0 = a0 * (0.20 ** 2)
    off = n_prod + n_inj * n_prod

    def logpost(th: torch.Tensor) -> torch.Tensor:
        lt_j, z, lt_ij = th[:n_prod], th[n_prod:off], th[off:off + n_inj * n_prod]
        if torch.any(lt_j < lt_min) or torch.any(lt_ij < lt_min):
            return torch.tensor(-float("inf"), dtype=dtype)
        qhat = _crm_simulate_torch(th, t, w, q0, n_inj, n_prod) * production_scale
        if torch.any(~torch.isfinite(qhat)):
            return torch.tensor(-float("inf"), dtype=dtype)
        ll = torch.tensor(0.0, dtype=dtype)
        for j in range(n_prod):
            mj = mask_t[:, j]
            qj = torch.clamp(qhat[:, j][mj], min=1e-12)
            resid = torch.log(y[:, j][mj]) - torch.log(qj)
            wj = float(rho_weights[j])
            ll = ll - (a0 + 0.5 * wj * n_obs[j]) * torch.log(b0 + 0.5 * wj * torch.sum(resid * resid))
        prior = -0.5 * torch.sum(((lt_j - log_tau_ref) / UQ_TAU_LOG_SD) ** 2)
        prior = prior - 0.5 * torch.sum(((lt_ij - log_tau_ref) / UQ_TAU_LOG_SD) ** 2)
        prior = prior - 0.5 * torch.sum((z / UQ_LAMBDA_LOGIT_SD) ** 2)
        return ll + prior

    def sigma_draw(th: torch.Tensor, rng: np.random.Generator) -> np.ndarray:
        with torch.no_grad():
            qhat = _crm_simulate_torch(th, t, w, q0, n_inj, n_prod) * production_scale
            sig = []
            for j in range(n_prod):
                mj = mask_t[:, j]
                resid = torch.log(y[:, j][mj]) - torch.log(torch.clamp(qhat[:, j][mj], min=1e-12))
                wj = float(rho_weights[j])
                shape = a0 + 0.5 * wj * n_obs[j]
                scale = b0 + 0.5 * wj * float(torch.sum(resid ** 2))
                sig.append(math.sqrt(_sample_inverse_gamma(shape, scale, rng)))
        return np.asarray(sig)

    return logpost, sigma_draw


def _posterior_crm_arrays(chain: np.ndarray, n_inj: int, n_prod: int):
    flat = chain.reshape(-1, chain.shape[-1])
    n_theta = flat.shape[1] - n_prod
    theta = flat[:, :n_theta]
    sigma = flat[:, n_theta:]
    tau_j = np.exp(theta[:, :n_prod])
    if n_inj == 0:
        lam = np.empty((flat.shape[0], 0, n_prod))
        tau_ij = np.empty((flat.shape[0], 0, n_prod))
    else:
        z = theta[:, n_prod:n_prod+n_inj*n_prod].reshape(flat.shape[0], n_inj, n_prod)
        lam = np.stack([_lambda_from_logit_slack(v) for v in z], axis=0)
        tau_ij = np.exp(theta[:, n_prod+n_inj*n_prod:]).reshape(flat.shape[0], n_inj, n_prod)
    return tau_j, lam, tau_ij, sigma


def _posterior_mean_crm_params(chain: np.ndarray, n_inj: int, n_prod: int) -> CRMParameters:
    tau_j, lam, tau_ij, _ = _posterior_crm_arrays(chain, n_inj, n_prod)
    return CRMParameters(tau_j=tau_j.mean(axis=0), lambda_ij=lam.mean(axis=0), tau_ij=tau_ij.mean(axis=0))


def _summ(name: str, vals: np.ndarray) -> Dict[str, object]:
    return {"parameter": name, "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "sd": float(np.std(vals, ddof=1)), "q2.5": float(np.quantile(vals, .025)),
            "q97.5": float(np.quantile(vals, .975))}


def _crm_posterior_summary(chain: np.ndarray, n_inj: int, n_prod: int,
                           producer_names: Sequence[str], injector_names: Sequence[str]) -> pd.DataFrame:
    tau_j, lam, tau_ij, sigma = _posterior_crm_arrays(chain, n_inj, n_prod)
    rows = [_summ(f"{p}::tau_j", tau_j[:, j]) for j, p in enumerate(producer_names)]
    for i, inj in enumerate(injector_names):
        for j, p in enumerate(producer_names):
            rows.append(_summ(f"{inj}->{p}::lambda", lam[:, i, j]))
            rows.append(_summ(f"{inj}->{p}::tau_ij", tau_ij[:, i, j]))
    rows += [_summ(f"{p}::sigma_CRM", sigma[:, j]) for j, p in enumerate(producer_names)]
    return pd.DataFrame(rows)


def _theta_names(n_inj: int, n_prod: int, producer_names: Sequence[str],
                 injector_names: Sequence[str]) -> List[str]:
    names = [f"{p}::log_tau_j" for p in producer_names]
    names += [f"{inj}->{prod}::lambda_logit" for inj in injector_names for prod in producer_names]
    names += [f"{inj}->{prod}::log_tau_ij" for inj in injector_names for prod in producer_names]
    names += [f"{p}::sigma_CRM" for p in producer_names]
    return names


def _crm_ml_forecast(model_name: str, params: CRMParameters, time: np.ndarray, injection: np.ndarray,
                     production: np.ndarray, n_hist: int, q0: np.ndarray, seed: int,
                     use_time: bool, models: Optional[list] = None):
    """Fit (or reuse) the per-producer CRM-ML models for one CRM parameter set and
    predict the forecast window. The CRM is simulated over history + forecast so
    its state at the forecast boundary is conditioned on the observed history."""
    resp = crm_simulate(params, time, injection, q0)  # rows correspond to time[1:]
    n_prod = production.shape[1]
    preds = np.empty((len(time) - n_hist, n_prod))
    fitted = []
    for j in range(n_prod):
        X_fore = build_crm_ml_features(time[n_hist:], injection[n_hist:], resp[n_hist-1:, j], use_time)
        if models is None:
            X_hist = build_crm_ml_features(time[1:n_hist], injection[1:n_hist], resp[:n_hist-1, j], use_time)
            y = production[1:n_hist, j]
            mask = _positive_active_mask(y)
            if int(mask.sum()) < 8:
                raise ValueError(f"Producer {j+1} has only {int(mask.sum())} positive rows; at least 8 are required.")
            m = fit_positive_crm_ml(model_name, X_hist[mask], y[mask], seed + j)
        else:
            m = models[j]
        fitted.append(m)
        preds[:, j] = positive_prediction(m, X_fore)
    return preds, fitted


def _oos_ml_calibration(model_name: str, data: FieldData, n_hist: int, seed: int, fraction: float,
                        crm_starts: int, crm_max_iter: int, use_time: bool):
    """Leak-free hold-out calibration of the ML residual scale.

    The CRM is recalibrated on the first `fraction` of history only; the ML model
    is trained on that block and scored on the remaining block, so neither stage
    has seen the calibration rows.  sigma = RMS of out-of-sample log residuals
    (includes bias); the mean residual is reported separately.
    """
    r = int(math.floor(fraction * n_hist))
    if r < 10:
        raise ValueError("History is too short for a leak-free calibration split (need >= 10 rows in the training block).")
    time, inj, prod = data.time[:n_hist], data.injection[:n_hist], data.production[:n_hist]
    cal_crm = CapacitanceResistanceModel(n_starts=crm_starts, max_iter=crm_max_iter,
                                         random_state=seed).fit(time[:r], inj[:r], prod[:r])
    resp = cal_crm.predict(time, inj, prod[0])  # rows -> time[1:]
    time_idx = np.arange(1, n_hist)
    records, sigma, bias = [], [], []
    for j, pname in enumerate(data.producer_names):
        X = build_crm_ml_features(time[1:], inj[1:], resp[:, j], use_time)
        y = prod[1:, j]
        pos = _positive_active_mask(y)
        tr = np.flatnonzero(pos & (time_idx < r))
        ca = np.flatnonzero(pos & (time_idx >= r))
        if len(tr) < 8 or len(ca) < 8:
            raise ValueError(
                f"Producer {pname}: calibration needs >= 8 positive rows before and after row {r} "
                f"(got {len(tr)} / {len(ca)}). Provide more history or fewer shut-in rows.")
        cal_model = fit_positive_crm_ml(model_name, X[tr], y[tr], seed + 10_000 + j)
        pred = positive_prediction(cal_model, X[ca])
        lr = np.log(y[ca]) - np.log(pred)
        s = float(np.sqrt(np.mean(lr ** 2)))
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"Producer {pname} produced a non-positive/non-finite residual sigma.")
        sigma.append(s)
        bias.append(float(np.mean(lr)))
        for k, o, p_, l_ in zip(ca, y[ca], pred, lr):
            records.append({"producer": pname, "historical_row": int(time_idx[k]),
                            "lead_from_split": int(time_idx[k] - r), "observed": float(o),
                            "crm_ml_prediction": float(p_), "log_residual": float(l_),
                            "calibration_role": "chronological_out_of_sample"})
    return pd.DataFrame(records), np.asarray(sigma), np.asarray(bias)


def _deterministic_forecast_metrics(observed: np.ndarray, predicted: np.ndarray,
                                    producer_names: Sequence[str], model: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "producer": pname, "model": model,
        "MAE_P50": mean_absolute_error(observed[:, j], predicted[:, j]),
        "RMSE_P50": root_mean_squared_error(observed[:, j], predicted[:, j]),
        "R2_P50": r_squared(observed[:, j], predicted[:, j]),
    } for j, pname in enumerate(producer_names)])


def _forecast_tables(producer_names: Sequence[str], ftime: np.ndarray, observed: np.ndarray,
                     det_pred: np.ndarray, mean_pred: np.ndarray, predictive: np.ndarray,
                     epi_sd: np.ndarray, ale_sd: np.ndarray, model: str):
    q10, q50, q90 = [np.quantile(predictive, p, axis=0) for p in (0.10, 0.50, 0.90)]
    lower, upper = np.quantile(predictive, [0.025, 0.975], axis=0)
    rows, metric_rows = [], []
    alpha = 0.05
    for j, pname in enumerate(producer_names):
        y = observed[:, j]
        lo, med, hi = lower[:, j], q50[:, j], upper[:, j]
        pos = y > 0
        covered = (y >= lo) & (y <= hi)
        pen = (hi - lo) + (2/alpha)*(lo - y)*(y < lo) + (2/alpha)*(y - hi)*(y > hi)
        for k, tval in enumerate(ftime):
            det = float(det_pred[k, j])
            rows.append({
                "time": float(tval), "producer": pname, "observed": float(y[k]),
                "observed_is_zero": bool(y[k] <= 0),
                "crm_ml_deterministic": det, "crm_ml_posterior_mean": float(mean_pred[k, j]),
                "P90_low": float(q10[k, j]), "P50": float(med[k]), "P10_high": float(q90[k, j]),
                "95pct_lower": float(lo[k]), "95pct_upper": float(hi[k]),
                "deterministic_error": det - float(y[k]), "deterministic_abs_error": abs(det - float(y[k])),
                "bayesian_error": float(med[k] - y[k]), "bayesian_abs_error": float(abs(med[k] - y[k])),
                "interval_width_95": float(hi[k] - lo[k]), "covered_95": bool(covered[k]),
                "epistemic_log_sd": float(epi_sd[k, j]), "aleatoric_log_sd": float(ale_sd[k, j]),
            })
        metric_rows.append({
            "producer": pname, "model": model,
            "MAE_P50": mean_absolute_error(y, med), "RMSE_P50": root_mean_squared_error(y, med),
            "R2_P50": r_squared(y, med),
            "PICP_95": float(np.mean(covered[pos])) if pos.any() else float("nan"),
            "MPIW_95": float(np.mean(hi - lo)),
            "interval_score_95": float(np.mean(pen[pos])) if pos.any() else float("nan"),
            "n_zero_observed": int((~pos).sum()),
            "mean_epistemic_log_sd": float(np.mean(epi_sd[:, j])),
            "mean_aleatoric_log_sd": float(np.mean(ale_sd[:, j])),
            "MAE_improvement_vs_deterministic":
                mean_absolute_error(y, det_pred[:, j]) - mean_absolute_error(y, med),
        })
    return pd.DataFrame(rows), pd.DataFrame(metric_rows)


def _run_precomputed_crm_uq(data: FieldData, n_forecast: int, model: str, mcmc_chains: int,
                            mcmc_samples: int, burn_in: int, thin: int, seed: int, verbose: bool,
                            use_time: bool, sigma_growth: float, predictive_draws: int) -> BayesianUQResult:
    """UQ for workbooks that already contain a q_CRM sheet.

    q_CRM is a fixed deterministic baseline; no injector parameters are invented.
    The selected ML model is trained on [time (optional), q_CRM]. Its OUT-OF-SAMPLE
    log predictions on a chronological hold-out block are used to calibrate
        log q_obs ~ Normal(alpha + beta * log q_ml, sigma_j)
    by HMC. Predictive draws use the posterior of (alpha, beta, sigma_j), so the
    interval reflects out-of-sample error (not in-sample fit).
    """
    if data.crm_baseline is None:
        raise ValueError("Pre-computed CRM UQ requires data.crm_baseline.")
    n_hist = len(data.time) - int(n_forecast)
    if n_hist < 20:
        raise ValueError("At least 20 historical rows are required before the forecast window.")
    qcrm = np.asarray(data.crm_baseline, float)
    if np.any(~np.isfinite(qcrm)) or np.any(qcrm < 0):
        raise ValueError("q_CRM contains invalid negative, NaN, or infinite values.")
    ml_name = model.split("-", 1)[1]
    hist_t, hist_qcrm, hist_y = data.time[:n_hist], qcrm[:n_hist], data.production[:n_hist]
    P = hist_y.shape[1]
    active = np.zeros(hist_y.shape, dtype=bool)
    for j in range(P):
        active[:, j] = _positive_active_mask(hist_y[:, j]) & (hist_qcrm[:, j] > 0)
        if active[:, j].sum() < 16:
            raise ValueError(f"Producer {j+1} needs at least 16 positive observed/q_CRM historical rows.")
    if verbose:
        print("CRM mode: pre-computed q_CRM baseline from workbook.")

    def make_X(t, qb):
        qb = np.asarray(qb, float)[:, None]
        return np.hstack(([np.asarray(t, float)[:, None]] if use_time else []) + [qb])

    forecast_t, forecast_qcrm, y_fore = data.time[n_hist:], qcrm[n_hist:], data.production[n_hist:]
    ml_models = [fit_positive_crm_ml(ml_name, make_X(hist_t, hist_qcrm[:, j])[active[:, j]],
                                     hist_y[:, j][active[:, j]], seed + j) for j in range(P)]
    det_pred = np.column_stack([positive_prediction(ml_models[j], make_X(forecast_t, forecast_qcrm[:, j]))
                                for j in range(P)])

    # Chronological out-of-sample predictions on the hold-out block.
    x_cal, y_cal, rms, bias, cal_records = [], [], [], [], []
    for j, pname in enumerate(data.producer_names):
        idx = np.flatnonzero(active[:, j])
        ntrain = max(8, int(np.floor(UQ_CALIBRATION_FRACTION * len(idx))))
        if len(idx) - ntrain < 8:
            ntrain = len(idx) - 8
        tr, ca = idx[:ntrain], idx[ntrain:]
        cm = fit_positive_crm_ml(ml_name, make_X(hist_t, hist_qcrm[:, j])[tr], hist_y[:, j][tr], seed + 10_000 + j)
        pred = positive_prediction(cm, make_X(hist_t, hist_qcrm[:, j])[ca])
        lr = np.log(hist_y[:, j][ca]) - np.log(pred)
        s = float(np.sqrt(np.mean(lr ** 2)))
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"Producer {j+1} has an invalid out-of-sample residual sigma.")
        rms.append(s); bias.append(float(np.mean(lr)))
        x_cal.append(np.log(pred)); y_cal.append(np.log(hist_y[:, j][ca]))
        for ii, yy, pp, rr in zip(ca, hist_y[:, j][ca], pred, lr):
            cal_records.append({"producer": pname, "historical_row": int(ii), "lead_from_split": int(ii - ca[0]),
                                "observed": float(yy), "crm_ml_prediction": float(pp), "log_residual": float(rr),
                                "calibration_role": "chronological_out_of_sample"})
    ml_sigma, ml_bias = np.asarray(rms), np.asarray(bias)

    xt = [torch.as_tensor(v, dtype=torch.float64) for v in x_cal]
    yt = [torch.as_tensor(v, dtype=torch.float64) for v in y_cal]
    log_sig_prior = math.log(0.2)

    def logpost(v: torch.Tensor) -> torch.Tensor:
        a, b, ls = v[0], v[1], v[2:]
        ll = torch.tensor(0.0, dtype=torch.float64)
        for j in range(P):
            resid = yt[j] - (a + b * xt[j])
            ll = ll - 0.5 * torch.sum((resid / torch.exp(ls[j])) ** 2) - yt[j].numel() * ls[j]
        prior = -0.5 * a ** 2 - 0.5 * ((b - 1.0) / 0.5) ** 2 - 0.5 * torch.sum(((ls - log_sig_prior) / 1.5) ** 2)
        return ll + prior

    chains, rates = [], []
    for c in range(mcmc_chains):
        rng0 = np.random.default_rng(seed + 100_000 * c)
        th0 = np.r_[0.0, 1.0, np.log(ml_sigma)] + rng0.normal(0, 0.05, 2 + P)
        d, rate, _ = _hmc_sample(logpost, th0, mcmc_samples, burn_in, thin, seed + 100_000 * c,
                                 step_size=0.02, leapfrog_steps=8)
        chains.append(d); rates.append(rate)
        if verbose:
            print(f"Calibration HMC chain {c+1}/{mcmc_chains}: acceptance={rate:.2%}")
    chain = np.stack(chains)
    flat = chain.reshape(-1, 2 + P)
    rng = np.random.default_rng(seed + 10_000_000)
    keep = rng.choice(len(flat), size=min(predictive_draws, len(flat)), replace=False)
    lead = np.arange(len(forecast_t))
    growth = np.sqrt(1.0 + sigma_growth * lead)[:, None]
    mus, preds = [], []
    for row in flat[keep]:
        mu = row[0] + row[1] * np.log(np.maximum(det_pred, 1e-30))
        mus.append(mu)
        preds.append(np.exp(mu + rng.normal(size=mu.shape) * np.exp(row[2:])[None, :] * growth))
    mus, predictive = np.stack(mus), np.stack(preds)
    epi_sd = np.std(mus, axis=0, ddof=1)
    ale_sd = np.exp(flat[:, 2:]).mean(axis=0)[None, :] * growth
    mean_pred = np.exp(mus.mean(axis=0))
    fc, mt = _forecast_tables(data.producer_names, forecast_t, y_fore, det_pred, mean_pred,
                              predictive, epi_sd, ale_sd, model)
    names = ["alpha", "beta"] + [f"log_sigma::{p}" for p in data.producer_names]
    post = pd.DataFrame([_summ(n, flat[:, k]) for k, n in enumerate(names)])
    pseudo = CRMParameters(np.ones(P), np.empty((0, P)), np.empty((0, P)))
    return BayesianUQResult(
        model=model + " [precomputed CRM]", forecast=fc, metrics=mt,
        baseline_metrics=_deterministic_forecast_metrics(y_fore, det_pred, data.producer_names, model + " deterministic"),
        posterior_summary=post, chain_samples=chain, rhat=_diagnostics(chain, names),
        calibration_fraction=UQ_CALIBRATION_FRACTION, mcmc_chains=mcmc_chains, mcmc_samples=mcmc_samples,
        burn_in=burn_in, thin=thin, seed=seed,
        acceptance_rates=pd.DataFrame({"chain": np.arange(1, mcmc_chains + 1), "HMC_acceptance": rates}),
        posterior_mean_params=pseudo, ml_sigma=ml_sigma, ml_bias=ml_bias,
        calibration_predictions=pd.DataFrame(cal_records),
        historical_production=pd.DataFrame({"time": data.time, **{n: data.production[:, j] for j, n in enumerate(data.producer_names)}}),
        settings={"precomputed": True, "use_time_feature": use_time, "sigma_growth": sigma_growth},
    )


def run_bayesian_uq(data: FieldData, n_forecast: int, model: str = DEFAULT_CRM_ML_MODEL,
                    mcmc_chains: int = MCMC_DEFAULT_CHAINS,
                    mcmc_samples: int = MCMC_DEFAULT_SAMPLES,
                    burn_in: int = MCMC_DEFAULT_BURN_IN,
                    thin: int = MCMC_DEFAULT_THIN,
                    seed: int = MCMC_DEFAULT_SEED,
                    crm_starts: int = 3, crm_max_iter: int = 300,
                    trim_leading_inactive: bool = False, verbose: bool = True,
                    hmc_step_size: float = 0.05, hmc_leapfrog_steps: int = 5,
                    predictive_draws: int = UQ_MAX_CRM_PREDICTIVE_DRAWS,
                    refit_ml_per_draw: bool = True, use_time_feature: bool = True,
                    sigma_growth: float = 0.0) -> BayesianUQResult:
    """Deterministic CRM -> Bayesian CRM (HMC) -> CRM-ML per posterior draw -> predictive UQ.

    trim_leading_inactive is accepted for backward compatibility and ignored:
    zero/shut-in rows are always excluded from the Lognormal rows.
    """
    if model not in CRM_ML_MODELS:
        raise ValueError(f"model must be one of {CRM_ML_MODELS}.")
    if not (0 < n_forecast < len(data.time)):
        raise ValueError("n_forecast must be between 1 and len(time)-1.")
    if mcmc_chains < 2 or mcmc_samples < 100 or burn_in < 100 or thin < 1:
        raise ValueError("MCMC requires >=2 chains, >=100 samples, >=100 burn-in and thin>=1.")
    if sigma_growth < 0 or predictive_draws < 20:
        raise ValueError("sigma_growth must be >= 0 and predictive_draws >= 20.")
    if data.crm_baseline is not None:
        return _run_precomputed_crm_uq(data, n_forecast, model, mcmc_chains, mcmc_samples, burn_in,
                                       thin, seed, verbose, use_time_feature, sigma_growth, predictive_draws)

    n_hist = len(data.time) - int(n_forecast)
    if n_hist < 20:
        raise ValueError("At least 20 historical rows are required before the forecast window.")
    hist_time, hist_inj, hist_prod = data.time[:n_hist], data.injection[:n_hist], data.production[:n_hist]
    n_inj, n_prod = data.injection.shape[1], data.production.shape[1]
    active_mask = np.zeros(hist_prod.shape, dtype=bool)
    for j in range(n_prod):
        active_mask[:, j] = _positive_active_mask(hist_prod[:, j])
        if int(active_mask[1:, j].sum()) < 8:
            raise ValueError(f"Producer {j+1} has fewer than 8 positive historical rows; Bayesian CRM-ML is not identifiable.")
    if verbose:
        print(f"Positive production rows used in the Lognormal likelihood: {int(active_mask.sum())}")
        print(f"Zero-production/shut-in rows retained but excluded from log-space modelling: {int((hist_prod == 0).sum())}")
        if trim_leading_inactive:
            print("Note: --trim-leading-inactive is deprecated and has no effect.")

    crm = CapacitanceResistanceModel(n_starts=crm_starts, max_iter=crm_max_iter, random_state=seed)
    crm.fit(hist_time, hist_inj, hist_prod)
    production_scale = float(crm._scale)
    tau_min = float(np.min(np.diff(hist_time)))
    if verbose:
        print(f"Deterministic CRM calibrated: objective={crm.objective_:.6g}; minimum tau={tau_min:g}")

    # Serial-correlation tempering weights from the deterministic CRM residuals.
    q_det_hist = crm.predict(hist_time, hist_inj, hist_prod[0])
    rho_w = np.ones(n_prod)
    for j in range(n_prod):
        m = active_mask[1:, j] & (q_det_hist[:, j] > 0)
        rho = float(np.clip(_lag1_autocorr(np.log(hist_prod[1:, j][m]) - np.log(q_det_hist[:, j][m])), 0.0, 0.9))
        rho_w[j] = max((1.0 - rho) / (1.0 + rho), 0.05)
    if verbose:
        print("Lag-1 effective-sample-size weights per producer:", np.round(rho_w, 3))

    logpost, sigma_draw = _make_crm_posterior(hist_time, hist_inj, hist_prod, production_scale,
                                              tau_min, active_mask, rho_w)
    th0 = _crm_theta_from_params(crm.params_)
    lt_min = math.log(tau_min)
    tau_idx = list(range(n_prod)) + list(range(n_prod + n_inj * n_prod, th0.size))
    chains, rates = [], []
    for c in range(mcmc_chains):
        seed_c = seed + 100_000 * c
        rng0 = np.random.default_rng(seed_c)
        start = None
        for attempt in range(20):  # dispersed starts; shrink jitter until the density is finite
            th = th0 + rng0.normal(0.0, 0.5 * (0.7 ** attempt), th0.shape)
            th[tau_idx] = np.maximum(th[tau_idx], lt_min + 1e-6)
            if np.isfinite(float(logpost(torch.as_tensor(th, dtype=torch.float64)))):
                start = th
                break
        if start is None:
            raise RuntimeError("Could not find a finite-density starting point for HMC.")
        draws, rate, eps = _hmc_sample(logpost, start, mcmc_samples, burn_in, thin, seed_c,
                                       step_size=hmc_step_size, leapfrog_steps=hmc_leapfrog_steps,
                                       extra_fn=sigma_draw, n_extra=n_prod)
        chains.append(draws); rates.append(rate)
        if verbose:
            print(f"Bayesian CRM HMC chain {c+1}/{mcmc_chains}: acceptance={rate:.2%}, final step={eps:.4f}")
    chain = np.stack(chains, axis=0)

    q0 = np.asarray(data.production[0], float)
    posterior_mean_params = _posterior_mean_crm_params(chain, n_inj, n_prod)
    model_name = model.split("-", 1)[1]
    observed_forecast = data.production[n_hist:]
    forecast_time = data.time[n_hist:]

    baseline_pred, _ = _crm_ml_forecast(model_name, crm.params_, data.time, data.injection, data.production,
                                        n_hist, q0, seed, use_time_feature)
    mean_pred, mean_models = _crm_ml_forecast(model_name, posterior_mean_params, data.time, data.injection,
                                              data.production, n_hist, q0, seed, use_time_feature)
    baseline_metrics = _deterministic_forecast_metrics(observed_forecast, baseline_pred,
                                                       data.producer_names, f"{model} deterministic CRM")

    cal_df, ml_sigma, ml_bias = _oos_ml_calibration(model_name, data, n_hist, seed + 700_000,
                                                    UQ_CALIBRATION_FRACTION, crm_starts, crm_max_iter,
                                                    use_time_feature)

    flat = chain.reshape(-1, chain.shape[-1])
    rng = np.random.default_rng(seed + 77)
    if len(flat) > predictive_draws:
        flat = flat[np.sort(rng.choice(len(flat), size=predictive_draws, replace=False))]
    n_theta = flat.shape[1] - n_prod
    ml_draws = np.empty((len(flat), n_forecast, n_prod))
    for d, row in enumerate(flat):
        p_d = _crm_params_from_theta(row[:n_theta], n_inj, n_prod)
        ml_draws[d], _ = _crm_ml_forecast(model_name, p_d, data.time, data.injection, data.production,
                                          n_hist, q0, seed, use_time_feature,
                                          models=None if refit_ml_per_draw else mean_models)
        if verbose and (d + 1) % 25 == 0:
            print(f"Propagated {d+1}/{len(flat)} posterior CRM draws through CRM-ML")

    lead = np.arange(n_forecast)
    ale_sd = ml_sigma[None, :] * np.sqrt(1.0 + sigma_growth * lead)[:, None]
    epi_sd = np.std(np.log(ml_draws), axis=0, ddof=1)
    predictive = ml_draws * np.exp(rng.normal(size=ml_draws.shape) * ale_sd[None, :, :])
    if np.any(~np.isfinite(predictive)) or np.any(predictive <= 0):
        raise RuntimeError("Posterior predictive simulation produced invalid values.")

    fc, mt = _forecast_tables(data.producer_names, forecast_time, observed_forecast, baseline_pred,
                              mean_pred, predictive, epi_sd, ale_sd, model)
    names = _theta_names(n_inj, n_prod, data.producer_names, data.injector_names)
    diag = _diagnostics(chain, names)
    post = _crm_posterior_summary(chain, n_inj, n_prod, data.producer_names, data.injector_names)
    ml_rows = pd.DataFrame({"parameter": [f"{p}::sigma_ML_oos" for p in data.producer_names],
                            "mean": ml_sigma, "median": ml_sigma, "sd": np.nan, "q2.5": np.nan, "q97.5": np.nan})
    post = pd.concat([post, ml_rows], ignore_index=True)
    result = BayesianUQResult(
        model=model, forecast=fc, metrics=mt, baseline_metrics=baseline_metrics,
        posterior_summary=post, chain_samples=chain, rhat=diag,
        calibration_fraction=UQ_CALIBRATION_FRACTION, mcmc_chains=mcmc_chains,
        mcmc_samples=mcmc_samples, burn_in=burn_in, thin=thin, seed=seed,
        acceptance_rates=pd.DataFrame({"chain": np.arange(1, mcmc_chains + 1), "HMC_acceptance": rates}),
        posterior_mean_params=posterior_mean_params, ml_sigma=ml_sigma, ml_bias=ml_bias,
        calibration_predictions=cal_df,
        historical_production=pd.DataFrame({"time": data.time, **{n: data.production[:, j] for j, n in enumerate(data.producer_names)}}),
        settings={"refit_ml_per_draw": refit_ml_per_draw, "use_time_feature": use_time_feature,
                  "sigma_growth": sigma_growth, "predictive_draws": int(len(flat)),
                  "rho_weights": rho_w.tolist()},
    )
    max_rhat = float(diag["R_hat"].max(skipna=True))
    min_ess = float(diag["ESS_bulk"].min(skipna=True))
    if np.isfinite(max_rhat) and max_rhat > UQ_RHAT_WARNING:
        warnings.warn(f"Maximum R-hat is {max_rhat:.3f}; increase MCMC iterations before treating intervals as final.", RuntimeWarning)
    if np.isfinite(min_ess) and min_ess < UQ_MIN_ESS_WARNING:
        warnings.warn(f"Minimum bulk ESS is {min_ess:.0f} (< {UQ_MIN_ESS_WARNING:.0f}); posterior summaries are noisy.", RuntimeWarning)
    return result


# 5b. VALIDATION UTILITIES

def rolling_origin_backtest(data: FieldData, n_forecast: int, n_origins: int = 3, **kwargs):
    """Repeat the whole pipeline at several forecast origins and pool coverage.

    Returns (per_origin_metrics, pooled_metrics). Pooled PICP uses every positive
    held-out observation across origins, which is far more informative than the
    ~n_forecast points of a single split.
    """
    per_origin, forecasts = [], []
    for k in range(n_origins):
        n_keep = len(data.time) - k * n_forecast
        if n_keep - n_forecast < 20:
            break
        res = run_bayesian_uq(_truncate_field(data, n_keep), n_forecast, verbose=False, **kwargs)
        m = res.metrics.copy(); m.insert(0, "origin_back_steps", k * n_forecast)
        per_origin.append(m)
        f = res.forecast.copy(); f.insert(0, "origin_back_steps", k * n_forecast)
        forecasts.append(f)
    if not per_origin:
        raise ValueError("Not enough history for a backtest at this forecast length.")
    allf = pd.concat(forecasts, ignore_index=True)
    pos = allf[~allf["observed_is_zero"]]
    pooled = pos.groupby("producer").agg(
        n_obs=("observed", "size"), PICP_95=("covered_95", "mean"),
        MPIW_95=("interval_width_95", "mean"), MAE_P50=("bayesian_abs_error", "mean")).reset_index()
    return pd.concat(per_origin, ignore_index=True), pooled


def synthetic_recovery_check(n_steps: int = 80, n_forecast: int = 12, seed: int = 1,
                             model: str = DEFAULT_CRM_ML_MODEL, chains: int = 2,
                             samples: int = 300, burn_in: int = 300, thin: int = 1,
                             noise_sd: float = 0.05) -> Dict[str, pd.DataFrame]:
    """Simulate from a known CRM + Lognormal noise, run the pipeline, and report
    forecast coverage and whether true parameters fall inside 95% posterior intervals.
    (lambda/tau can be weakly identified; misses here are informative, not bugs.)"""
    rng = np.random.default_rng(seed)
    time = np.arange(n_steps, dtype=float) + 1.0
    truth = CRMParameters(tau_j=np.array([8.0, 12.0]),
                          lambda_ij=np.array([[0.6, 0.2], [0.1, 0.7]]),
                          tau_ij=np.array([[5.0, 15.0], [10.0, 6.0]]))
    inj = 100.0 + 20.0 * np.sin(np.arange(n_steps)[:, None] / np.array([6.0, 9.0])[None, :]) \
        + rng.normal(0, 3.0, (n_steps, 2))
    q0 = np.array([60.0, 50.0])
    q = np.vstack([q0, crm_simulate(truth, time, inj, q0)]) * np.exp(rng.normal(0, noise_sd, (n_steps, 2)))
    data = FieldData(time=time, production=q, injection=inj, distances=np.ones((2, 2)),
                     producer_names=["P-01", "P-02"], injector_names=["I-01", "I-02"])
    res = run_bayesian_uq(data, n_forecast, model, chains, samples, burn_in, thin, seed, verbose=False)
    rows = []
    for i, iname in enumerate(data.injector_names):
        for j, pname in enumerate(data.producer_names):
            for label, tv in (("lambda", truth.lambda_ij[i, j]), ("tau_ij", truth.tau_ij[i, j])):
                r = res.posterior_summary[res.posterior_summary["parameter"] == f"{iname}->{pname}::{label}"].iloc[0]
                rows.append({"parameter": r["parameter"], "true": float(tv), "post_mean": r["mean"],
                             "q2.5": r["q2.5"], "q97.5": r["q97.5"], "covered": bool(r["q2.5"] <= tv <= r["q97.5"])})
    return {"parameter_recovery": pd.DataFrame(rows), "forecast_metrics": res.metrics, "diagnostics": res.rhat}


# 6. FIGURES AND EXPORTS

def plot_production_history(data: FieldData, n_hist: Optional[int] = None) -> plt.Figure:
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
    df = result.forecast[result.forecast["producer"] == producer].sort_values("time")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if df.empty:
        ax.set_title(f"{producer}: no forecast observations available")
        fig.tight_layout()
        return fig
    ax.plot(df["time"], df["observed"], lw=2.0, label="Observed production")
    ax.plot(df["time"], df["crm_ml_deterministic"], lw=1.5, linestyle=":", label="Deterministic CRM-ML")
    ax.plot(df["time"], df["P50"], lw=2.0, linestyle="--", label="Bayesian P50")
    ax.fill_between(df["time"].to_numpy(float), df["95pct_lower"].to_numpy(float),
                    df["95pct_upper"].to_numpy(float), alpha=0.20, label="95% predictive interval")
    ax.axvline(float(df["time"].min()), linestyle="--", lw=1.4, label="Forecast start ($t_f$)")
    ax.set_xlabel(f"Time [{data.time_unit}]")
    ax.set_ylabel(f"Production rate [{data.rate_unit}]")
    ax.set_title(f"{producer} — production and forecast")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_forecast_test(result: BayesianUQResult, data: FieldData) -> plt.Figure:
    df = result.forecast
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for pname in df["producer"].unique():
        d = df[df["producer"] == pname].sort_values("time")
        ax.plot(d["time"], d["observed"], lw=1.8, label=f"Observed test - {pname}")
        ax.plot(d["time"], d["crm_ml_deterministic"], lw=1.2, linestyle=":", label=f"Deterministic CRM-ML - {pname}")
        ax.plot(d["time"], d["P50"], lw=1.8, linestyle="--", label=f"Bayesian P50 - {pname}")
        ax.fill_between(d["time"].to_numpy(float), d["95pct_lower"].to_numpy(float),
                        d["95pct_upper"].to_numpy(float), alpha=0.20, label=f"95% interval - {pname}")
    if len(df):
        ax.axvline(float(df["time"].min()), linestyle="--", lw=1.5, label="Forecast start (t_f)")
    ax.set_xlabel(f"Time [{data.time_unit}]")
    ax.set_ylabel(f"Production rate [{data.rate_unit}]")
    ax.set_title("Held-out production test: deterministic CRM-ML vs Bayesian CRM-ML")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_ml_calibration(result: BayesianUQResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    cal = result.calibration_predictions
    for pname in cal["producer"].unique():
        d = cal[cal["producer"] == pname]
        ax.plot(d["historical_row"], d["log_residual"], marker=".", linestyle="-", label=pname)
    ax.axhline(0.0, linestyle="--", lw=1.0)
    ax.set_xlabel("Historical row")
    ax.set_ylabel("log(observed) - log(CRM-ML prediction)")
    ax.set_title("Leak-free chronological out-of-sample residuals used for aleatoric UQ")
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
    names = list(result.rhat["parameter"])
    if "alpha" in names:
        picks = [("alpha", 0, lambda x: x), ("beta", 1, lambda x: x), (names[2], 2, lambda x: x)]
    else:
        n_prod = len(result.posterior_mean_params.tau_j)
        picks = [(names[0] + " (exp)", 0, np.exp)]
        if p - n_prod > 2 * n_prod:  # injectors present
            picks.append((names[n_prod], n_prod, lambda x: x))
        picks.append((names[-1], p - 1, lambda x: x))
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.4 * len(picks) + 1), sharex=True, squeeze=False)
    for ax, (label, idx, tf) in zip(axes[:, 0], picks):
        for c in range(chain.shape[0]):
            ax.plot(tf(chain[c, :, idx]), lw=0.7, alpha=0.7, label=f"Chain {c+1}" if ax is axes[0, 0] else None)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Retained MCMC draw")
    axes[0, 0].set_title(f"HMC trace diagnostics - {result.model}")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def export_result(result: BayesianUQResult, folder: Union[str, Path], rate_unit: str) -> List[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result.forecast.to_csv(folder / "uq_forecast.csv", index=False)
    result.metrics.to_csv(folder / "uq_metrics.csv", index=False)
    result.baseline_metrics.to_csv(folder / "deterministic_baseline_metrics.csv", index=False)
    result.posterior_summary.to_csv(folder / "uq_posterior_summary.csv", index=False)
    result.rhat.to_csv(folder / "uq_rhat_ess.csv", index=False)
    result.acceptance_rates.to_csv(folder / "uq_acceptance_rates.csv", index=False)
    pd.DataFrame({"producer": result.forecast["producer"].unique(), "sigma_ML_oos_rms": result.ml_sigma,
                  "mean_log_residual_bias": result.ml_bias if result.ml_bias is not None else np.nan}
                 ).to_csv(folder / "uq_ml_aleatoric_sigma.csv", index=False)
    result.calibration_predictions.to_csv(folder / "uq_ml_calibration_predictions.csv", index=False)
    result.historical_production.to_csv(folder / "historical_production.csv", index=False)
    pd.DataFrame(result.settings.items(), columns=["setting", "value"]).astype(str).to_csv(folder / "run_settings.csv", index=False)
    pd.DataFrame(result.posterior_mean_params.lambda_ij).to_csv(folder / "posterior_mean_lambda.csv", index=False)
    pd.DataFrame(result.posterior_mean_params.tau_ij).to_csv(folder / "posterior_mean_tau_ij.csv", index=False)
    pd.DataFrame({"tau_j": result.posterior_mean_params.tau_j}).to_csv(folder / "posterior_mean_tau_j.csv", index=False)
    np.savez_compressed(folder / "uq_mcmc_samples.npz", chain_samples=result.chain_samples)
    hp = result.historical_production
    stub = FieldData(time=hp["time"].to_numpy(float), production=hp.drop(columns=["time"]).to_numpy(float),
                     injection=np.empty((len(hp), 0)), distances=np.empty((0, len(hp.columns) - 1)),
                     producer_names=list(hp.columns[1:]), injector_names=[])
    figs = {"uq_forecast.png": plot_uq_forecast(result, rate_unit),
            "uq_mcmc_diagnostics.png": plot_mcmc_diagnostics(result),
            "uq_forecast_test_with_interval.png": plot_forecast_test(result, stub),
            "uq_ml_calibration_residuals.png": plot_ml_calibration(result)}
    for fname, fig in figs.items():
        fig.savefig(folder / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return sorted(folder.iterdir())


def _print_report(result: BayesianUQResult) -> None:
    print("\n" + "=" * 72)
    print("BAYESIAN UQ CRM-ML PRODUCTION FORECAST")
    print("=" * 72)
    print(f"Selected method: {result.model}")
    print(f"MCMC: {result.mcmc_chains} chains x {result.mcmc_samples} retained draws, burn-in={result.burn_in}, thin={result.thin}")
    print(f"Calibration split: first {result.calibration_fraction:.0%} of history trains, remainder scores (leak-free)")
    print(f"Settings: {result.settings}")
    print("\nDeterministic baseline metrics:")
    print(result.baseline_metrics.round(4).to_string(index=False))
    print("\nBayesian posterior-predictive metrics:")
    print(result.metrics.round(4).to_string(index=False))
    print("\nHMC acceptance (post burn-in):")
    print(result.acceptance_rates.round(4).to_string(index=False))
    print("\nMCMC diagnostics (R-hat, bulk ESS):")
    print(result.rhat.round(3).to_string(index=False))


# 7. CLI / STREAMLIT GUI

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Bayesian UQ CRM-ML production forecasting (v2)")
    parser.add_argument("--data", help="Excel workbook with Time, q_Observed, w_Injection and Distances")
    parser.add_argument("--forecast", type=int, default=12, help="Final time steps treated as forecast (default: 12)")
    parser.add_argument("--model", choices=CRM_ML_MODELS, default=DEFAULT_CRM_ML_MODEL)
    parser.add_argument("--crm-starts", type=int, default=3)
    parser.add_argument("--crm-max-iter", type=int, default=300)
    parser.add_argument("--mcmc-chains", type=int, default=MCMC_DEFAULT_CHAINS)
    parser.add_argument("--mcmc-samples", type=int, default=MCMC_DEFAULT_SAMPLES)
    parser.add_argument("--mcmc-burnin", type=int, default=MCMC_DEFAULT_BURN_IN)
    parser.add_argument("--mcmc-thin", type=int, default=MCMC_DEFAULT_THIN)
    parser.add_argument("--hmc-step-size", type=float, default=0.05)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=5)
    parser.add_argument("--predictive-draws", type=int, default=UQ_MAX_CRM_PREDICTIVE_DRAWS,
                        help="Posterior CRM draws propagated through CRM-ML (default 200)")
    parser.add_argument("--no-refit", action="store_true",
                        help="Do not refit the ML model per posterior draw (faster, understates epistemic UQ)")
    parser.add_argument("--no-time-feature", action="store_true",
                        help="Drop raw time from ML features (avoids tree flat-extrapolation in time)")
    parser.add_argument("--sigma-growth", type=float, default=0.0,
                        help="Widen sigma with lead k: sigma*sqrt(1+g*k). User-chosen sensitivity; default 0")
    parser.add_argument("--backtest", type=int, default=0, metavar="N",
                        help="Also run an N-origin rolling-origin backtest and report pooled PICP")
    parser.add_argument("--synthetic-check", action="store_true",
                        help="Run a known-truth synthetic recovery/coverage check and exit")
    parser.add_argument("--seed", type=int, default=MCMC_DEFAULT_SEED)
    parser.add_argument("--trim-leading-inactive", action="store_true", help="Deprecated; no effect.")
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
    if args.synthetic_check:
        out = synthetic_recovery_check(model=args.model, seed=args.seed)
        for k, v in out.items():
            print(f"\n{k}:\n{v.round(4).to_string(index=False)}")
        return
    if not args.data:
        raise ValueError("--data is required. Use --write-template to create a workbook template.")

    data = load_field_data(args.data)
    print(f"Loaded {len(data.time)} rows, {data.injection.shape[1]} injectors, {data.production.shape[1]} producers.")
    for note in data.notes:
        print("Note:", note)
    common = dict(model=args.model, mcmc_chains=args.mcmc_chains, mcmc_samples=args.mcmc_samples,
                  burn_in=args.mcmc_burnin, thin=args.mcmc_thin, seed=args.seed,
                  hmc_step_size=args.hmc_step_size, hmc_leapfrog_steps=args.hmc_leapfrog_steps,
                  crm_starts=args.crm_starts, crm_max_iter=args.crm_max_iter,
                  predictive_draws=args.predictive_draws, refit_ml_per_draw=not args.no_refit,
                  use_time_feature=not args.no_time_feature, sigma_growth=args.sigma_growth)
    result = run_bayesian_uq(data, n_forecast=args.forecast, trim_leading_inactive=args.trim_leading_inactive,
                             verbose=not args.quiet, **common)
    _print_report(result)
    paths = export_result(result, args.out, data.rate_unit)
    if args.backtest > 0:
        per_origin, pooled = rolling_origin_backtest(data, args.forecast, args.backtest, **common)
        per_origin.to_csv(Path(args.out) / "backtest_per_origin.csv", index=False)
        pooled.to_csv(Path(args.out) / "backtest_pooled.csv", index=False)
        print("\nRolling-origin backtest (pooled over origins):")
        print(pooled.round(4).to_string(index=False))
    print(f"\nWrote {len(paths)} output files to {Path(args.out).resolve()}")


def run_gui() -> None:
    import streamlit as st

    st.set_page_config(page_title="Bayesian CRM-ML UQ Forecast", layout="wide")
    st.title("Bayesian CRM-ML Production Forecasting")
    st.caption("Posterior over CRM parameters → CRM-ML (refit per draw) → Lognormal predictive uncertainty")

    with st.sidebar:
        st.header("1. Data")
        upload = st.file_uploader("Upload Excel workbook (.xlsx)", type=["xlsx"])
        with tempfile.TemporaryDirectory() as tmp:
            template = write_template(Path(tmp) / "crm_ml_uq_template.xlsx").read_bytes()
            st.download_button("Download workbook template", template, "crm_ml_uq_template.xlsx",
                               use_container_width=True)
        st.header("2. CRM-ML method")
        model = st.selectbox("CRM-ML method used for UQ", CRM_ML_MODELS, index=0)
        forecast = st.number_input("Forecast length (final samples)", min_value=1, value=12, step=1)
        crm_starts = st.slider("CRM optimisation restarts", 1, 8, 3)
        use_time = st.checkbox("Use raw time as an ML feature (paper)", value=True)
        refit = st.checkbox("Refit ML for every posterior draw", value=True)
        growth = st.number_input("Sigma growth per lead step", 0.0, 1.0, 0.0, 0.01)
        draws = st.number_input("Posterior draws propagated", 20, 1000, UQ_MAX_CRM_PREDICTIVE_DRAWS, 20)
        st.header("3. Bayesian UQ / HMC")
        chains = st.slider("MCMC chains", 2, 20, MCMC_DEFAULT_CHAINS)
        samples = st.number_input("Retained samples per chain", 100, 10000, MCMC_DEFAULT_SAMPLES, 100)
        burnin = st.number_input("Burn-in iterations", 100, 20000, MCMC_DEFAULT_BURN_IN, 100)
        thin = st.number_input("Thinning interval", 1, 20, MCMC_DEFAULT_THIN, 1)
        hmc_step = st.number_input("Initial HMC step size", 0.005, 0.50, 0.05, 0.005, format="%.3f")
        hmc_leapfrog = st.number_input("HMC leapfrog steps", 1, 20, 5, 1)
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

    fig = plot_production_history(data, max(1, len(data.time) - int(forecast)))
    st.subheader("Production overview")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    with st.expander("Input data summary", expanded=True):
        st.dataframe(data.summary(), use_container_width=True)
    with st.expander("Input notes"):
        for note in data.notes:
            st.write(note)

    if run:
        try:
            with st.spinner("Calibrating CRM, sampling the posterior, refitting CRM-ML per draw..."):
                st.session_state["uq_result"] = run_bayesian_uq(
                    data, int(forecast), model, int(chains), int(samples), int(burnin), int(thin), int(seed),
                    crm_starts=int(crm_starts), verbose=True, hmc_step_size=float(hmc_step),
                    hmc_leapfrog_steps=int(hmc_leapfrog), predictive_draws=int(draws),
                    refit_ml_per_draw=bool(refit), use_time_feature=bool(use_time),
                    sigma_growth=float(growth))
        except Exception as exc:
            st.error(f"Run failed: {exc}")
            return

    result = st.session_state.get("uq_result")
    if result is None:
        st.info("Run Bayesian UQ to display the result tabs.")
        return

    st.subheader(f"Results — {result.model}")
    max_rhat = float(result.rhat["R_hat"].max(skipna=True))
    min_ess = float(result.rhat["ESS_bulk"].min(skipna=True))
    mean_picp = float(result.metrics["PICP_95"].mean())
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Max R-hat", f"{max_rhat:.3f}")
    c2.metric("Min bulk ESS", f"{min_ess:.0f}")
    c3.metric("Mean 95% PICP", f"{mean_picp:.1%}")
    c4.metric("Mean 95% MPIW", f"{float(result.metrics['MPIW_95'].mean()):.3f}")
    c5.metric("Bayesian P50 MAE", f"{float(result.metrics['MAE_P50'].mean()):.3f}")
    c6.metric("MAE improvement", f"{float(result.metrics['MAE_improvement_vs_deterministic'].mean()):+.3f}")
    if max_rhat > UQ_RHAT_WARNING or min_ess < UQ_MIN_ESS_WARNING:
        st.warning("R-hat > 1.10 or ESS < 100 for at least one parameter. Run longer before treating intervals as final.")
    else:
        st.success("MCMC checks passed (R-hat ≤ 1.10, ESS ≥ 100).")
    st.caption("PICP is computed on positive observations only. With a short forecast window it is a rough indicator; use the CLI --backtest for pooled coverage.")

    tabs = st.tabs(["Overview", "Production & Forecast", "Detailed Comparison", "UQ / Calibration",
                    "MCMC Diagnostics", "Posterior Parameters", "Downloads"])
    with tabs[0]:
        comparison = result.baseline_metrics.merge(
            result.metrics[["producer", "MAE_P50", "RMSE_P50", "R2_P50", "PICP_95", "MPIW_95",
                            "interval_score_95", "MAE_improvement_vs_deterministic"]],
            on="producer", suffixes=("_deterministic", "_bayesian"))
        st.dataframe(comparison.round(4), use_container_width=True)
        st.markdown("### Uncertainty decomposition (log scale)")
        st.dataframe(result.metrics[["producer", "mean_epistemic_log_sd", "mean_aleatoric_log_sd"]].round(5),
                     use_container_width=True)
        st.dataframe(pd.DataFrame({"producer": data.producer_names, "sigma_ML_oos_rms": result.ml_sigma,
                                   "mean_log_residual_bias": result.ml_bias}).round(5), use_container_width=True)
    with tabs[1]:
        for producer in data.producer_names:
            with st.container(border=True):
                f = plot_producer_forecast(result, producer, data)
                st.pyplot(f, use_container_width=True)
                plt.close(f)
        st.dataframe(result.forecast.round(4), use_container_width=True)
    with tabs[2]:
        cols = ["time", "producer", "observed", "crm_ml_deterministic", "deterministic_error", "P50",
                "bayesian_error", "P90_low", "P10_high", "95pct_lower", "95pct_upper", "interval_width_95",
                "covered_95", "epistemic_log_sd", "aleatoric_log_sd"]
        st.dataframe(result.forecast[cols].round(4), use_container_width=True)
        st.dataframe(result.metrics.round(4), use_container_width=True)
    with tabs[3]:
        f = plot_ml_calibration(result)
        st.pyplot(f, use_container_width=True)
        plt.close(f)
        st.dataframe(result.calibration_predictions.round(5), use_container_width=True)
    with tabs[4]:
        f = plot_mcmc_diagnostics(result)
        st.pyplot(f, use_container_width=True)
        plt.close(f)
        st.dataframe(result.rhat.round(4), use_container_width=True)
        st.dataframe(result.acceptance_rates.round(4), use_container_width=True)
    with tabs[5]:
        st.dataframe(result.posterior_summary.round(5), use_container_width=True)
        if len(data.injector_names) and result.posterior_mean_params.lambda_ij.size:
            st.markdown("### Posterior-mean connectivity λ")
            st.dataframe(pd.DataFrame(result.posterior_mean_params.lambda_ij, index=data.injector_names,
                                      columns=data.producer_names).round(5), use_container_width=True)
            st.markdown("### Posterior-mean response times τ_ij")
            st.dataframe(pd.DataFrame(result.posterior_mean_params.tau_ij, index=data.injector_names,
                                      columns=data.producer_names).round(5), use_container_width=True)
        st.markdown("### Posterior-mean producer response times τ_j")
        st.dataframe(pd.DataFrame({"producer": data.producer_names,
                                   "tau_j": result.posterior_mean_params.tau_j}).round(5), use_container_width=True)
    with tabs[6]:
        with tempfile.TemporaryDirectory() as tmp:
            files = export_result(result, tmp, data.rate_unit)
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fpath in files:
                    zf.write(fpath, arcname=fpath.name)
            st.download_button("Download complete Bayesian UQ results (ZIP)", buf.getvalue(),
                               "bayesian_crm_uq_results.zip", use_container_width=True)


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