#!/usr/bin/env python3
"""crm_ml_bayes_uq.py: Bayesian uncertainty quantification for CRM + machine learning (CRM-ML).

What it does, in order
  1-3. Exactly the same Bayesian CRM layer as crm_bayes_uq.py: deterministic
       calibration, NUTS posterior over every lambda_ij, tau_ij, tau_j, and
       propagation of every posterior draw through the CRM.
  4.   Adds an ML model chosen with --ml (xgb, elm, mlp, nusvm; several allowed).
       The ML model learns what the CRM misses, using CRM outputs that change with
       time and with the posterior draw, so every draw gives a different CRM-ML
       forecast.  P10 / P50 / P90 come from those per-draw forecasts plus noise.
  5.   Saves one plot per producer per model, and metrics for history and holdout.

The last --forecast rows are never used for fitting, sampling, training or choosing anything.
See crm_ml_bayes_uq.md for the full documentation.

Usage
  python crm_ml_bayes_uq.py --data buffalo_field.xlsx --ml xgb
  python crm_ml_bayes_uq.py --data buffalo_field.xlsx --ml xgb elm mlp nusvm
  python crm_ml_bayes_uq.py --self-test

Requires: numpy pandas scipy torch jax numpyro xgboost scikit-learn matplotlib openpyxl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import torch
from jax import lax
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.optimize import minimize
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. DATA LOADING, CRM FORWARD MODEL AND DETERMINISTIC CALIBRATION
# =============================================================================
# Article-faithful CRMP core (Ogali & Orodu 2025, Eq. 1 without BHP term).
#
# Version 2 of the CRM used for Bayesian UQ.  Differences from uq-crm.py (v1),
# each traced to the audit of 2026-10-10:
#
# * Producer-specific start time t0_j = first non-zero rate in the calibration
#   window (article Eq. 1 / MATLAB `nonZ`).  v1 started every producer at row 0.
# * Connectivity constraint is injector-wise, sum_j lambda_ij <= 1 (article
#   constraint (ii) / MATLAB `Ay`).  v1 constrained producer-wise.
# * tau bounds: tau_min = 1 day expressed in the data's time unit (article uses
#   1/30.44 month for Buffalo), tau_max = 1000 * N (MATLAB `UBx`).
# * Deterministic calibration is concurrent (all parameters at once, as the
#   article states), uses exact autograd gradients, several starts, and reports
#   convergence status for every start.

DT = torch.float64


# --------------------------------------------------------------------------- data
@dataclass
class CRMData:
    """Container for the observed production and injection series used by the CRM pipeline.

    Parameters:
        time: Strictly increasing time index for each observation row.
        production: Production-rate matrix with shape [N, J].
        injection: Injection-rate matrix with shape [N, I].
        producers: Producer labels aligned with the production columns.
        injectors: Injector labels aligned with the injection columns.
        distances: Optional producer-injector distance matrix for diagnostics.

    Returns:
        A validated CRMData instance that is passed into the forward model and calibration routines.

    Side effects:
        None beyond storing the loaded data arrays and metadata.
    """

    time: np.ndarray            # [N]
    production: np.ndarray      # [N, J]
    injection: np.ndarray       # [N, I]
    producers: list[str]
    injectors: list[str]
    distances: pd.DataFrame | None = None

    @property
    def n_inj(self):
        """Return the number of injectors represented in the loaded workbook.

        Parameters:
            None.

        Returns:
            The integer column count of the injection matrix.

        Side effects:
            None.
        """
        return self.injection.shape[1]

    @property
    def n_prod(self):
        """Return the number of producers represented in the loaded workbook.

        Parameters:
            None.

        Returns:
            The integer column count of the production matrix.

        Side effects:
            None.
        """
        return self.production.shape[1]

    def head(self, n: int) -> "CRMData":
        """Return the leading n rows of the dataset while preserving the original metadata.

        Parameters:
            n: Number of leading rows to retain.

        Returns:
            A CRMData slice containing the truncated time series and the original labels.

        Side effects:
            None; the original dataset remains unchanged.
        """
        return CRMData(self.time[:n], self.production[:n], self.injection[:n],
                       self.producers, self.injectors, self.distances)


def _sheet(xls: pd.ExcelFile, names: list[str]) -> str:
    """Resolve the first matching worksheet name from a list of likely aliases.

    Parameters:
        xls: Excel workbook already opened by pandas.
        names: Candidate sheet names to search for.

    Returns:
        The matched worksheet name used by the loader.

    Side effects:
        Raises ValueError if the workbook does not contain any expected sheet.
    """
    low = {s.strip().lower(): s for s in xls.sheet_names}
    for n in names:
        if n in low:
            return low[n]
    raise ValueError(f"none of {names} in {xls.sheet_names}")


def _numeric_block(raw: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Normalise a worksheet into column labels and a numeric matrix.

    Parameters:
        raw: Raw worksheet extracted from Excel.

    Returns:
        A tuple of (column names, numeric matrix) after dropping null-only rows and units rows.

    Side effects:
        None; the function only reorganises the worksheet data.
    """
    raw = raw.dropna(how="all").reset_index(drop=True)
    cols = [str(c).strip() for c in raw.iloc[0]]
    body = raw.iloc[1:].apply(pd.to_numeric, errors="coerce")
    # A units row such as "[MSTB/day]" becomes all-NaN and is dropped here.
    body = body.loc[body.notna().all(axis=1)]
    return cols, np.array(body.to_numpy(float), copy=True)


def load_excel(path) -> CRMData:
    """Load and validate the CRM workbook from disk.

    Parameters:
        path: File system path to the Excel workbook containing time, production, and injection data.

    Returns:
        A CRMData object with the validated arrays and metadata aligned to the workbook.

    Side effects:
        Reads the workbook file and raises ValueError when the required sheets or numeric constraints are invalid.
    """
    xls = pd.ExcelFile(path)
    _, t = _numeric_block(xls.parse(_sheet(xls, ["time", "t"]), header=None))
    qn, q = _numeric_block(xls.parse(_sheet(xls, ["q_observed", "production"]), header=None))
    wn, w = _numeric_block(xls.parse(_sheet(xls, ["w_injection", "injection"]), header=None))
    dist = None
    if "distances" in [s.lower() for s in xls.sheet_names]:
        d = xls.parse(_sheet(xls, ["distances"]), header=None).dropna(how="all")
        dist = pd.DataFrame(d.iloc[2:, 1:].to_numpy(float), index=d.iloc[2:, 0].astype(str).str.strip(),
                            columns=[str(c).strip() for c in d.iloc[0, 1:]])
    t = t[:, 0]
    if not (len(t) == len(q) == len(w)):
        raise ValueError("time, production and injection lengths differ")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing")
    if np.any(q < 0) or np.any(w < 0) or not np.isfinite(q).all() or not np.isfinite(w).all():
        raise ValueError("negative or non-finite rates")
    return CRMData(t, q, w, qn, wn, dist)


def producer_starts(production: np.ndarray) -> np.ndarray:
    """Find the first strictly positive production row for each producer.

    Parameters:
        production: Production-rate matrix with shape [N, J].

    Returns:
        An integer array of start-row indices, one per producer.

    Side effects:
        Raises ValueError if any producer has no positive production in the calibration window.
    """
    pos = production > 0
    if not pos.any(axis=0).all():
        bad = np.where(~pos.any(axis=0))[0]
        raise ValueError(f"producers {bad.tolist()} never produce in the calibration window")
    return pos.argmax(axis=0)


# ----------------------------------------------------------------------- forward
class CRMForward:
    """Vectorised Eq. 1 on a fixed timeline.

    Prediction for producer j at row n (n > s_j):
        q_j(s_j) exp(-(t_n - t_{s_j})/tau_j)
      + sum_i lambda_ij sum_{m=s_j+1}^{n} [e^{-(t_n-t_m)/tau_ij} - e^{-(t_n-t_{m-1})/tau_ij}] w_i(t_m)
    Row s_j returns the observed initial rate; rows n < s_j return 0
    (exactly as MATLAB q_CRM).  Starts and q_j(s_j) are fixed by the
    calibration window and reused for forecasting.
    """

    def __init__(self, time, injection, starts, q_start):
        """Precompute the lagged CRM tensors needed by the forward pass.

        Parameters:
            time: Observation time index used by the model.
            injection: Injection-rate matrix with shape [N, I].
            starts: Producer start rows, one per producer.
            q_start: Initial production rate at each producer start.

        Returns:
            None.

        Side effects:
            Allocates torch tensors and caches time-lag arrays on the instance.
        """
        t = torch.as_tensor(time, dtype=DT)
        self.N = t.numel()
        self.w = torch.as_tensor(injection, dtype=DT)               # [N, I]
        self.starts = torch.as_tensor(starts, dtype=torch.long)      # [J]
        self.q0 = torch.as_tensor(q_start, dtype=DT)                 # [J]
        n = torch.arange(self.N)
        tprev = torch.cat([t[:1], t[:-1]])
        D = t[:, None] - t[None, :]            # t_n - t_m
        Dp = t[:, None] - tprev[None, :]       # t_n - t_{m-1}
        lower = (n[None, :] <= n[:, None]) & (n[None, :] >= 1)
        # active[j, n, m]: m in (s_j, n]
        active = lower[None] & (n[None, None, :] > self.starts[:, None, None])
        self.active = active.to(DT)                                  # [J, N, M]
        self.D = torch.where(lower, D, torch.zeros_like(D))
        self.Dp = torch.where(lower, Dp, torch.zeros_like(Dp))
        self.after = (n[None, :] > self.starts[:, None]).T.to(DT)    # [N, J] rows n > s_j
        self.at = (n[None, :] == self.starts[:, None]).T.to(DT)      # [N, J]
        self.elapsed = (t[:, None] - t[self.starts][None, :]).clamp(min=0)  # [N, J]
        self.wm = self.w                                             # w_i(t_m)

    def __call__(self, tau_j, lam, tau_ij):
        """Evaluate the CRM forecast for a candidate parameter set.

        Parameters:
            tau_j: Producer decay times with shape [J].
            lam: Injector-to-producer connectivity matrix with shape [I, J].
            tau_ij: Injector-specific decay times with shape [I, J].

        Returns:
            A production forecast array with shape [N, J].

        Side effects:
            Uses autograd-enabled torch tensors when gradient tracking is active; no external state is modified.
        """
        prod = self.q0[None, :] * torch.exp(-self.elapsed / tau_j[None, :])
        # kernel[i, j, n, m]
        tau = tau_ij[:, :, None, None]
        ker = (torch.exp(-self.D / tau) - torch.exp(-self.Dp / tau)) * self.active[None]
        resp = torch.einsum("ijnm,mi->ijn", ker, self.wm)              # [I, J, N]
        inj = torch.einsum("ij,ijn->nj", lam, resp)
        return self.at * self.q0[None, :] + self.after * (prod + inj)

    def contributions(self, lam, tau_ij):
        """Return the per-injector CRM contribution tensor for feature generation.

        Parameters:
            lam: Injector-to-producer connectivity matrix with shape [I, J].
            tau_ij: Injector-specific decay times with shape [I, J].

        Returns:
            A tensor with shape [N, J, I] encoding the injector contribution at each time and producer.

        Side effects:
            None; it is a pure transformation of the cached time-lag kernels.
        """
        tau = tau_ij[:, :, None, None]
        ker = (torch.exp(-self.D / tau) - torch.exp(-self.Dp / tau)) * self.active[None]
        resp = torch.einsum("ijnm,mi->ijn", ker, self.wm)
        return (lam[:, :, None] * resp).permute(2, 1, 0) * self.after[:, :, None]


# ---------------------------------------------------------- deterministic CRM fit
@dataclass
class CRMParams:
    """Bundle the physical CRM parameters needed by the forward model.

    Parameters:
        tau_j: Producer decay times with shape [J].
        lam: Injector-to-producer connectivity matrix with shape [I, J].
        tau_ij: Injector-specific decay times with shape [I, J].

    Returns:
        A parameter container ready to be converted into torch tensors or saved to disk.

    Side effects:
        None beyond storing the arrays.
    """

    tau_j: np.ndarray
    lam: np.ndarray
    tau_ij: np.ndarray

    def torch(self):
        """Convert the arrays to the torch tensors expected by the CRM forward model.

        Parameters:
            None.

        Returns:
            A tuple (tau_j, lam, tau_ij) as torch tensors in the project dtype.

        Side effects:
            None; it allocates temporary tensor objects only.
        """
        return (torch.as_tensor(np.array(self.tau_j), dtype=DT), torch.as_tensor(np.array(self.lam), dtype=DT),
                torch.as_tensor(np.array(self.tau_ij), dtype=DT))


@dataclass
class FitResult:
    """Store the deterministic CRM calibration outcome and per-start diagnostics.

    Parameters:
        params: Best-fit CRM parameter set.
        objective: Optimisation objective at the best solution.
        starts_table: Per-start optimisation metrics.
        tau_bounds: Search bounds applied to the tau parameters.
        solutions: Candidate solutions discovered across the multi-start run.

    Returns:
        A summary object for the deterministic fit stage.

    Side effects:
        None beyond storing the optimisation results.
    """

    params: CRMParams
    objective: float
    starts_table: pd.DataFrame = field(repr=False)
    tau_bounds: tuple[float, float] = (0.0, 0.0)
    solutions: list = field(default_factory=list, repr=False)   # (objective, CRMParams) per start


def fit_deterministic(data: CRMData, tau_min: float, tau_max: float | None = None,
                      n_starts: int = 8, max_iter: int = 1000, seed: int = 2025) -> FitResult:
    """Fit the CRM parameters by constrained least-squares on the observed history.

    Parameters:
        data: Calibration dataset containing the history window.
        tau_min: Lower bound for tau_j and tau_ij values.
        tau_max: Upper bound for tau_j and tau_ij values; defaults to 1000 * N when None.
        n_starts: Number of multi-start optimisation attempts.
        max_iter: Maximum SLSQP iterations per start.
        seed: Random seed used for the non-deterministic initial guesses.

    Returns:
        A FitResult object containing the best fit, objective value, and per-start diagnostics.

    Side effects:
        Runs the constrained optimizer and allocates intermediate torch tensors, but does not write files.
    """
    I, J, N = data.n_inj, data.n_prod, len(data.time)
    tau_max = 1000.0 * N if tau_max is None else tau_max
    s = producer_starts(data.production)
    fwd = CRMForward(data.time, data.injection, s, data.production[s, np.arange(J)])
    y = torch.as_tensor(data.production, dtype=DT)
    mask = fwd.after                                  # residuals only where the model predicts
    denom = float(N * J)                              # Eq. 2 normaliser (constant)
    lo, hi = np.log(tau_min), np.log(tau_max)
    nl = I * J

    def unpack(x):
        x = torch.as_tensor(x, dtype=DT)
        return torch.exp(x[:J]), x[J:J + nl].reshape(I, J), torch.exp(x[J + nl:]).reshape(I, J)

    def f_and_g(x):
        xt = torch.tensor(x, dtype=DT, requires_grad=True)
        tj, lam, tij = unpack(xt)
        r = (y - fwd(tj, lam, tij)) * mask
        f = (r * r).sum() / denom
        g, = torch.autograd.grad(f, xt)
        return float(f.detach()), g.numpy()

    A = np.zeros((I, 2 * nl + J))
    for i in range(I):
        A[i, J + i * J: J + (i + 1) * J] = 1.0
    cons = {"type": "ineq", "fun": lambda x: 1.0 - A @ x, "jac": lambda x: -A}
    bounds = [(lo, hi)] * J + [(0.0, 1.0)] * nl + [(lo, hi)] * nl

    rng = np.random.default_rng(seed)
    rows, best, sols = [], None, []
    for k in range(n_starts):
        x0 = np.empty(J + 2 * nl)
        if k == 0:   # MATLAB initial guess: tau = 10, lambda = 1/J - 1e-5
            x0[:J] = np.log(np.clip(10.0, tau_min, tau_max))
            x0[J:J + nl] = 1.0 / J - 1e-5
            x0[J + nl:] = np.log(np.clip(10.0, tau_min, tau_max))
        else:
            x0[:J] = rng.uniform(np.log(max(tau_min, 0.1)), np.log(min(tau_max, 10 * N)), J)
            x0[J:J + nl] = (rng.dirichlet(np.ones(J + 1), size=I)[:, :J] * 0.95).ravel()
            x0[J + nl:] = rng.uniform(np.log(max(tau_min, 0.1)), np.log(min(tau_max, 10 * N)), nl)
        # SLSQP enforces the injector-wise sum_j lambda_ij <= 1 constraint while fitting the log-parameters.
        res = minimize(f_and_g, x0, jac=True, method="SLSQP", bounds=bounds, constraints=[cons],
                       options={"maxiter": max_iter, "ftol": 1e-12})
        viol = max(0.0, float((A @ res.x - 1).max()))
        tj_k, lam_k, tij_k = (v.detach().numpy() for v in unpack(res.x))
        sols.append((float(res.fun), CRMParams(tj_k, np.clip(lam_k, 0, 1), tij_k)))
        rows.append({"start": k, "objective": float(res.fun), "success": bool(res.success),
                     "status": int(res.status), "message": str(res.message), "nit": int(res.nit),
                     "max_constraint_violation": viol})
        if np.isfinite(res.fun) and (best is None or res.fun < best[1]):
            best = (res.x, float(res.fun))
    tj, lam, tij = (v.detach().numpy() for v in unpack(best[0]))
    table = pd.DataFrame(rows)
    table["rel_gap_to_best"] = table.objective / best[1] - 1.0
    return FitResult(CRMParams(tj, np.clip(lam, 0, 1), tij), best[1], table, (tau_min, tau_max),
                     sorted(sols, key=lambda s: s[0]))


def predict(data_full: CRMData, calib: CRMData, p: CRMParams) -> np.ndarray:
    """Forecast the full timeline using the calibration-window starts and the fitted CRM parameters.

    Parameters:
        data_full: Full dataset, including the holdout window.
        calib: Calibration dataset whose starts define the producer start rows.
        p: CRM parameter set to evaluate.

    Returns:
        An array of shape [N, J] giving the modelled production trajectory on the full timeline.

    Side effects:
        None beyond the temporary torch forward pass executed in no-grad mode.
    """
    J = calib.n_prod
    s = producer_starts(calib.production)
    fwd = CRMForward(data_full.time, data_full.injection, s, calib.production[s, np.arange(J)])
    with torch.no_grad():
        return fwd(*p.torch()).numpy()


# =============================================================================
# 2. BAYESIAN CRM (NUTS), PROPAGATION AND POSTERIOR PREDICTIVE
# =============================================================================
# Bayesian CRMP calibration with NUTS (NumPyro / JAX).
#
# Why NumPyro rather than the hand-written PyTorch HMC of v1: the CRM
# recurrence written with `lax.scan` and JIT costs well under a millisecond per
# gradient, versus ~13-55 ms in PyTorch, so we can afford NUTS with adapted
# step size and diagonal mass matrix, 4 chains, and long enough runs for honest
# R-hat / ESS / divergence diagnostics.  The JAX forward model is tested for
# equality against the torch forward (which is itself tested against a
# line-by-line MATLAB transcription).
#
# Model (per producer j, injector i):
#   log tau_j, log tau_ij ~ Normal(log 10, sd) truncated to [log tau_min, log tau_max]
#   (lambda_i1..lambda_iJ) ~ Dirichlet(alpha), with sum_j lambda_ij == 1 exactly
#   There is NO slack component in this Bayesian parameterisation. This differs from
#   deterministic calibration below, which permits sum_j lambda_ij <= 1; the audit CSV
#   makes this difference visible because it can shift the CRM features used by ML.
#   sigma_j ~ HalfNormal(scale_j)
#   normal:    q_obs ~ Normal(q_crm, sigma_j)                   rows after each producer's start
#   lognormal: log q_obs ~ Normal(log q_crm, sigma_j)           rows after start with q_obs > 0

def crm_jax(tau_j, lam, tau_ij, t, w, starts, q0):
    """Evaluate the CRM forward model in JAX using the same discrete-time recurrence as the torch implementation.

    Parameters:
        tau_j: Producer decay times with shape [J].
        lam: Injector-to-producer connectivity matrix with shape [I, J].
        tau_ij: Injector-specific decay times with shape [I, J].
        t: Time grid with shape [N].
        w: Injection matrix with shape [N, I].
        starts: Producer start-row indices with shape [J].
        q0: Initial production values at each producer start.

    Returns:
        A production forecast array with shape [N, J].

    Side effects:
        None; it is a pure JAX computation graph.
    """
    N = t.shape[0]
    n = jnp.arange(N)
    after = (n[:, None] > starts[None, :]).astype(t.dtype)          # [N, J]
    at = (n[:, None] == starts[None, :]).astype(t.dtype)
    elapsed = jnp.clip(t[:, None] - t[starts][None, :], 0.0)
    prod = q0[None, :] * jnp.exp(-elapsed / tau_j[None, :])
    c = jnp.exp(-jnp.diff(t)[:, None, None] / tau_ij[None])        # [N-1, I, J]

    def step(S, xs):
        c_n, w_n, a_n = xs
        S = a_n[None, :] * (c_n * S + (1.0 - c_n) * w_n[:, None])
        return S, S

    _, Ss = lax.scan(step, jnp.zeros(tau_ij.shape, t.dtype), (c, w[1:], after[1:]))
    resp = jnp.concatenate([jnp.zeros((1,) + tau_ij.shape, t.dtype), Ss])   # [N, I, J]
    inj = jnp.einsum("ij,nij->nj", lam, resp)
    return at * q0[None, :] + after * (prod + inj)


def crm_contrib_jax(lam, tau_ij, t, w, starts):
    """Compute the per-injector contribution tensor used in the machine-learning feature set.

    Parameters:
        lam: Injector-to-producer connectivity matrix with shape [I, J].
        tau_ij: Injector-specific decay times with shape [I, J].
        t: Time grid with shape [N].
        w: Injection matrix with shape [N, I].
        starts: Producer start-row indices with shape [J].

    Returns:
        A tensor with shape [N, J, I] containing the injector contributions for each producer.

    Side effects:
        None; it builds a pure JAX array from the supplied inputs.
    """
    N = t.shape[0]
    after = (jnp.arange(N)[:, None] > starts[None, :]).astype(t.dtype)
    c = jnp.exp(-jnp.diff(t)[:, None, None] / tau_ij[None])

    def step(S, xs):
        c_n, w_n, a_n = xs
        S = a_n[None, :] * (c_n * S + (1.0 - c_n) * w_n[:, None])
        return S, S

    _, Ss = lax.scan(step, jnp.zeros(tau_ij.shape, t.dtype), (c, w[1:], after[1:]))
    resp = jnp.concatenate([jnp.zeros((1,) + tau_ij.shape, t.dtype), Ss])
    return jnp.transpose(lam[None] * resp, (0, 2, 1))


def make_model(t, w, starts, q0, y, likelihood, tau_min, tau_max, tau_mu=np.log(10.0), tau_sd=2.0,
               alpha=1.0, sigma_scale=None):
    """Construct the NumPyro Bayesian CRM model for a chosen observation likelihood.

    Parameters:
        t: Time index with shape [N].
        w: Injection-rate matrix with shape [N, I].
        starts: Producer start rows with shape [J].
        q0: Initial production rates at each producer start.
        y: Observed production matrix with shape [N, J].
        likelihood: Observation model name; one of "normal", "lognormal", or "ar1".
        tau_min: Lower bound for all tau parameters in the latent domain.
        tau_max: Upper bound for all tau parameters in the latent domain.
        tau_mu: Prior mean for the log-tau parameters.
        tau_sd: Prior scale for the log-tau parameters.
        alpha: Dirichlet concentration parameter for the injector allocation.
        sigma_scale: Optional per-producer observation-noise scale for the HalfNormal prior.

    Returns:
        A tuple (model, n_obs, n_zero) where model is the NumPyro model function, n_obs is the number of valid post-start rows, and n_zero is the number of rows with non-positive production in the masked likelihood.

    Side effects:
        None; it only composes the probabilistic model structure and the observation mask.
    """
    I, J = w.shape[1], y.shape[1]
    N = len(t)
    t, w, starts, q0, y = (jnp.asarray(a) for a in (t, w, starts, q0, y))
    after = np.arange(N)[:, None] > np.asarray(starts)[None, :]
    prev_after = np.zeros_like(after)
    prev_after[1:] = after[:-1]
    if likelihood in ("normal", "ar1"):
        mask = jnp.asarray(after)
        ytr = y
    elif likelihood == "lognormal":
        mask = jnp.asarray(after & (np.asarray(y) > 0))
        ytr = jnp.log(jnp.where(mask, y, 1.0))
    else:
        raise ValueError(likelihood)
    if sigma_scale is None:
        sigma_scale = np.ones(J)
    lo, hi = np.log(tau_min), np.log(tau_max)

    def model():
        prior = dist.Normal(tau_mu, tau_sd)
        log_tau_j = numpyro.sample("log_tau_j", dist.Uniform(lo, hi).expand([J]).to_event(1))
        log_tau_ij = numpyro.sample("log_tau_ij", dist.Uniform(lo, hi).expand([I, J]).to_event(2))
        numpyro.factor("tau_prior", prior.log_prob(log_tau_j).sum() + prior.log_prob(log_tau_ij).sum())
        
        # REMOVED the + 1 to enforce sum(lambda) == 1.0 exactly (no slack)
        lam_full = numpyro.sample("lam_full", dist.Dirichlet(alpha * jnp.ones(J)).expand([I]).to_event(1))
        
        sigma = numpyro.sample("sigma", dist.HalfNormal(jnp.asarray(sigma_scale)).to_event(1))
        
        # Using lam_full directly since it is now exactly size J
        mu = crm_jax(jnp.exp(log_tau_j), lam_full, jnp.exp(log_tau_ij), t, w, starts, q0)
        
        if likelihood == "ar1":
            # Conditional AR(1) residual likelihood: e_n = y_n - mu_n,
            # e_n | e_{n-1} ~ N(phi_j e_{n-1}, sigma_j) for rows whose previous row is
            # also after the producer start; the first modelled row uses the
            # stationary sd sigma_j / sqrt(1 - phi_j^2).
            phi = numpyro.sample("phi", dist.Uniform(-0.99, 0.99).expand([J]).to_event(1))
            e = jnp.where(mask, ytr - mu, 0.0)
            e_prev = jnp.concatenate([jnp.zeros((1, J)), e[:-1]], axis=0)
            cond = jnp.asarray(prev_after)
            loc = mu + jnp.where(cond, phi[None, :] * e_prev, 0.0)
            scale = jnp.where(cond, sigma[None, :], sigma[None, :] / jnp.sqrt(1.0 - phi[None, :] ** 2))
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample("obs", dist.Normal(loc, scale), obs=ytr)
            return
            
        if likelihood == "normal":
            loc = mu
        else:
            loc = jnp.log(jnp.clip(mu, 1e-8))
            
        with numpyro.handlers.mask(mask=mask):
            numpyro.sample("obs", dist.Normal(loc, sigma[None, :]), obs=ytr)
    return model, int(np.asarray(mask).sum()), int((after & (np.asarray(y) <= 0)).sum())


def init_values(p, J, sigma, tau_min=None, tau_max=None, phi=None):
    """Construct interior NUTS starting values from a deterministic CRM solution."""
    if tau_min is not None:
        lo, hi = np.log(tau_min) + 1e-3, np.log(tau_max) - 1e-3
        p = type(p)(np.exp(np.clip(np.log(p.tau_j), lo, hi)), p.lam, np.exp(np.clip(np.log(p.tau_ij), lo, hi)))
    
    lam = np.clip(p.lam, 1e-4, None)
    # Force the exact sum to 1.0 without the slack variable
    full = lam / lam.sum(1, keepdims=True)
    
    return {"log_tau_j": jnp.log(jnp.asarray(p.tau_j)), "log_tau_ij": jnp.log(jnp.asarray(p.tau_ij)),
            "lam_full": jnp.asarray(full), "sigma": jnp.asarray(sigma),
            **({} if phi is None else {"phi": jnp.asarray(np.clip(phi, -0.9, 0.9))})}

def run_nuts(model, inits, warmup, samples, seed, target_accept=0.9, max_tree_depth=8, progress=False):
    """Run NUTS sampling for each chain and gather the posterior output.

    Parameters:
        model: NumPyro model to sample from.
        inits: Initial values for each chain.
        warmup: Number of warmup draws per chain.
        samples: Number of posterior draws retained per chain.
        seed: Base PRNG seed for the chain set.
        target_accept: NUTS target acceptance probability.
        max_tree_depth: Maximum tree depth allowed by the sampler.
        progress: Whether to show the per-chain progress bar.

    Returns:
        A tuple (grouped_samples, info) containing the stacked sample dictionary and the detailed per-chain diagnostics.

    Side effects:
        Runs the MCMC routine, which is computationally expensive and mutates the internal sampling state of the MCMC object.
    """
    chains, divs, info = [], [], []
    for c, init in enumerate(inits):
        kernel = NUTS(model, init_strategy=init_to_value(values=init), target_accept_prob=target_accept,
                      max_tree_depth=max_tree_depth, dense_mass=False)
        mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=progress)
        # This chain-level sampler exploration is the Bayesian uncertainty quantification step required for the posterior forecast.
        mcmc.run(jax.random.PRNGKey(seed + 1000 * c), extra_fields=("diverging", "num_steps", "accept_prob"))
        ex = mcmc.get_extra_fields()
        chains.append(mcmc.get_samples())
        divs.append(int(np.sum(ex["diverging"])))
        info.append({"chain": c, "divergences": divs[-1], "mean_accept": float(np.mean(ex["accept_prob"])),
                     "mean_leapfrog": float(np.mean(ex["num_steps"])), "max_leapfrog": int(np.max(ex["num_steps"]))})
    grouped = {k: np.stack([np.asarray(ch[k]) for ch in chains]) for k in chains[0]}
    return grouped, info


def diagnostics(grouped):
    """Summarise the posterior diagnostics for each scalar parameter in the grouped chain output.

    Parameters:
        grouped: Stacked posterior samples keyed by parameter name.

    Returns:
        A list of records with each parameter's R-hat and bulk ESS.

    Side effects:
        None; it only reads the grouped samples and computes summary statistics.
    """
    s = summary(grouped, group_by_chain=True)
    rows = []
    for name, d in s.items():
        rh = np.ravel(d["r_hat"]); ne = np.ravel(d["n_eff"])
        for k in range(rh.size):
            rows.append({"param": f"{name}[{k}]", "r_hat": float(rh[k]), "ess_bulk": float(ne[k])})
    return rows


def physical(grouped, J):
    """Convert the latent posterior draws into the physical CRM parameterisation.

    Parameters:
        grouped: Dictionary of stacked posterior samples from the MCMC run.
        J: Number of producers in the model.

    Returns:
        A tuple (tau_j, lam, tau_ij, sigma) with the physical parameter values for every draw.

    Side effects:
        None; it only transforms the latent unconstrained values to physically interpretable ones.
    """
    flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in grouped.items()}
    return (np.exp(flat["log_tau_j"]), flat["lam_full"][:, :, :J], np.exp(flat["log_tau_ij"]), flat["sigma"])


def propagate(draws, t, w, starts, q0):
    """Evaluate the CRM forward model on every posterior draw in the supplied parameter set.

    Parameters:
        draws: Tuple (tau_j, lam, tau_ij, sigma) for the posterior draws.
        t: Time index with shape [N].
        w: Injection-rate matrix with shape [N, I].
        starts: Producer start rows with shape [J].
        q0: Initial production rates at each producer start.

    Returns:
        A NumPy array of shape [D, N, J] containing the CRM trajectory for each posterior draw.

    Side effects:
        JIT-compiles the batched forward pass before executing it. No external state is mutated.
    """
    tau_j, lam, tau_ij, _ = draws
    f = jax.jit(jax.vmap(lambda a, b, c: crm_jax(a, b, c, jnp.asarray(t), jnp.asarray(w),
                                                 jnp.asarray(starts), jnp.asarray(q0))))
    return np.asarray(f(jnp.asarray(tau_j), jnp.asarray(lam), jnp.asarray(tau_ij)))


def predictive(mu, sigma, likelihood, starts, seed=0):
    """Posterior predictive: add observation noise to each draw's CRM trajectory.

    Normal predictive is truncated at zero (rates cannot be negative); the
    fraction of truncated values is returned so it can be reported.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(mu.shape)
    if likelihood == "normal":
        y = mu + sigma[:, None, :] * eps
        frac_neg = float(np.mean(y < 0))
        y = np.maximum(y, 0.0)
    else:
        y = np.clip(mu, 1e-8, None) * np.exp(sigma[:, None, :] * eps)
        frac_neg = 0.0
    N = mu.shape[1]
    before = np.arange(N)[:, None] <= np.asarray(starts)[None, :]
    y = np.where(before[None], mu, y)
    return y, frac_neg


def predictive_ar1(mu, sigma, phi, y, H, starts, seed=0):
    """AR(1) posterior predictive.

    History rows: one-step-ahead predictions mu_n + phi e_{n-1} + noise, using the
    observed previous residual (labelled as one-step-ahead, not a free run).
    Holdout rows (n >= H): multi-step simulation started from the last *historical*
    residual e_{H-1}; no holdout observation is used.  Truncated at zero.
    """
    rng = np.random.default_rng(seed)
    D, N, J = mu.shape
    st = np.asarray(starts)
    after = np.arange(N)[:, None] > st[None, :]
    out = np.array(mu, copy=True)
    e_obs = np.where(after[None], y[None, :N] - mu, 0.0)          # only rows < H are used below
    sd0 = sigma / np.sqrt(1 - phi ** 2)
    for n in range(1, H):
        prev_ok = after[n - 1]
        loc = mu[:, n] + np.where(prev_ok[None], phi * e_obs[:, n - 1], 0.0)
        sc = np.where(prev_ok[None], sigma, sd0)
        out[:, n] = np.where(after[n][None], loc + sc * rng.standard_normal((D, J)), mu[:, n])
    e = np.where(after[H - 1][None], y[None, H - 1] - mu[:, H - 1], 0.0)
    for n in range(H, N):
        e = phi * e + sigma * rng.standard_normal((D, J))
        out[:, n] = mu[:, n] + e
    frac_neg = float(np.mean(out[:, H:] < 0))
    return np.maximum(out, 0.0), frac_neg


# =============================================================================
# 3. POINT AND PROBABILISTIC METRICS
# =============================================================================
# Point and probabilistic forecast metrics, per producer and field.

def point(y, m):
    """Compute the scalar point-forecast metrics for an observed series and a matched prediction series.

    Parameters:
        y: Observed values with shape [N] or [N, J].
        m: Modelled values aligned to y.

    Returns:
        A dictionary with the mean absolute error, RMSE, and two R-squared summaries.

    Side effects:
        None; the function only computes summary statistics from the supplied arrays.
    """
    e = y - m
    out = {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2))}
    sst = np.sum((y - y.mean()) ** 2)
    out["R2_det"] = 1 - np.sum(e ** 2) / sst if sst > 0 else np.nan          # coefficient of determination
    c = np.corrcoef(y, m)[0, 1] if np.std(y) > 0 and np.std(m) > 0 else np.nan
    out["R2_corr"] = c ** 2                                                  # article Eq. 5
    return out


def interval(y, lo, hi, alpha):
    """Compute the interval coverage and the Winkler interval score for a predictive band.

    Parameters:
        y: Observed values with shape [N].
        lo: Lower predictive bound with shape [N].
        hi: Upper predictive bound with shape [N].
        alpha: Tail probability associated with the interval width.

    Returns:
        A dictionary containing interval coverage, average width, and interval score.

    Side effects:
        None; it synthesises forecast accuracy statistics from the supplied bounds.
    """
    width = hi - lo
    score = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)  # Winkler / interval score
    return {"PICP": np.mean((y >= lo) & (y <= hi)), "width": np.mean(width), "IS": np.mean(score)}


def crps_samples(y, S, max_draws=400, seed=0):
    """Approximate the CRPS from a sampled predictive ensemble.

    Parameters:
        y: Observed value or series with shape [N].
        S: Predictive samples with shape [D, N] or [D, N, ...].
        max_draws: Maximum ensemble size used before random downsampling.
        seed: Random seed used when subsampling a large ensemble.

    Returns:
        The mean CRPS value across the observed entries.

    Side effects:
        Draws a random subset when the ensemble exceeds max_draws.
    """
    if S.shape[0] > max_draws:
        S = S[np.random.default_rng(seed).choice(S.shape[0], max_draws, replace=False)]
    t1 = np.mean(np.abs(S - y[None]), axis=0)
    Ss = np.sort(S, axis=0)
    d = Ss.shape[0]
    k = np.arange(1, d + 1)[:, None]
    t2 = np.sum((2 * k - d - 1) * Ss, axis=0) / d ** 2          # = 0.5 E|X-X'|
    return float(np.mean(t1 - t2))


def table(y_all, point_pred, rows_by_producer, producers, samples=None, label=""):
    """Construct the per-producer and pooled summary metrics for one evaluation window.

    Parameters:
        y_all: Full observed production matrix with shape [N, J].
        point_pred: Point forecast matrix aligned to y_all.
        rows_by_producer: List of row index arrays, one per producer, for the evaluation window.
        producers: Producer names aligned to the columns of y_all.
        samples: Optional predictive sample array with shape [D, N, J].
        label: Human-readable label for the evaluation period.

    Returns:
        A pandas DataFrame containing the per-producer and pooled metrics for the supplied period.

    Side effects:
        None; the function only aggregates the forecast and sample summaries into a tabular report.
    """
    recs = []
    pooled_y, pooled_m, pooled_S = [], [], []
    for j, name in enumerate(producers):
        r = rows_by_producer[j]
        if len(r) < 2:
            continue
        y, m = y_all[r, j], point_pred[r, j]
        rec = {"period": label, "producer": name, "n": len(r), **point(y, m)}
        if samples is not None:
            S = samples[:, r, j]
            for a in (0.2, 0.05):
                lo, hi = np.quantile(S, [a / 2, 1 - a / 2], axis=0)
                rec.update({f"{k}_{int(100 * (1 - a))}": v for k, v in interval(y, lo, hi, a).items()})
            rec["CRPS"] = crps_samples(y, S)
            pooled_S.append(S)
        recs.append(rec)
        pooled_y.append(y); pooled_m.append(m)
    y, m = np.concatenate(pooled_y), np.concatenate(pooled_m)
    rec = {"period": label, "producer": "FIELD(pooled)", "n": len(y), **point(y, m)}
    if samples is not None:
        S = np.concatenate(pooled_S, axis=1)
        for a in (0.2, 0.05):
            lo, hi = np.quantile(S, [a / 2, 1 - a / 2], axis=0)
            rec.update({f"{k}_{int(100 * (1 - a))}": v for k, v in interval(y, lo, hi, a).items()})
        rec["CRPS"] = crps_samples(y, S)
    recs.append(rec)
    return pd.DataFrame(recs)


# =============================================================================
# 4. MACHINE-LEARNING LAYER (XGB, ELM, MLP, NuSVM) ON TOP OF THE BAYESIAN CRM
# =============================================================================
# Why the ML layer is built this way
# ----------------------------------
# The article feeds lambda_ij, tau_ij and tau_j to the ML model as input columns.
# In a per-producer model those columns hold the SAME value on every row, so the
# ML model cannot learn anything from them and swapping in a different posterior
# draw changes nothing (we measured exactly 0.0 change for XGB).  Uncertainty in
# the CRM parameters would then never reach the ML forecast.
#
# Here the CRM enters through inputs that change with time AND with theta:
#     x_j(n; theta) = [ q_crm_j(n; theta),  lambda_ij * F(w_i; tau_ij)(n) for each i,  w_i(n) ]
# where F is the CRM's exponential filter of injector i's rate.  The ML model
# learns the part of production the CRM misses (the residual y - q_crm).  For a
# posterior draw theta_d the hybrid forecast is
#     q_hybrid(n; theta_d) = q_crm(n; theta_d) + g(x(n; theta_d)).
# The model g is trained once, on features from the deterministic CRM fit
# (one coherent parameter set), using history rows only.  Every posterior draw is then pushed
# through BOTH the CRM and g, so each draw gives a different hybrid trajectory.
# Hyperparameters and the hybrid noise level are chosen on a time-ordered split
# inside the history (last 20% of each producer's history), never on the holdout.

ML_CHOICES = ("xgb", "elm", "mlp", "nusvm")
ML_NAMES = {"xgb": "CRM-XGB", "elm": "CRM-ELM", "mlp": "CRM-MLP", "nusvm": "CRM-NuSVM"}

XGB_PARAMS = dict(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                  min_child_weight=5, reg_lambda=1.0, random_state=0, n_jobs=1)


class ELMRegressor(BaseEstimator, RegressorMixin):
    """Extreme Learning Machine: random tanh hidden layer, ridge-solved output weights.

    Hidden weights and biases are drawn once from U(-1, 1) (as in the article's
    description of ELM) and never trained; only the output layer is fitted, by
    regularised least squares (a small ridge term keeps the solve stable).
    """

    def __init__(self, n_hidden=50, alpha=1e-2, random_state=0):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.random_state = random_state

    def _hidden(self, X):
        return np.tanh(X @ self.W_ + self.b_)

    def fit(self, X, y):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.uniform(-1.0, 1.0, (X.shape[1], self.n_hidden))
        self.b_ = rng.uniform(-1.0, 1.0, self.n_hidden)
        Hm = self._hidden(X)
        A = Hm.T @ Hm + self.alpha * np.eye(self.n_hidden)
        self.beta_ = np.linalg.solve(A, Hm.T @ np.asarray(y, float))
        return self

    def predict(self, X):
        return self._hidden(np.asarray(X, float)) @ self.beta_


# Small hyperparameter grids.  For every producer, the setting with the lowest
# validation RMSE on the LAST 20% OF THAT PRODUCER'S HISTORY is chosen (time-ordered,
# no shuffling), then refitted on the full history.  The holdout is never used.
# The grids span weak to strong regularisation because, without it, ELM and MLP
# turn small changes in the CRM inputs into very large swings in the output.
HP_GRID = {
    "xgb": [dict(max_depth=d, n_estimators=n) for d in (2, 3) for n in (100, 300)],
    "elm": [dict(n_hidden=h, alpha=a) for h in (10, 25, 50) for a in (0.1, 1.0, 10.0)],
    "mlp": [dict(alpha=a) for a in (1e-3, 1e-1, 1.0, 10.0)],
    "nusvm": [dict(C=c) for c in (0.1, 1.0, 10.0)],
}


def make_ml(name, seed=0, **hp):
    """Return an unfitted regressor.  Inputs and targets are standardised for the
    scale-sensitive models (ELM, MLP, NuSVM); XGB is scale-invariant."""
    if name == "xgb":
        return xgb.XGBRegressor(**{**XGB_PARAMS, "random_state": seed, **hp})
    if name == "elm":
        base = ELMRegressor(n_hidden=hp.get("n_hidden", 25), alpha=hp.get("alpha", 1.0), random_state=seed)
    elif name == "mlp":
        # One hidden layer with 10 neurons, as selected in the article.
        base = MLPRegressor(hidden_layer_sizes=(10,), activation="tanh", alpha=hp.get("alpha", 1e-1),
                            max_iter=3000, random_state=seed)
    elif name == "nusvm":
        base = NuSVR(nu=0.5, C=hp.get("C", 1.0), kernel="rbf", gamma="scale")
    else:
        raise ValueError(f"unknown ML model {name!r}; choose from {ML_CHOICES}")
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), base),
                                      transformer=StandardScaler())


def ml_features(mu, contrib, w, j, rows):
    """mu [N,J], contrib [N,J,I], w [N,I] -> X [len(rows), 1 + 2I] for producer j."""
    return np.column_stack([mu[rows, j], contrib[rows, j, :], w[rows, :]])


class ResidualML:
    """Per-producer ML model of the CRM residual, trained on history rows only."""

    def __init__(self, name, seed=0, inner_frac=0.8):
        self.name, self.seed, self.inner_frac = name, seed, inner_frac
        self.models, self.sigma = {}, {}
        self.diagnostics = []

    def fit(self, y, mu_hat, contrib_hat, w, starts, H):
        """Fit one residual model per producer and audit it against an uncorrected CRM.

        A corrector is retained only when its inner chronological validation RMSE is
        strictly below the CRM-only validation RMSE. This prevents a weak corrector
        from being forced into the hybrid merely because one must be selected.
        """
        self.chosen = {}
        self.diagnostics = []
        for j in range(y.shape[1]):
            rows = np.arange(starts[j] + 1, H)
            if len(rows) < 4:
                raise ValueError(
                    f"producer {j}: only {len(rows)} post-start history rows; "
                    "need at least 4 to tune a residual model safely"
                )
            X = ml_features(mu_hat, contrib_hat, w, j, rows)
            r = y[rows, j] - mu_hat[rows, j]
            k = max(1, min(len(rows) - 1, int(len(rows) * self.inner_frac)))
            crm_train_rmse = float(np.sqrt(np.mean(r[:k] ** 2)))
            crm_val_rmse = float(np.sqrt(np.mean(r[k:] ** 2)))
            best = None
            for hp in HP_GRID[self.name]:
                inner = make_ml(self.name, self.seed, **hp).fit(X[:k], r[:k])
                corrected_val_rmse = float(np.sqrt(np.mean((r[k:] - inner.predict(X[k:])) ** 2)))
                corrected_train_rmse = float(np.sqrt(np.mean((r[:k] - inner.predict(X[:k])) ** 2)))
                if best is None or corrected_val_rmse < best[0]:
                    best = (corrected_val_rmse, hp, corrected_train_rmse)

            # Also evaluate the selected candidate after refitting on the full history.
            # This is a diagnostic in-sample number, not an unbiased performance estimate.
            candidate = make_ml(self.name, self.seed, **best[1]).fit(X, r)
            candidate_train_rmse = float(np.sqrt(np.mean((r - candidate.predict(X)) ** 2)))
            improvement = crm_val_rmse - best[0]
            use_corrector = bool(improvement > max(1e-12, 1e-6 * crm_val_rmse))
            self.models[j] = candidate if use_corrector else None
            self.sigma[j] = best[0] if use_corrector else crm_val_rmse
            self.chosen[j] = {"enabled": use_corrector, "hyperparameters": best[1],
                              "crm_inner_validation_RMSE": crm_val_rmse,
                              "corrected_inner_validation_RMSE": best[0]}
            used_train_rmse = candidate_train_rmse if use_corrector else float(np.sqrt(np.mean(r ** 2)))
            self.diagnostics.append({
                "model": self.name, "producer_index": j, "n_history_rows": int(len(rows)),
                "n_inner_train": int(k), "n_inner_validation": int(len(rows) - k),
                "crm_inner_train_RMSE": crm_train_rmse,
                "candidate_corrected_inner_train_RMSE": best[2],
                "crm_inner_validation_RMSE": crm_val_rmse,
                "candidate_corrected_inner_validation_RMSE": best[0],
                "inner_validation_RMSE_improvement": improvement,
                "inner_validation_RMSE_improvement_pct": (100.0 * improvement / crm_val_rmse
                                                            if crm_val_rmse > 0 else np.nan),
                "candidate_corrected_full_history_in_sample_RMSE": candidate_train_rmse,
                "crm_full_history_RMSE": float(np.sqrt(np.mean(r ** 2))),
                "used_corrected_full_history_RMSE": used_train_rmse,
                "corrector_enabled": use_corrector,
                "selected_hyperparameters": json.dumps(best[1], sort_keys=True),
            })
        return self

    def predict_draws(self, mu, contrib, w, starts):
        """mu [D,N,J], contrib [D,N,J,I] -> hybrid mean [D,N,J].

        A producer with no validation improvement receives no correction (CRM only).
        """
        D, N, J = mu.shape
        out = mu.copy()
        for j, m in self.models.items():
            if m is None:
                continue
            rows = np.arange(starts[j] + 1, N)
            X = np.concatenate([ml_features(mu[d], contrib[d], w, j, rows) for d in range(D)])
            out[:, rows, j] = mu[:, rows, j] + m.predict(X).reshape(D, len(rows))
        return out


def ml_only(name, y, w, starts, H, N, seed=0, inner_frac=0.8):
    """Plain per-producer ML on injection rates + months since start (the article's
    'ML' comparison, minus well distances which are constant per producer).
    Same history-only hyperparameter selection as the hybrid, for a fair comparison."""
    out = np.zeros((N, y.shape[1]))
    for j in range(y.shape[1]):
        rows = np.arange(starts[j] + 1, N)
        tr = rows[rows < H]
        X = np.column_stack([w, np.arange(N) - starts[j]])
        k = int(len(tr) * inner_frac)
        best = min(HP_GRID[name], key=lambda hp: np.mean(
            (y[tr[k:], j] - make_ml(name, seed, **hp).fit(X[tr[:k]], y[tr[:k], j]).predict(X[tr[k:]])) ** 2))
        out[rows, j] = make_ml(name, seed, **best).fit(X[tr], y[tr, j]).predict(X[rows])
    return out


# =============================================================================
# 5. PERCENTILES, BEST-FITTING DRAW, PLOTS AND THE END-TO-END PIPELINE
# =============================================================================

def log(msg):
    """Print a timestamped diagnostic message to standard output.

    Parameters:
        msg: Message text to emit to the terminal.

    Returns:
        None.

    Side effects:
        Writes to stdout in a machine-readable log format.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_name(s):
    """Convert a label to a filesystem-safe basename by replacing unsupported characters.

    Parameters:
        s: Input string to sanitise.

    Returns:
        A version of the string suitable for plot filenames and CSV naming.

    Side effects:
        None; it only transforms the provided string.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).strip("_")


def best_fitting_draw(mu, y, starts, H):
    """Index of the posterior draw whose CRM trajectory has the lowest history MSE
    (article Eq. 2), using rows after each producer's start and before H only.

    Each posterior draw is already a joint combination (lambda, tau_ij, tau_j), so
    this answers "which combination fits the training data best" without mixing
    parameters from different draws (mixing would break the correlations the
    posterior learned).  It is a reference point estimate, not the uncertainty."""
    N = mu.shape[1]
    after = (np.arange(N)[:, None] > np.asarray(starts)[None, :]) & (np.arange(N)[:, None] < H)
    err = ((mu - y[None]) ** 2 * after[None]).sum(axis=(1, 2)) / after.sum()
    return int(np.argmin(err)), err


def percentiles(S):
    """S [D, N, J] -> dict of P10, P50, P90 (statistical percentiles: P10 is the low curve)."""
    q = np.quantile(S, [0.10, 0.50, 0.90], axis=0)
    return {"P10": q[0], "P50": q[1], "P90": q[2]}


def p_labels(convention):
    """Map the statistical percentiles to the labels shown on plots and in CSVs.

    percentile (default): P10 = 10th percentile (low), P90 = 90th percentile (high).
    reserves (SPE/PRMS style): P90 = value exceeded with 90% probability (low),
    P10 = value exceeded with 10% probability (high).  Same curves, swapped names."""
    if convention == "reserves":
        return {"P10": "P90", "P50": "P50", "P90": "P10"}
    return {"P10": "P10", "P50": "P50", "P90": "P90"}


def plot_producer(path, t, H, obs, start, band, param_band, best, title, labels, ref=None, ref_label=None):
    """One figure for one producer: observed rates, Bayesian P10/P50/P90, the
    parameter-only P10-P90 band, the best-fitting posterior draw, holdout marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, blue_dark, orange, ink, muted = "#2a78d6", "#184f95", "#eb6834", "#1f1f1e", "#8a8984"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axvspan(t[H] - 0.5, t[-1] + 0.5, color=muted, alpha=0.08, lw=0)
    ax.axvline(t[H] - 0.5, color=muted, lw=1, ls="--")
    ax.text(t[H] - 0.5, 1.0, "  holdout", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=muted)
    ax.fill_between(t, band["P10"], band["P90"], color=blue, alpha=0.15, lw=0,
                    label=f"{labels['P10']}-{labels['P90']} predictive (parameters + noise)")
    if param_band is not None:
        ax.fill_between(t, param_band["P10"], param_band["P90"], color=blue, alpha=0.35, lw=0,
                        label=f"{labels['P10']}-{labels['P90']} from CRM parameters only")
    ax.plot(t, band["P90"], color=blue_dark, lw=1.0, ls="--", label=labels["P90"])
    ax.plot(t, band["P50"], color=blue, lw=1.0, label=labels["P50"])
    ax.plot(t, band["P10"], color=blue_dark, lw=1.0, ls=":", label=labels["P10"])
    if ref is not None:
        ax.plot(t, ref, color=muted, lw=1.0, label=ref_label)
    ax.plot(t, best, color=orange, lw=1.0, ls="-.", label="best-fitting posterior draw (history)")
    ax.plot(t[start:], obs[start:], color=ink, lw=1.0, marker="o", ms=2.5, label="observed")
    ax.set_title(title, fontsize=11, color=ink, loc="left")
    ax.set_xlabel("Time (Unit)")
    ax.set_ylabel("q (Unit)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=2400)
    plt.close(fig)


def plot_parameter_posteriors(path, tau_j, lam, tau_ij, j, pname, injectors):
    """Plot the posterior distributions of CRM parameters for a specific producer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    I = len(injectors)
    fig = plt.figure(figsize=(10, 3 * (I + 1)))
    
    # Plot tau_j on the top row spanning all columns
    ax_tau_j = plt.subplot(I + 1, 1, 1)
    ax_tau_j.hist(tau_j, bins=50, density=True, color="#2a78d6", alpha=0.7, edgecolor='none')
    ax_tau_j.set_title(f"Producer Time Constant: tau_j")
    ax_tau_j.grid(alpha=0.15)
    ax_tau_j.set_ylabel("Density")
    ax_tau_j.set_xlabel("Value")
    
    # Plot lambda_ij and tau_ij for each injector
    for i, iname in enumerate(injectors):
        ax_lam = plt.subplot(I + 1, 2, 2 * (i + 1) + 1)
        ax_lam.hist(lam[:, i], bins=50, density=True, color="#184f95", alpha=0.7, edgecolor='none')
        ax_lam.set_title(f"Connectivity: lambda ({iname} -> {pname})")
        ax_lam.grid(alpha=0.15)
        ax_lam.set_ylabel("Density")
        ax_lam.set_xlabel("Value")
        
        ax_tij = plt.subplot(I + 1, 2, 2 * (i + 1) + 2)
        ax_tij.hist(tau_ij[:, i], bins=50, density=True, color="#eb6834", alpha=0.7, edgecolor='none')
        ax_tij.set_title(f"Delay: tau_ij ({iname} -> {pname})")
        ax_tij.grid(alpha=0.15)
        ax_tij.set_ylabel("Density")
        ax_tij.set_xlabel("Value")
    
    fig.suptitle(f"Posterior Parameter Distributions: {pname}", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=2400)
    plt.close(fig)


def build_parser(description):
    """Create the CLI argument parser for the CRM-ML Bayesian analysis workflow.

    Parameters:
        description: Human-readable description shown in the help text.

    Returns:
        An argparse.ArgumentParser configured for the CRM-ML runner and self-test mode.

    Side effects:
        None beyond constructing the parser object; it does not parse arguments or write files.
    """
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", help="Excel workbook with sheets time, q_observed, w_injection")
    ap.add_argument("--forecast", type=int, default=12, help="number of final rows held out (never used for fitting)")
    ap.add_argument("--tau-min", type=float, default=1 / 30.44,
                    help="minimum tau in data time units (article: 1 day = 1/30.44 month)")
    ap.add_argument("--starts", type=int, default=8, help="deterministic multi-start count")
    ap.add_argument("--max-iter", type=int, default=1000, help="SLSQP iterations per start")
    ap.add_argument("--likelihood", default="normal", choices=["normal", "ar1", "lognormal"],
                    help="observation model for the Bayesian CRM")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000, help="posterior draws kept per chain")
    ap.add_argument("--tau-prior-sd", type=float, default=2.0, help="sd of the Normal(log 10, sd) prior on log tau")
    ap.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration for each injector's lambda row")
    ap.add_argument("--p-convention", default="percentile", choices=["percentile", "reserves"],
                    help="label curves as statistical percentiles (P10 low) or reserves style (P90 low)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default=None, help="output folder")
    ap.add_argument("--self-test", action="store_true", help="run the built-in regression tests and exit")
    return ap


def run_pipeline(a, ml_models=("xgb",)):
    """Run the full CRM-ML workflow: deterministic CRM fit, Bayesian posterior sampling, and optional ML residual model fit.

    Parameters:
        a: Parsed CLI arguments containing the dataset path, model choices, and output settings.
        ml_models: Iterable of ML model names to fit on top of the Bayesian CRM forecast.

    Returns:
        A summary dictionary containing the convergence indicators, output paths, and model statistics.

    Side effects:
        Reads the input workbook, writes CSVs/plots into the output directory, and may take several minutes when sampling the posterior.
    """
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    json.dump(vars(a), open(out / "run_settings.json", "w"), indent=1)
    labels = p_labels(a.p_convention)

    d = load_excel(a.data)
    N, I, J = len(d.time), d.n_inj, d.n_prod
    H = N - a.forecast
    if H < 10:
        raise ValueError("forecast too long: fewer than 10 history rows left")
    hist = d.head(H)
    s = producer_starts(hist.production)
    q0 = hist.production[s, np.arange(J)]
    y = d.production
    rows_hist = [np.arange(s[j] + 1, H) for j in range(J)]
    rows_fc = [np.arange(H, N) for _ in range(J)]
    log(f"N={N} history={H} holdout={N - H}; injectors={I} producers={J}")
    log(f"producer start rows: {dict(zip(d.producers, s.tolist()))}")

    # ---- Step 1: deterministic CRM (baseline and sampler start points)
    import pickle
    cache = out / f"det_fit_H{H}_s{a.starts}_it{a.max_iter}_seed{a.seed}_tmin{a.tau_min:.6g}.pkl"
    t0 = time.time()
    if cache.exists():
        fit = pickle.load(open(cache, "rb"))
        log(f"deterministic fit loaded from {cache.name}")
    else:
        fit = fit_deterministic(hist, a.tau_min, None, a.starts, a.max_iter, a.seed)
        pickle.dump(fit, open(cache, "wb"))
    fit.starts_table.to_csv(out / "deterministic_starts.csv", index=False)
    log(f"deterministic CRM: best objective {fit.objective:.5g} ({time.time() - t0:.0f}s); "
        f"converged starts {int(fit.starts_table.success.sum())}/{a.starts}")
    det = predict(d, hist, fit.params)

    # ---- Step 2: Bayesian posterior of (lambda_ij, tau_ij, tau_j) with NUTS
    lik = a.likelihood
    resid = []
    for j in range(J):
        r = rows_hist[j]
        if lik in ("normal", "ar1"):
            resid.append(np.std(y[r, j] - det[r, j]))
        else:
            pos = r[y[r, j] > 0]
            resid.append(np.std(np.log(y[pos, j]) - np.log(np.clip(det[pos, j], 1e-8, None))))
    resid = np.maximum(np.array(resid), 1e-3)
    scale = (np.array([np.std(y[rows_hist[j], j]) for j in range(J)]) if lik in ("normal", "ar1") else np.ones(J))
    model, n_obs, n_zero = make_model(hist.time, hist.injection, s, q0, hist.production, lik, a.tau_min,
                                      4.5 * H, np.log(10.0), a.tau_prior_sd, a.alpha, scale)
    phi0 = None
    if lik == "ar1":
        phi0 = np.array([np.corrcoef((y[rows_hist[j], j] - det[rows_hist[j], j])[:-1],
                                     (y[rows_hist[j], j] - det[rows_hist[j], j])[1:])[0, 1] for j in range(J)])
        resid = resid * np.sqrt(1 - np.clip(phi0, -0.9, 0.9) ** 2)
    sols = [sp for _, sp in fit.solutions]
    inits = [init_values(sols[c % len(sols)], J, resid, a.tau_min, 4.5 * H, phi0) for c in range(a.chains)]
    log(f"NUTS ({lik} likelihood): {a.chains} chains x ({a.warmup} warmup + {a.samples} draws), {n_obs} data rows")
    t0 = time.time()
    grouped, info = run_nuts(model, inits, a.warmup, a.samples, a.seed)
    diag = pd.DataFrame(diagnostics(grouped))
    n_div = sum(x["divergences"] for x in info)
    converged = bool(diag.r_hat.max() < 1.01 and diag.ess_bulk.min() > 400 and n_div == 0)
    log(f"NUTS done ({time.time() - t0:.0f}s): max R-hat {diag.r_hat.max():.3f}, min ESS {diag.ess_bulk.min():.0f}, "
        f"divergences {n_div} -> {'CONVERGED' if converged else 'NOT CONVERGED: treat results as provisional'}")
    diag.to_csv(out / "mcmc_diagnostics.csv", index=False)
    pd.DataFrame(info).to_csv(out / "mcmc_chains.csv", index=False)
    np.savez_compressed(out / "posterior_samples.npz", **grouped)

    tau_j, lam, tau_ij, sig = physical(grouped, J)
    phi = grouped["phi"].reshape(-1, J) if lik == "ar1" else None
    D = len(tau_j)
    
    # NEW CODE: Generate and save posterior distributions for CRM parameters
    for j, pname in enumerate(d.producers):
        f_post = out / "plots" / f"Posterior_Params_{safe_name(pname)}.svg"
        plot_parameter_posteriors(f_post, tau_j[:, j], lam[:, :, j], tau_ij[:, :, j], j, pname, d.injectors)

    names = ([f"tau_j[{n}]" for n in d.producers] + [f"lambda[{i}->{p}]" for i in d.injectors for p in d.producers]
             + [f"tau_ij[{i}->{p}]" for i in d.injectors for p in d.producers] + [f"sigma[{n}]" for n in d.producers]
             + ([f"phi[{n}]" for n in d.producers] if phi is not None else []))
    flat = np.column_stack([tau_j, lam.reshape(D, -1), tau_ij.reshape(D, -1), sig] + ([phi] if phi is not None else []))
    qs = np.quantile(flat, [0.10, 0.50, 0.90], axis=0)
    pd.DataFrame({"parameter": names, labels["P10"]: qs[0], labels["P50"]: qs[1], labels["P90"]: qs[2],
                  "mean": flat.mean(0), "sd": flat.std(0),
                  "deterministic": np.r_[fit.params.tau_j, fit.params.lam.ravel(), fit.params.tau_ij.ravel(),
                                         np.full(J, np.nan)] .tolist() + ([np.nan] * J if phi is not None else [])}
                 ).to_csv(out / "posterior_parameters.csv", index=False)

    # ---- Step 3: push every posterior draw through the CRM on the full timeline
    mu = propagate((tau_j, lam, tau_ij, sig), d.time, d.injection, s, q0)            # [D, N, J]
    if lik == "ar1":
        ypred, _ = predictive_ar1(mu, sig, phi, y, H, s, seed=a.seed)
    else:
        ypred, _ = predictive(mu, sig, lik, s, seed=a.seed)
    k_best, _ = best_fitting_draw(mu, y, s, H)
    best = mu[k_best]
    pd.DataFrame({"parameter": names, "best_fitting_draw": flat[k_best]}).to_csv(out / "best_fitting_draw.csv", index=False)
    pp, pr = percentiles(ypred), percentiles(mu)
    # Preserve full holdout predictive draws so a later stacking stage can combine
    # predictive distributions rather than trying to reconstruct them from P10/P50/P90.
    predictive_export = {"time": np.asarray(d.time[H:]),
                         "producers": np.asarray(d.producers, dtype=str),
                         "crm_predictive": np.asarray(ypred[:, H:, :]),
                         "crm_parameter_only": np.asarray(mu[:, H:, :])}
    log(f"propagated {D} posterior draws; best-fitting draw index {k_best}")

    # Audit the deterministic-vs-Bayesian connectivity constraint used by this run.
    det_lam_sum = np.asarray(fit.params.lam).sum(axis=1)
    post_lam_sum = lam.sum(axis=2)  # [draw, injector]; Dirichlet without slack forces 1.0
    pd.DataFrame({"injector": d.injectors,
                  "deterministic_lambda_sum": det_lam_sum,
                  "posterior_lambda_sum_mean": post_lam_sum.mean(axis=0),
                  "posterior_lambda_sum_sd": post_lam_sum.std(axis=0),
                  "posterior_lambda_sum_min": post_lam_sum.min(axis=0),
                  "posterior_lambda_sum_max": post_lam_sum.max(axis=0)}).to_csv(
                      out / "connectivity_constraint_audit.csv", index=False)

    tables = []
    for label, rows in (("history", rows_hist), ("holdout", rows_fc)):
        tables += [table(y, det, rows, d.producers, label=label).assign(model="CRM deterministic"),
                   table(y, best, rows, d.producers, label=label).assign(model="CRM best-fitting draw"),
                   table(y, pp["P50"], rows, d.producers, ypred, label).assign(model="Bayesian CRM predictive"),
                   table(y, pr["P50"], rows, d.producers, mu, label).assign(model="Bayesian CRM parameters only")]

    long = []
    for j, pname in enumerate(d.producers):
        for n in range(N):
            long.append({"producer": pname, "time": d.time[n], "period": "history" if n < H else "holdout",
                         "observed": y[n, j], "crm_deterministic": det[n, j], "crm_best_fitting_draw": best[n, j],
                         **{f"crm_{labels[k]}": pp[k][n, j] for k in ("P10", "P50", "P90")},
                         **{f"crm_param_only_{labels[k]}": pr[k][n, j] for k in ("P10", "P50", "P90")}})
    ts = pd.DataFrame(long)

    plots = []
    for j, pname in enumerate(d.producers):
        f = out / "plots" / f"CRM_{safe_name(pname)}.svg"
        plot_producer(f, d.time, H, y[:, j], s[j], {k: v[:, j] for k, v in pp.items()},
                      {k: v[:, j] for k, v in pr.items()}, best[:, j],
                      f"{pname}: Bayesian CRM ({lik} likelihood)", labels)
        plots.append(f)

    # ---- Step 4 (ML program only): ML layer, every draw pushed through CRM + ML
    summary_ml = {}
    if ml_models:
        Dm = min(a.ml_draws, D)
        idx = np.random.default_rng(a.seed).choice(D, Dm, replace=False)
        cfun = jax.jit(jax.vmap(lambda l, tij: crm_contrib_jax(l, tij, jnp.asarray(d.time), jnp.asarray(d.injection),
                                                              jnp.asarray(s))))
        contrib = np.asarray(cfun(jnp.asarray(lam[idx]), jnp.asarray(tau_ij[idx])))         # [Dm, N, J, I]
        # The ML model is trained on CRM features from ONE coherent parameter set: the
        # article-style deterministic calibration.  (Taking each parameter's posterior
        # median separately would mix values from different draws into a combination
        # the posterior never proposed.)
        mu_ref = det
        contrib_ref = np.asarray(crm_contrib_jax(jnp.asarray(fit.params.lam), jnp.asarray(fit.params.tau_ij),
                                                 jnp.asarray(d.time), jnp.asarray(d.injection), jnp.asarray(s)))
        before = np.arange(N)[:, None] <= s[None, :]
        for name in ml_models:
            t0 = time.time()
            tag = ML_NAMES[name]
            hy = ResidualML(name, a.seed).fit(y, mu_ref, contrib_ref, d.injection, s, H)
            diag_ml = pd.DataFrame(hy.diagnostics)
            diag_ml["producer"] = [d.producers[int(j)] for j in diag_ml["producer_index"]]
            diag_ml.to_csv(out / f"{name}_residual_diagnostics.csv", index=False)
            hyb = hy.predict_draws(mu[idx], contrib, d.injection, s)                          # [Dm, N, J]
            # Store predictive samples over the outer holdout for subsequent stacking.
            predictive_export[f"{name}_predictive"] = None  # populated after observation noise is added
            # Quantify the feature-distribution shift between the deterministic reference
            # used to train the corrector and the Bayesian CRM draws to which it is applied.
            shift_rows = []
            for j, pname in enumerate(d.producers):
                rr = rows_hist[j]
                if len(rr) == 0:
                    continue
                qshift = np.sqrt(np.mean((mu[idx][:, rr, j] - mu_ref[rr, j][None, :]) ** 2, axis=1))
                cdelta = contrib[:, rr, j, :] - contrib_ref[rr, j, :][None, :, :]
                cshift = np.sqrt(np.mean(cdelta ** 2, axis=(1, 2)))
                ref_scale = float(np.std(mu_ref[rr, j]))
                shift_rows.append({"model": name, "producer": pname,
                                   "crm_feature_RMSE_shift_mean": float(qshift.mean()),
                                   "crm_feature_RMSE_shift_median": float(np.median(qshift)),
                                   "crm_feature_RMSE_shift_p90": float(np.quantile(qshift, 0.90)),
                                   "crm_feature_shift_over_reference_sd": float(qshift.mean() / ref_scale) if ref_scale > 0 else np.nan,
                                   "injector_contribution_RMSE_shift_mean": float(cshift.mean()),
                                   "injector_contribution_RMSE_shift_p90": float(np.quantile(cshift, 0.90))})
            pd.DataFrame(shift_rows).to_csv(out / f"{name}_posterior_feature_shift.csv", index=False)
            sh = np.array([hy.sigma[j] for j in range(J)])
            rng = np.random.default_rng(a.seed + 7)
            hyb_pred = np.where(before[None], hyb, np.maximum(hyb + sh[None, None, :] * rng.standard_normal(hyb.shape), 0.0))
            predictive_export[f"{name}_predictive"] = np.asarray(hyb_pred[:, H:, :])
            ph, phr = percentiles(hyb_pred), percentiles(hyb)
            k_hb, _ = best_fitting_draw(hyb, y, s, H)
            mlo = ml_only(name, y, d.injection, s, H, N, a.seed)
            for label, rows in (("history", rows_hist), ("holdout", rows_fc)):
                tables += [table(y, ph["P50"], rows, d.producers, hyb_pred, label).assign(model=f"{tag} predictive"),
                           table(y, phr["P50"], rows, d.producers, hyb, label).assign(model=f"{tag} parameters only"),
                           table(y, mlo, rows, d.producers, label=label).assign(model=f"{name.upper()} only (no CRM)")]
            for k in ("P10", "P50", "P90"):
                ts[f"{name}_{labels[k]}"] = np.concatenate([ph[k][:, j] for j in range(J)])
            ts[f"{name}_best_fitting_draw"] = np.concatenate([hyb[k_hb][:, j] for j in range(J)])
            spread = hyb[:, H:, :].std(0).mean(0)
            summary_ml[tag] = {"draws_propagated": int(Dm), "hybrid_noise_sd": dict(zip(d.producers, sh.round(4).tolist())),
                               "chosen_hyperparameters": {d.producers[j]: hy.chosen[j] for j in range(J)},
                               "holdout_sd_of_mean_across_draws": dict(zip(d.producers, spread.round(4).tolist()))}
            for j, pname in enumerate(d.producers):
                f = out / "plots" / f"{tag}_{safe_name(pname)}.svg"
                plot_producer(f, d.time, H, y[:, j], s[j], {k: v[:, j] for k, v in ph.items()},
                              {k: v[:, j] for k, v in phr.items()}, hyb[k_hb][:, j],
                              f"{pname}: {tag} on Bayesian CRM posterior ({lik} likelihood)", labels,
                              ref=pp["P50"][:, j], ref_label=f"CRM-only {labels['P50']}")
                plots.append(f)
            log(f"{tag}: {Dm} draws through CRM + {name.upper()} ({time.time() - t0:.0f}s)")

    np.savez_compressed(out / "holdout_predictive_draws.npz", **predictive_export)
    ts.to_csv(out / "percentile_timeseries.csv", index=False)
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(out / "metrics.csv", index=False)
    summary = {"likelihood": lik, "history_rows": H, "holdout_rows": N - H, "posterior_draws": D,
               "deterministic_objective": fit.objective,
               "deterministic_max_injector_lambda_sum": float(np.asarray(fit.params.lam).sum(axis=1).max()),
               "bayesian_connectivity_sum_is_exactly_one": True,
               "max_rhat": float(diag.r_hat.max()),
               "min_ess_bulk": float(diag.ess_bulk.min()), "divergences": n_div, "converged": converged,
               "max_injector_lambda_sum": float(lam.sum(2).max()), "best_fitting_draw_index": k_best, **summary_ml}
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)

    field = res[res.producer == "FIELD(pooled)"]
    cols = [c for c in ["model", "period", "MAE", "RMSE", "R2_det", "PICP_80", "PICP_95", "width_80", "CRPS"] if c in field]
    print(field[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log(f"wrote {len(plots)} producer plots to {out / 'plots'}")
    if not converged:
        log("WARNING: convergence gate not met (R-hat < 1.01, ESS > 400, 0 divergences). Do not report as final.")
    return summary


# =============================================================================
# 6. BUILT-IN REGRESSION TESTS  (python3 crm_ml_bayes_uq.py --self-test)
# =============================================================================

def _matlab_q_crm(tau_j, lam, tau_ij, t, I, q):
    """Line-by-line transcription of the lecturer's MATLAB q_CRM (no BHP term), 0-based."""
    NN, II = I.shape
    Q = np.zeros((NN, len(tau_j)))
    for j in range(len(tau_j)):
        s = int(np.argmax(q[:, j] > 0))
        Q[s, j] = q[s, j]
        for n in range(s + 1, NN):
            val = q[s, j] * np.exp((t[s] - t[n]) / tau_j[j])
            for i in range(II):
                acc = sum((np.exp((t[m] - t[n]) / tau_ij[i, j]) - np.exp((t[m - 1] - t[n]) / tau_ij[i, j])) * I[m, i]
                          for m in range(s + 1, n + 1))
                val += lam[i, j] * acc
            Q[n, j] = val
    return Q


def _toy_case(seed=1, N=30, I=3, J=4):
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.r_[1.0, rng.uniform(0.3, 2.0, N - 1)])
    w = rng.uniform(0, 50, (N, I))
    q = rng.uniform(1, 30, (N, J))
    q[:7, 1] = 0.0     # late-starting producer
    q[:2, 3] = 0.0
    p = CRMParams(rng.uniform(0.5, 20, J), rng.dirichlet(np.ones(J + 1), I)[:, :J], rng.uniform(0.5, 20, (I, J)))
    return t, w, q, p


def _t_forward_matches_matlab():
    t, w, q, p = _toy_case()
    s = producer_starts(q)
    got = CRMForward(t, w, s, q[s, np.arange(4)])(*p.torch()).detach().numpy()
    np.testing.assert_allclose(got, _matlab_q_crm(p.tau_j, p.lam, p.tau_ij, t, w, q), rtol=1e-10, atol=1e-10)


def _t_zero_before_start():
    t, w, q, p = _toy_case()
    s = producer_starts(q)
    out = CRMForward(t, w, s, q[s, np.arange(4)])(*p.torch()).detach().numpy()
    assert np.all(out[:7, 1] == 0.0) and np.all(out[:2, 3] == 0.0)


def _t_autograd_matches_finite_difference():
    t, w, q, p = _toy_case()
    s = producer_starts(q)
    fwd = CRMForward(t, w, s, q[s, np.arange(4)])
    x = torch.tensor(np.r_[np.log(p.tau_j), p.lam.ravel(), np.log(p.tau_ij).ravel()], dtype=DT, requires_grad=True)

    def f(x):
        return (fwd(torch.exp(x[:4]), x[4:16].reshape(3, 4), torch.exp(x[16:]).reshape(3, 4)) ** 2).sum()

    g, = torch.autograd.grad(f(x), x)
    fd = []
    for k in range(x.numel()):
        e = torch.zeros_like(x); e[k] = 1e-6
        fd.append(float((f(x.detach() + e) - f(x.detach() - e)) / 2e-6))
    np.testing.assert_allclose(g.numpy(), fd, rtol=1e-5, atol=1e-5)


def _t_jax_equals_torch():
    t, w, q, p = _toy_case(seed=3, N=40)
    s = producer_starts(q); q0 = q[s, np.arange(4)]
    ref = CRMForward(t, w, s, q0)(*p.torch()).detach().numpy()
    got = np.asarray(crm_jax(*(jnp.asarray(v) for v in (p.tau_j, p.lam, p.tau_ij, t, w, s, q0))))
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


def _t_contributions_sum_to_injection_term():
    t, w, q, p = _toy_case(seed=3, N=40)
    s = producer_starts(q); q0 = q[s, np.arange(4)]
    full = np.asarray(crm_jax(*(jnp.asarray(v) for v in (p.tau_j, p.lam, p.tau_ij, t, w, s, q0))))
    noinj = np.asarray(crm_jax(*(jnp.asarray(v) for v in (p.tau_j, 0 * p.lam, p.tau_ij, t, w, s, q0))))
    c = np.asarray(crm_contrib_jax(*(jnp.asarray(v) for v in (p.lam, p.tau_ij, t, w, s))))
    np.testing.assert_allclose(c.sum(-1), full - noinj, atol=1e-10)


def _t_draws_change_crm_trajectories():
    t, w, q, p = _toy_case(seed=3, N=40)
    s = producer_starts(q); q0 = q[s, np.arange(4)]
    rng = np.random.default_rng(0)
    tj = p.tau_j[None] * np.exp(rng.normal(0, .3, (5, 4)))
    lam = np.stack([rng.dirichlet(np.ones(5), 3)[:, :4] for _ in range(5)])
    tij = p.tau_ij[None] * np.exp(rng.normal(0, .3, (5, 3, 4)))
    mu = propagate((tj, lam, tij, None), t, w, s, q0)
    after = np.arange(40)[:, None] > s[None, :]
    assert np.all(mu.std(0)[after] > 0)


def _t_prior_draws_respect_injector_constraint():
    from numpyro.infer import Predictive
    t, w, q, p = _toy_case()
    s = producer_starts(q)
    model, *_ = make_model(t, w, s, q[s, np.arange(4)], q, "normal", 0.1, 1e4)
    lam = np.asarray(Predictive(model, num_samples=200)(jax.random.PRNGKey(0))["lam_full"])[:, :, :4]
    assert lam.min() >= 0 and lam.sum(-1).max() <= 1 + 1e-9


def _t_ar1_predictive_ignores_holdout():
    rng = np.random.default_rng(1)
    mu = rng.uniform(5, 10, (50, 30, 2)); y = rng.uniform(5, 10, (30, 2))
    args = (np.full((50, 2), 0.5), np.full((50, 2), 0.6))
    a_, _ = predictive_ar1(mu, *args, y, 24, np.array([0, 3]), seed=4)
    y2 = y.copy(); y2[24:] *= 10
    b_, _ = predictive_ar1(mu, *args, y2, 24, np.array([0, 3]), seed=4)
    np.testing.assert_array_equal(a_, b_)


def _t_ar1_loglik_matches_manual():
    from numpyro import handlers
    from scipy.stats import norm
    t, w, q, p = _toy_case(seed=3, N=40)
    s = producer_starts(q); q0 = q[s, np.arange(4)]
    model, *_ = make_model(t, w, s, q0, q, "ar1", 0.1, 1e4, sigma_scale=np.ones(4))
    init = init_values(p, 4, np.full(4, 2.0), 0.1, 1e4, phi=np.full(4, 0.5))
    tr = handlers.trace(handlers.substitute(model, data=init)).get_trace()
    ll = float(tr["obs"]["fn"].log_prob(tr["obs"]["value"]).sum())
    pp_ = CRMParams(np.exp(np.asarray(init["log_tau_j"])), np.asarray(init["lam_full"])[:, :4],
                    np.exp(np.asarray(init["log_tau_ij"])))
    mu = CRMForward(t, w, s, q0)(*pp_.torch()).detach().numpy()
    tot = 0.0
    for j in range(4):
        e = q[:, j] - mu[:, j]
        for n in range(s[j] + 1, len(t)):
            tot += (norm.logpdf(e[n], 0.5 * e[n - 1], 2.0) if n - 1 > s[j]
                    else norm.logpdf(e[n], 0.0, 2.0 / np.sqrt(0.75)))
    np.testing.assert_allclose(ll, tot, rtol=1e-8)


def _t_best_fitting_draw_uses_history_only():
    rng = np.random.default_rng(2)
    mu = rng.uniform(0, 10, (20, 30, 2)); y = rng.uniform(0, 10, (30, 2)); s = np.array([0, 4])
    k1, _ = best_fitting_draw(mu, y, s, 24)
    y2 = y.copy(); y2[24:] = 1e6
    k2, _ = best_fitting_draw(mu, y2, s, 24)
    assert k1 == k2


def _t_percentile_labels():
    assert p_labels("percentile")["P10"] == "P10" and p_labels("reserves")["P10"] == "P90"
    q = percentiles(np.arange(1000.0).reshape(1000, 1, 1))
    assert q["P10"][0, 0] < q["P50"][0, 0] < q["P90"][0, 0]


def _ml_toy(seed=0):
    rng = np.random.default_rng(seed)
    N, I, J = 120, 3, 2
    w = rng.uniform(0, 20, (N, I))
    mu = np.cumsum(rng.normal(0, 1, (N, J)), 0) + 20
    contrib = rng.uniform(0, 5, (N, J, I))
    y = np.maximum(mu + 0.5 * contrib.sum(-1) - 5 + rng.normal(0, 1, (N, J)), 0)
    return N, I, J, w, mu, contrib, y


def _t_article_style_constant_columns_are_inert():
    """Article design: per-producer lambda/tau inputs are constant columns -> no effect on XGB."""
    N, I, J, w, mu, contrib, y = _ml_toy()
    const = np.tile([0.2, 0.3, 0.1, 5.0, 8.0, 2.0], (N, 1))
    m = make_ml("xgb").fit(np.column_stack([w, const]), y[:, 0])
    assert np.array_equal(m.predict(np.column_stack([w, const])), m.predict(np.column_stack([w, const * 3.0])))


def _t_every_ml_model_carries_draw_uncertainty():
    N, I, J, w, mu, contrib, y = _ml_toy()
    s = np.array([0, 10])
    for name in ML_CHOICES:
        hy = ResidualML(name).fit(y, mu, contrib, w, s, H=100)
        draws_mu = np.stack([mu, mu * 1.1])
        draws_c = np.stack([contrib, contrib * 1.3])
        out = hy.predict_draws(draws_mu, draws_c, w, s)
        assert np.abs(out[0, 100:] - out[1, 100:]).max() > 0.1, name
        assert all(v > 0 for v in hy.sigma.values()), name


def _t_ml_never_sees_holdout_targets():
    N, I, J, w, mu, contrib, y = _ml_toy()
    s = np.array([0, 10])
    y2 = y.copy(); y2[100:] *= 5
    for name in ML_CHOICES:
        a_ = ResidualML(name).fit(y, mu, contrib, w, s, 100).predict_draws(mu[None], contrib[None], w, s)
        b_ = ResidualML(name).fit(y2, mu, contrib, w, s, 100).predict_draws(mu[None], contrib[None], w, s)
        np.testing.assert_allclose(a_, b_, err_msg=name)


def run_self_tests():
    """Execute the built-in regression tests and report the pass/fail summary.

    Parameters:
        None.

    Returns:
        True when every internal test passes; otherwise False.

    Side effects:
        Runs the internal self-test suite and prints a pass/fail report to stdout.
    """
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("_t_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name[3:]}  ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"FAIL  {name[3:]}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed", flush=True)
    return failed == 0


# =============================================================================
# 7. COMMAND LINE
# =============================================================================

def main():
    """CLI entry point for the CRM-ML analysis workflow and the built-in self-test runner.

    Parameters:
        None. All configuration comes from command-line flags.

    Returns:
        None.

    Side effects:
        Parses user arguments, may run the self-test suite, and executes the analysis pipeline, writing output files to disk.
    """
    ap = build_parser("Bayesian UQ for CRM + machine learning (CRM-XGB / ELM / MLP / NuSVM).")
    ap.add_argument("--ml", nargs="+", default=["xgb"], choices=list(ML_CHOICES),
                    help="ML model(s) to put on top of the Bayesian CRM; each gets its own plots and metrics")
    ap.add_argument("--ml-draws", type=int, default=1000,
                    help="posterior draws pushed through CRM + ML (random subset if fewer than available)")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_tests() else 1)
    if not a.data:
        ap.error("--data is required")
    a.out = a.out or "results_crm_ml"
    run_pipeline(a, ml_models=tuple(dict.fromkeys(a.ml)))


if __name__ == "__main__":
    main()