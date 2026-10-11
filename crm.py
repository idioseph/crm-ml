#!/usr/bin/env python3
"""crm_bayes_uq.py: Bayesian uncertainty quantification for CRM calibration (CRM only, no ML).

What it does, in order
  1. Calibrates the article's CRMP (Ogali & Orodu 2025, Eq. 1, no BHP term) by least
     squares on the history rows only: producer-specific start times, injector-wise
     sum_j lambda_ij <= 1, tau between 1 day and 1000*N, several starts.
  2. Builds a Bayesian posterior over every lambda_ij, tau_ij and tau_j (plus noise sd)
     and samples it with NUTS.  Each posterior draw is one JOINT combination of all
     CRM parameters that is consistent with the history.
  3. Pushes every draw through the CRM over history + holdout, adds observation
     noise, and reports P10 / P50 / P90 curves per producer, the best-fitting draw,
     and metrics (MAE, RMSE, R2, interval coverage, interval score, CRPS) for the
     history and the held-out rows separately.
  4. Saves one plot per producer.

The last --forecast rows are never used for fitting, sampling, or choosing anything.
See crm_bayes_uq.md for the full documentation.

Usage
  python crm_bayes_uq.py --data buffalo_field.xlsx --forecast 12
  python crm_bayes_uq.py --data buffalo_field.xlsx --likelihood ar1 --warmup 2000 --samples 2000
  python crm_bayes_uq.py --data buffalo_field.xlsx --synthetic-check --reps 3
  python crm_bayes_uq.py --self-test

Requires: numpy pandas scipy torch jax numpyro matplotlib openpyxl
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

jax.config.update("jax_enable_x64", True)


# 1. DATA LOADING, CRM FORWARD MODEL AND DETERMINISTIC CALIBRATION

DT = torch.float64

# --------------------------------------------------------------------------- data
@dataclass
class CRMData:
    """Container for the observed production/injection history used by the CRM models.

    Parameters:
        time: Strictly increasing time index for each row.
        production: Rate matrix with shape [N, J] for J producers.
        injection: Injection matrix with shape [N, I] for I injectors.
        producers: Producer names aligned with the production columns.
        injectors: Injector names aligned with the injection columns.
        distances: Optional distance matrix between producers and injectors.

    Returns:
        CRMData instance storing the raw history and metadata required by the forward model.

    Side effects:
        None beyond object construction; this class only stores data.
    """

    time: np.ndarray            # [N]
    production: np.ndarray      # [N, J]
    injection: np.ndarray       # [N, I]
    producers: list[str]
    injectors: list[str]
    distances: pd.DataFrame | None = None

    @property
    def n_inj(self):
        """Return the number of injectors in the loaded data table.

        Parameters:
            None.

        Returns:
            The column count of the injection matrix.

        Side effects:
            None.
        """
        return self.injection.shape[1]

    @property
    def n_prod(self):
        """Return the number of producers in the loaded data table.

        Parameters:
            None.

        Returns:
            The column count of the production matrix.

        Side effects:
            None.
        """
        return self.production.shape[1]

    def head(self, n: int) -> "CRMData":
        """Slice the dataset to its first n rows while preserving metadata.

        Parameters:
            n: Number of leading rows to keep.

        Returns:
            A CRMData object containing the truncated time series and the original labels.

        Side effects:
            None; the original dataset remains unchanged.
        """
        return CRMData(self.time[:n], self.production[:n], self.injection[:n],
                       self.producers, self.injectors, self.distances)


def _sheet(xls: pd.ExcelFile, names: list[str]) -> str:
    """Resolve the first matching worksheet name from a list of likely aliases.

    Parameters:
        xls: Excel workbook already opened by pandas.
        names: Candidate sheet names or aliases to search for.

    Returns:
        The exact workbook sheet name matched by alias.

    Side effects:
        Raises ValueError if no expected sheet is present.
    """
    low = {s.strip().lower(): s for s in xls.sheet_names}
    for n in names:
        if n in low:
            return low[n]
    raise ValueError(f"none of {names} in {xls.sheet_names}")


def _numeric_block(raw: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Normalise a worksheet to a header row plus numeric body matrix.

    Parameters:
        raw: Raw worksheet extracted from Excel.

    Returns:
        A tuple of (column names, numeric value matrix) excluding all-null rows and the
        units row such as [MSTB/day].

    Side effects:
        None; it only cleans and converts the worksheet contents.
    """
    raw = raw.dropna(how="all").reset_index(drop=True)
    cols = [str(c).strip() for c in raw.iloc[0]]
    body = raw.iloc[1:].apply(pd.to_numeric, errors="coerce")
    # A units row such as "[MSTB/day]" becomes all-NaN and is dropped here.
    body = body.loc[body.notna().all(axis=1)]
    return cols, np.array(body.to_numpy(float), copy=True)


def load_excel(path) -> CRMData:
    """Load and validate an Excel workbook containing the CRM input timeseries.

    Parameters:
        path: File path to the workbook containing time, production, and injection sheets.

    Returns:
        A CRMData object containing validated time, production, and injection arrays.

    Side effects:
        Reads the workbook from disk and raises ValueError on structural or data-quality issues.
    """
    # Open the workbook and locate the canonical sheet names used by the project.
    xls = pd.ExcelFile(path)
    _, t = _numeric_block(xls.parse(_sheet(xls, ["time", "t"]), header=None))
    qn, q = _numeric_block(xls.parse(_sheet(xls, ["q_observed", "production"]), header=None))
    wn, w = _numeric_block(xls.parse(_sheet(xls, ["w_injection", "injection"]), header=None))
    dist = None
    # Keep the optional distance matrix when the workbook includes it for diagnostics.
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
    """Compute the first positive production row for each producer.

    Parameters:
        production: Matrix of shape [N, J] containing producer production rates.

    Returns:
        An array of start-row indices for each producer, matching the MATLAB-style first positive rate rule.

    Side effects:
        Raises ValueError if any producer never has a positive rate in the calibration window.
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
        """Precompute the time-indexed CRM kernels used by the torch forward pass.

        Parameters:
            time: Timeline array for the entire dataset.
            injection: Injection-rate matrix with shape [N, I].
            starts: Producer start rows, one per producer.
            q_start: Initial production rate at each producer start.

        Returns:
            None. The instance stores precomputed tensors for efficient forecasting.

        Side effects:
            Allocates torch tensors and cached time-lag matrices on the object.
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
        """Evaluate the CRM production forecast for a candidate parameter set.

        Parameters:
            tau_j: Producer decay times with shape [J].
            lam: Injector-to-producer connectivity matrix with shape [I, J].
            tau_ij: Injector-specific delay times with shape [I, J].

        Returns:
            Forecasted production rates with shape [N, J] for all rows and producers.

        Side effects:
            Uses autograd-enabled torch tensors when gradient tracking is enabled; no external state is mutated.
        """
        prod = self.q0[None, :] * torch.exp(-self.elapsed / tau_j[None, :])
        # Kernel for each injector/producer pair over the entire time grid.
        tau = tau_ij[:, :, None, None]
        ker = (torch.exp(-self.D / tau) - torch.exp(-self.Dp / tau)) * self.active[None]
        resp = torch.einsum("ijnm,mi->ijn", ker, self.wm)              # [I, J, N]
        inj = torch.einsum("ij,ijn->nj", lam, resp)
        return self.at * self.q0[None, :] + self.after * (prod + inj)

    def contributions(self, lam, tau_ij):
        """Return the per-injector CRM contribution used for ML feature generation.

        Parameters:
            lam: Injector-to-producer connectivity matrix [I, J].
            tau_ij: Injector-specific decay times [I, J].

        Returns:
            Contribution tensor with shape [N, J, I] that preserves the injector attribution.

        Side effects:
            None; it is a pure tensor transformation.
        """
        tau = tau_ij[:, :, None, None]
        ker = (torch.exp(-self.D / tau) - torch.exp(-self.Dp / tau)) * self.active[None]
        resp = torch.einsum("ijnm,mi->ijn", ker, self.wm)
        return (lam[:, :, None] * resp).permute(2, 1, 0) * self.after[:, :, None]


# ---------------------------------------------------------- deterministic CRM fit
@dataclass
class CRMParams:
    """Bundle the CRM parameters used by the deterministic and Bayesian routines.

    Parameters:
        tau_j: Producer decay times with shape [J].
        lam: Injector connectivity matrix with shape [I, J].
        tau_ij: Injector-specific delay times with shape [I, J].

    Returns:
        A parameter object ready to be passed into the CRM forward model.

    Side effects:
        None beyond storing parameter arrays.
    """

    tau_j: np.ndarray
    lam: np.ndarray
    tau_ij: np.ndarray

    def torch(self):
        """Convert the parameter arrays to the torch tensors expected by the forward model.

        Parameters:
            None.

        Returns:
            A tuple (tau_j, lam, tau_ij) as torch tensors on the project dtype.

        Side effects:
            None; it only creates new tensor views.
        """
        return (torch.as_tensor(np.array(self.tau_j), dtype=DT), torch.as_tensor(np.array(self.lam), dtype=DT),
                torch.as_tensor(np.array(self.tau_ij), dtype=DT))


@dataclass
class FitResult:
    """Summary object for the deterministic least-squares calibration run.

    Parameters:
        params: Best-fit CRMParams values.
        objective: Value of the calibration objective at the best solution.
        starts_table: Per-start optimisation diagnostics.
        tau_bounds: Search bounds used for tau_j and tau_ij.
        solutions: Ordered list of all candidate solutions found across starts.

    Returns:
        The deterministic calibration output package.

    Side effects:
        None beyond storing optimisation results.
    """

    params: CRMParams
    objective: float
    starts_table: pd.DataFrame = field(repr=False)
    tau_bounds: tuple[float, float] = (0.0, 0.0)
    solutions: list = field(default_factory=list, repr=False)   # (objective, CRMParams) per start


def fit_deterministic(data: CRMData, tau_min: float, tau_max: float | None = None,
                      n_starts: int = 8, max_iter: int = 1000, seed: int = 2025) -> FitResult:
    """Concurrent least-squares calibration of the CRM parameters on the supplied history.

    Parameters:
        data: CRMData object containing the calibration window.
        tau_min: Lower bound on the producer and injector decay times.
        tau_max: Upper bound; if None, defaults to 1000 * N.
        n_starts: Number of multi-start optimisations to run.
        max_iter: Maximum number of SLSQP iterations per start.
        seed: Random seed used for the non-deterministic initial guesses.

    Returns:
        A FitResult with the best-fit CRMParams, the objective value, and metrics for each start.

    Side effects:
        Runs the convex-constraint optimisation routine and writes no external files.
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

    # Run the constrained least-squares solver from several start points to reduce local minima risk.
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
    """Forecast the full timeline using the calibration-window starts and fitted CRM parameters.

    Parameters:
        data_full: Full dataset including the holdout period.
        calib: Calibration dataset whose start rows are reused for the forecast.
        p: CRM parameter set to evaluate.

    Returns:
        A NumPy array with shape [N, J] containing the modelled production trajectory.

    Side effects:
        None; it performs a forward pass using torch's no-grad mode.
    """
    J = calib.n_prod
    s = producer_starts(calib.production)
    fwd = CRMForward(data_full.time, data_full.injection, s, calib.production[s, np.arange(J)])
    with torch.no_grad():
        return fwd(*p.torch()).numpy()


# 2. BAYESIAN CRM (NUTS), PROPAGATION AND POSTERIOR PREDICTIVE
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
#   (lambda_i1..lambda_iJ, slack_i) ~ Dirichlet(alpha)          injector-wise mass balance
#   sigma_j ~ HalfNormal(scale_j)
#   normal:    q_obs ~ Normal(q_crm, sigma_j)                   rows after each producer's start
#   lognormal: log q_obs ~ Normal(log q_crm, sigma_j)           rows after start with q_obs > 0

def crm_jax(tau_j, lam, tau_ij, t, w, starts, q0):
    """Evaluate the CRM forward model in JAX using the exact discrete-time recurrence.

    Parameters:
        tau_j: Producer decay times with shape [J].
        lam: Connectivity matrix with shape [I, J].
        tau_ij: Injector-specific decay times with shape [I, J].
        t: Time grid of shape [N].
        w: Injection matrix of shape [N, I].
        starts: Producer start indices with shape [J].
        q0: Initial production values at each producer start with shape [J].

    Returns:
        Forecasted production rates with shape [N, J].

    Side effects:
        None; it builds a pure JAX computation graph.
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

    # Scan the time recurrence forward so each row is conditioned on the prior injector state.
    _, Ss = lax.scan(step, jnp.zeros(tau_ij.shape, t.dtype), (c, w[1:], after[1:]))
    resp = jnp.concatenate([jnp.zeros((1,) + tau_ij.shape, t.dtype), Ss])   # [N, I, J]
    inj = jnp.einsum("ij,nij->nj", lam, resp)
    return at * q0[None, :] + after * (prod + inj)


def crm_contrib_jax(lam, tau_ij, t, w, starts):
    """Compute the per-injector CRM contribution in JAX for diagnostic and feature use.

    Parameters:
        lam: Injector connectivity matrix [I, J].
        tau_ij: Injector-specific decay times [I, J].
        t: Time grid [N].
        w: Injection matrix [N, I].
        starts: Producer start rows [J].

    Returns:
        A tensor with shape [N, J, I] containing the filtered injector contribution for each producer.

    Side effects:
        None; the function is a pure numerical transform.
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
    """Construct the NumPyro generative model for the Bayesian CRM calibration.

    Parameters:
        t: Observation time grid.
        w: Injection matrix [N, I].
        starts: Producer start rows [J].
        q0: Initial production values at each producer start [J].
        y: Observed production matrix [N, J].
        likelihood: One of "normal", "lognormal", or "ar1".
        tau_min: Lower bound applied to tau_j and tau_ij in the prior.
        tau_max: Upper bound applied to tau_j and tau_ij in the prior.
        tau_mu: Prior mean for log tau values.
        tau_sd: Prior sd for log tau values.
        alpha: Dirichlet concentration for the injector mass-allocation vector.
        sigma_scale: Optional per-producer observation-noise scales.

    Returns:
        A tuple (model, n_obs, n_zero) where model is a NumPyro probabilistic function, n_obs is the
        number of post-start observational rows, and n_zero counts zero-valued rows in the holdout mask.

    Side effects:
        Builds the model object used by MCMC; no sampling is performed here.
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
        lam_full = numpyro.sample("lam_full", dist.Dirichlet(alpha * jnp.ones(J + 1)).expand([I]).to_event(1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(jnp.asarray(sigma_scale)).to_event(1))
        mu = crm_jax(jnp.exp(log_tau_j), lam_full[:, :J], jnp.exp(log_tau_ij), t, w, starts, q0)
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
    """Convert a deterministic CRM fit into plausible NUTS initial values.

    Parameters:
        p: Deterministic CRMParams solution used as the starting point.
        J: Number of producers.
        sigma: Per-producer observation-noise scale for the initial state.
        tau_min: Optional lower bound used to keep values inside the model support.
        tau_max: Optional upper bound used to keep values inside the model support.
        phi: Optional AR(1) coefficients for the initial AR model.

    Returns:
        A dictionary with the log-parameter initial values accepted by NumPyro.

    Side effects:
        None; it only transforms arrays into a valid sampling configuration.
    """
    if tau_min is not None:
        lo, hi = np.log(tau_min) + 1e-3, np.log(tau_max) - 1e-3
        p = type(p)(np.exp(np.clip(np.log(p.tau_j), lo, hi)), p.lam, np.exp(np.clip(np.log(p.tau_ij), lo, hi)))
    lam = np.clip(p.lam, 1e-4, None)
    slack = np.clip(1.0 - lam.sum(1, keepdims=True), 1e-3, None)
    full = np.concatenate([lam, slack], axis=1)
    full = full / full.sum(1, keepdims=True)
    return {"log_tau_j": jnp.log(jnp.asarray(p.tau_j)), "log_tau_ij": jnp.log(jnp.asarray(p.tau_ij)),
            "lam_full": jnp.asarray(full), "sigma": jnp.asarray(sigma),
            **({} if phi is None else {"phi": jnp.asarray(np.clip(phi, -0.9, 0.9))})}


def run_nuts(model, inits, warmup, samples, seed, target_accept=0.9, max_tree_depth=8, progress=False):
    """Run one NUTS chain per initial state and aggregate the posterior draws.

    Parameters:
        model: NumPyro model function.
        inits: Initial value dictionaries for each chain.
        warmup: Number of warmup draws per chain.
        samples: Number of retained posterior draws per chain.
        seed: Base random seed for the chain initialisation.
        target_accept: HMC target acceptance probability.
        max_tree_depth: Maximum NUTS tree depth.
        progress: Whether to show chain progress output.

    Returns:
        A tuple (grouped_samples, info) where grouped_samples is a dictionary of stacked arrays and info is
        a per-chain diagnostics summary.

    Side effects:
        Executes the MCMC sampling routine, which is the main computational workload of the pipeline.
    """
    chains, divs, info = [], [], []
    for c, init in enumerate(inits):
        # Each chain gets a unique JAX key so the posterior sample set remains independent across runs.
        kernel = NUTS(model, init_strategy=init_to_value(values=init), target_accept_prob=target_accept,
                      max_tree_depth=max_tree_depth, dense_mass=False)
        mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=progress)
        mcmc.run(jax.random.PRNGKey(seed + 1000 * c), extra_fields=("diverging", "num_steps", "accept_prob"))
        ex = mcmc.get_extra_fields()
        chains.append(mcmc.get_samples())
        divs.append(int(np.sum(ex["diverging"])))
        info.append({"chain": c, "divergences": divs[-1], "mean_accept": float(np.mean(ex["accept_prob"])),
                     "mean_leapfrog": float(np.mean(ex["num_steps"])), "max_leapfrog": int(np.max(ex["num_steps"]))})
    grouped = {k: np.stack([np.asarray(ch[k]) for ch in chains]) for k in chains[0]}
    return grouped, info


def diagnostics(grouped):
    """Summarise the posterior draws by scalar parameter using NumPyro's R-hat and bulk ESS diagnostics.

    Parameters:
        grouped: Stacked posterior samples keyed by parameter name.

    Returns:
        A list of rows containing the parameter name, R-hat, and bulk ESS.

    Side effects:
        None; this function only aggregates diagnostics from the posterior samples.
    """
    s = summary(grouped, group_by_chain=True)
    rows = []
    for name, d in s.items():
        rh = np.ravel(d["r_hat"]); ne = np.ravel(d["n_eff"])
        for k in range(rh.size):
            rows.append({"param": f"{name}[{k}]", "r_hat": float(rh[k]), "ess_bulk": float(ne[k])})
    return rows


def physical(grouped, J):
    """Convert unconstrained latent draws into interpretable physical CRM parameters.

    Parameters:
        grouped: Posterior sample dictionary from NumPyro.
        J: Number of producers.

    Returns:
        A tuple (tau_j, lam, tau_ij, sigma) in the physical parameterisation used for prediction.

    Side effects:
        None; it transforms the posterior arrays into a numeric representation.
    """
    flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in grouped.items()}
    return (np.exp(flat["log_tau_j"]), flat["lam_full"][:, :, :J], np.exp(flat["log_tau_ij"]), flat["sigma"])


def propagate(draws, t, w, starts, q0):
    """Batch-propagate many posterior draws through the CRM forward model.

    Parameters:
        draws: Tuple (tau_j, lam, tau_ij, sigma) for multiple posterior draws.
        t: Time grid [N].
        w: Injection matrix [N, I].
        starts: Producer start rows [J].
        q0: Initial rate values at each producer start [J].

    Returns:
        A NumPy array of shape [D, N, J] containing the CRM trajectories for every posterior draw.

    Side effects:
        JIT-compiles and executes the batched JAX forward evaluation.
    """
    tau_j, lam, tau_ij, _ = draws
    f = jax.jit(jax.vmap(lambda a, b, c: crm_jax(a, b, c, jnp.asarray(t), jnp.asarray(w),
                                                 jnp.asarray(starts), jnp.asarray(q0))))
    return np.asarray(f(jnp.asarray(tau_j), jnp.asarray(lam), jnp.asarray(tau_ij)))


def predictive(mu, sigma, likelihood, starts, seed=0):
    """Add observation noise to CRM trajectories to create a posterior predictive draw.

    Parameters:
        mu: CRM mean trajectory array with shape [D, N, J].
        sigma: Observation-noise scale array with shape [D, J].
        likelihood: Either "normal" or "lognormal".
        starts: Producer start rows [J].
        seed: Random seed controlling the noise realisation.

    Returns:
        A tuple (y_pred, frac_neg), where y_pred is the simulated observed production and frac_neg is the
        fraction of negative values in the normal predictive case.

    Side effects:
        Draws random Gaussian noise and truncates negative rates for the normal model.
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
    """Generate the AR(1) posterior predictive distribution with one-step residual dependence.

    Parameters:
        mu: CRM mean trajectory array [D, N, J].
        sigma: Observation-noise scale [D, J].
        phi: AR(1) coefficient array [D, J].
        y: Observed production array [N, J].
        H: History cutoff row index; rows before H are treated as observed history.
        starts: Producer start rows [J].
        seed: Random seed used for the simulation noise.

    Returns:
        A tuple (simulated_y, frac_neg) containing the AR(1) predictive time series and the fraction of negative values.

    Side effects:
        Simulates a noisy autoregressive process for the held-out rows and clips values at zero.
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


# 3. POINT AND PROBABILISTIC METRICS
# Point and probabilistic forecast metrics, per producer and field.

def point(y, m):
    """Compute the standard point-forecast accuracy metrics for a single target series.

    Parameters:
        y: Observed values.
        m: Modelled values aligned to y.

    Returns:
        A dictionary of MAE, RMSE, and R^2 summary metrics.

    Side effects:
        None; it only computes descriptive statistics on the arrays.
    """
    e = y - m
    out = {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2))}
    sst = np.sum((y - y.mean()) ** 2)
    out["R2_det"] = 1 - np.sum(e ** 2) / sst if sst > 0 else np.nan          # coefficient of determination
    c = np.corrcoef(y, m)[0, 1] if np.std(y) > 0 and np.std(m) > 0 else np.nan
    out["R2_corr"] = c ** 2                                                  # article Eq. 5
    return out


def interval(y, lo, hi, alpha):
    """Compute coverage and Winkler interval score for an observation interval.

    Parameters:
        y: Observed values.
        lo: Lower interval bound.
        hi: Upper interval bound.
        alpha: Tail probability used for the interval width scaling.

    Returns:
        A dictionary with PICP, mean width, and interval score.

    Side effects:
        None; it only summarises the predictive interval quality.
    """
    width = hi - lo
    score = width + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)  # Winkler / interval score
    return {"PICP": np.mean((y >= lo) & (y <= hi)), "width": np.mean(width), "IS": np.mean(score)}


def crps_samples(y, S, max_draws=400, seed=0):
    """Approximate the CRPS from a sample approximation of the predictive distribution.

    Parameters:
        y: Observed values with shape [N].
        S: Sample matrix with shape [D, N].
        max_draws: Maximum number of posterior draws used for the approximation.
        seed: Random seed used when sub-sampling large ensembles.

    Returns:
        The mean CRPS across the observations.

    Side effects:
        Draws a random subset of samples if the ensemble is larger than max_draws.
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
    """Build a metrics table for one forecast period across all producers and the pooled field total.

    Parameters:
        y_all: Full observed production array [N, J].
        point_pred: Point forecast array [N, J].
        rows_by_producer: Sequence of row-index arrays for each producer in the evaluation period.
        producers: Producer names aligned with the columns.
        samples: Optional predictive sample array [D, N, J].
        label: Period label such as "history" or "holdout".

    Returns:
        A pandas DataFrame summarising the point and probabilistic metrics for the period.

    Side effects:
        None; it constructs and returns a reporting table.
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


# 4. PERCENTILES, BEST-FITTING DRAW, PLOTS AND THE END-TO-END PIPELINE

def log(msg):
    """Emit a timestamped log line for the user-facing diagnostics stream.

    Parameters:
        msg: Message text to print.

    Returns:
        None.

    Side effects:
        Writes to stdout with a timestamp.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_name(s):
    """Convert a producer label into a filesystem-safe stem for plot names.

    Parameters:
        s: Raw producer name.

    Returns:
        A sanitised string that preserves alphanumerics, dashes, and underscores.

    Side effects:
        None; it only transforms the string values.
    """
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).strip("_")


def best_fitting_draw(mu, y, starts, H):
    """Identify the posterior draw with the lowest in-sample history RMSE.

    Parameters:
        mu: Posterior CRM trajectories with shape [D, N, J].
        y: Observed production matrix [N, J].
        starts: Producer start rows [J].
        H: History cutoff index; only rows below H are considered when scoring fit quality.

    Returns:
        A tuple (best_index, errors) where best_index is the winning draw and errors contains the per-draw MSE values.

    Side effects:
        None; it only ranks posterior draws.
    """
    N = mu.shape[1]
    after = (np.arange(N)[:, None] > np.asarray(starts)[None, :]) & (np.arange(N)[:, None] < H)
    err = ((mu - y[None]) ** 2 * after[None]).sum(axis=(1, 2)) / after.sum()
    return int(np.argmin(err)), err


def percentiles(S):
    """Convert a sample ensemble into the P10, P50, and P90 summary curves.

    Parameters:
        S: Ensemble of trajectories with shape [D, N, J].

    Returns:
        A dictionary mapping "P10", "P50", and "P90" to their percentile curves.

    Side effects:
        None.
    """
    q = np.quantile(S, [0.10, 0.50, 0.90], axis=0)
    return {"P10": q[0], "P50": q[1], "P90": q[2]}


def p_labels(convention):
    """Map percentile conventions to labels used in plots and CSV exports.

    Parameters:
        convention: Either "percentile" or "reserves".

    Returns:
        A dictionary that renames the summary curves for the selected plotting convention.

    Side effects:
        None.
    """
    if convention == "reserves":
        return {"P10": "P90", "P50": "P50", "P90": "P10"}
    return {"P10": "P10", "P50": "P50", "P90": "P90"}


def plot_producer(path, t, H, obs, start, band, param_band, best, title, labels, ref=None, ref_label=None):
    """Create a single producer plot showing observed data, posterior predictive bands, and the best fit.

    Parameters:
        path: Output file path for the PNG plot.
        t: Time axis for the full dataset.
        H: History cutoff row.
        obs: Observed production data for the producer.
        start: Producer start row.
        band: Predictive percentiles with keys P10/P50/P90.
        param_band: Parameter-only percentiles with keys P10/P50/P90.
        best: Best-fit history trajectory for the producer.
        title: Plot title.
        labels: Percentage label convention for display.
        ref: Optional reference curve.
        ref_label: Label for the optional reference curve.

    Returns:
        None. The plot is saved to disk at path.

    Side effects:
        Creates a PNG plot on the filesystem using matplotlib.
    """
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
    ax.plot(t, band["P90"], color=blue_dark, lw=1.2, ls="--", label=labels["P90"])
    ax.plot(t, band["P50"], color=blue, lw=2.2, label=labels["P50"])
    ax.plot(t, band["P10"], color=blue_dark, lw=1.2, ls=":", label=labels["P10"])
    if ref is not None:
        ax.plot(t, ref, color=muted, lw=1.2, label=ref_label)
    ax.plot(t, best, color=orange, lw=1.4, ls="-.", label="best-fitting posterior draw (history)")
    ax.plot(t[start:], obs[start:], color=ink, lw=1, marker="o", ms=2.5, label="observed")
    ax.set_title(title, fontsize=11, color=ink, loc="left")
    ax.set_xlabel("time (data time units)")
    ax.set_ylabel("liquid production rate")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser(description):
    """Create the command-line interface for the CRM uncertainty-quantification workflow.

    Parameters:
        description: Short description shown in the argparse help text.

    Returns:
        An ArgumentParser configured with all runtime options for calibration and reporting.

    Side effects:
        Builds the CLI configuration object, but does not parse user input or run the model.
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


def run_pipeline(a):
    """Run the full deterministic-fit to Bayesian-posterior CRM workflow and export all outputs.

    Parameters:
        a: Parsed argparse namespace containing the input data path, forecast horizon, convergence settings,
           likelihood choice, and output directory.

    Returns:
        A summary dictionary describing the run diagnostics and reported metrics.

    Side effects:
        Reads the workbook from disk, writes output CSV/JSON/PNG artifacts to a results directory, and prints runtime logs.
    """
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    # Save the exact CLI config so later runs can be reproduced from the output directory.
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

    # ---- Step 1: deterministic CRM (article-faithful baseline and sampler start points)
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
    if lik in ("normal", "ar1"):
        scale = np.empty(J, dtype=float)
        for j in range(J):
            scale[j] = np.std(y[rows_hist[j], j])
    else:
        scale = np.ones(J, dtype=float)
    model, n_obs, n_zero = make_model(hist.time, hist.injection, s, q0, hist.production, lik, a.tau_min,
                                      1000.0 * H, np.log(10.0), a.tau_prior_sd, a.alpha, scale)
    phi0 = None
    if lik == "ar1":
        phi0 = np.array([np.corrcoef((y[rows_hist[j], j] - det[rows_hist[j], j])[:-1],
                                     (y[rows_hist[j], j] - det[rows_hist[j], j])[1:])[0, 1] for j in range(J)])
        resid = resid * np.sqrt(1 - np.clip(phi0, -0.9, 0.9) ** 2)
    sols = [sp for _, sp in fit.solutions]
    inits = [init_values(sols[c % len(sols)], J, resid, a.tau_min, 1000.0 * H, phi0) for c in range(a.chains)]
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
    log(f"propagated {D} posterior draws; best-fitting draw index {k_best}")

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
        f = out / "plots" / f"CRM_{safe_name(pname)}.png"
        plot_producer(f, d.time, H, y[:, j], s[j], {k: v[:, j] for k, v in pp.items()},
                      {k: v[:, j] for k, v in pr.items()}, best[:, j],
                      f"{pname}: Bayesian CRM ({lik} likelihood)", labels)
        plots.append(f)

    summary_ml = {}

    ts.to_csv(out / "percentile_timeseries.csv", index=False)
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(out / "metrics.csv", index=False)
    summary = {"likelihood": lik, "history_rows": H, "holdout_rows": N - H, "posterior_draws": D,
               "deterministic_objective": fit.objective, "max_rhat": float(diag.r_hat.max()),
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


def synthetic_check(a):
    """Run a synthetic recovery study using CRM-generated data and the same calibration workflow.

    Parameters:
        a: Parsed CLI arguments including the generation noise and replicate count.

    Returns:
        A list of dictionary records summarising each synthetic replicate's parameter recovery and coverage.

    Side effects:
        Reads the data workbook, performs a Monte Carlo synthetic recovery pass, and writes a JSON artifact.
    """
    d = load_excel(a.data)
    N, I, J = len(d.time), d.n_inj, d.n_prod
    H = N - a.forecast
    s = producer_starts(d.production[:H])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = []
    for rep in range(a.reps):
        rng = np.random.default_rng(100 + rep)
        true = CRMParams(np.exp(rng.uniform(np.log(1), np.log(30), J)),
                         rng.dirichlet(np.full(J + 1, 0.5), I)[:, :J],
                         np.exp(rng.uniform(np.log(0.5), np.log(30), (I, J))))
        q0 = d.production[s, np.arange(J)]
        mu = CRMForward(d.time, d.injection, s, q0)(*true.torch()).detach().numpy()
        after = np.arange(N)[:, None] > s[None, :]
        y = np.where(after, np.maximum(mu + a.noise * rng.standard_normal(mu.shape), 0), d.production * (~after))
        syn = CRMData(d.time, y, d.injection, d.producers, d.injectors)
        hist = syn.head(H)
        fit = fit_deterministic(hist, a.tau_min, None, min(a.starts, 4), a.max_iter, rep)
        model, *_ = make_model(hist.time, hist.injection, s, q0, hist.production, "normal", a.tau_min,
                               1000.0 * H, sigma_scale=np.full(J, 3.0))
        n_ch = min(a.chains, len(fit.solutions))
        inits = [init_values(sp, J, np.full(J, a.noise), a.tau_min, 1000.0 * H) for _, sp in fit.solutions[:n_ch]]
        grouped, info = run_nuts(model, inits, a.warmup, a.samples, rep)
        diag = diagnostics(grouped)
        tj, lam, tij, sig = physical(grouped, J)

        def cover(draws, truth):
            lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)
            return float(np.mean((truth >= lo) & (truth <= hi)))

        pm = propagate((tj, lam, tij, sig), d.time, d.injection, s, q0)
        yp, _ = predictive(pm, sig, "normal", s, rep)
        tab = table(y, np.median(pm, 0), [np.arange(H, N)] * J, d.producers, yp, "holdout")
        rec = {"rep": rep,
               "det_lambda_MAE": float(np.abs(fit.params.lam - true.lam).mean()),
               "lambda_cov90": cover(lam, true.lam), "tau_j_cov90": cover(tj, true.tau_j),
               "tau_ij_cov90": cover(tij, true.tau_ij),
               "tau_ij_cov90_where_lambda>0.05": cover(tij[:, true.lam > 0.05], true.tau_ij[true.lam > 0.05]),
               "sigma_cov90": cover(sig, np.full(J, a.noise)),
               "holdout_PICP95": float(tab.iloc[-1]["PICP_95"]), "holdout_PICP80": float(tab.iloc[-1]["PICP_80"]),
               "max_rhat": max(r["r_hat"] for r in diag), "min_ess": min(r["ess_bulk"] for r in diag),
               "divergences": sum(x["divergences"] for x in info)}
        log(json.dumps(rec))
        recs.append(rec)
    json.dump(recs, open(out / "synthetic_check.json", "w"), indent=1)
    return recs


# 5. SYNTHETIC RECOVERY CHECK
# Generates rates from the CRM with known (tau_j, lambda, tau_ij) using the
# Buffalo injection history and producer start rows, adds Normal noise,
# calibrates on the first N-12 rows with the same code, and reports
# (a) deterministic recovery error, (b) whether posterior 90% intervals contain
# the truth, (c) holdout predictive coverage.  If the method cannot recover
# parameters it generated itself, Buffalo-scale parameter claims are not
# defensible.

def synthetic_check(a):
    d = load_excel(a.data)
    N, I, J = len(d.time), d.n_inj, d.n_prod
    H = N - a.forecast
    s = producer_starts(d.production[:H])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = []
    for rep in range(a.reps):
        rng = np.random.default_rng(100 + rep)
        true = CRMParams(np.exp(rng.uniform(np.log(1), np.log(30), J)),
                         rng.dirichlet(np.full(J + 1, 0.5), I)[:, :J],
                         np.exp(rng.uniform(np.log(0.5), np.log(30), (I, J))))
        q0 = d.production[s, np.arange(J)]
        mu = CRMForward(d.time, d.injection, s, q0)(*true.torch()).detach().numpy()
        after = np.arange(N)[:, None] > s[None, :]
        y = np.where(after, np.maximum(mu + a.noise * rng.standard_normal(mu.shape), 0), d.production * (~after))
        syn = CRMData(d.time, y, d.injection, d.producers, d.injectors)
        hist = syn.head(H)
        fit = fit_deterministic(hist, a.tau_min, None, min(a.starts, 4), a.max_iter, rep)
        model, *_ = make_model(hist.time, hist.injection, s, q0, hist.production, "normal", a.tau_min,
                               1000.0 * H, sigma_scale=np.full(J, 3.0))
        n_ch = min(a.chains, len(fit.solutions))
        inits = [init_values(sp, J, np.full(J, a.noise), a.tau_min, 1000.0 * H) for _, sp in fit.solutions[:n_ch]]
        grouped, info = run_nuts(model, inits, a.warmup, a.samples, rep)
        diag = diagnostics(grouped)
        tj, lam, tij, sig = physical(grouped, J)

        def cover(draws, truth):
            lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)
            return float(np.mean((truth >= lo) & (truth <= hi)))

        pm = propagate((tj, lam, tij, sig), d.time, d.injection, s, q0)
        yp, _ = predictive(pm, sig, "normal", s, rep)
        tab = table(y, np.median(pm, 0), [np.arange(H, N)] * J, d.producers, yp, "holdout")
        rec = {"rep": rep,
               "det_lambda_MAE": float(np.abs(fit.params.lam - true.lam).mean()),
               "lambda_cov90": cover(lam, true.lam), "tau_j_cov90": cover(tj, true.tau_j),
               "tau_ij_cov90": cover(tij, true.tau_ij),
               "tau_ij_cov90_where_lambda>0.05": cover(tij[:, true.lam > 0.05], true.tau_ij[true.lam > 0.05]),
               "sigma_cov90": cover(sig, np.full(J, a.noise)),
               "holdout_PICP95": float(tab.iloc[-1]["PICP_95"]), "holdout_PICP80": float(tab.iloc[-1]["PICP_80"]),
               "max_rhat": max(r["r_hat"] for r in diag), "min_ess": min(r["ess_bulk"] for r in diag),
               "divergences": sum(x["divergences"] for x in info)}
        log(json.dumps(rec))
        recs.append(rec)
    json.dump(recs, open(out / "synthetic_check.json", "w"), indent=1)
    return recs


# 6. BUILT-IN REGRESSION TESTS  (python3 crm_bayes_uq.py --self-test)

def _matlab_q_crm(tau_j, lam, tau_ij, t, I, q):
    """Transcribe the MATLAB CRM formula exactly for numerical regression checks.

    Parameters:
        tau_j: Producer decay times.
        lam: Injector-to-producer connectivity matrix.
        tau_ij: Injector-specific decay times.
        t: Time grid.
        I: Injection-rate matrix.
        q: Observed production matrix.

    Returns:
        The MATLAB-equivalent CRM trajectory matrix Q with one column per producer.

    Side effects:
        None; it is a reference implementation used only by the self-tests.
    """
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
    """Build a synthetic CRM toy dataset for numerical regression tests.

    Parameters:
        seed: Random seed for reproducibility.
        N: Number of time rows.
        I: Number of injectors.
        J: Number of producers.

    Returns:
        A tuple (t, w, q, p) containing the synthetic time grid, injections, production data, and true parameters.

    Side effects:
        None; it produces deterministic synthetic inputs for unit tests.
    """
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.r_[1.0, rng.uniform(0.3, 2.0, N - 1)])
    w = rng.uniform(0, 50, (N, I))
    q = rng.uniform(1, 30, (N, J))
    q[:7, 1] = 0.0     # late-starting producer
    q[:2, 3] = 0.0
    p = CRMParams(rng.uniform(0.5, 20, J), rng.dirichlet(np.ones(J + 1), I)[:, :J], rng.uniform(0.5, 20, (I, J)))
    return t, w, q, p


def _t_forward_matches_matlab():
    """Assert that the torch CRM forward pass matches the MATLAB reference implementation."""
    t, w, q, p = _toy_case()
    s = producer_starts(q)
    got = CRMForward(t, w, s, q[s, np.arange(4)])(*p.torch()).detach().numpy()
    np.testing.assert_allclose(got, _matlab_q_crm(p.tau_j, p.lam, p.tau_ij, t, w, q), rtol=1e-10, atol=1e-10)


def _t_zero_before_start():
    """Assert that the CRM model emits zeros before each producer's start row."""
    t, w, q, p = _toy_case()
    s = producer_starts(q)
    out = CRMForward(t, w, s, q[s, np.arange(4)])(*p.torch()).detach().numpy()
    assert np.all(out[:7, 1] == 0.0) and np.all(out[:2, 3] == 0.0)


def _t_autograd_matches_finite_difference():
    """Verify the exact autograd gradient matches its finite-difference approximation."""
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
    """Check that the JAX forward pass is numerically equivalent to the torch reference."""
    t, w, q, p = _toy_case(seed=3, N=40)
    s = producer_starts(q); q0 = q[s, np.arange(4)]
    ref = CRMForward(t, w, s, q0)(*p.torch()).detach().numpy()
    got = np.asarray(crm_jax(*(jnp.asarray(v) for v in (p.tau_j, p.lam, p.tau_ij, t, w, s, q0))))
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


def _t_contributions_sum_to_injection_term():
    """Confirm that the injector contribution tensor reconstructs the full CRM injection term."""
    t, w, q, p = _toy_case(seed=3, N=40)
    s = producer_starts(q); q0 = q[s, np.arange(4)]
    full = np.asarray(crm_jax(*(jnp.asarray(v) for v in (p.tau_j, p.lam, p.tau_ij, t, w, s, q0))))
    noinj = np.asarray(crm_jax(*(jnp.asarray(v) for v in (p.tau_j, 0 * p.lam, p.tau_ij, t, w, s, q0))))
    c = np.asarray(crm_contrib_jax(*(jnp.asarray(v) for v in (p.lam, p.tau_ij, t, w, s))))
    np.testing.assert_allclose(c.sum(-1), full - noinj, atol=1e-10)


def _t_draws_change_crm_trajectories():
    """Ensure that different posterior draws produce different CRM trajectories."""
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
    """Verify the prior allows only feasible injector allocations that sum to at most one."""
    from numpyro.infer import Predictive
    t, w, q, p = _toy_case()
    s = producer_starts(q)
    model, *_ = make_model(t, w, s, q[s, np.arange(4)], q, "normal", 0.1, 1e4)
    lam = np.asarray(Predictive(model, num_samples=200)(jax.random.PRNGKey(0))["lam_full"])[:, :, :4]
    assert lam.min() >= 0 and lam.sum(-1).max() <= 1 + 1e-9


def _t_ar1_predictive_ignores_holdout():
    """Check that the AR(1) simulation ignores holdout observations when generating the predictive path."""
    rng = np.random.default_rng(1)
    mu = rng.uniform(5, 10, (50, 30, 2)); y = rng.uniform(5, 10, (30, 2))
    args = (np.full((50, 2), 0.5), np.full((50, 2), 0.6))
    a_, _ = predictive_ar1(mu, *args, y, 24, np.array([0, 3]), seed=4)
    y2 = y.copy(); y2[24:] *= 10
    b_, _ = predictive_ar1(mu, *args, y2, 24, np.array([0, 3]), seed=4)
    np.testing.assert_array_equal(a_, b_)


def _t_ar1_loglik_matches_manual():
    """Confirm the AR(1) log-likelihood uses the expected one-step conditional Gaussian structure."""
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
    """Ensure the best-fitting posterior draw ignores the holdout period when scoring history fit."""
    rng = np.random.default_rng(2)
    mu = rng.uniform(0, 10, (20, 30, 2)); y = rng.uniform(0, 10, (30, 2)); s = np.array([0, 4])
    k1, _ = best_fitting_draw(mu, y, s, 24)
    y2 = y.copy(); y2[24:] = 1e6
    k2, _ = best_fitting_draw(mu, y2, s, 24)
    assert k1 == k2


def _t_percentile_labels():
    """Check the percentile label convention is consistent for both reporting styles."""
    assert p_labels("percentile")["P10"] == "P10" and p_labels("reserves")["P10"] == "P90"
    q = percentiles(np.arange(1000.0).reshape(1000, 1, 1))
    assert q["P10"][0, 0] < q["P50"][0, 0] < q["P90"][0, 0]


def run_self_tests():
    """Execute the built-in numerical regression suite and return whether all checks pass."""
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


# 7. COMMAND LINE

def main():
    """Parse CLI options and execute either the main pipeline or the synthetic-recovery/self-test branches.

    Parameters:
        None; arguments are read from sys.argv via argparse.

    Returns:
        None or a synthetic-check result object depending on the selected CLI mode.

    Side effects:
        Reads command-line arguments, runs the CRM workflow, and exits the process when requested by the self-test flag.
    """
    ap = build_parser("Bayesian UQ for CRM calibration (CRM only).")
    ap.add_argument("--synthetic-check", action="store_true",
                    help="recover known parameters from CRM-generated data instead of the normal run")
    ap.add_argument("--reps", type=int, default=3, help="synthetic check: replicates")
    ap.add_argument("--noise", type=float, default=1.0, help="synthetic check: Normal noise sd")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_tests() else 1)
    if not a.data:
        ap.error("--data is required")
    a.out = a.out or "results_crm"
    if a.synthetic_check:
        return synthetic_check(a)
    run_pipeline(a)


if __name__ == "__main__":
    main()