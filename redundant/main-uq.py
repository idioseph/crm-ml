#!/usr/bin/env python3
"""Bayesian parameter-level uncertainty quantification for a first-order CRM (v2).

CRM-ML is deliberately NOT part of this program.

Workflow
--------
Excel data -> history/forecast split -> producer-specific initial states
-> deterministic CRM calibration (SLSQP, exact torch gradients)
-> Bayesian posterior over (tau_j, lambda_ij, tau_ij)
-> MAP polish (start point only) -> adaptive HMC, several chains
-> parameter diagnostics -> propagate EVERY posterior draw through the CRM over the
   complete historical + forecast timeline -> functional diagnostics
-> P10/P50/P90 and central 95% parameter-induced intervals
-> CSV outputs and ONE PNG PER PRODUCER (no combined plot).

Design rules that make the deterministic and Bayesian stages the SAME forward model
-----------------------------------------------------------------------------------
* There is exactly one CRM implementation: `crm_forward` (torch, float64). SLSQP
  calibration, the HMC log-posterior and the final propagation all call it.
* A producer starts at its first positive historical observation (its initial state).
  The injection response of that producer starts from zero at that row, exactly like
  the classical CRM at t0. Before the start the prediction is undefined (NaN in outputs).
* Production and injection are divided by the same history scale in every stage, and
  the observed data are divided by it too (log-residuals are scale-free).

Likelihood: log y ~ Normal(log q_CRM, sigma_j^2) on strictly positive rows after the
producer start, with sigma_j^2 ~ InvGamma(a0, b0) integrated out analytically.
Optional lag-1 tempering weight w_j = (1-rho)/(1+rho) (power likelihood; on by default,
switch off with --no-lag1-tempering). Zeros stay in the data/outputs and are excluded
from the likelihood; nothing replaces them.

Interpretation
--------------
The P10/P50/P90 and 95% bands are induced by the posterior of the CRM parameters under
the stated model, priors and likelihood. They are not full predictive intervals. The CSV
additionally contains `predictive_*` columns that add a Lognormal observation-noise draw
(sigma_j from its conditional posterior) so that you can judge real-world coverage; the
plots show only the parameter-level band.
Intervals are meaningful only if the convergence checks pass.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

TOTAL_STAGES = 8
EPS = 1e-12
DTYPE = torch.float64
DIVERGENCE_ENERGY = 1000.0   # |energy error| above this, on a valid trajectory, is a divergence
TAIL_ESS_MIN = 100.0
MIN_MEAN_ACCEPT = 0.50
MAX_DIVERGENCE_FRACTION = 0.01
A0 = 2.0                     # Inverse-Gamma shape for sigma^2 (fixed hyperparameter)
B0 = A0 * 0.20 ** 2          # Inverse-Gamma scale for sigma^2 (fixed hyperparameter)


# -----------------------------------------------------------------------------
# Console helpers
# -----------------------------------------------------------------------------


def stage(number: int, message: str) -> float:
    print("\n" + "=" * 72, flush=True)
    print(f"[{number}/{TOTAL_STAGES}] {message}", flush=True)
    print("=" * 72, flush=True)
    return time.perf_counter()


def progress(message: str) -> None:
    print(f"    {message}", flush=True)


def finished(start_time: float, message: str = "Completed") -> None:
    elapsed = time.perf_counter() - start_time
    label = f"{elapsed:.1f} seconds" if elapsed < 60 else f"{elapsed / 60:.2f} minutes"
    progress(f"✓ {message} ({label})")


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class CRMParameters:
    tau_j: np.ndarray        # [P]
    lambda_ij: np.ndarray    # [I, P]
    tau_ij: np.ndarray       # [I, P]


@dataclass
class CRMData:
    time: np.ndarray
    production: np.ndarray
    injection: np.ndarray
    producer_names: list
    injector_names: list


@dataclass
class HMCResult:
    draws: np.ndarray
    acceptance_rate: float
    mean_accept_prob: float
    final_step_size: float
    divergences: int
    invalid_proposals: int
    warmup_divergences: int
    warmup_invalid: int
    unique_fraction: float
    final_log_posterior: float


# -----------------------------------------------------------------------------
# Excel loading (strict alignment, nothing is silently dropped)
# -----------------------------------------------------------------------------


def load_excel(path) -> CRMData:
    """Load aligned Time, Production and Injection sheets.

    Header row = names. Non-numeric rows below the header (for example a unit row such
    as [MSTB/day]) are ignored. A row that is only partially numeric is an error.
    """
    xls = pd.ExcelFile(path)

    def find(aliases):
        mapping = {str(s).strip().lower(): str(s) for s in xls.sheet_names}
        for alias in aliases:
            if alias.lower() in mapping:
                return mapping[alias.lower()]
        raise ValueError(f"Missing required sheet. Tried {aliases}; found {xls.sheet_names}")

    def parse(sheet_name):
        raw = xls.parse(sheet_name, header=None).dropna(how="all").reset_index(drop=True)
        if raw.empty:
            raise ValueError(f"Sheet '{sheet_name}' is empty.")
        columns = [str(v).strip() for v in raw.iloc[0]]
        numeric = raw.iloc[1:].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(how="all").reset_index(drop=True)
        if numeric.empty:
            raise ValueError(f"Sheet '{sheet_name}' has no numeric data rows.")
        bad = numeric.isna().any(axis=1).to_numpy()
        if bad.any():
            rows = (np.flatnonzero(bad) + 2).tolist()
            shown = ", ".join(str(r) for r in rows[:10])
            raise ValueError(
                f"Sheet '{sheet_name}' has incomplete numeric rows near data rows {shown}. "
                "Fix the workbook instead of silently deleting rows."
            )
        numeric.columns = columns
        return numeric

    t = parse(find(["time", "t", "date", "days"]))
    q = parse(find(["q_observed", "production", "q_obs", "q"]))
    w = parse(find(["w_injection", "injection", "w_inj", "inj"]))
    if len(t) != len(q) or len(t) != len(w):
        raise ValueError(
            f"Time, production and injection must have the same number of rows; got {len(t)}, {len(q)}, {len(w)}."
        )
    time_values = t.iloc[:, 0].to_numpy(float)
    production = q.to_numpy(float)
    injection = w.to_numpy(float)
    if np.any(~np.isfinite(time_values)) or np.any(np.diff(time_values) <= 0):
        raise ValueError("Time must be finite and strictly increasing.")
    if np.any(~np.isfinite(production)) or np.any(production < 0):
        raise ValueError("Production contains NaN, infinity or negative values.")
    if np.any(~np.isfinite(injection)) or np.any(injection < 0):
        raise ValueError("Injection contains NaN, infinity or negative values.")
    if production.shape[1] < 1 or injection.shape[1] < 1:
        raise ValueError("At least one producer and one injector are required.")
    return CRMData(time_values, production, injection, list(q.columns), list(w.columns))


def first_positive_indices(production_history: np.ndarray) -> np.ndarray:
    """First strictly positive historical row of every producer (-1 if none)."""
    n_prod = production_history.shape[1]
    starts = np.full(n_prod, -1, dtype=int)
    for j in range(n_prod):
        positive = np.flatnonzero(production_history[:, j] > 0)
        if positive.size:
            starts[j] = int(positive[0])
    return starts


# -----------------------------------------------------------------------------
# The ONE CRM forward model (torch, float64)
# -----------------------------------------------------------------------------


@dataclass
class Context:
    dt: torch.Tensor        # [N]      N = T-1
    signal: torch.Tensor    # [N, I]   scaled injection at rows 1..T-1
    q_start: torch.Tensor   # [P]      scaled initial state of every producer
    started: torch.Tensor   # [T, P]   bool, row >= producer start
    resp_on: torch.Tensor   # [N, P]   float, row > producer start (response switched on)
    elapsed: torch.Tensor   # [T, P]   time since producer start, clamped at 0


def make_context(time_values, injection_scaled, q_start_scaled, producer_start) -> Context:
    t = torch.tensor(np.asarray(time_values, dtype=float), dtype=DTYPE)
    n_time = t.numel()
    start = torch.tensor(np.asarray(producer_start, dtype=np.int64))
    rows = torch.arange(n_time)[:, None]
    started = rows >= start[None, :]
    resp_on = (rows > start[None, :])[1:].to(DTYPE)
    elapsed = torch.clamp(t[:, None] - t[start][None, :], min=0.0)
    return Context(
        dt=torch.diff(t),
        signal=torch.tensor(np.asarray(injection_scaled, dtype=float)[1:], dtype=DTYPE),
        q_start=torch.tensor(np.asarray(q_start_scaled, dtype=float), dtype=DTYPE),
        started=started,
        resp_on=resp_on,
        elapsed=elapsed,
    )


def crm_forward(tau_j: torch.Tensor, lam: torch.Tensor, tau_ij: torch.Tensor, ctx: Context) -> torch.Tensor:
    """First-order CRM for all producers over the whole context timeline (scaled units).

        q_j(t_n) = q_j(start_j) exp(-(t_n - t_start_j)/tau_j) + sum_i lambda_ij R_ij(n)
        R_ij(n)  = a R_ij(n-1) + (1-a) I_i(n),   a = exp(-dt_n / tau_ij),  R = 0 at start_j

    The recursion is evaluated in closed form in log space:
        R_n = exp(-C_n + logcumsumexp_k(log src_k + C_k)),  C = cumsum(dt/tau),
        src_k = (1-a_k) I_k * [row k after the producer start]
    All exponents are non-positive after combination, so nothing overflows.
    Rows before a producer's start return the floor EPS (outputs mark them as NaN).
    """
    natural = torch.where(
        ctx.started,
        ctx.q_start[None, :] * torch.exp(-ctx.elapsed / tau_j[None, :]),
        torch.zeros_like(ctx.elapsed),
    )
    x = ctx.dt[:, None, None] / tau_ij[None, :, :]                              # [N, I, P]
    src = (-torch.expm1(-x)) * ctx.signal[:, :, None] * ctx.resp_on[:, None, :]  # [N, I, P]
    cum = torch.cumsum(x, dim=0)
    log_src = torch.log(torch.clamp(src, min=1e-300))
    response = torch.exp(-cum + torch.logcumsumexp(log_src + cum, dim=0))      # [N, I, P]
    tail = natural[1:] + torch.sum(response * lam[None, :, :], dim=1)
    out = torch.cat([natural[:1], tail], dim=0)
    out = torch.where(ctx.started, out, torch.zeros_like(out))
    return torch.clamp(out, min=EPS)


def forward_numpy(params: CRMParameters, ctx: Context) -> np.ndarray:
    with torch.no_grad():
        q = crm_forward(
            torch.tensor(np.asarray(params.tau_j, float), dtype=DTYPE),
            torch.tensor(np.asarray(params.lambda_ij, float), dtype=DTYPE),
            torch.tensor(np.asarray(params.tau_ij, float), dtype=DTYPE),
            ctx,
        )
    return q.numpy()


# -----------------------------------------------------------------------------
# Connectivity transformations (softmax with a fixed zero-logit slack category)
# -----------------------------------------------------------------------------


def lam_to_z_start(lam: np.ndarray, delta: float = 0.05) -> np.ndarray:
    """Logits for an initial state. Boundary solutions (lambda = 0, row sum = 1) are
    softened toward the uniform distribution so that the start is not on a flat,
    saturated part of the softmax. Used only to initialise optimisation/HMC."""
    lam = np.asarray(lam, dtype=float)
    n_inj, n_prod = lam.shape
    out = np.empty_like(lam)
    for i in range(n_inj):
        row = np.clip(lam[i], 0.0, None)
        if row.sum() > 1.0:
            row = row / row.sum() * 0.90
        slack = max(1.0 - float(row.sum()), 0.0)
        p = np.r_[row, slack]
        p = p / p.sum()
        p = (1.0 - delta) * p + delta / (n_prod + 1)
        p = p / p.sum()
        out[i] = np.log(p[:-1] / p[-1])
    return out


def softmax_slack_np(z: np.ndarray) -> np.ndarray:
    """z [..., I, P] -> lambda [..., I, P] with slack logit fixed at 0."""
    z = np.asarray(z, dtype=float)
    aug = np.concatenate([z, np.zeros(z.shape[:-1] + (1,))], axis=-1)
    aug = aug - aug.max(axis=-1, keepdims=True)
    e = np.exp(aug)
    return (e / e.sum(axis=-1, keepdims=True))[..., :-1]


# -----------------------------------------------------------------------------
# Deterministic calibration (SLSQP with exact gradients from torch)
# -----------------------------------------------------------------------------


def fit_deterministic(
    ctx: Context,
    y_scaled: np.ndarray,
    fit_mask: np.ndarray,
    tmin: float,
    span: float,
    n_inj: int,
    n_prod: int,
    starts: int,
    max_iter: int,
    seed: int,
    mode: str,
):
    """Constrained least squares of the CRM on the history.

    x = [log tau_j (P), lambda_ij (I*P), log tau_ij (I*P)]
    bounds: log tau >= log(tmin), 0 <= lambda <= 1; constraint: sum_j lambda_ij <= 1.
    `log-positive` minimises the mean squared LOG residual on the same rows that enter the
    Bayesian likelihood (consistent with it); `linear-positive` uses linear residuals.
    """
    P, I = n_prod, n_inj
    lam_off = P
    tau_off = P + I * P
    n_par = P + 2 * I * P
    log_tmin = math.log(tmin)
    bounds = [(log_tmin, None)] * P + [(0.0, 1.0)] * (I * P) + [(log_tmin, None)] * (I * P)
    A = np.zeros((I, n_par))
    for i in range(I):
        A[i, lam_off + i * P: lam_off + (i + 1) * P] = 1.0
    constraints = {"type": "ineq", "fun": lambda x: 1.0 - A @ x, "jac": lambda x: -A}

    y1 = torch.tensor(np.asarray(y_scaled, float)[1:], dtype=DTYPE)
    maskf = torch.tensor(np.asarray(fit_mask, dtype=float), dtype=DTYPE)
    log_y1 = torch.log(torch.where(maskf > 0, y1, torch.ones_like(y1)))  # placeholder never used: masked out
    n_use = max(float(maskf.sum()), 1.0)

    def fun(x):
        xt = torch.tensor(np.asarray(x, float), dtype=DTYPE, requires_grad=True)
        tau_j = torch.exp(xt[:P])
        lam = xt[lam_off:tau_off].reshape(I, P)
        tau_ij = torch.exp(xt[tau_off:]).reshape(I, P)
        q = crm_forward(tau_j, lam, tau_ij, ctx)[1:]
        if mode == "log-positive":
            resid = (log_y1 - torch.log(q)) * maskf
        else:
            resid = (y1 - q) * maskf
        loss = torch.sum(resid ** 2) / n_use
        if not bool(torch.isfinite(loss)):
            return 1e10, np.zeros(n_par)
        loss.backward()
        grad = xt.grad.numpy().copy()
        if not np.all(np.isfinite(grad)):
            return 1e10, np.zeros(n_par)
        return float(loss.detach()), grad

    rng = np.random.default_rng(seed)
    lower_start = log_tmin + 0.05
    upper_log = math.log(max(10.0 * tmin, span))
    best = None
    for k in range(starts):
        x0 = np.empty(n_par)
        if k == 0:
            init = max(math.log(max(2.0 * tmin, 0.1 * span)), lower_start)
            x0[:P] = init
            x0[lam_off:tau_off] = 0.6 / max(P, 1)
            x0[tau_off:] = init
        else:
            x0[:P] = rng.uniform(lower_start, upper_log, P)
            x0[lam_off:tau_off] = (0.6 * rng.dirichlet(np.ones(P), size=I)).ravel()
            x0[tau_off:] = rng.uniform(lower_start, upper_log, I * P)
        res = minimize(
            fun, x0, jac=True, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": int(max_iter), "ftol": 1e-10},
        )
        progress(f"Calibration start {k + 1}/{starts}: objective={res.fun:.8g}, success={res.success}")
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("All deterministic CRM calibration starts failed.")
    x = best.x
    params = CRMParameters(
        tau_j=np.exp(x[:P]),
        lambda_ij=np.clip(x[lam_off:tau_off].reshape(I, P), 0.0, 1.0),
        tau_ij=np.exp(x[tau_off:]).reshape(I, P),
    )
    return params, float(best.fun), bool(best.success), str(best.message)


def lag1_weight(log_residual: np.ndarray) -> float:
    x = np.asarray(log_residual, dtype=float)
    if x.size < 4 or np.std(x) <= 0:
        return 1.0
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    if not np.isfinite(rho):
        return 1.0
    rho = float(np.clip(rho, 0.0, 0.90))
    return max((1.0 - rho) / (1.0 + rho), 0.05)


# -----------------------------------------------------------------------------
# Bayesian model in standardised coordinates
# -----------------------------------------------------------------------------


class BayesModel:
    """theta = [u (P), zs (I*P), v (I*P)] with
        log tau_j  = log_ref + tau_sd * u,   log tau_ij = log_ref + tau_sd * v,
        logit z    = lambda_sd * zs,         lambda = softmax([z, 0])[:P].
    Priors are u, zs, v ~ N(0, 1), i.e. log tau ~ N(log_ref, tau_sd^2), z ~ N(0, lambda_sd^2),
    truncated at log tau >= log(tmin). tau_sd, lambda_sd, log_ref, A0, B0 are fixed modelling
    hyperparameters; none of them is learned from the production data.
    """

    def __init__(self, ctx_hist, y_scaled, mask_hist, tau_sd, lambda_sd, tmin, span, rho_weights, n_inj, n_prod):
        self.ctx = ctx_hist
        self.I, self.P = n_inj, n_prod
        self.tau_sd, self.lambda_sd = float(tau_sd), float(lambda_sd)
        self.log_tmin = math.log(tmin)
        self.log_ref = 0.5 * (math.log(tmin) + math.log(span))
        y1 = torch.tensor(np.asarray(y_scaled, float)[1:], dtype=DTYPE)
        self.maskf = torch.tensor(np.asarray(mask_hist, dtype=float), dtype=DTYPE)
        self.log_y1 = torch.log(torch.where(self.maskf > 0, y1, torch.ones_like(y1)))  # masked placeholder
        self.log_y1_np = self.log_y1.numpy().copy()
        self.mask_np = np.asarray(mask_hist, dtype=bool)
        self.n_obs = self.maskf.sum(dim=0)
        self.weight = torch.tensor(np.asarray(rho_weights, float), dtype=DTYPE)
        self.zero_col = torch.zeros((n_inj, 1), dtype=DTYPE)
        IP = n_inj * n_prod
        self.dim = n_prod + 2 * IP
        self.tau_idx = list(range(n_prod)) + list(range(n_prod + IP, n_prod + 2 * IP))
        self.tau_lower_std = (self.log_tmin - self.log_ref) / self.tau_sd

    def split(self, theta):
        P, I = self.P, self.I
        IP = I * P
        u = theta[:P]
        zs = theta[P:P + IP].reshape(I, P)
        v = theta[P + IP:P + 2 * IP].reshape(I, P)
        return u, zs, v

    def physical(self, theta):
        u, zs, v = self.split(theta)
        lt_j = self.log_ref + self.tau_sd * u
        lt_ij = self.log_ref + self.tau_sd * v
        lam = torch.softmax(torch.cat([self.lambda_sd * zs, self.zero_col], dim=1), dim=1)[:, :self.P]
        return lt_j, lt_ij, torch.exp(lt_j), lam, torch.exp(lt_ij)

    def log_prior(self, theta):
        u, zs, v = self.split(theta)
        return -0.5 * (torch.sum(u ** 2) + torch.sum(zs ** 2) + torch.sum(v ** 2))

    def log_lik(self, q_hist_scaled):
        """Marginal (sigma^2 integrated out) tempered Lognormal log-likelihood.
        q_hist_scaled: model output over the history rows [T_hist, P]."""
        resid = (self.log_y1 - torch.log(q_hist_scaled[1:])) * self.maskf
        rss = torch.sum(resid ** 2, dim=0)
        return -torch.sum((A0 + 0.5 * self.weight * self.n_obs) * torch.log(B0 + 0.5 * self.weight * rss))

    def logpost(self, theta):
        lt_j, lt_ij, tau_j, lam, tau_ij = self.physical(theta)
        if bool((lt_j < self.log_tmin - 1e-9).any()) or bool((lt_ij < self.log_tmin - 1e-9).any()):
            return torch.tensor(-float("inf"), dtype=DTYPE)
        q = crm_forward(tau_j, lam, tau_ij, self.ctx)
        return self.log_lik(q) + self.log_prior(theta)

    def theta_from_params(self, params: CRMParameters) -> np.ndarray:
        u = (np.log(np.maximum(params.tau_j, EPS)) - self.log_ref) / self.tau_sd
        zs = lam_to_z_start(params.lambda_ij) / self.lambda_sd
        v = (np.log(np.maximum(params.tau_ij, EPS)) - self.log_ref) / self.tau_sd
        theta = np.concatenate([u, zs.ravel(), v.ravel()])
        theta[self.tau_idx] = np.maximum(theta[self.tau_idx], self.tau_lower_std)
        return theta


def make_vg(model: BayesModel) -> Callable:
    """(log posterior, gradient) or (-inf, None) when the state is invalid."""

    def vg(theta: torch.Tensor):
        q = theta.detach().clone().requires_grad_(True)
        lp = model.logpost(q)
        if not bool(torch.isfinite(lp)):
            return float("-inf"), None
        (grad,) = torch.autograd.grad(lp, q)
        if not bool(torch.isfinite(grad).all()):
            return float("-inf"), None
        return float(lp.detach()), grad.detach()

    return vg


def map_polish(model: BayesModel, theta0: np.ndarray, maxiter: int) -> np.ndarray:
    """L-BFGS-B on the log-posterior in the HMC coordinates (tau bounds as true bounds).
    Used only to find a good starting region; HMC samples the same target."""
    d = theta0.size
    bounds = [(None, None)] * d
    for i in model.tau_idx:
        bounds[i] = (model.tau_lower_std, None)

    def fun(x):
        v = torch.tensor(np.asarray(x, float), dtype=DTYPE, requires_grad=True)
        lp = model.logpost(v)
        if not bool(torch.isfinite(lp)):
            return 1e12, np.zeros(d)
        (g,) = torch.autograd.grad(lp, v)
        g = g.numpy().copy()
        if not np.all(np.isfinite(g)):
            return 1e12, np.zeros(d)
        return -float(lp.detach()), -g

    f0, _ = fun(theta0)
    res = minimize(fun, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter)})
    if np.isfinite(res.fun) and res.fun < f0:
        return np.asarray(res.x, float)
    return np.asarray(theta0, float)


# -----------------------------------------------------------------------------
# HMC: Stan-style warm-up (dual averaging + windowed diagonal metric), jittered L
# -----------------------------------------------------------------------------


class DualAveraging:
    def __init__(self, step: float, target: float, gamma: float = 0.05, t0: float = 10.0, kappa: float = 0.75):
        self.target, self.gamma, self.t0, self.kappa = float(target), gamma, t0, kappa
        self.restart(step)

    def restart(self, step: float) -> None:
        self.mu = math.log(10.0 * step)
        self.m = 0
        self.h_bar = 0.0
        self.log_eps = math.log(step)
        self.log_eps_bar = math.log(step)

    @staticmethod
    def _clip(eps: float) -> float:
        return float(np.clip(eps, 1e-5, 1.5))

    def update(self, accept_prob: float) -> float:
        self.m += 1
        eta = 1.0 / (self.m + self.t0)
        self.h_bar = (1.0 - eta) * self.h_bar + eta * (self.target - accept_prob)
        self.log_eps = self.mu - math.sqrt(self.m) / self.gamma * self.h_bar
        w = self.m ** (-self.kappa)
        self.log_eps_bar = w * self.log_eps + (1.0 - w) * self.log_eps_bar
        return self._clip(math.exp(self.log_eps))

    def final(self) -> float:
        return self._clip(math.exp(self.log_eps_bar))


def run_hmc(vg, theta0, samples, burnin, thin, step_size, n_leapfrog, jitter, target_accept, seed) -> HMCResult:
    """HMC with a diagonal metric.

    Hamiltonian: H = -logp + 0.5 * sum(p^2 * inv_mass), p ~ N(0, 1/inv_mass), q += eps * inv_mass * p.
    inv_mass is the estimated POSTERIOR VARIANCE (Stan convention).
    Warm-up: 15% step size only, then doubling windows of metric estimation (running estimate
    reset after every window), last 10% step size only. Dual averaging restarts after every
    metric update. The kernel is frozen after warm-up. The number of leapfrog steps is drawn
    uniformly each iteration. Wall (invalid) rejections are counted separately from divergences.
    """
    if samples < 20 or burnin < 100 or thin < 1:
        raise ValueError("Use at least 20 retained samples, 100 burn-in iterations and thin >= 1.")
    rng = np.random.default_rng(int(seed))
    q = torch.tensor(np.asarray(theta0, float), dtype=DTYPE)
    lp, g = vg(q)
    if g is None:
        raise RuntimeError("HMC start has an invalid posterior density or gradient.")
    d = q.numel()
    inv_mass = torch.ones(d, dtype=DTYPE)
    eps = float(step_size)
    da = DualAveraging(eps, target_accept)

    total = burnin + samples * thin
    init_buf = int(0.15 * burnin)
    adapt_end = burnin - int(0.10 * burnin)
    window = max(50, burnin // 10)
    pos, ends = init_buf, []
    while True:
        end = pos + window
        if end + 2 * window > adapt_end:
            ends.append(adapt_end)
            break
        ends.append(end)
        pos, window = end, window * 2
    end_idx = 0
    w_mean = torch.zeros(d, dtype=DTYPE)
    w_m2 = torch.zeros(d, dtype=DTYPE)
    w_n = 0

    lo = max(1, int(round(n_leapfrog * (1.0 - jitter))))
    hi = max(lo, int(round(n_leapfrog * (1.0 + jitter))))

    retained = np.empty((samples, d))
    saved = 0
    n_acc = 0
    acc_sum = 0.0
    div_post = inv_post = div_warm = inv_warm = 0
    checkpoints = set(np.unique(np.linspace(1, total, min(10, total)).astype(int)).tolist())

    for it in range(total):
        L = int(rng.integers(lo, hi + 1))
        p0 = torch.tensor(rng.standard_normal(d), dtype=DTYPE) / torch.sqrt(inv_mass)
        h0 = -lp + 0.5 * float(torch.sum(p0 * p0 * inv_mass))

        qn = q
        pn = p0 + 0.5 * eps * g
        gn = g
        lpn = lp
        valid = True
        for i in range(L):
            qn = qn + eps * inv_mass * pn
            lpn, g_new = vg(qn)
            if g_new is None:
                valid = False
                break
            gn = g_new
            if i < L - 1:
                pn = pn + eps * gn

        accepted = False
        accept_prob = 0.0
        d_h = math.inf
        if valid:
            pn = pn + 0.5 * eps * gn
            h1 = -lpn + 0.5 * float(torch.sum(pn * pn * inv_mass))
            d_h = h1 - h0
            if math.isfinite(d_h):
                accept_prob = 1.0 if d_h <= 0.0 else math.exp(-d_h)
                if rng.random() < accept_prob:
                    q, lp, g = qn, lpn, gn
                    accepted = True
        divergent = valid and ((not math.isfinite(d_h)) or d_h > DIVERGENCE_ENERGY)

        if it < burnin:
            if not valid:
                inv_warm += 1
            if divergent:
                div_warm += 1
            eps = da.update(accept_prob)
            if init_buf <= it < adapt_end:
                w_n += 1
                delta = q - w_mean
                w_mean = w_mean + delta / w_n
                w_m2 = w_m2 + delta * (q - w_mean)
                if end_idx < len(ends) and it + 1 == ends[end_idx]:
                    var = w_m2 / max(w_n - 1, 1)
                    var = (w_n / (w_n + 5.0)) * var + 1e-3 * (5.0 / (w_n + 5.0))
                    inv_mass = torch.clamp(var, 1e-4, 1e4)
                    w_mean = torch.zeros(d, dtype=DTYPE)
                    w_m2 = torch.zeros(d, dtype=DTYPE)
                    w_n = 0
                    end_idx += 1
                    da.restart(eps)
            if it == burnin - 1:
                eps = da.final()
        else:
            if not valid:
                inv_post += 1
            if divergent:
                div_post += 1
            if accepted:
                n_acc += 1
            acc_sum += accept_prob
            if (it - burnin + 1) % thin == 0:
                retained[saved] = q.numpy()
                saved += 1

        if (it + 1) in checkpoints:
            phase = "warm-up" if it < burnin else "sampling"
            progress(f"{phase}: {it + 1}/{total} ({100.0 * (it + 1) / total:.0f}%) | step={eps:.4g} | accept_prob={accept_prob:.3f}")

    if saved != samples:
        raise RuntimeError(f"HMC retained {saved} draws; expected {samples}.")
    n_post = max(samples * thin, 1)
    return HMCResult(
        draws=retained,
        acceptance_rate=n_acc / n_post,
        mean_accept_prob=acc_sum / n_post,
        final_step_size=eps,
        divergences=div_post,
        invalid_proposals=inv_post,
        warmup_divergences=div_warm,
        warmup_invalid=inv_warm,
        unique_fraction=np.unique(retained, axis=0).shape[0] / samples,
        final_log_posterior=float(lp),
    )


# -----------------------------------------------------------------------------
# Convergence diagnostics
# -----------------------------------------------------------------------------


def split_chains(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    m, n, d = a.shape
    half = n // 2
    if half < 4:
        raise ValueError("At least 8 retained draws per chain are required for diagnostics.")
    return a[:, :2 * half, :].reshape(2 * m, half, d)


def rhat(a: np.ndarray) -> np.ndarray:
    """Split R-hat. Constant / non-finite parameters return inf (they fail)."""
    s = split_chains(a)
    m, n, d = s.shape
    within = s.var(axis=1, ddof=1).mean(axis=0)
    between = n * s.mean(axis=1).var(axis=0, ddof=1)
    var_hat = ((n - 1) / n) * within + between / n
    out = np.full(d, np.inf)
    ok = np.isfinite(within) & np.isfinite(var_hat) & (within > 1e-300)
    out[ok] = np.sqrt(var_hat[ok] / within[ok])
    return out


def _ess_series(x: np.ndarray) -> float:
    """Multi-chain ESS of one scalar series x [m, n] (Geyer initial monotone sequence)."""
    m, n = x.shape
    if n < 4:
        return 1.0
    cm = x.mean(axis=1)
    w = float(x.var(axis=1, ddof=1).mean())
    if not np.isfinite(w) or w <= 1e-300:
        return 1.0
    var_plus = (n - 1) / n * w + (float(cm.var(ddof=1)) if m > 1 else 0.0)
    f = np.fft.rfft(x - cm[:, None], n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n=2 * n, axis=1)[:, :n] / n
    rho = 1.0 - (w - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pair_sum, prev, t = 0.0, np.inf, 0
    while t + 1 < n:
        pair = float(rho[t] + rho[t + 1])
        if pair < 0.0:
            break
        pair = min(pair, prev)
        prev = pair
        pair_sum += pair
        t += 2
    tau = -1.0 + 2.0 * pair_sum
    return float(np.clip(m * n / max(tau, 1e-12), 1.0, m * n))


def ess_bulk(a: np.ndarray) -> np.ndarray:
    s = split_chains(a)
    return np.array([_ess_series(s[:, :, k]) for k in range(s.shape[2])])


def ess_tail(a: np.ndarray) -> np.ndarray:
    """min ESS of the indicators x <= q05 and x <= q95 (tail behaviour, i.e. the intervals)."""
    s = split_chains(a)
    out = np.empty(s.shape[2])
    for k in range(s.shape[2]):
        x = s[:, :, k]
        q05, q95 = np.quantile(x, [0.05, 0.95])
        out[k] = min(_ess_series((x <= q05).astype(float)), _ess_series((x <= q95).astype(float)))
    return out


def theta_names(producers, injectors):
    return (
        [f"{p}::std_log_tau_j" for p in producers]
        + [f"{i}->{p}::std_lambda_logit" for i in injectors for p in producers]
        + [f"{i}->{p}::std_log_tau_ij" for i in injectors for p in producers]
    )


def _summ(name, vals):
    return {
        "parameter": name, "mean": float(np.mean(vals)), "median": float(np.median(vals)),
        "sd": float(np.std(vals, ddof=1)), "q2.5": float(np.quantile(vals, 0.025)),
        "q97.5": float(np.quantile(vals, 0.975)),
    }


def posterior_summary(flat, model: BayesModel, producers, injectors) -> pd.DataFrame:
    D = flat.shape[0]
    P, I = model.P, model.I
    IP = I * P
    tau_j = np.exp(model.log_ref + model.tau_sd * flat[:, :P])
    z = model.lambda_sd * flat[:, P:P + IP].reshape(D, I, P)
    lam = softmax_slack_np(z)
    tau_ij = np.exp(model.log_ref + model.tau_sd * flat[:, P + IP:]).reshape(D, I, P)
    rows = [_summ(f"{p}::tau_j", tau_j[:, j]) for j, p in enumerate(producers)]
    for i, inj in enumerate(injectors):
        for j, p in enumerate(producers):
            rows.append(_summ(f"{inj}->{p}::lambda", lam[:, i, j]))
            rows.append(_summ(f"{inj}->{p}::tau_ij", tau_ij[:, i, j]))
    slack = 1.0 - lam.sum(axis=2)
    for i, inj in enumerate(injectors):
        rows.append(_summ(f"{inj}::slack_(1-sum_lambda)", slack[:, i]))
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots (one figure per producer, no combined figure)
# -----------------------------------------------------------------------------


def safe_filename(name: str) -> str:
    keep = [c if (c.isalnum() or c in "._-") else "_" for c in str(name)]
    return "".join(keep).strip("._") or "producer"


def save_individual_plots(ts: pd.DataFrame, producers, out_dir: Path, boundary: float) -> None:
    """Save two per-producer plot sets using the original rows without smoothing.

    Set 1: every historical and forecast row, with a vertical split marker.
    Set 2: forecast rows only, preserving their original time values.

    Markers show each source row. Lines only connect adjacent recorded points;
    there is no interpolation, resampling, rolling mean, or smoothing.
    """
    plot_sets = [
        ("history_and_forecast", ts.copy(), "History + forecast"),
        ("forecast_only", ts.loc[ts["period"].astype(str).str.lower().eq("forecast")].copy(), "Forecast only"),
    ]

    for folder_name, plot_data, set_title in plot_sets:
        plot_dir = out_dir / "wells" / folder_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        used = set()

        for name in producers:
            d = plot_data.loc[plot_data["producer"] == name].sort_values("time", kind="stable")
            if d.empty:
                progress(f"No rows for {name} in plot set '{folder_name}'; skipped.")
                continue

            x = d["time"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(11, 6.2))

            # Parameter uncertainty band comes directly from saved CSV columns.
            lower = d["95pct_lower"].to_numpy(dtype=float)
            upper = d["95pct_upper"].to_numpy(dtype=float)
            if np.isfinite(lower).any() and np.isfinite(upper).any():
                ax.fill_between(x, lower, upper, alpha=0.16,
                                label="95% parameter interval", zorder=1)

            # Exact datapoints are marked; the line simply connects consecutive rows.
            observed = d["observed"].to_numpy(dtype=float)
            ax.plot(x, observed, marker="o", markersize=3.5, linewidth=0.9,
                    label="Observed production", zorder=4)

            p50 = d["P50"].to_numpy(dtype=float)
            ax.plot(x, p50, marker="o", markersize=2.8, linewidth=1.1,
                    linestyle="--", label="Bayesian CRM P50", zorder=3)

            if "deterministic_crm" in d.columns:
                deterministic = d["deterministic_crm"].to_numpy(dtype=float)
                if np.isfinite(deterministic).any():
                    ax.plot(x, deterministic, marker=".", markersize=3.0, linewidth=0.9,
                            label="Deterministic CRM", zorder=2)

            if folder_name == "history_and_forecast":
                ax.axvline(boundary, linestyle=":", linewidth=1.3, color="black",
                           label="Forecast begins", zorder=5)

            ax.set_xlabel("Time (original data values)")
            ax.set_ylabel("Production rate (input units)")
            ax.set_title(f"{name}: {set_title}")
            ax.grid(True, which="major", alpha=0.22, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=8.5, frameon=False, loc="best")
            fig.tight_layout()

            fname = safe_filename(name)
            if fname in used:
                fname = f"{fname}_{len(used)}"
            used.add(fname)
            path = plot_dir / f"{fname}.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            progress(f"Saved {folder_name} plot for {name}: {path}")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Bayesian parameter-level CRM UQ (adaptive HMC)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--forecast", type=int, required=True)
    ap.add_argument("--mcmc-chains", type=int, default=4)
    ap.add_argument("--mcmc-samples", type=int, default=1000)
    ap.add_argument("--mcmc-burnin", type=int, default=1500)
    ap.add_argument("--mcmc-thin", type=int, default=1)
    ap.add_argument("--hmc-step-size", type=float, default=0.05, help="initial step (adapted in warm-up)")
    ap.add_argument("--hmc-leapfrog-steps", type=int, default=20, help="mean trajectory length")
    ap.add_argument("--hmc-leapfrog-jitter", type=float, default=0.3, help="fractional jitter of L")
    ap.add_argument("--hmc-target-accept", type=float, default=0.80)
    ap.add_argument("--tau-log-sd", type=float, default=2.0)
    ap.add_argument("--lambda-logit-sd", type=float, default=1.5)
    ap.add_argument("--calibration-objective", choices=["log-positive", "linear-positive"], default="log-positive")
    ap.add_argument("--crm-starts", type=int, default=3)
    ap.add_argument("--crm-max-iter", type=int, default=400)
    ap.add_argument("--map-iter", type=int, default=200)
    ap.add_argument("--start-jitter", type=float, default=0.5, help="sd of chain start dispersion (standardised units)")
    ap.add_argument("--no-lag1-tempering", action="store_true")
    ap.add_argument("--rhat-threshold", type=float, default=1.05)
    ap.add_argument("--ess-threshold", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default="bayesian_crm_parameter_uq")
    args = ap.parse_args()

    if args.forecast < 1:
        ap.error("--forecast must be at least 1")
    if args.mcmc_chains < 2:
        ap.error("--mcmc-chains must be at least 2")
    if args.mcmc_samples < 20 or args.mcmc_burnin < 100 or args.mcmc_thin < 1:
        ap.error("need --mcmc-samples >= 20, --mcmc-burnin >= 100, --mcmc-thin >= 1")
    if args.hmc_step_size <= 0 or args.hmc_leapfrog_steps < 1 or not (0.0 <= args.hmc_leapfrog_jitter < 1.0):
        ap.error("invalid HMC step size / leapfrog settings")
    if not (0.0 < args.hmc_target_accept < 1.0):
        ap.error("--hmc-target-accept must be in (0, 1)")
    if args.tau_log_sd <= 0 or args.lambda_logit_sd <= 0:
        ap.error("prior standard deviations must be positive")

    print("\n" + "=" * 72, flush=True)
    print("BAYESIAN CRM PARAMETER-LEVEL UNCERTAINTY QUANTIFICATION (v2)", flush=True)
    print("CRM-ML: EXCLUDED", flush=True)
    print("=" * 72, flush=True)

    # ---------------------------------------------------------------- 1. data
    t0 = stage(1, "Loading and validating data...")
    data = load_excel(args.data)
    n_total = len(data.time)
    P, I = len(data.producer_names), len(data.injector_names)
    progress(f"Time points: {n_total}; producers: {P}; injectors: {I}")
    finished(t0, "Data loaded")

    # ---------------------------------------------------------------- 2. split
    t0 = stage(2, "Preparing historical + forecast timeline...")
    n_history = n_total - args.forecast
    if n_history < 20:
        raise ValueError("At least 20 historical rows are required before the forecast window.")
    hist_time = data.time[:n_history]
    hist_prod = data.production[:n_history]
    hist_inj = data.injection[:n_history]
    producer_start = first_positive_indices(hist_prod)
    if np.any(producer_start < 0):
        bad = [data.producer_names[j] for j in np.flatnonzero(producer_start < 0)]
        raise ValueError(f"No positive historical production found for: {bad}")

    scale = max(float(np.mean(hist_prod)), EPS)          # one scale for every stage
    q_start_phys = hist_prod[producer_start, np.arange(P)]
    q_start_scaled = q_start_phys / scale
    inj_scaled_full = data.injection / scale
    y_hist_scaled = hist_prod / scale
    ctx_hist = make_context(hist_time, inj_scaled_full[:n_history], q_start_scaled, producer_start)
    ctx_full = make_context(data.time, inj_scaled_full, q_start_scaled, producer_start)
    rows_idx = np.arange(n_history)[:, None]
    mask_hist = ((hist_prod > 0) & (rows_idx > producer_start[None, :]))[1:]     # rows 1..n_history-1
    n_obs = mask_hist.sum(axis=0)
    if np.any(n_obs < 8):
        bad = [data.producer_names[j] for j in np.flatnonzero(n_obs < 8)]
        raise ValueError(f"Producer(s) {bad} have fewer than 8 positive observations after their start row.")
    tmin = float(np.min(np.diff(hist_time)))
    span = max(float(hist_time[-1] - hist_time[0]), 1.01 * tmin)

    progress(f"Historical rows: {n_history}; forecast rows: {args.forecast}")
    progress(f"Last historical time: {data.time[n_history - 1]:g}; first forecast time: {data.time[n_history]:g}")
    progress(f"Production/injection scale (history mean production): {scale:.8g}")
    for j, name in enumerate(data.producer_names):
        progress(f"{name}: start row {producer_start[j] + 1}, time {hist_time[producer_start[j]]:g}, "
                 f"q_start={q_start_phys[j]:g}, positive likelihood rows={int(n_obs[j])}")
    finished(t0, "Timeline prepared")

    # ---------------------------------------------------------------- 3. deterministic CRM
    t0 = stage(3, "Deterministic CRM calibration...")
    progress(f"Objective: {args.calibration_objective}; starts: {args.crm_starts}; max iterations: {args.crm_max_iter}")
    det_params, det_obj, det_ok, det_msg = fit_deterministic(
        ctx_hist, y_hist_scaled, mask_hist, tmin, span, I, P,
        args.crm_starts, args.crm_max_iter, args.seed, args.calibration_objective)
    progress(f"Best objective: {det_obj:.8g}; success={det_ok}; message={det_msg}")
    det_q_full_scaled = forward_numpy(det_params, ctx_full)
    det_pred_full = det_q_full_scaled * scale
    det_q_hist_scaled = det_q_full_scaled[:n_history]
    finished(t0, "Deterministic CRM calibrated")

    # ---------------------------------------------------------------- 4. posterior + MAP start
    t0 = stage(4, "Building Bayesian posterior...")
    rho_weights = np.ones(P)
    if not args.no_lag1_tempering:
        for j in range(P):
            m = mask_hist[:, j] & (det_q_hist_scaled[1:, j] > EPS)
            if int(m.sum()) >= 4:
                resid = np.log(y_hist_scaled[1:, j][m]) - np.log(det_q_hist_scaled[1:, j][m])
                rho_weights[j] = lag1_weight(resid)
    model = BayesModel(ctx_hist, y_hist_scaled, mask_hist, args.tau_log_sd, args.lambda_logit_sd,
                       tmin, span, rho_weights, I, P)
    vg = make_vg(model)
    progress(f"HMC dimension: {model.dim}; minimum tau: {tmin:g}; reference tau: {math.exp(model.log_ref):g}")
    progress(f"Fixed hyperparameters: log(tau) prior SD={args.tau_log_sd:g}, lambda-logit prior SD={args.lambda_logit_sd:g}, "
             f"sigma^2 ~ InvGamma({A0:g}, {B0:g}) (not learned from data)")
    progress("Likelihood: Lognormal on positive rows after each producer start; sigma^2 integrated out")
    progress("Lag-1 tempering weights: " + (np.array2string(np.round(rho_weights, 3), separator=", ")
                                              if not args.no_lag1_tempering else "disabled (all 1)"))

    theta_det = model.theta_from_params(det_params)
    lp_det, g_det = vg(torch.tensor(theta_det, dtype=DTYPE))
    if g_det is None:
        raise RuntimeError("The deterministic solution has an invalid posterior density.")
    progress(f"Log posterior at deterministic solution: {lp_det:.6g}")
    theta_map = map_polish(model, theta_det, args.map_iter) if args.map_iter > 0 else theta_det.copy()
    margin = 0.05 / model.tau_sd
    theta_map[model.tau_idx] = np.maximum(theta_map[model.tau_idx], model.tau_lower_std + margin)
    lp_map, g_map = vg(torch.tensor(theta_map, dtype=DTYPE))
    if g_map is None or lp_map < lp_det:
        theta_map, lp_map = theta_det.copy(), lp_det
        theta_map[model.tau_idx] = np.maximum(theta_map[model.tau_idx], model.tau_lower_std + margin)
        lp_map, g_map = vg(torch.tensor(theta_map, dtype=DTYPE))
        if g_map is None:
            raise RuntimeError("Could not find a valid HMC starting region.")
    progress(f"Log posterior at MAP start: {lp_map:.6g}")
    finished(t0, "Bayesian posterior constructed")

    # ---------------------------------------------------------------- 5. HMC
    t0 = stage(5, "Running adaptive Hamiltonian Monte Carlo...")
    progress(f"Chains: {args.mcmc_chains}; retained per chain: {args.mcmc_samples}; warm-up: {args.mcmc_burnin}; thin: {args.mcmc_thin}")
    progress(f"Leapfrog steps: {args.hmc_leapfrog_steps} (jitter {args.hmc_leapfrog_jitter:g}); target accept: {args.hmc_target_accept:g}")
    results = []
    for c in range(args.mcmc_chains):
        seed_c = args.seed + 100_000 * c
        progress(f"Starting chain {c + 1}/{args.mcmc_chains}")
        rng_c = np.random.default_rng(seed_c + 7)
        start = None
        for attempt in range(30):
            jit = args.start_jitter * (0.8 ** attempt)
            cand = theta_map + rng_c.normal(0.0, jit, theta_map.size)
            cand[model.tau_idx] = np.maximum(cand[model.tau_idx], model.tau_lower_std + margin)
            _, g_c = vg(torch.tensor(cand, dtype=DTYPE))
            if g_c is not None:
                start = cand
                break
        if start is None:
            start = theta_map.copy()
        res = run_hmc(vg, start, args.mcmc_samples, args.mcmc_burnin, args.mcmc_thin, args.hmc_step_size,
                      args.hmc_leapfrog_steps, args.hmc_leapfrog_jitter, args.hmc_target_accept, seed_c)
        results.append(res)
        progress(f"Chain {c + 1}: acceptance={res.acceptance_rate:.2%}, mean_accept_prob={res.mean_accept_prob:.2%}, "
                 f"step={res.final_step_size:.4g}, divergences={res.divergences}, wall/invalid={res.invalid_proposals}, "
                 f"unique={res.unique_fraction:.2%}")
    chains = np.stack([r.draws for r in results], axis=0)          # [C, S, D]
    C, S, D = chains.shape
    finished(t0, "HMC sampling completed")

    # ---------------------------------------------------------------- 6. parameter diagnostics
    t0 = stage(6, "Checking MCMC diagnostics (parameters)...")
    names = theta_names(data.producer_names, data.injector_names)
    rhat_v, ess_b, ess_t = rhat(chains), ess_bulk(chains), ess_tail(chains)
    diag_df = pd.DataFrame({"parameter": names, "R_hat": rhat_v, "ESS_bulk": ess_b, "ESS_tail": ess_t})
    progress(f"Max R-hat: {np.max(rhat_v):.4g}; min bulk ESS: {np.min(ess_b):.1f}; min tail ESS: {np.min(ess_t):.1f}")
    finished(t0, "Parameter diagnostics calculated")

    # ---------------------------------------------------------------- 7. propagation + functional diagnostics
    t0 = stage(7, "Propagating EVERY posterior draw over the full historical + forecast timeline...")
    flat = chains.reshape(-1, D)
    n_draws = flat.shape[0]
    T = n_total
    preds = np.empty((n_draws, T, P))
    log_post = np.empty(n_draws)
    ckpt = set(np.unique(np.linspace(1, n_draws, min(10, n_draws)).astype(int)).tolist())
    with torch.no_grad():
        for k in range(n_draws):
            th = torch.tensor(flat[k], dtype=DTYPE)
            _, _, tau_j, lam, tau_ij = model.physical(th)
            q_full = crm_forward(tau_j, lam, tau_ij, ctx_full)
            preds[k] = q_full.numpy()
            log_post[k] = float(model.log_lik(q_full[:n_history]) + model.log_prior(th))
            if (k + 1) in ckpt:
                progress(f"Propagation: {k + 1}/{n_draws} ({100.0 * (k + 1) / n_draws:.0f}%)")
    preds *= scale                                                  # physical units

    pr = preds.reshape(C, S, T, P)
    f_last = np.log(np.maximum(pr[:, :, n_history - 1, :], EPS))
    f_fmean = np.log(np.maximum(pr[:, :, n_history:, :].mean(axis=2), EPS))
    f_end = np.log(np.maximum(pr[:, :, -1, :], EPS))
    func = np.concatenate([log_post.reshape(C, S, 1), f_last, f_fmean, f_end], axis=2)
    func_names = (["log_posterior"]
                  + [f"{p}::log_q_last_history" for p in data.producer_names]
                  + [f"{p}::log_q_forecast_mean" for p in data.producer_names]
                  + [f"{p}::log_q_final" for p in data.producer_names])
    f_rhat, f_ess = rhat(func), ess_bulk(func)
    func_df = pd.DataFrame({"quantity": func_names, "R_hat": f_rhat, "ESS_bulk": f_ess})
    progress(f"Functional max R-hat: {np.max(f_rhat):.4g}; functional min ESS: {np.min(f_ess):.1f}")

    total_post = max(S * args.mcmc_thin, 1) * C
    div_frac = sum(r.divergences for r in results) / total_post
    checks = [
        ("max_parameter_R_hat", float(np.max(rhat_v)), args.rhat_threshold, bool(np.all(np.isfinite(rhat_v)) and np.max(rhat_v) <= args.rhat_threshold)),
        ("min_parameter_bulk_ESS", float(np.min(ess_b)), args.ess_threshold, bool(np.all(np.isfinite(ess_b)) and np.min(ess_b) >= args.ess_threshold)),
        ("min_parameter_tail_ESS", float(np.min(ess_t)), TAIL_ESS_MIN, bool(np.all(np.isfinite(ess_t)) and np.min(ess_t) >= TAIL_ESS_MIN)),
        ("max_functional_R_hat", float(np.max(f_rhat)), args.rhat_threshold, bool(np.all(np.isfinite(f_rhat)) and np.max(f_rhat) <= args.rhat_threshold)),
        ("min_functional_ESS", float(np.min(f_ess)), TAIL_ESS_MIN, bool(np.all(np.isfinite(f_ess)) and np.min(f_ess) >= TAIL_ESS_MIN)),
        ("divergence_fraction", float(div_frac), MAX_DIVERGENCE_FRACTION, bool(div_frac <= MAX_DIVERGENCE_FRACTION)),
        ("min_chain_mean_accept_prob", float(min(r.mean_accept_prob for r in results)), MIN_MEAN_ACCEPT,
         bool(min(r.mean_accept_prob for r in results) >= MIN_MEAN_ACCEPT)),
    ]
    conv_df = pd.DataFrame(checks, columns=["check", "value", "threshold", "passed"])
    converged = bool(conv_df["passed"].all())
    for _, row in conv_df.iterrows():
        progress(f"{'PASS' if row['passed'] else 'FAIL'}  {row['check']}: {row['value']:.4g} (threshold {row['threshold']:g})")
    if converged:
        progress("✓ All MCMC quality checks passed.")
    else:
        progress("WARNING: MCMC quality checks failed. Do NOT interpret the intervals as a valid posterior yet.")
    finished(t0, "Propagation and functional diagnostics completed")

    # ---------------------------------------------------------------- 8. outputs
    t0 = stage(8, "Generating and saving outputs...")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    started_full = np.arange(T)[:, None] >= producer_start[None, :]
    qs = np.quantile(preds, [0.025, 0.10, 0.50, 0.90, 0.975], axis=0)          # [5, T, P]
    qs[:, ~started_full] = np.nan                                              # no prediction before a producer starts
    lo95, p10, p50, p90, hi95 = qs

    # optional predictive columns: Lognormal observation noise, sigma_j | theta from its conditional posterior
    rng_n = np.random.default_rng(args.seed + 424242)
    pred_hist = np.maximum(preds[:, 1:n_history, :] / scale, EPS)
    resid = (model.log_y1_np[None, :, :] - np.log(pred_hist)) * mask_hist[None, :, :]
    rss = np.sum(resid ** 2, axis=1)                                           # [D, P]
    shape_p = A0 + 0.5 * rho_weights * n_obs                                   # [P]
    scale_p = B0 + 0.5 * rho_weights[None, :] * rss                            # [D, P]
    sigma2 = scale_p / rng_n.gamma(shape_p[None, :], size=(n_draws, P))
    noisy = preds * np.exp(np.sqrt(sigma2)[:, None, :] * rng_n.standard_normal((n_draws, T, P)))
    pq = np.quantile(noisy, [0.025, 0.50, 0.975], axis=0)
    pq[:, ~started_full] = np.nan
    del noisy

    rows = []
    for k in range(T):
        period = "historical" if k < n_history else "forecast"
        for j, name in enumerate(data.producer_names):
            rows.append({
                "time": float(data.time[k]), "period": period, "producer": name,
                "observed": float(data.production[k, j]),
                "observed_is_zero": bool(data.production[k, j] <= 0),
                "model_active": bool(started_full[k, j]),
                "P10_low": float(p10[k, j]), "P50": float(p50[k, j]), "P90_high": float(p90[k, j]),
                "95pct_lower": float(lo95[k, j]), "95pct_upper": float(hi95[k, j]),
                "interval_width_95": float(hi95[k, j] - lo95[k, j]),
                "predictive_95_lower": float(pq[0, k, j]), "predictive_P50": float(pq[1, k, j]),
                "predictive_95_upper": float(pq[2, k, j]),
                "deterministic_crm": float(det_pred_full[k, j]) if started_full[k, j] else float("nan"),
                "producer_start_row": int(producer_start[j] + 1),
            })
    ts = pd.DataFrame(rows)
    ts.to_csv(out_dir / "crm_parameter_uq_timeseries.csv", index=False)

    metric_rows = []
    fidx = np.arange(n_history, T)
    for j, name in enumerate(data.producer_names):
        y = data.production[fidx, j]
        m = (y > 0) & started_full[fidx, j]
        if not m.any():
            metric_rows.append({"producer": name, "n_forecast_positive": 0})
            continue
        yy = y[m]
        metric_rows.append({
            "producer": name, "n_forecast_positive": int(m.sum()),
            "MAE_P50": float(np.mean(np.abs(p50[fidx, j][m] - yy))),
            "MAE_deterministic": float(np.mean(np.abs(det_pred_full[fidx, j][m] - yy))),
            "RMSE_P50": float(np.sqrt(np.mean((p50[fidx, j][m] - yy) ** 2))),
            "PICP_param_95": float(np.mean((yy >= lo95[fidx, j][m]) & (yy <= hi95[fidx, j][m]))),
            "PICP_predictive_95": float(np.mean((yy >= pq[0, fidx, j][m]) & (yy <= pq[2, fidx, j][m]))),
            "MPIW_param_95": float(np.mean(hi95[fidx, j][m] - lo95[fidx, j][m])),
            "MPIW_predictive_95": float(np.mean(pq[2, fidx, j][m] - pq[0, fidx, j][m])),
        })
    pd.DataFrame(metric_rows).to_csv(out_dir / "forecast_metrics.csv", index=False)

    pd.DataFrame(
        [{"time": float(data.time[k]), **{n: (float(det_pred_full[k, j]) if started_full[k, j] else float("nan"))
                                         for j, n in enumerate(data.producer_names)}} for k in range(T)]
    ).to_csv(out_dir / "deterministic_crm_timeseries.csv", index=False)

    np.savez_compressed(out_dir / "crm_parameter_hmc_samples.npz", chains=chains)
    diag_df.to_csv(out_dir / "mcmc_diagnostics.csv", index=False)
    func_df.to_csv(out_dir / "mcmc_functional_diagnostics.csv", index=False)
    conv_df.to_csv(out_dir / "convergence_summary.csv", index=False)
    pd.DataFrame({
        "chain": np.arange(1, C + 1),
        "acceptance_rate": [r.acceptance_rate for r in results],
        "mean_accept_prob": [r.mean_accept_prob for r in results],
        "final_step_size": [r.final_step_size for r in results],
        "divergences_post_warmup": [r.divergences for r in results],
        "wall_invalid_post_warmup": [r.invalid_proposals for r in results],
        "divergences_warmup": [r.warmup_divergences for r in results],
        "wall_invalid_warmup": [r.warmup_invalid for r in results],
        "unique_fraction": [r.unique_fraction for r in results],
        "final_log_posterior": [r.final_log_posterior for r in results],
    }).to_csv(out_dir / "hmc_chain_summary.csv", index=False)
    posterior_summary(flat, model, data.producer_names, data.injector_names).to_csv(
        out_dir / "crm_parameter_posterior_summary.csv", index=False)
    pd.DataFrame({
        "producer": data.producer_names, "start_row": producer_start + 1,
        "start_time": hist_time[producer_start], "initial_production": q_start_phys,
        "positive_likelihood_rows": n_obs, "lag1_weight": rho_weights,
    }).to_csv(out_dir / "producer_initial_states.csv", index=False)
    settings = {
        "forecast_rows": args.forecast, "historical_rows": n_history, "chains": args.mcmc_chains,
        "samples_per_chain": args.mcmc_samples, "warmup": args.mcmc_burnin, "thin": args.mcmc_thin,
        "hmc_step_initial": args.hmc_step_size, "hmc_leapfrog": args.hmc_leapfrog_steps,
        "hmc_leapfrog_jitter": args.hmc_leapfrog_jitter, "hmc_target_accept": args.hmc_target_accept,
        "tau_log_sd": args.tau_log_sd, "lambda_logit_sd": args.lambda_logit_sd,
        "tau_min": tmin, "tau_reference": math.exp(model.log_ref), "a0": A0, "b0": B0,
        "calibration_objective": args.calibration_objective, "crm_starts": args.crm_starts,
        "map_iter": args.map_iter, "start_jitter": args.start_jitter,
        "lag1_tempering": not args.no_lag1_tempering, "history_scale": scale,
        "converged": converged, "seed": args.seed,
    }
    pd.DataFrame({"setting": list(settings.keys()), "value": list(settings.values())}).to_csv(
        out_dir / "run_settings.csv", index=False)

    save_individual_plots(ts, data.producer_names, out_dir, float(data.time[n_history - 1]))
    finished(t0, "Outputs generated")

    print("\n" + "=" * 72, flush=True)
    print("RUN COMPLETE", flush=True)
    print("=" * 72, flush=True)
    print(f"Results: {out_dir.resolve()}", flush=True)
    print(f"MCMC quality checks passed: {converged}", flush=True)
    print("CRM-ML is excluded. Intervals are induced by uncertainty in the CRM physical parameters.", flush=True)
    if not converged:
        print("WARNING: do not treat these intervals as validated UQ until the checks pass "
              "(see convergence_summary.csv).", flush=True)


if __name__ == "__main__":
    main()