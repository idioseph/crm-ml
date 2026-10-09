#!/usr/bin/env python3
"""Bayesian parameter-level uncertainty quantification for a producer-based CRM.

This implementation deliberately excludes CRM-ML.

Scientific objective
--------------------
Quantify how uncertainty in the physical CRM calibration parameters propagates to
uncertainty in production predictions. The uncertain parameters are:
    * producer response times, tau_j
    * injector-producer connectivities, lambda_ij
    * injector-producer response times, tau_ij

The implementation follows a producer-based first-order CRM structure while using
producer-specific first-active-production states. The latter is important for field
data in which some producers have leading zero/shut-in rows: a producer whose first
observed rate occurs later should not have tau_j identified from an artificial q0=0.

Pipeline
--------
1. Load and strictly align Time, Production and Injection sheets.
2. Split history and the requested forecast horizon.
3. Detect the first positive historical production observation for each producer.
4. Calibrate the deterministic CRM on the historical period with SLSQP.
5. Use the calibrated result as the starting point for a short Bayesian MAP
   preconditioning step in transformed coordinates.
6. Construct a Lognormal production likelihood with an analytically marginalised
   Inverse-Gamma prior on sigma_j^2. A lag-1 effective-sample correction can temper
   serially correlated CRM residuals.
7. Sample the posterior with adaptive HMC using:
       - standardised transformed coordinates,
       - reasonable-step-size search,
       - dual-averaging warm-up,
       - diagonal or dense mass-matrix adaptation,
       - randomized trajectory length,
       - multiple independent chains.
8. Check parameter and functional convergence.
9. Propagate every retained posterior parameter draw through the complete
   historical-plus-forecast timeline.
10. Calculate P10/P50/P90 and central 95% parameter-induced intervals.
11. Save machine-readable outputs and one plot per producer only.

Important interpretation
------------------------
The reported 95% intervals represent uncertainty induced by the posterior of the
CRM parameters under the stated model, priors and likelihood. They are not a claim
that the CRM has zero model-form error, nor are they automatically full predictive
intervals including an independently simulated observation-noise draw.
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


# -----------------------------------------------------------------------------
# Global numerical settings
# -----------------------------------------------------------------------------

TOTAL_STAGES = 8
EPS = 1e-12
HMC_DIVERGENCE_THRESHOLD = 100.0


# -----------------------------------------------------------------------------
# Console helpers
# -----------------------------------------------------------------------------


def stage(number: int, message: str) -> float:
    """Print a pipeline stage heading and return its start timestamp."""
    print("\n" + "=" * 72, flush=True)
    print(f"[{number}/{TOTAL_STAGES}] {message}", flush=True)
    print("=" * 72, flush=True)
    return time.perf_counter()


def progress(message: str) -> None:
    """Print an indented progress message immediately."""
    print(f"    {message}", flush=True)


def finished(start_time: float, message: str = "Completed") -> None:
    """Print elapsed time for a pipeline stage."""
    elapsed = time.perf_counter() - start_time
    label = f"{elapsed:.1f} seconds" if elapsed < 60 else f"{elapsed / 60:.2f} minutes"
    progress(f"✓ {message} ({label})")


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class CRMParameters:
    """Physical CRM parameters."""

    tau_j: np.ndarray
    lambda_ij: np.ndarray
    tau_ij: np.ndarray


@dataclass
class CRMData:
    """Aligned field data."""

    time: np.ndarray
    production: np.ndarray
    injection: np.ndarray
    producer_names: list[str]
    injector_names: list[str]


@dataclass
class HMCResult:
    """One HMC chain and its sampling diagnostics."""

    draws: np.ndarray
    acceptance_rate: float
    mean_acceptance_probability: float
    final_step_size: float
    # Post-warm-up diagnostics only; warm-up counts are reported separately in HMC output.
    divergences: int
    invalid_proposals: int
    warmup_divergences: int
    warmup_invalid_proposals: int
    unique_fraction: float
    final_log_posterior: float


# -----------------------------------------------------------------------------
# Excel loading and strict alignment
# -----------------------------------------------------------------------------


def load_excel(path: str | Path) -> CRMData:
    """Load aligned Time, Production and Injection sheets from Excel.

    Unlike the earlier implementation, incomplete rows are not independently
    dropped from each sheet. That prevents silent time-series misalignment.
    """

    xls = pd.ExcelFile(path)

    def find(aliases: list[str]) -> str:
        mapping = {str(s).strip().lower(): str(s) for s in xls.sheet_names}
        for alias in aliases:
            if alias.lower() in mapping:
                return mapping[alias.lower()]
        raise ValueError(f"Missing required sheet. Tried {aliases}; found {xls.sheet_names}")

    def parse(sheet_name: str) -> pd.DataFrame:
        raw = xls.parse(sheet_name, header=None)
        raw = raw.dropna(how="all").reset_index(drop=True)
        if raw.empty:
            raise ValueError(f"Sheet '{sheet_name}' is empty.")

        columns = [str(v).strip() for v in raw.iloc[0]]
        body = raw.iloc[1:].copy()
        numeric = body.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(how="all").reset_index(drop=True)

        if numeric.empty:
            raise ValueError(f"Sheet '{sheet_name}' has no numeric data rows.")

        bad = numeric.isna().any(axis=1).to_numpy()
        if bad.any():
            rows = (np.flatnonzero(bad) + 2).tolist()
            shown = ", ".join(str(r) for r in rows[:10])
            suffix = "..." if len(rows) > 10 else ""
            raise ValueError(
                f"Sheet '{sheet_name}' contains incomplete numeric rows at Excel rows "
                f"{shown}{suffix}. Fix the workbook instead of silently deleting rows."
            )

        numeric.columns = columns
        return numeric

    t = parse(find(["time", "t", "date", "days"]))
    q = parse(find(["q_observed", "production", "q_obs", "q"]))
    w = parse(find(["w_injection", "injection", "w_inj", "inj"]))

    if len(t) != len(q) or len(t) != len(w):
        raise ValueError(
            "Time, production and injection sheets must contain the same number of rows. "
            f"Found {len(t)}, {len(q)}, {len(w)}."
        )

    time_values = t.iloc[:, 0].to_numpy(float)
    production = q.to_numpy(float)
    injection = w.to_numpy(float)

    if np.any(~np.isfinite(time_values)) or np.any(np.diff(time_values) <= 0):
        raise ValueError("Time must be finite and strictly increasing.")
    if np.any(~np.isfinite(production)) or np.any(production < 0):
        raise ValueError("Production contains NaN, infinity, or negative values.")
    if np.any(~np.isfinite(injection)) or np.any(injection < 0):
        raise ValueError("Injection contains NaN, infinity, or negative values.")
    if production.ndim != 2 or injection.ndim != 2:
        raise ValueError("Production and injection must be two-dimensional tables.")
    if production.shape[1] < 1 or injection.shape[1] < 1:
        raise ValueError("At least one producer and one injector are required.")

    return CRMData(
        time=time_values,
        production=production,
        injection=injection,
        producer_names=list(q.columns),
        injector_names=list(w.columns),
    )


# -----------------------------------------------------------------------------
# Producer start handling
# -----------------------------------------------------------------------------


def first_positive_indices(production_history: np.ndarray) -> np.ndarray:
    """Return the first positive historical row for every producer.

    The first positive observation becomes that producer's CRM initial state. This
    avoids forcing tau_j to act on q0=0 for producers that were not yet producing at
    the beginning of the dataset. A later zero/shut-in is still retained and can be
    excluded from the Lognormal likelihood without resetting the CRM state.
    """

    production_history = np.asarray(production_history, dtype=float)
    n_rows, n_producers = production_history.shape
    starts = np.full(n_producers, -1, dtype=int)

    for j in range(n_producers):
        positive = np.flatnonzero(production_history[:, j] > 0)
        if positive.size:
            starts[j] = int(positive[0])

    return starts


# -----------------------------------------------------------------------------
# CRM forward model
# -----------------------------------------------------------------------------


def crm_simulate(
    parameters: CRMParameters,
    time_values: np.ndarray,
    injection: np.ndarray,
    production_history: np.ndarray,
    producer_start: np.ndarray,
) -> np.ndarray:
    """Evaluate the parameterised CRM over the complete supplied timeline.

    For producer j, the natural production term starts at its first positive
    historical observation t_start,j:

        q_nat,j(t_n) = q_j(t_start,j) exp(-(t_n-t_start,j)/tau_j)

    for n >= start_j.

    Each injector-producer response obeys the exact first-order recurrence:

        R_ij,n = a_ij,n R_ij,n-1 + (1-a_ij,n) I_i,n,
        a_ij,n = exp(-Delta t_n / tau_ij)

    and the total producer rate is:

        q_j,n = q_nat,j,n + sum_i lambda_ij R_ij,n.

    Before a producer's first active observation its predicted production is zero.
    """

    time_values = np.asarray(time_values, dtype=float)
    injection = np.asarray(injection, dtype=float)
    production_history = np.asarray(production_history, dtype=float)
    producer_start = np.asarray(producer_start, dtype=int)

    n_time = len(time_values)
    n_injectors, n_producers = parameters.lambda_ij.shape

    if injection.shape != (n_time, n_injectors):
        raise ValueError("Injection dimensions do not match CRM parameters.")
    if production_history.shape[1] != n_producers:
        raise ValueError("Production dimensions do not match CRM parameters.")
    if producer_start.size != n_producers:
        raise ValueError("producer_start length does not match number of producers.")
    if np.any(producer_start < 0):
        bad = np.flatnonzero(producer_start < 0).tolist()
        raise ValueError(f"Producer(s) {bad} have no positive historical production.")

    dt = np.diff(time_values)
    output = np.zeros((n_time, n_producers), dtype=float)
    response_state = np.zeros((n_injectors, n_producers), dtype=float)

    # Per-producer starting rates and start times are fixed from historical data.
    q_start = production_history[producer_start, np.arange(n_producers)]
    start_time = time_values[producer_start]

    for n in range(n_time):
        if n == 0:
            active = n >= producer_start
            output[n, active] = q_start[active]
            continue

        active = n >= producer_start

        # Reset response states for producers that have not yet become active.
        response_state[:, ~active] = 0.0

        coefficient = np.exp(-dt[n - 1] / parameters.tau_ij)
        response_state[:, active] = (
            coefficient[:, active] * response_state[:, active]
            + (1.0 - coefficient[:, active]) * injection[n, :, None]
        )

        natural = np.zeros(n_producers, dtype=float)
        natural[active] = q_start[active] * np.exp(
            -(time_values[n] - start_time[active]) / parameters.tau_j[active]
        )

        output[n] = natural + np.sum(parameters.lambda_ij * response_state, axis=0)
        output[n, ~active] = 0.0

    return np.maximum(output, EPS)


# -----------------------------------------------------------------------------
# Torch CRM forward model
# -----------------------------------------------------------------------------


def crm_simulate_torch(
    log_tau_j: torch.Tensor,
    z: torch.Tensor,
    log_tau_ij: torch.Tensor,
    time_values: torch.Tensor,
    injection_scaled: torch.Tensor,
    q_start_scaled: torch.Tensor,
    producer_start: torch.Tensor,
) -> torch.Tensor:
    """Differentiable CRM using a time-only recurrence.

    z is already in physical logit coordinates. The softmax-with-slack mapping is
    carried out here so HMC remains unconstrained.
    """

    dtype = time_values.dtype
    n_time = time_values.numel()
    n_injectors, n_producers = z.shape

    tau_j = torch.exp(log_tau_j)
    tau_ij = torch.exp(log_tau_ij)

    augmented = torch.cat(
        [z, torch.zeros((n_injectors, 1), dtype=dtype, device=time_values.device)],
        dim=1,
    )
    lambda_ij = torch.softmax(augmented, dim=1)[:, :n_producers]

    dt = torch.diff(time_values)
    response_state = torch.zeros(
        (n_injectors, n_producers), dtype=dtype, device=time_values.device
    )

    rows = []
    for n in range(n_time):
        # Python indexing of producer_start is constant metadata and does not affect
        # gradients.
        active_bool = torch.tensor(
            n >= producer_start.detach().cpu().numpy(),
            dtype=torch.bool,
            device=time_values.device,
        )

        if n == 0:
            current = torch.where(active_bool, q_start_scaled, torch.zeros_like(q_start_scaled))
            rows.append(current)
            continue

        response_state = torch.where(
            active_bool[None, :],
            response_state,
            torch.zeros_like(response_state),
        )

        coefficient = torch.exp(-dt[n - 1] / tau_ij)
        response_state_active = (
            coefficient * response_state
            + (1.0 - coefficient) * injection_scaled[n, :, None]
        )
        response_state = torch.where(
            active_bool[None, :],
            response_state_active,
            torch.zeros_like(response_state_active),
        )

        start_times = time_values[producer_start]
        elapsed = time_values[n] - start_times
        natural = q_start_scaled * torch.exp(-elapsed / tau_j)
        natural = torch.where(active_bool, natural, torch.zeros_like(natural))

        current = natural + torch.sum(lambda_ij * response_state, dim=0)
        current = torch.where(active_bool, current, torch.zeros_like(current))
        rows.append(current)

    return torch.clamp(torch.stack(rows, dim=0), min=EPS)


# -----------------------------------------------------------------------------
# Connectivity transformations
# -----------------------------------------------------------------------------


def lam_to_z(lam: np.ndarray) -> np.ndarray:
    """Convert physical connectivity rows to logits against a slack category."""

    lam = np.asarray(lam, dtype=float)
    n_injectors, n_producers = lam.shape
    result = np.empty_like(lam)

    for i in range(n_injectors):
        row = np.clip(lam[i], 1e-12, None)
        if row.sum() >= 1.0:
            row = row / row.sum() * 0.95
        slack = max(1.0 - float(row.sum()), 1e-12)
        probabilities = np.r_[row, slack]
        probabilities /= probabilities.sum()
        result[i] = np.log(probabilities[:-1] / probabilities[-1])

    return result


def lam_to_z_start(lam: np.ndarray, delta: float = 0.05) -> np.ndarray:
    """Soften boundary solutions before HMC initialisation."""

    lam = np.asarray(lam, dtype=float)
    n_injectors, n_producers = lam.shape
    output = np.empty_like(lam)

    for i in range(n_injectors):
        row = np.clip(lam[i], 0.0, None)
        if row.sum() > 1.0:
            row = row / row.sum() * 0.90
        slack = max(1.0 - float(row.sum()), 0.0)
        probabilities = np.r_[row, slack]
        probabilities /= probabilities.sum()
        probabilities = (1.0 - delta) * probabilities + delta / (n_producers + 1)
        probabilities /= probabilities.sum()
        output[i] = np.log(probabilities[:-1] / probabilities[-1])

    return output


def z_to_lam(z: np.ndarray) -> np.ndarray:
    """Map logits plus a fixed zero-logit slack category to connectivity."""

    z = np.asarray(z, dtype=float)
    output = np.empty_like(z)

    for i in range(z.shape[0]):
        augmented = np.r_[z[i], 0.0]
        augmented -= augmented.max()
        p = np.exp(augmented)
        p /= p.sum()
        output[i] = p[:-1]

    return output


# -----------------------------------------------------------------------------
# Deterministic calibration
# -----------------------------------------------------------------------------


class Calibrator:
    """SLSQP calibration of the physical CRM."""

    def __init__(self, starts: int, max_iter: int, seed: int, objective_mode: str):
        self.starts = max(1, int(starts))
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.objective_mode = objective_mode
        self.params_: Optional[CRMParameters] = None
        self.objective_ = np.inf
        self.scale_ = 1.0
        self.success_ = False
        self.message_ = ""

    def _unpack(self, x: np.ndarray, n_injectors: int, n_producers: int) -> CRMParameters:
        p = 0
        tau_j = np.exp(x[p:p + n_producers])
        p += n_producers
        lam = x[p:p + n_injectors * n_producers].reshape(n_injectors, n_producers)
        p += n_injectors * n_producers
        tau_ij = np.exp(x[p:p + n_injectors * n_producers]).reshape(n_injectors, n_producers)
        return CRMParameters(tau_j=tau_j, lambda_ij=lam, tau_ij=tau_ij)

    def fit(
        self,
        time_values: np.ndarray,
        injection: np.ndarray,
        production: np.ndarray,
        producer_start: np.ndarray,
    ) -> "Calibrator":
        time_values = np.asarray(time_values, dtype=float)
        injection = np.asarray(injection, dtype=float)
        production = np.asarray(production, dtype=float)

        n_rows, n_injectors = injection.shape
        n_producers = production.shape[1]
        tmin = float(np.min(np.diff(time_values)))
        span = float(time_values[-1] - time_values[0])
        if tmin <= 0 or span <= 0:
            raise ValueError("Time spacing must be positive and history span must be non-zero.")

        self.scale_ = max(float(np.mean(np.abs(production))), EPS)
        q_scaled = production / self.scale_
        w_scaled = injection / self.scale_
        q_start = production[producer_start, np.arange(n_producers)] / self.scale_

        log_tau_min = math.log(tmin)
        n_parameters = n_producers + 2 * n_injectors * n_producers
        lambda_offset = n_producers
        tau_ij_offset = n_producers + n_injectors * n_producers

        bounds = (
            [(log_tau_min, None)] * n_producers
            + [(0.0, 1.0)] * (n_injectors * n_producers)
            + [(log_tau_min, None)] * (n_injectors * n_producers)
        )

        constraint_matrix = np.zeros((n_injectors, n_parameters), dtype=float)
        for i in range(n_injectors):
            start = lambda_offset + i * n_producers
            constraint_matrix[i, start:start + n_producers] = 1.0

        constraint_kwargs = {
            "constraints": {
                "type": "ineq",
                "fun": lambda x: 1.0 - constraint_matrix @ x,
                "jac": lambda x: -constraint_matrix,
            }
        }

        # The first positive row is an initial state. Subsequent active positive rows
        # contribute to the calibration objective; leading zeros are not artificial
        # errors against q=0 because the producer had not started.
        rows = np.arange(n_rows)
        fit_mask = np.zeros_like(production, dtype=bool)
        for j in range(n_producers):
            fit_mask[:, j] = (rows > producer_start[j]) & (production[:, j] > 0)
        fit_mask_post = fit_mask[1:]

        def objective(x: np.ndarray) -> float:
            params = self._unpack(x, n_injectors, n_producers)
            prediction = crm_simulate(
                params,
                time_values,
                w_scaled,
                production / self.scale_,
                producer_start,
            )

            if self.objective_mode == "linear-positive":
                residual = q_scaled - prediction
                return float(np.mean((residual[fit_mask]) ** 2))

            if self.objective_mode == "linear-all":
                return float(np.mean((q_scaled[1:] - prediction[1:]) ** 2))

            if self.objective_mode == "log-positive":
                residual = np.log(np.clip(q_scaled, EPS, None)) - np.log(np.clip(prediction, EPS, None))
                return float(np.mean((residual[fit_mask]) ** 2))

            raise ValueError(f"Unknown calibration objective: {self.objective_mode}")

        rng = np.random.default_rng(self.seed)
        best_x = None
        best_f = np.inf
        best_success = False
        best_message = ""

        upper_init = max(10.0 * tmin, span)
        upper_log = math.log(max(upper_init, 1.05 * tmin))
        lower_start_log = log_tau_min + 0.05

        for start_number in range(self.starts):
            x0 = np.empty(n_parameters, dtype=float)

            if start_number == 0:
                initial_tau = max(2.0 * tmin, 0.1 * span)
                initial_log = max(math.log(initial_tau), lower_start_log)
                x0[:n_producers] = initial_log
                x0[lambda_offset:tau_ij_offset] = 0.6 / max(n_producers, 1)
                x0[tau_ij_offset:] = initial_log
            else:
                x0[:n_producers] = rng.uniform(
                    lower_start_log,
                    upper_log,
                    n_producers,
                )
                x0[lambda_offset:tau_ij_offset] = (
                    0.6
                    * rng.dirichlet(np.ones(n_producers), size=n_injectors)
                ).ravel()
                x0[tau_ij_offset:] = rng.uniform(
                    lower_start_log,
                    upper_log,
                    n_injectors * n_producers,
                )

            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                options={
                    "maxiter": self.max_iter,
                    "ftol": 1e-9,
                    "disp": False,
                },
                **constraint_kwargs,
            )

            progress(
                f"Calibration start {start_number + 1}/{self.starts}: "
                f"objective={result.fun:.8g}, success={result.success}"
            )

            if np.isfinite(result.fun) and result.fun < best_f:
                best_x = result.x.copy()
                best_f = float(result.fun)
                best_success = bool(result.success)
                best_message = str(result.message)

        if best_x is None:
            raise RuntimeError("All deterministic CRM calibration starts failed.")

        self.params_ = self._unpack(best_x, n_injectors, n_producers)
        self.objective_ = best_f
        self.success_ = best_success
        self.message_ = best_message
        return self


# -----------------------------------------------------------------------------
# Bayesian transformed-coordinate representation
# -----------------------------------------------------------------------------


def theta_from_params(
    parameters: CRMParameters,
    log_tau_reference: float,
    tau_sd: float,
    lambda_sd: float,
) -> np.ndarray:
    """Convert physical parameters into standardised HMC coordinates."""

    log_tau_j = np.log(np.maximum(parameters.tau_j, EPS))
    log_tau_ij = np.log(np.maximum(parameters.tau_ij, EPS))
    z = lam_to_z_start(parameters.lambda_ij)

    return np.concatenate([
        (log_tau_j - log_tau_reference) / tau_sd,
        (z / lambda_sd).ravel(),
        ((log_tau_ij - log_tau_reference) / tau_sd).ravel(),
    ])


def physical_from_theta(
    theta: np.ndarray | torch.Tensor,
    n_injectors: int,
    n_producers: int,
    log_tau_reference: float,
    tau_sd: float,
    lambda_sd: float,
) -> CRMParameters:
    """Convert standardised HMC coordinates into physical CRM parameters."""

    if isinstance(theta, torch.Tensor):
        t = theta
        u = t[:n_producers]
        z_std = t[n_producers:n_producers + n_injectors * n_producers].reshape(n_injectors, n_producers)
        v = t[n_producers + n_injectors * n_producers:]
        log_tau_j = log_tau_reference + tau_sd * u
        z = lambda_sd * z_std
        log_tau_ij = log_tau_reference + tau_sd * v.reshape(n_injectors, n_producers)
        return log_tau_j, z, log_tau_ij

    theta = np.asarray(theta, dtype=float)
    u = theta[:n_producers]
    z_std = theta[n_producers:n_producers + n_injectors * n_producers].reshape(n_injectors, n_producers)
    v = theta[n_producers + n_injectors * n_producers:]
    log_tau_j = log_tau_reference + tau_sd * u
    z = lambda_sd * z_std
    log_tau_ij = log_tau_reference + tau_sd * v.reshape(n_injectors, n_producers)
    return CRMParameters(
        tau_j=np.exp(log_tau_j),
        lambda_ij=z_to_lam(z),
        tau_ij=np.exp(log_tau_ij),
    )


def _inverse_softplus(value: np.ndarray) -> np.ndarray:
    """Stable inverse softplus for positive numpy values."""
    value = np.asarray(value, dtype=float)
    return np.where(
        value > 20.0,
        value,
        np.log(np.expm1(np.maximum(value, 1e-12))),
    )


# -----------------------------------------------------------------------------
# Bayesian posterior and MAP preconditioning
# -----------------------------------------------------------------------------


def lag1_weight(log_residual: np.ndarray) -> float:
    """Return the lag-1 effective-sample correction factor."""

    x = np.asarray(log_residual, dtype=float)
    if x.size < 4 or np.std(x) <= 0:
        return 1.0
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    if not np.isfinite(rho):
        return 1.0
    rho = float(np.clip(rho, 0.0, 0.90))
    return max((1.0 - rho) / (1.0 + rho), 0.05)


def make_logpost(
    history_time: np.ndarray,
    history_injection: np.ndarray,
    history_production: np.ndarray,
    producer_start: np.ndarray,
    tau_sd: float,
    lambda_sd: float,
    use_lag1_tempering: bool,
    deterministic_parameters: CRMParameters,
):
    """Construct the differentiable parameter-level Bayesian CRM posterior."""

    dtype = torch.float64
    scale = max(float(np.mean(np.abs(history_production))), EPS)
    tmin = float(np.min(np.diff(history_time)))
    span = max(float(history_time[-1] - history_time[0]), 1.01 * tmin)
    log_tau_reference = 0.5 * (math.log(tmin) + math.log(span))
    log_tau_min = math.log(tmin)

    t = torch.tensor(np.asarray(history_time, dtype=float), dtype=dtype)
    w = torch.tensor(np.asarray(history_injection / scale, dtype=float), dtype=dtype)
    q_scaled = np.asarray(history_production / scale, dtype=float)
    q_start = torch.tensor(
        q_scaled[producer_start, np.arange(history_production.shape[1])],
        dtype=dtype,
    )
    y = torch.tensor(q_scaled, dtype=dtype)
    start_tensor = torch.tensor(producer_start, dtype=torch.long)

    n_injectors = history_injection.shape[1]
    n_producers = history_production.shape[1]

    rows = np.arange(len(history_time))[:, None]
    positive = history_production > 0
    active_after_start = rows > producer_start[None, :]
    likelihood_mask_np = positive & active_after_start
    likelihood_mask_np = likelihood_mask_np[1:]
    likelihood_mask = torch.tensor(likelihood_mask_np, dtype=torch.bool)
    mask_float = likelihood_mask.to(dtype)
    n_observed = mask_float.sum(dim=0)

    if torch.any(n_observed <= 0):
        bad = np.flatnonzero(n_observed.numpy() <= 0).tolist()
        raise ValueError(f"Producer(s) {bad} have no positive observations after their start state.")

    # Inverse-Gamma prior on sigma^2. sigma^2 is analytically marginalised.
    a0 = 2.0
    b0 = a0 * 0.20**2

    rho_weights = np.ones(n_producers, dtype=float)
    if use_lag1_tempering:
        det_q = crm_simulate(
            deterministic_parameters,
            history_time,
            history_injection / scale,
            history_production / scale,
            producer_start,
        )
        for j in range(n_producers):
            m = likelihood_mask_np[:, j] & (det_q[1:, j] > 0)
            if int(m.sum()) >= 4:
                residual = (
                    np.log(history_production[1:, j][m] / scale)
                    - np.log(np.maximum(det_q[1:, j][m], EPS))
                )
                rho_weights[j] = lag1_weight(residual)

    def log_posterior(theta: torch.Tensor) -> torch.Tensor:
        u_j = theta[:n_producers]
        z_std = theta[
            n_producers:n_producers + n_injectors * n_producers
        ].reshape(n_injectors, n_producers)
        v_ij = theta[n_producers + n_injectors * n_producers:].reshape(
            n_injectors, n_producers
        )

        log_tau_j = log_tau_reference + tau_sd * u_j
        log_tau_ij = log_tau_reference + tau_sd * v_ij

        # Hard physical support: tau >= minimum sampling interval.
        if torch.any(log_tau_j < log_tau_min) or torch.any(log_tau_ij < log_tau_min):
            return torch.tensor(-float("inf"), dtype=dtype)

        z = lambda_sd * z_std
        qhat = crm_simulate_torch(
            log_tau_j,
            z,
            log_tau_ij,
            t,
            w,
            q_start,
            start_tensor,
        )

        residual = torch.log(torch.clamp(y[1:], min=EPS)) - torch.log(qhat[1:])
        rss = torch.sum((residual * mask_float) ** 2, dim=0)

        weight = torch.tensor(rho_weights, dtype=dtype)
        log_likelihood = torch.sum(
            -(
                a0 + 0.5 * weight * n_observed
            )
            * torch.log(
                b0 + 0.5 * weight * rss
            )
        )

        # Because theta is standardised, the priors are standard Normal up to constants.
        prior = -0.5 * torch.sum(u_j ** 2)
        prior = prior - 0.5 * torch.sum(z_std ** 2)
        prior = prior - 0.5 * torch.sum(v_ij ** 2)

        return log_likelihood + prior

    metadata = {
        "scale": scale,
        "tmin": tmin,
        "span": span,
        "log_tau_reference": log_tau_reference,
        "tau_reference": math.exp(log_tau_reference),
        "a0": a0,
        "b0": b0,
        "rho_weights": rho_weights,
        "likelihood_mask": likelihood_mask_np,
        "n_observed": n_observed.numpy(),
    }
    return log_posterior, metadata


def map_refine(
    log_posterior: Callable[[torch.Tensor], torch.Tensor],
    theta0: np.ndarray,
    n_injectors: int,
    n_producers: int,
    log_tau_reference: float,
    log_tau_min: float,
    tau_sd: float,
    lambda_sd: float,
    steps: int,
    learning_rate: float,
) -> np.ndarray:
    """Perform a short MAP preconditioning pass without changing HMC's target.

    A smooth positive transform is used only for optimisation of the initial state:
        log_tau = log_tau_min + softplus(a)
    with the corresponding Jacobian included. The final HMC target remains the
    original posterior in standardised log-tau/logit coordinates.
    """

    if steps <= 0:
        return np.asarray(theta0, dtype=float)

    theta0 = np.asarray(theta0, dtype=float)
    u0 = theta0[:n_producers]
    z0 = theta0[n_producers:n_producers + n_injectors * n_producers]
    v0 = theta0[n_producers + n_injectors * n_producers:]

    log_tau_j0 = log_tau_reference + tau_sd * u0
    log_tau_ij0 = log_tau_reference + tau_sd * v0
    a0 = _inverse_softplus(np.maximum(np.exp(log_tau_j0) - np.exp(log_tau_min), 1e-8))
    # The above inverse is in physical tau space; convert back to the corresponding
    # optimisation log-tau representation through exp(log_tau). This deliberately
    # uses the stable identity tau = tau_min + softplus(a).

    a = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(v0, dtype=torch.float64, requires_grad=True)
    zstd = torch.tensor(z0, dtype=torch.float64, requires_grad=True)

    optimiser = torch.optim.Adam([a, zstd, b], lr=learning_rate)
    best_state = None
    best_value = float("inf")

    tau_min = math.exp(log_tau_min)
    tau_ref = math.exp(log_tau_reference)

    for step in range(1, steps + 1):
        optimiser.zero_grad(set_to_none=True)

        tau_j = tau_min + torch.nn.functional.softplus(a)
        tau_ij = tau_min + torch.nn.functional.softplus(b)
        log_tau_j = torch.log(tau_j)
        log_tau_ij = torch.log(tau_ij)

        u = (log_tau_j - log_tau_reference) / tau_sd
        v = (log_tau_ij - log_tau_reference) / tau_sd
        theta = torch.cat([u, zstd, v])

        base = log_posterior(theta)
        if not torch.isfinite(base):
            # Smooth fallback: the softplus transform itself guarantees valid tau,
            # so this path should only occur if the forward model becomes non-finite.
            loss = torch.tensor(1e8, dtype=torch.float64, requires_grad=True)
        else:
            # Jacobian for the optimisation-only transform from a,b to log tau.
            # d log(tau)/da = sigmoid(a) / tau.
            jac = torch.sum(torch.nn.functional.logsigmoid(a) - torch.log(tau_j))
            jac = jac + torch.sum(torch.nn.functional.logsigmoid(b) - torch.log(tau_ij))
            loss = -(base + jac)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([a, zstd, b], max_norm=5.0)
        optimiser.step()

        value = float(loss.detach())
        if np.isfinite(value) and value < best_value:
            best_value = value
            with torch.no_grad():
                u_best = (torch.log(tau_j) - log_tau_reference) / tau_sd
                v_best = (torch.log(tau_ij) - log_tau_reference) / tau_sd
                best_state = torch.cat([u_best, zstd, v_best]).cpu().numpy().copy()

        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            progress(f"MAP preconditioning: {step}/{steps}, negative log target={value:.6g}")

    if best_state is None:
        return theta0
    return best_state


# -----------------------------------------------------------------------------
# HMC linear algebra helpers
# -----------------------------------------------------------------------------


def _make_mass_matrix(
    positions: np.ndarray,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Build a regularised diagonal or dense mass matrix."""

    positions = np.asarray(positions, dtype=float)
    d = positions.shape[1]

    if mode == "diagonal":
        variance = np.var(positions, axis=0, ddof=1)
        variance = np.where(np.isfinite(variance), variance, 1.0)
        variance = np.clip(variance, 0.05, 20.0)
        variance = 0.90 * variance + 0.10
        mass_np = np.diag(variance)
    elif mode == "dense":
        covariance = np.cov(positions, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance)
        if covariance.shape != (d, d) or not np.all(np.isfinite(covariance)):
            variance = np.var(positions, axis=0, ddof=1)
            variance = np.clip(np.where(np.isfinite(variance), variance, 1.0), 0.05, 20.0)
            mass_np = np.diag(variance)
        else:
            diag = np.diag(np.diag(covariance))
            # Shrink the sample covariance toward its diagonal for stability.
            mass_np = 0.75 * covariance + 0.25 * diag
            mass_np = 0.5 * (mass_np + mass_np.T)
    else:
        raise ValueError("mass matrix mode must be 'diagonal' or 'dense'.")

    eigenvalues, eigenvectors = np.linalg.eigh(mass_np)
    eigenvalues = np.clip(eigenvalues, 0.05, 20.0)
    mass_np = (eigenvectors * eigenvalues) @ eigenvectors.T
    mass_np += 1e-6 * np.eye(d)
    inverse_np = np.linalg.inv(mass_np)
    chol_np = np.linalg.cholesky(mass_np)

    return (
        torch.tensor(mass_np, dtype=torch.float64),
        torch.tensor(inverse_np, dtype=torch.float64),
        chol_np,
    )


def _kinetic(momentum: torch.Tensor, inverse_mass: torch.Tensor) -> torch.Tensor:
    """Diagonal/dense kinetic energy."""
    return 0.5 * torch.dot(momentum, inverse_mass @ momentum)


def _sample_momentum(chol_mass: np.ndarray, dimension: int, rng: np.random.Generator) -> torch.Tensor:
    """Sample p ~ N(0, M) using the Cholesky factor of M."""
    z = torch.tensor(rng.normal(size=dimension), dtype=torch.float64)
    chol = torch.tensor(chol_mass, dtype=torch.float64)
    return chol @ z


# -----------------------------------------------------------------------------
# HMC sampler
# -----------------------------------------------------------------------------


def _one_leapfrog_trajectory(
    log_posterior: Callable[[torch.Tensor], torch.Tensor],
    current: torch.Tensor,
    step_size: float,
    leapfrog_steps: int,
    mass: torch.Tensor,
    inverse_mass: torch.Tensor,
    chol_mass: np.ndarray,
    rng: np.random.Generator,
):
    """Run one HMC trajectory."""

    position = current.detach().clone().requires_grad_(True)
    momentum0 = _sample_momentum(chol_mass, position.numel(), rng)

    logp0 = log_posterior(position)
    if not torch.isfinite(logp0):
        raise RuntimeError("Current HMC state has non-finite posterior density.")
    grad = torch.autograd.grad(logp0, position)[0]

    momentum = momentum0 + 0.5 * step_size * grad
    proposal = position
    proposal_logp = logp0
    valid = True

    for leapfrog in range(leapfrog_steps):
        proposal = (
            proposal
            + step_size * (inverse_mass @ momentum)
        ).detach().requires_grad_(True)

        proposal_logp = log_posterior(proposal)
        if not torch.isfinite(proposal_logp):
            valid = False
            break

        proposal_grad = torch.autograd.grad(proposal_logp, proposal)[0]

        if leapfrog < leapfrog_steps - 1:
            momentum = momentum + step_size * proposal_grad

    if not valid:
        return current, False, 0.0, math.inf, False

    momentum = momentum + 0.5 * step_size * proposal_grad
    momentum = -momentum

    h0 = -logp0.detach() + _kinetic(momentum0, inverse_mass)
    h1 = -proposal_logp.detach() + _kinetic(momentum, inverse_mass)
    energy_error = float(h1 - h0)

    if not np.isfinite(energy_error):
        return current, False, 0.0, math.inf, False

    log_accept = min(0.0, float(h0 - h1))
    accept_prob = math.exp(log_accept)
    accepted = math.log(max(rng.random(), 1e-300)) < log_accept

    return (
        proposal.detach() if accepted else current,
        accepted,
        accept_prob,
        energy_error,
        True,
    )


def _find_reasonable_epsilon(
    log_posterior: Callable[[torch.Tensor], torch.Tensor],
    current: torch.Tensor,
    initial_epsilon: float,
    mass: torch.Tensor,
    inverse_mass: torch.Tensor,
    chol_mass: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Find a numerically reasonable initial HMC step size."""

    epsilon = float(np.clip(initial_epsilon, 1e-5, 0.1))
    target_log_accept = math.log(0.5)

    # Try a one-step trajectory. The current state is not permanently changed.
    accepted_state, _, accept_prob, energy_error, valid = _one_leapfrog_trajectory(
        log_posterior,
        current,
        epsilon,
        1,
        mass,
        inverse_mass,
        chol_mass,
        rng,
    )

    if not valid or not np.isfinite(accept_prob) or not np.isfinite(energy_error):
        direction = -1
    else:
        direction = 1 if math.log(max(accept_prob, 1e-300)) > target_log_accept else -1

    log_epsilon = math.log(epsilon)
    for _ in range(20):
        if direction == 1:
            if not valid or math.log(max(accept_prob, 1e-300)) < target_log_accept:
                break
            epsilon *= 2.0
        else:
            if valid and math.log(max(accept_prob, 1e-300)) > target_log_accept:
                break
            epsilon *= 0.5

        epsilon = float(np.clip(epsilon, 1e-6, 0.1))
        _, _, accept_prob, energy_error, valid = _one_leapfrog_trajectory(
            log_posterior,
            current,
            epsilon,
            1,
            mass,
            inverse_mass,
            chol_mass,
            rng,
        )
        log_epsilon = math.log(epsilon)

        if epsilon <= 1e-6 or epsilon >= 0.1:
            break

    return float(np.clip(math.exp(log_epsilon), 1e-6, 0.1))


def hmc(
    log_posterior: Callable[[torch.Tensor], torch.Tensor],
    start: np.ndarray,
    samples: int,
    burnin: int,
    thin: int,
    initial_step_size: float,
    leapfrog_steps: int,
    seed: int,
    target_accept: float,
    mass_mode: str,
    leapfrog_jitter: int,
):
    """Run adaptive HMC and freeze the kernel after warm-up."""

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    current = torch.tensor(np.asarray(start, dtype=float), dtype=torch.float64)
    dimension = current.numel()

    # Start from identity metric in the standardised parameter coordinates.
    mass = torch.eye(dimension, dtype=torch.float64)
    inverse_mass = torch.eye(dimension, dtype=torch.float64)
    chol_mass = np.eye(dimension, dtype=float)

    # Reasonable step size search is much safer than blindly starting at a user epsilon.
    step_size = _find_reasonable_epsilon(
        log_posterior,
        current,
        initial_step_size,
        mass,
        inverse_mass,
        chol_mass,
        rng,
    )

    # Dual-averaging constants.
    da_target = float(target_accept)
    da_gamma = 0.05
    da_t0 = 10.0
    da_kappa = 0.75
    da_mu = math.log(10.0 * step_size)
    da_hbar = 0.0
    da_log_eps_bar = math.log(step_size)
    da_m = 0

    total_iterations = burnin + samples * thin
    checkpoints = set(
        np.unique(
            np.linspace(1, total_iterations, min(10, total_iterations), dtype=int)
        ).tolist()
    )

    warm_positions: list[np.ndarray] = []
    retained: list[np.ndarray] = []
    accepted_post = 0
    accept_probs_post: list[float] = []
    # Diagnostics that determine whether the *retained posterior sample* is usable.
    # Warm-up problems are tracked separately because adaptation is explicitly allowed
    # to have rejected/unstable proposals while it is learning the geometry.
    divergences_warmup = 0
    divergences_post = 0
    invalid_warmup = 0
    invalid_post = 0

    # Metric is adapted once near 60% of warm-up, then dual averaging restarts so that
    # the step size is compatible with the new geometry.
    metric_iteration = int(0.60 * burnin) if burnin >= 100 else None

    for iteration in range(1, total_iterations + 1):
        if leapfrog_jitter > 0:
            L = int(
                rng.integers(
                    max(1, leapfrog_steps - leapfrog_jitter),
                    leapfrog_steps + leapfrog_jitter + 1,
                )
            )
        else:
            L = int(leapfrog_steps)

        # Small trajectory-length jitter is independent of the current state and keeps
        # a fixed-kernel sampler after warm-up while reducing resonance.
        current_eps = step_size
        proposal, accepted, accept_prob, energy_error, valid = _one_leapfrog_trajectory(
            log_posterior,
            current,
            current_eps,
            L,
            mass,
            inverse_mass,
            chol_mass,
            rng,
        )
        current = proposal

        is_divergence = (
            (np.isfinite(energy_error) and abs(energy_error) > HMC_DIVERGENCE_THRESHOLD)
            or not np.isfinite(energy_error)
        )

        if iteration <= burnin:
            if not valid:
                invalid_warmup += 1
            if is_divergence:
                divergences_warmup += 1
            warm_positions.append(current.numpy().copy())

            da_m += 1
            eta = 1.0 / (da_m + da_t0)
            da_hbar = (
                (1.0 - eta) * da_hbar
                + eta * (da_target - accept_prob)
            )
            log_eps = da_mu - math.sqrt(da_m) / da_gamma * da_hbar
            weight = da_m ** (-da_kappa)
            da_log_eps_bar = weight * log_eps + (1.0 - weight) * da_log_eps_bar
            step_size = float(np.clip(math.exp(log_eps), 1e-6, 0.1))

            # Adapt the mass matrix once enough actual movement has been observed.
            if metric_iteration is not None and iteration == metric_iteration:
                warm = np.asarray(warm_positions, dtype=float)
                begin = max(10, warm.shape[0] // 4)
                metric_samples = warm[begin:]
                if metric_samples.shape[0] >= 50:
                    mass, inverse_mass, chol_mass = _make_mass_matrix(metric_samples, mass_mode)
                    progress(
                        f"{mass_mode.title()} mass adaptation: "
                        f"median variance={np.median(np.diag(mass.numpy())):.6g}"
                    )

                    # Restart dual averaging after changing the metric.
                    da_mu = math.log(10.0 * step_size)
                    da_hbar = 0.0
                    da_log_eps_bar = math.log(step_size)
                    da_m = 0

        else:
            if not valid:
                invalid_post += 1
            if is_divergence:
                divergences_post += 1
            if accepted:
                accepted_post += 1
            accept_probs_post.append(float(accept_prob))
            if (iteration - burnin) % thin == 0:
                retained.append(current.numpy().copy())

        # Freeze the smoothed dual-averaging step at the end of warm-up.
        if burnin > 0 and iteration == burnin:
            step_size = float(np.clip(math.exp(da_log_eps_bar), 1e-6, 0.1))

        if iteration in checkpoints:
            phase = "burn-in" if iteration <= burnin else "sampling"
            progress(
                f"{phase}: {iteration}/{total_iterations} "
                f"({100.0 * iteration / total_iterations:.0f}%) "
                f"| step={step_size:.6g} "
                f"| accept_prob={accept_prob:.3f}"
            )

    draws = np.asarray(retained, dtype=float)
    if draws.shape != (samples, dimension):
        raise RuntimeError(
            f"HMC retained shape {draws.shape}; expected {(samples, dimension)}."
        )

    acceptance_rate = accepted_post / max(samples * thin, 1)
    mean_acceptance_probability = float(np.mean(accept_probs_post)) if accept_probs_post else 0.0
    unique_fraction = (
        np.unique(draws, axis=0).shape[0] / max(len(draws), 1)
    )
    final_logp = float(log_posterior(torch.tensor(draws[-1], dtype=torch.float64)))

    return HMCResult(
        draws=draws,
        acceptance_rate=acceptance_rate,
        mean_acceptance_probability=mean_acceptance_probability,
        final_step_size=step_size,
        divergences=divergences_post,
        invalid_proposals=invalid_post,
        warmup_divergences=divergences_warmup,
        warmup_invalid_proposals=invalid_warmup,
        unique_fraction=unique_fraction,
        final_log_posterior=final_logp,
    )


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def _split_chains(chains: np.ndarray) -> np.ndarray:
    """Split chains into two halves for split-R-hat and ESS."""

    chains = np.asarray(chains, dtype=float)
    n_chains, n_draws, n_parameters = chains.shape
    if n_draws < 8:
        raise ValueError("At least 8 retained draws per chain are required for diagnostics.")
    half = n_draws // 2
    return chains[:, :2 * half].reshape(2 * n_chains, half, n_parameters)


def rhat(chains: np.ndarray) -> np.ndarray:
    """Compute split-R-hat."""

    split = _split_chains(chains)
    m, n, d = split.shape
    means = split.mean(axis=1)
    vars_ = split.var(axis=1, ddof=1)
    within = vars_.mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    variance_hat = ((n - 1) / n) * within + between / n

    result = np.full(d, np.inf, dtype=float)
    valid = (
        np.isfinite(within)
        & np.isfinite(variance_hat)
        & (within > 1e-12)
        & (variance_hat > 1e-12)
    )
    result[valid] = np.sqrt(variance_hat[valid] / within[valid])

    constant = (
        np.isfinite(within)
        & np.isfinite(variance_hat)
        & (within <= 1e-12)
        & (variance_hat <= 1e-12)
    )
    result[constant] = np.inf
    return result


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance sequence."""

    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n == 0:
        return np.empty(0)
    return np.correlate(x, x, mode="full")[n - 1:] / n


def ess(chains: np.ndarray) -> np.ndarray:
    """Approximate bulk ESS using split chains and Geyer positive truncation."""

    split = _split_chains(chains)
    m, n, d = split.shape
    total = m * n

    vars_ = split.var(axis=1, ddof=1)
    within = vars_.mean(axis=0)
    means = split.mean(axis=1)
    between = n * means.var(axis=0, ddof=1)
    variance_hat = ((n - 1) / n) * within + between / n

    output = np.ones(d, dtype=float)

    for p in range(d):
        if (
            not np.isfinite(variance_hat[p])
            or variance_hat[p] <= 1e-12
            or not np.isfinite(within[p])
            or within[p] <= 1e-12
        ):
            output[p] = 1.0
            continue

        acovs = []
        for c in range(m):
            ac = _autocovariance(split[c, :, p])
            if len(ac) == n:
                acovs.append(ac)
        if not acovs:
            output[p] = 1.0
            continue

        mean_acov = np.mean(np.stack(acovs), axis=0)
        rho = 1.0 - (within[p] - mean_acov) / variance_hat[p]
        rho[0] = 1.0

        rho_sum = 0.0
        for k in range(1, len(rho) - 1, 2):
            pair = float(rho[k] + rho[k + 1])
            if not np.isfinite(pair) or pair <= 0:
                break
            rho_sum += pair

        output[p] = float(np.clip(total / max(1.0 + 2.0 * rho_sum, 1.0), 1.0, total))

    return output


def parameter_names(producer_names: list[str], injector_names: list[str]) -> list[str]:
    """Names for the standardised HMC coordinates."""

    return (
        [f"{p}::std_log_tau_j" for p in producer_names]
        + [
            f"{i}->{p}::std_lambda_logit"
            for i in injector_names
            for p in producer_names
        ]
        + [
            f"{i}->{p}::std_log_tau_ij"
            for i in injector_names
            for p in producer_names
        ]
    )


def physical_parameter_names(producer_names: list[str], injector_names: list[str]) -> list[str]:
    """Names for physical posterior parameters."""

    return (
        [f"{p}::tau_j" for p in producer_names]
        + [
            f"{i}->{p}::lambda"
            for i in injector_names
            for p in producer_names
        ]
        + [
            f"{i}->{p}::tau_ij"
            for i in injector_names
            for p in producer_names
        ]
    )


def functional_diagnostics(
    chains: np.ndarray,
    log_posterior: Callable[[torch.Tensor], torch.Tensor],
    time_values: np.ndarray,
    injection: np.ndarray,
    production: np.ndarray,
    producer_start: np.ndarray,
    metadata: dict,
    tau_sd: float,
    lambda_sd: float,
    producer_names: list[str],
    n_keep: int = 120,
) -> pd.DataFrame:
    """Diagnose quantities the forecast actually uses."""

    m, n, d = chains.shape
    n_theta = d
    indices = np.unique(np.linspace(0, n - 1, min(n_keep, n), dtype=int))
    n_prod = production.shape[1]
    n_inj = injection.shape[1]

    scale = float(metadata["scale"])
    t = torch.tensor(np.asarray(time_values, dtype=float), dtype=torch.float64)
    w = torch.tensor(np.asarray(injection / scale, dtype=float), dtype=torch.float64)
    q_scaled = production / scale
    q_start = torch.tensor(
        q_scaled[producer_start, np.arange(n_prod)], dtype=torch.float64
    )
    starts = torch.tensor(producer_start, dtype=torch.long)

    rows = np.empty((m, len(indices), 1 + 2 * n_prod), dtype=float)
    with torch.no_grad():
        for c in range(m):
            for k, idx in enumerate(indices):
                th = torch.tensor(chains[c, idx], dtype=torch.float64)
                rows[c, k, 0] = float(log_posterior(th))
                log_tau_j, z, log_tau_ij = physical_from_theta(
                    th,
                    n_inj,
                    n_prod,
                    metadata["log_tau_reference"],
                    tau_sd,
                    lambda_sd,
                )
                q = crm_simulate_torch(
                    log_tau_j,
                    z,
                    log_tau_ij,
                    t,
                    w,
                    q_start,
                    starts,
                ).numpy() * scale
                # Last history row is the boundary diagnostic; forecast mean is all
                # rows from n_history onward and is supplied by the caller through
                # the selected time array.
                rows[c, k, 1:1 + n_prod] = np.log(np.maximum(q[-1], EPS))
                rows[c, k, 1 + n_prod:] = np.log(
                    np.maximum(q.mean(axis=0), EPS)
                )

    names = (
        ["log_posterior"]
        + [f"{p}::log_crm_last_time" for p in producer_names]
        + [f"{p}::log_crm_mean_all_time" for p in producer_names]
    )

    diagnostic_rhat = rhat(rows)
    diagnostic_ess = ess(rows)
    return pd.DataFrame(
        {
            "quantity": names,
            "R_hat": diagnostic_rhat,
            "ESS_bulk": diagnostic_ess,
        }
    )


# -----------------------------------------------------------------------------
# Summary and metrics
# -----------------------------------------------------------------------------


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination."""

    y = np.asarray(observed, float)
    p = np.asarray(predicted, float)
    denominator = np.sum((y - y.mean()) ** 2)
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - p) ** 2) / denominator)


def deterministic_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    producer_names: list[str],
    active_mask: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Compute point metrics per producer on the requested active mask."""

    rows = []
    for j, name in enumerate(producer_names):
        m = active_mask[:, j]
        y = observed[m, j]
        p = predicted[m, j]
        if len(y) == 0:
            rows.append({
                "producer": name,
                "model": model_name,
                "n": 0,
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
            })
            continue
        rows.append({
            "producer": name,
            "model": model_name,
            "n": int(len(y)),
            "MAE": float(np.mean(np.abs(y - p))),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "R2": r_squared(y, p),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def safe_filename(name: str) -> str:
    """Make a producer name safe for a Windows/Unix filename."""

    keep = []
    for char in str(name):
        if char.isalnum() or char in "._-":
            keep.append(char)
        elif char in " /\\":
            keep.append("_")
        else:
            keep.append("_")
    result = "".join(keep).strip("._")
    return result or "producer"


def save_individual_plots(
    rows: pd.DataFrame,
    producer_names: list[str],
    output_directory: Path,
    forecast_boundary: float,
) -> None:
    """Save one independent figure per producer and no combined figure."""

    plot_directory = output_directory / "wells"
    plot_directory.mkdir(parents=True, exist_ok=True)

    for producer in producer_names:
        d = rows.loc[rows["producer"] == producer].sort_values("time")
        if d.empty:
            continue

        figure, axis = plt.subplots(figsize=(10, 5.5))
        x = d["time"].to_numpy(float)
        observed = d["observed"].to_numpy(float)
        p50 = d["P50"].to_numpy(float)
        lower = d["95pct_lower"].to_numpy(float)
        upper = d["95pct_upper"].to_numpy(float)

        axis.plot(x, observed, linewidth=1.4, label="Observed production")
        axis.plot(x, p50, linewidth=1.8, linestyle="--", label="CRM P50")
        axis.fill_between(
            x,
            lower,
            upper,
            alpha=0.20,
            label="95% parameter uncertainty",
        )
        axis.axvline(
            forecast_boundary,
            linestyle="--",
            linewidth=1.2,
            label="Forecast boundary",
        )

        axis.set_xlabel("Time")
        axis.set_ylabel("Production rate")
        axis.set_title(f"{producer} — Bayesian CRM parameter-level UQ")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
        figure.tight_layout()

        path = plot_directory / f"{safe_filename(producer)}.png"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        progress(f"Saved individual well plot: {path}")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian parameter-level CRM UQ using adaptive HMC"
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--forecast", type=int, required=True)

    parser.add_argument("--mcmc-chains", type=int, default=4)
    parser.add_argument("--mcmc-samples", type=int, default=1000)
    parser.add_argument("--mcmc-burnin", type=int, default=1500)
    parser.add_argument("--mcmc-thin", type=int, default=1)

    parser.add_argument("--hmc-step-size", type=float, default=0.005)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=10)
    parser.add_argument("--hmc-leapfrog-jitter", type=int, default=2)
    parser.add_argument("--hmc-target-accept", type=float, default=0.80)
    parser.add_argument("--hmc-mass-matrix", choices=["diagonal", "dense"], default="dense")

    parser.add_argument("--tau-log-sd", type=float, default=1.5)
    parser.add_argument("--lambda-logit-sd", type=float, default=1.5)

    parser.add_argument(
        "--calibration-objective",
        choices=["linear-positive", "linear-all", "log-positive"],
        default="linear-positive",
    )
    parser.add_argument("--crm-starts", type=int, default=3)
    parser.add_argument("--crm-max-iter", type=int, default=500)

    parser.add_argument("--map-steps", type=int, default=75)
    parser.add_argument("--map-learning-rate", type=float, default=0.03)

    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--out", default="bayesian_crm_parameter_uq")
    parser.add_argument("--no-lag1-tempering", action="store_true")

    args = parser.parse_args()

    if args.forecast < 1:
        parser.error("--forecast must be at least 1")
    if args.mcmc_chains < 2:
        parser.error("--mcmc-chains must be at least 2")
    if args.mcmc_samples < 20:
        parser.error("--mcmc-samples must be at least 20")
    if args.mcmc_burnin < 100:
        parser.error("--mcmc-burnin must be at least 100")
    if args.mcmc_thin < 1:
        parser.error("--mcmc-thin must be at least 1")
    if args.hmc_step_size <= 0:
        parser.error("--hmc-step-size must be positive")
    if args.hmc_leapfrog_steps < 1:
        parser.error("--hmc-leapfrog-steps must be at least 1")
    if args.hmc_target_accept <= 0 or args.hmc_target_accept >= 1:
        parser.error("--hmc-target-accept must be between 0 and 1")
    if args.tau_log_sd <= 0 or args.lambda_logit_sd <= 0:
        parser.error("Prior standard deviations must be positive")

    print("\n" + "=" * 72, flush=True)
    print("BAYESIAN CRM PARAMETER-LEVEL UNCERTAINTY QUANTIFICATION", flush=True)
    print("CRM-ML: EXCLUDED", flush=True)
    print("=" * 72, flush=True)

    # ------------------------------------------------------------------
    # Stage 1: data
    # ------------------------------------------------------------------
    start_time = stage(1, "Loading and validating data...")
    data = load_excel(args.data)
    progress(f"Time points: {len(data.time)}")
    progress(f"Producers: {len(data.producer_names)}")
    progress(f"Injectors: {len(data.injector_names)}")
    progress(
        f"Production scale used for calibration/Bayesian likelihood: "
        f"{np.mean(np.abs(data.production)):.8g}"
    )
    finished(start_time, "Data loading completed")

    # ------------------------------------------------------------------
    # Stage 2: split history/forecast and determine producer starts
    # ------------------------------------------------------------------
    start_time = stage(2, "Preparing historical + forecast timeline...")
    n_total = len(data.time)
    n_history = n_total - args.forecast
    if n_history < 20:
        raise ValueError("At least 20 historical rows are required before the forecast window.")

    history = CRMData(
        time=data.time[:n_history],
        production=data.production[:n_history],
        injection=data.injection[:n_history],
        producer_names=data.producer_names,
        injector_names=data.injector_names,
    )

    producer_start = first_positive_indices(history.production)
    if np.any(producer_start < 0):
        bad_names = [
            history.producer_names[j]
            for j in np.flatnonzero(producer_start < 0)
        ]
        raise ValueError(f"No positive historical production found for: {bad_names}")

    progress(f"Historical rows: {n_history}")
    progress(f"Forecast rows: {args.forecast}")
    progress(f"Last historical time: {data.time[n_history - 1]:g}")
    progress(f"First forecast time: {data.time[n_history]:g}")

    for j, name in enumerate(history.producer_names):
        progress(
            f"{name}: first positive historical production at "
            f"row {producer_start[j] + 1}, time {history.time[producer_start[j]]:g}, "
            f"q0={history.production[producer_start[j], j]:g}"
        )
    finished(start_time, "Timeline and producer initial states prepared")

    # ------------------------------------------------------------------
    # Stage 3: deterministic calibration
    # ------------------------------------------------------------------
    start_time = stage(3, "Deterministic CRM calibration...")
    progress(f"Objective: {args.calibration_objective}")
    progress(f"Optimisation starts: {args.crm_starts}")
    progress(f"Maximum iterations per start: {args.crm_max_iter}")

    calibrator = Calibrator(
        starts=args.crm_starts,
        max_iter=args.crm_max_iter,
        seed=args.seed,
        objective_mode=args.calibration_objective,
    ).fit(
        history.time,
        history.injection,
        history.production,
        producer_start,
    )

    progress(f"Best calibration objective: {calibrator.objective_:.8g}")
    progress(f"Best optimiser success flag: {calibrator.success_}")
    progress(f"Best optimiser message: {calibrator.message_}")
    finished(start_time, "Deterministic CRM calibration completed")

    if calibrator.params_ is None:
        raise RuntimeError("Deterministic calibration produced no parameters.")

    # ------------------------------------------------------------------
    # Stage 4: Bayesian posterior and MAP preconditioning
    # ------------------------------------------------------------------
    start_time = stage(4, "Building Bayesian posterior...")

    history_scale = calibrator.scale_
    tmin = float(np.min(np.diff(history.time)))
    span = max(float(history.time[-1] - history.time[0]), 1.01 * tmin)
    log_tau_reference = 0.5 * (math.log(tmin) + math.log(span))
    tau_reference = math.exp(log_tau_reference)

    log_posterior, posterior_meta = make_logpost(
        history.time,
        history.injection,
        history.production,
        producer_start,
        args.tau_log_sd,
        args.lambda_logit_sd,
        use_lag1_tempering=not args.no_lag1_tempering,
        deterministic_parameters=calibrator.params_,
    )

    theta_start = theta_from_params(
        calibrator.params_,
        log_tau_reference,
        args.tau_log_sd,
        args.lambda_logit_sd,
    )

    # Give the initial state a small deterministic perturbation only after converting
    # to standardised coordinates, where the prior scales are comparable.
    initial_logp = float(
        log_posterior(torch.tensor(theta_start, dtype=torch.float64))
    )
    if not np.isfinite(initial_logp):
        raise RuntimeError("Initial Bayesian posterior density is non-finite.")

    progress(f"HMC parameter dimension: {theta_start.size}")
    progress(f"Minimum tau: {tmin:g}")
    progress(f"Reference tau: {tau_reference:g}")
    progress(f"log(tau) prior SD: {args.tau_log_sd:g}")
    progress(f"lambda-logit prior SD: {args.lambda_logit_sd:g}")
    progress("Likelihood: Lognormal production model")
    progress("sigma^2: Inverse-Gamma prior, analytically marginalised")
    progress(
        "Lag-1 residual tempering: "
        + ("enabled" if not args.no_lag1_tempering else "disabled")
    )
    progress(
        "Lag-1 effective-sample weights: "
        + np.array2string(np.round(posterior_meta["rho_weights"], 3), separator=", ")
    )
    progress(f"Initial posterior log density: {initial_logp:.6g}")

    if args.map_steps > 0:
        theta_map = map_refine(
            log_posterior,
            theta_start,
            len(data.injector_names),
            len(data.producer_names),
            log_tau_reference,
            math.log(tmin),
            args.tau_log_sd,
            args.lambda_logit_sd,
            args.map_steps,
            args.map_learning_rate,
        )
    else:
        theta_map = theta_start.copy()

    map_logp = float(log_posterior(torch.tensor(theta_map, dtype=torch.float64)))
    progress(f"MAP-preconditioned posterior log density: {map_logp:.6g}")
    finished(start_time, "Bayesian posterior constructed")

    # ------------------------------------------------------------------
    # Stage 5: HMC
    # ------------------------------------------------------------------
    start_time = stage(5, "Running adaptive Hamiltonian Monte Carlo...")
    progress(f"Chains: {args.mcmc_chains}")
    progress(f"Retained samples per chain: {args.mcmc_samples}")
    progress(f"Burn-in: {args.mcmc_burnin}")
    progress(f"Thinning: {args.mcmc_thin}")
    progress(f"Initial step size request: {args.hmc_step_size}")
    progress(f"Leapfrog steps: {args.hmc_leapfrog_steps} ± {args.hmc_leapfrog_jitter}")
    progress(f"Target acceptance: {args.hmc_target_accept}")
    progress(f"Mass matrix: {args.hmc_mass_matrix}")

    chains = []
    chain_results: list[HMCResult] = []

    # Four chains are recommended for the final experiment. Each chain gets a distinct
    # moderate perturbation of the MAP state in standardised coordinates.
    for chain_number in range(args.mcmc_chains):
        progress(f"Starting chain {chain_number + 1}/{args.mcmc_chains}")
        rng = np.random.default_rng(args.seed + 100_000 * chain_number)
        start = theta_map + rng.normal(0.0, 0.20, size=theta_map.size)

        # Make sure the perturbed time coordinates remain above the physical tau wall.
        tau_coordinates = np.r_[
            np.arange(len(data.producer_names)),
            np.arange(
                len(data.producer_names) + len(data.injector_names) * len(data.producer_names),
                theta_map.size,
            ),
        ]
        log_tau = log_tau_reference + args.tau_log_sd * start[tau_coordinates]
        start[tau_coordinates] = np.maximum(
            start[tau_coordinates],
            (math.log(tmin) + 0.02 - log_tau_reference) / args.tau_log_sd,
        )

        result = hmc(
            log_posterior=log_posterior,
            start=start,
            samples=args.mcmc_samples,
            burnin=args.mcmc_burnin,
            thin=args.mcmc_thin,
            initial_step_size=args.hmc_step_size,
            leapfrog_steps=args.hmc_leapfrog_steps,
            seed=args.seed + 100_000 * chain_number,
            target_accept=args.hmc_target_accept,
            mass_mode=args.hmc_mass_matrix,
            leapfrog_jitter=args.hmc_leapfrog_jitter,
        )

        chains.append(result.draws)
        chain_results.append(result)
        progress(
            f"Chain {chain_number + 1}: acceptance={result.acceptance_rate:.2%}, "
            f"mean_accept_prob={result.mean_acceptance_probability:.2%}, "
            f"final_step={result.final_step_size:.6g}, "
            f"post-warmup divergences={result.divergences}, "
            f"post-warmup invalid={result.invalid_proposals}, "
            f"unique_fraction={result.unique_fraction:.2%}"
        )

    chains_array = np.stack(chains, axis=0)
    finished(start_time, "Adaptive HMC sampling completed")

    # ------------------------------------------------------------------
    # Stage 6: diagnostics
    # ------------------------------------------------------------------
    start_time = stage(6, "Checking MCMC diagnostics...")

    rhat_values = rhat(chains_array)
    ess_values = ess(chains_array)
    names = parameter_names(data.producer_names, data.injector_names)

    diagnostics = pd.DataFrame(
        {
            "parameter": names,
            "R_hat": rhat_values,
            "ESS_bulk": ess_values,
        }
    )

    max_rhat = float(np.max(rhat_values))
    min_ess = float(np.min(ess_values))
    total_divergences = int(sum(r.divergences for r in chain_results))
    max_repeated_fraction = float(max(1.0 - r.unique_fraction for r in chain_results))

    progress(f"Maximum R-hat: {max_rhat:.6g}")
    progress(f"Minimum bulk ESS: {min_ess:.6g}")
    progress(f"Total large-energy-error divergences: {total_divergences}")
    for i, result in enumerate(chain_results, start=1):
        progress(f"Chain {i} acceptance: {result.acceptance_rate:.2%}")
        progress(f"Chain {i} unique retained-state fraction: {result.unique_fraction:.2%}")

    functional = functional_diagnostics(
        chains_array,
        log_posterior,
        data.time,
        data.injection,
        data.production,
        producer_start,
        posterior_meta,
        args.tau_log_sd,
        args.lambda_logit_sd,
        data.producer_names,
    )
    functional_max_rhat = float(np.max(functional["R_hat"].to_numpy()))
    functional_min_ess = float(np.min(functional["ESS_bulk"].to_numpy()))
    progress(f"Maximum functional R-hat: {functional_max_rhat:.6g}")
    progress(f"Minimum functional ESS: {functional_min_ess:.6g}")

    operational_pass = bool(
        np.all(np.isfinite(rhat_values))
        and np.all(np.isfinite(ess_values))
        and max_rhat <= 1.05
        and min_ess >= 200
        and total_divergences == 0
        and max_repeated_fraction <= 0.50
        and np.isfinite(functional_max_rhat)
        and functional_max_rhat <= 1.05
    )

    if operational_pass:
        progress("✓ MCMC operational quality checks passed.")
    else:
        progress("WARNING: MCMC operational quality checks are not all satisfied.")
        progress("Do not treat the resulting intervals as final validated UQ yet.")

    finished(start_time, "MCMC diagnostics calculated")

    # ------------------------------------------------------------------
    # Stage 7: posterior propagation over complete timeline
    # ------------------------------------------------------------------
    start_time = stage(7, "Propagating posterior CRM parameters...")

    flat = chains_array.reshape(-1, chains_array.shape[-1])
    total_draws = len(flat)
    progress(f"Posterior draws: {total_draws}")

    predictions = np.empty(
        (total_draws, len(data.time), len(data.producer_names)),
        dtype=float,
    )

    propagation_checkpoints = set(
        np.unique(
            np.linspace(1, total_draws, min(10, total_draws), dtype=int)
        ).tolist()
    )

    # Convert observed producer starts to scaled initial production for the physical
    # and differentiable forward models.
    q_start_physical = data.production[producer_start, np.arange(len(data.producer_names))]

    for draw_number, theta in enumerate(flat, start=1):
        params = physical_from_theta(
            theta,
            len(data.injector_names),
            len(data.producer_names),
            posterior_meta["log_tau_reference"],
            args.tau_log_sd,
            args.lambda_logit_sd,
        )
        predictions[draw_number - 1] = crm_simulate(
            params,
            data.time,
            data.injection / history_scale,
            data.production / history_scale,
            producer_start,
        ) * history_scale

        if draw_number in propagation_checkpoints:
            progress(
                f"Propagation: {draw_number}/{total_draws} "
                f"({100.0 * draw_number / total_draws:.0f}%)"
            )

    p10, p50, p90 = np.quantile(predictions, [0.10, 0.50, 0.90], axis=0)
    lower95, upper95 = np.quantile(predictions, [0.025, 0.975], axis=0)
    finished(start_time, "Posterior propagation completed")

    # ------------------------------------------------------------------
    # Stage 8: outputs
    # ------------------------------------------------------------------
    start_time = stage(8, "Generating and saving outputs...")

    output_directory = Path(args.out)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Time-series output includes t0 and every historical/forecast row.
    rows = []
    for k, current_time in enumerate(data.time):
        period = "historical" if k < n_history else "forecast"
        for j, producer in enumerate(data.producer_names):
            rows.append(
                {
                    "time": float(current_time),
                    "period": period,
                    "producer": producer,
                    "observed": float(data.production[k, j]),
                    "P10_low": float(p10[k, j]),
                    "P50": float(p50[k, j]),
                    "P90_high": float(p90[k, j]),
                    "95pct_lower": float(lower95[k, j]),
                    "95pct_upper": float(upper95[k, j]),
                    "interval_width_95": float(upper95[k, j] - lower95[k, j]),
                    "producer_start_row": int(producer_start[j] + 1),
                    "producer_start_time": float(data.time[producer_start[j]]),
                }
            )

    timeseries = pd.DataFrame(rows)
    timeseries.to_csv(output_directory / "crm_parameter_uq_timeseries.csv", index=False)

    # Deterministic CRM path over the complete timeline.
    deterministic_prediction = crm_simulate(
        calibrator.params_,
        data.time,
        data.injection / history_scale,
        data.production / history_scale,
        producer_start,
    ) * history_scale

    pd.DataFrame(
        [
            {
                "time": float(data.time[k]),
                **{
                    data.producer_names[j]: float(deterministic_prediction[k, j])
                    for j in range(len(data.producer_names))
                },
            }
            for k in range(len(data.time))
        ]
    ).to_csv(output_directory / "deterministic_crm_timeseries.csv", index=False)

    # Historical active positive metric mask for deterministic CRM.
    hist_rows = np.arange(n_history)[:, None]
    historical_active_positive = (
        (hist_rows > producer_start[None, :])
        & (data.production[:n_history] > 0)
    )
    det_hist_metrics = deterministic_metrics(
        data.production[:n_history],
        deterministic_prediction[:n_history],
        data.producer_names,
        historical_active_positive,
        "deterministic CRM",
    )
    det_hist_metrics.to_csv(output_directory / "deterministic_crm_historical_metrics.csv", index=False)

    # Forecast point metrics include every forecast row. This does not change the
    # Bayesian likelihood and is only an external comparison table.
    forecast_metrics_mask = np.ones((args.forecast, len(data.producer_names)), dtype=bool)
    det_forecast_metrics = deterministic_metrics(
        data.production[n_history:],
        deterministic_prediction[n_history:],
        data.producer_names,
        forecast_metrics_mask,
        "deterministic CRM forecast",
    )
    det_forecast_metrics.to_csv(output_directory / "deterministic_crm_forecast_metrics.csv", index=False)

    # Raw HMC states.
    np.savez_compressed(
        output_directory / "crm_parameter_hmc_samples.npz",
        chains=chains_array,
    )

    diagnostics.to_csv(output_directory / "mcmc_diagnostics.csv", index=False)
    functional.to_csv(output_directory / "mcmc_functional_diagnostics.csv", index=False)

    pd.DataFrame(
        {
            "chain": np.arange(1, len(chain_results) + 1),
            "acceptance_rate": [r.acceptance_rate for r in chain_results],
            "mean_acceptance_probability": [r.mean_acceptance_probability for r in chain_results],
            "final_step_size": [r.final_step_size for r in chain_results],
            "divergences": [r.divergences for r in chain_results],
            "invalid_proposals_post_warmup": [r.invalid_proposals for r in chain_results],
            "warmup_divergences": [r.warmup_divergences for r in chain_results],
            "warmup_invalid_proposals": [r.warmup_invalid_proposals for r in chain_results],
            "unique_fraction": [r.unique_fraction for r in chain_results],
            "final_log_posterior": [r.final_log_posterior for r in chain_results],
        }
    ).to_csv(output_directory / "hmc_acceptance.csv", index=False)

    # Physical posterior parameter summary.
    physical_samples = []
    for theta in flat:
        params = physical_from_theta(
            theta,
            len(data.injector_names),
            len(data.producer_names),
            posterior_meta["log_tau_reference"],
            args.tau_log_sd,
            args.lambda_logit_sd,
        )
        physical_samples.append(
            np.concatenate([
                params.tau_j.ravel(),
                params.lambda_ij.ravel(),
                params.tau_ij.ravel(),
            ])
        )
    physical_samples = np.asarray(physical_samples, dtype=float)
    pnames = physical_parameter_names(data.producer_names, data.injector_names)

    pd.DataFrame(
        {
            "parameter": pnames,
            "mean": np.mean(physical_samples, axis=0),
            "median": np.median(physical_samples, axis=0),
            "sd": np.std(physical_samples, axis=0, ddof=1),
            "q2.5": np.quantile(physical_samples, 0.025, axis=0),
            "q97.5": np.quantile(physical_samples, 0.975, axis=0),
        }
    ).to_csv(output_directory / "crm_parameter_posterior_summary.csv", index=False)

    pd.DataFrame(
        {
            "producer": data.producer_names,
            "first_positive_history_row": producer_start + 1,
            "first_positive_time": data.time[producer_start],
            "initial_production": data.production[producer_start, np.arange(len(data.producer_names))],
            "positive_history_rows_used": posterior_meta["n_observed"],
            "lag1_effective_sample_weight": posterior_meta["rho_weights"],
        }
    ).to_csv(output_directory / "producer_initial_states.csv", index=False)

    # Save concise run settings so the experiment is reproducible.
    settings = {
        "forecast_rows": args.forecast,
        "historical_rows": n_history,
        "chains": args.mcmc_chains,
        "samples_per_chain": args.mcmc_samples,
        "burnin": args.mcmc_burnin,
        "thin": args.mcmc_thin,
        "hmc_step_size_initial": args.hmc_step_size,
        "hmc_leapfrog_steps": args.hmc_leapfrog_steps,
        "hmc_leapfrog_jitter": args.hmc_leapfrog_jitter,
        "hmc_target_accept": args.hmc_target_accept,
        "hmc_mass_matrix": args.hmc_mass_matrix,
        "tau_log_sd": args.tau_log_sd,
        "lambda_logit_sd": args.lambda_logit_sd,
        "tau_reference": tau_reference,
        "tau_min": tmin,
        "a0": posterior_meta["a0"],
        "b0": posterior_meta["b0"],
        "calibration_objective": args.calibration_objective,
        "crm_starts": args.crm_starts,
        "crm_max_iter": args.crm_max_iter,
        "map_steps": args.map_steps,
        "map_learning_rate": args.map_learning_rate,
        "lag1_tempering": not args.no_lag1_tempering,
        "operational_checks_passed": operational_pass,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "functional_max_rhat": functional_max_rhat,
        "functional_min_ess_bulk": functional_min_ess,
        "total_divergences": total_divergences,
        "history_scale": history_scale,
    }
    pd.DataFrame(
        {"setting": list(settings.keys()), "value": list(settings.values())}
    ).to_csv(output_directory / "run_settings.csv", index=False)

    # The only figures produced are individual producer figures.
    save_individual_plots(
        timeseries,
        data.producer_names,
        output_directory,
        float(data.time[n_history - 1]),
    )

    finished(start_time, "Outputs generated")

    print("\n" + "=" * 72, flush=True)
    print("RUN COMPLETE", flush=True)
    print("=" * 72, flush=True)
    print(f"Results: {output_directory.resolve()}", flush=True)
    print(f"Maximum R-hat: {max_rhat:.6g}", flush=True)
    print(f"Minimum bulk ESS: {min_ess:.6g}", flush=True)
    print(f"Functional maximum R-hat: {functional_max_rhat:.6g}", flush=True)
    print(f"Total divergences: {total_divergences}", flush=True)
    print(f"MCMC operational checks passed: {operational_pass}", flush=True)
    print(
        "CRM-ML is excluded. Intervals are induced by uncertainty in the CRM "
        "physical parameters.",
        flush=True,
    )
    if not operational_pass:
        print(
            "WARNING: Do not treat these intervals as final validated UQ until "
            "MCMC quality checks are satisfactory.",
            flush=True,
        )


if __name__ == "__main__":
    main()
