#!/usr/bin/env python3
"""Bayesian CRM parameter-level uncertainty quantification.

The program deliberately excludes CRM-ML.  It calibrates a deterministic CRM,
constructs a Bayesian posterior over CRM parameters, samples that posterior with
Hamiltonian Monte Carlo (HMC), propagates the sampled parameters through the CRM
on the complete historical-plus-forecast timeline, and reports production
quantiles and 95% posterior intervals.
"""

# Standard-library imports are kept explicit so each dependency is easy to identify.
import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

# NumPy and pandas handle numerical arrays and tabular input/output respectively.
import numpy as np
import pandas as pd

# PyTorch is used here because HMC needs gradients of the log-posterior.
import torch

# Matplotlib is configured for non-interactive file output because this script is
# normally executed from a terminal rather than from a notebook.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SciPy supplies the constrained deterministic optimizer used to obtain a sensible
# starting point before Bayesian sampling begins.
from scipy.optimize import minimize


# Keeping the number of console stages in one place prevents stage labels from
# becoming inconsistent when the pipeline is edited later.
TOTAL_STAGES = 8


def stage(number: int, message: str) -> float:
    """Print a clearly separated pipeline stage and return its start time."""
    print("\n" + "=" * 72, flush=True)
    print(f"[{number}/{TOTAL_STAGES}] {message}", flush=True)
    print("=" * 72, flush=True)
    return time.perf_counter()


def progress(message: str) -> None:
    """Print an indented progress message immediately, even when output is buffered."""
    print(f"    {message}", flush=True)


def finished(start_time: float, message: str = "Completed") -> None:
    """Report how long a pipeline stage took."""
    elapsed = time.perf_counter() - start_time
    duration = f"{elapsed:.1f} seconds" if elapsed < 60 else f"{elapsed / 60:.2f} minutes"
    progress(f"✓ {message} ({duration})")


@dataclass
class CRMParameters:
    """Physical CRM parameters for all producers and injector-producer pairs."""

    # tau_j controls the producer's response/decline time scale.
    tau_j: np.ndarray

    # lambda_ij describes the fractional connectivity from injector i to producer j.
    lambda_ij: np.ndarray

    # tau_ij controls the response time between injector i and producer j.
    tau_ij: np.ndarray


@dataclass
class CRMData:
    """Container holding the time series and well names read from Excel."""

    # Time must be strictly increasing because the CRM uses successive time steps.
    time: np.ndarray

    # Production is arranged as rows=time points and columns=producers.
    production: np.ndarray

    # Injection is arranged as rows=time points and columns=injectors.
    injection: np.ndarray

    # Human-readable producer names retained for output files and plots.
    producer_names: list[str]

    # Human-readable injector names retained for parameter summaries.
    injector_names: list[str]


def load_excel(path: str | Path) -> CRMData:
    """Load time, production, and injection sheets from the supplied workbook."""

    # Open the workbook once so all sheet parsing uses the same ExcelFile object.
    xls = pd.ExcelFile(path)

    def find(names: list[str]) -> str:
        """Find a sheet using several accepted case-insensitive aliases."""
        # Build a normalized lookup so Time, time, and TIME are treated alike.
        mapping = {str(sheet).strip().lower(): str(sheet) for sheet in xls.sheet_names}

        # Return the first alias that exists in the workbook.
        for name in names:
            if name in mapping:
                return mapping[name]

        # Fail early with the actual workbook sheet names instead of producing a
        # less useful error later in the numerical code.
        raise ValueError(f"Missing sheet; tried {names}. Found {xls.sheet_names}")

    def parse(name: str) -> pd.DataFrame:
        """Read one sheet whose first non-empty row contains column names."""
        # Read without assuming pandas should interpret the first row as a header.
        raw = xls.parse(name, header=None)

        # Remove completely empty rows and rebuild a clean integer index.
        raw = raw.dropna(how="all").reset_index(drop=True)

        # Treat the first remaining row as the user-supplied column names.
        columns = [str(value).strip() for value in raw.iloc[0]]

        # Convert the remaining cells to numbers; non-numeric cells become NaN.
        body = raw.iloc[1:].apply(pd.to_numeric, errors="coerce")

        # Keep only rows where every expected numeric field is present.
        body = body.loc[body.notna().all(axis=1)].reset_index(drop=True)
        body.columns = columns
        return body

    # Accept common names while still requiring separate time, production, and
    # injection data sources.
    t = parse(find(["time", "t", "date", "days"]))
    q = parse(find(["q_observed", "production", "q_obs", "q"]))
    w = parse(find(["w_injection", "injection", "w_inj", "inj"]))

    # Convert the three tables into numerical arrays used by the model.
    time_values = t.iloc[:, 0].to_numpy(float)
    production = q.to_numpy(float)
    injection = w.to_numpy(float)

    # All three tables must describe exactly the same number of time points.
    if len(time_values) != len(production) or len(time_values) != len(injection):
        raise ValueError("Time, production and injection lengths differ.")

    # Time must be finite and strictly increasing so every CRM interval has a
    # positive duration.
    if np.any(~np.isfinite(time_values)) or np.any(np.diff(time_values) <= 0):
        raise ValueError("Time must be finite and strictly increasing.")

    # Production and injection cannot contain missing, infinite, or negative values.
    if (
        np.any(~np.isfinite(production))
        or np.any(production < 0)
        or np.any(~np.isfinite(injection))
        or np.any(injection < 0)
    ):
        raise ValueError("Invalid negative/NaN data.")

    # Preserve the workbook column names so exported results remain interpretable.
    return CRMData(
        time_values,
        production,
        injection,
        list(q.columns),
        list(w.columns),
    )


def exp_filter(signal: np.ndarray, dt: np.ndarray, tau: float) -> np.ndarray:
    """Apply a first-order exponential response filter to an injection signal."""

    # The state starts at zero before the first injection observation.
    previous = 0.0

    # Allocate the output once rather than repeatedly growing a Python list.
    output = np.empty_like(signal, dtype=float)

    # Each coefficient describes how much of the previous response survives the
    # current time interval.
    for n, coefficient in enumerate(np.exp(-dt / tau)):
        previous = coefficient * previous + (1.0 - coefficient) * signal[n]
        output[n] = previous

    return output


def crm_simulate(
    parameters: CRMParameters,
    time_values: np.ndarray,
    injection: np.ndarray,
    q0: np.ndarray,
) -> np.ndarray:
    """Evaluate the first-order CRM from the initial production state onward."""

    # I is the number of injectors and P is the number of producers.
    n_injectors, n_producers = parameters.lambda_ij.shape

    # dt is the interval between observations; tr measures elapsed time from the
    # initial production observation used as the starting state.
    dt = np.diff(time_values)
    elapsed = time_values[1:] - time_values[0]

    # The first term represents natural/producer response from the initial rate.
    output = q0[None, :] * np.exp(
        -elapsed[:, None] / parameters.tau_j[None, :]
    )

    # Each injector-producer pair contributes a filtered injection response scaled
    # by its connectivity coefficient.
    for injector in range(n_injectors):
        for producer in range(n_producers):
            response = exp_filter(
                injection[1:, injector],
                dt,
                parameters.tau_ij[injector, producer],
            )
            output[:, producer] += (
                parameters.lambda_ij[injector, producer] * response
            )

    # The likelihood works in log space, so prevent numerical log(0) values while
    # preserving the model's positive-production representation.
    return np.maximum(output, 1e-12)


def lam_to_z(lam: np.ndarray) -> np.ndarray:
    """Convert injector-wise constrained connectivity fractions into logits."""

    # The previous CRM implementation constrains the connectivity contributions
    # for each injector, so the transformation must operate across producers
    # within each injector row.
    lam = np.asarray(lam, dtype=float)
    n_injectors, n_producers = lam.shape
    z = np.empty_like(lam)

    # The slack term represents the portion of an injector's connectivity that is
    # not assigned to the listed producers. This guarantees that each injector's
    # connectivity weights remain non-negative and sum to at most one.
    for injector in range(n_injectors):
        row = np.clip(lam[injector], 1e-10, None)
        slack = max(1.0 - float(row.sum()), 1e-10)
        probabilities = np.r_[row, slack]
        probabilities /= probabilities.sum()
        z[injector] = np.log(probabilities[:-1] / probabilities[-1])

    return z


def z_to_lam(z: np.ndarray) -> np.ndarray:
    """Map injector-wise unconstrained logits back to valid CRM fractions."""

    # The softmax is applied across each injector's producer connections plus one
    # slack category, matching the deterministic CRM constraint used for fitting.
    z = np.asarray(z, dtype=float)
    n_injectors, n_producers = z.shape
    output = np.empty_like(z)

    # Add one fixed zero-logit slack category to every injector row. Removing that
    # category leaves lambda values whose row sum is automatically <= 1.
    for injector in range(n_injectors):
        augmented = np.r_[z[injector], 0.0]
        augmented -= augmented.max()
        probabilities = np.exp(augmented)
        probabilities /= probabilities.sum()
        output[injector] = probabilities[:-1]

    return output


def theta_to_par(theta: np.ndarray, n_injectors: int, n_producers: int) -> CRMParameters:
    """Transform an HMC state vector back into physical CRM parameters."""

    # Exponentiation guarantees positive producer and interwell time constants.
    tau_j = np.exp(theta[:n_producers])

    # The middle block contains injector-producer connectivity logits.
    z = theta[n_producers : n_producers + n_injectors * n_producers].reshape(
        n_injectors, n_producers
    )

    # The final block contains logarithmic interwell response times.
    tau_ij = np.exp(
        theta[n_producers + n_injectors * n_producers :]
    ).reshape(n_injectors, n_producers)

    # Convert logits back to physically constrained connectivity fractions.
    lambda_ij = z_to_lam(z)

    return CRMParameters(tau_j, lambda_ij, tau_ij)


class Calibrator:
    """Deterministically calibrate CRM parameters before Bayesian sampling."""

    def __init__(self, starts: int = 3, max_iter: int = 500, seed: int = 2025):
        # Multiple optimization starts reduce dependence on a single local optimum.
        self.starts = max(1, int(starts))
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.params_: CRMParameters | None = None
        self.objective_ = np.inf
        self.scale_ = 1.0

    def unpack(
        self,
        x: np.ndarray,
        n_injectors: int,
        n_producers: int,
    ) -> CRMParameters:
        """Convert the optimizer's parameter vector into physical CRM parameters."""

        # The first block contains log(tau_j), which keeps producer response times
        # strictly positive while SLSQP works in an unconstrained log coordinate.
        tau_j = np.exp(x[:n_producers])

        # The second block contains injector-to-producer connectivity values directly.
        # The SLSQP constraints below enforce the same injector-wise connectivity
        # restriction used by the earlier CRM implementation.
        lambda_ij = x[
            n_producers : n_producers + n_injectors * n_producers
        ].reshape(n_injectors, n_producers)

        # The final block contains log(tau_ij), again ensuring positive response times.
        tau_ij = np.exp(
            x[n_producers + n_injectors * n_producers :]
        ).reshape(n_injectors, n_producers)

        return CRMParameters(tau_j, lambda_ij, tau_ij)

    def fit(
        self,
        time_values: np.ndarray,
        injection: np.ndarray,
        production: np.ndarray,
    ) -> "Calibrator":
        """Fit the deterministic CRM using the same SLSQP structure as the prior implementation."""

        # Convert all inputs to ordinary floating-point arrays so SciPy receives a
        # predictable numerical representation.
        time_values, injection, production = map(
            lambda x: np.asarray(x, float),
            (time_values, injection, production),
        )

        # Identify the injector/producer dimensions and the smallest sampling interval.
        n_rows, n_injectors = injection.shape
        if time_values.size != n_rows or production.shape[0] != n_rows:
            raise ValueError("Time, production and injection must have equal lengths.")
        if production.ndim != 2 or n_rows < 10:
            raise ValueError("Production must be 2D and at least 10 time steps are required.")

        n_producers = production.shape[1]
        tmin = float(np.min(np.diff(time_values)))
        if tmin <= 0:
            raise ValueError("The minimum sampling interval must be positive.")

        # Match the earlier CRM implementation: tau_max is optional rather than an
        # artificial 100-times-history upper bound imposed by the new parameter-only code.
        # This lets the calibration determine a long response time when the data require it.
        tau_max = None

        # Scale production and injection by the mean absolute production rate. This is
        # the same numerical scaling used in the previous CRM implementation.
        self.scale_ = max(float(np.mean(np.abs(production))), 1e-12)
        q = production / self.scale_
        w = injection / self.scale_

        # Work with log(tau) bounds so every response time remains at least one sampling
        # interval and therefore stays physically/numerically meaningful.
        lo = np.log(tmin)
        hi = None if tau_max is None else np.log(tau_max)
        n_par = n_producers + 2 * n_injectors * n_producers

        # Parameter ordering is: log(tau_j), lambda_ij, log(tau_ij).
        bounds = (
            [(lo, hi)] * n_producers
            + [(0.0, 1.0)] * (n_injectors * n_producers)
            + [(lo, hi)] * (n_injectors * n_producers)
        )

        # Build one linear inequality for each injector. Each injector's total
        # connectivity to all producers must not exceed one.
        lambda_offset = n_producers
        constraint_matrix = np.zeros((n_injectors, n_par))
        for injector in range(n_injectors):
            constraint_matrix[
                injector,
                lambda_offset + injector * n_producers :
                lambda_offset + (injector + 1) * n_producers,
            ] = 1.0

        # SLSQP cannot be given an empty constraint set, so only add this block when
        # at least one injector exists.
        optimizer_options = {}
        if n_injectors > 0:
            optimizer_options["constraints"] = {
                "type": "ineq",
                "fun": lambda x: 1.0 - constraint_matrix @ x,
                "jac": lambda x: -constraint_matrix,
            }

        # The initial production observation is the starting state of the CRM recurrence.
        q0 = q[0]

        def objective(x: np.ndarray) -> float:
            """Return mean squared CRM error on the scaled historical series."""

            # Convert the optimizer vector into physical parameters and simulate the CRM.
            parameters = self.unpack(x, n_injectors, n_producers)
            prediction = crm_simulate(parameters, time_values, w, q0)

            # Compare every post-initial historical prediction with its observation.
            return float(np.mean((q[1:] - prediction) ** 2))

        # Use a fixed generator so the randomized starts are reproducible.
        rng = np.random.default_rng(self.seed)
        best_x = None
        best_f = np.inf

        # Run the requested number of local SLSQP optimizations and keep the best
        # finite solution rather than trusting the first successful start.
        for start in range(self.starts):
            x0 = np.empty(n_par)

            if start == 0:
                # The first start mirrors the previous implementation: tau begins at
                # ten sampling intervals and each injector initially receives a total
                # connectivity of 0.8 distributed equally across the producers.
                initial = np.log(10.0 * tmin)
                if hi is not None:
                    initial = min(initial, hi)
                x0[:n_producers] = initial
                x0[
                    lambda_offset :
                    lambda_offset + n_injectors * n_producers
                ] = 0.8 / max(n_producers, 1)
                x0[
                    lambda_offset + n_injectors * n_producers :
                ] = initial
            else:
                # Alternative starts explore a broad range of response times rather
                # than making every optimization begin near the same point.
                upper_init = max(10.0, time_values[-1] - time_values[0]) * tmin
                # Apply an upper tau bound only when one has actually been supplied.
                # tau_max is intentionally None in the current implementation because
                # we do not want to impose an artificial upper limit on the CRM response
                # time constants.
                if tau_max is not None:
                    upper_init = min(upper_init, float(tau_max))
                upper_log = np.log(max(tmin * 1.01, upper_init))

                x0[:n_producers] = rng.uniform(lo, upper_log, n_producers)

                if n_injectors > 0:
                    # A Dirichlet draw creates positive connectivity values whose
                    # total is exactly 0.9 for each injector, keeping every start feasible.
                    x0[
                        lambda_offset :
                        lambda_offset + n_injectors * n_producers
                    ] = (
                        0.9
                        * rng.dirichlet(
                            np.ones(n_producers),
                            size=n_injectors,
                        )
                    ).ravel()

                x0[
                    lambda_offset + n_injectors * n_producers :
                ] = rng.uniform(lo, upper_log, n_injectors * n_producers)

            # SLSQP handles both the individual parameter bounds and the injector-wise
            # connectivity constraints.
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": self.max_iter, "ftol": 1e-12},
                **optimizer_options,
            )

            # Retain the best finite result even if SciPy reports a non-success status;
            # this matches the robust selection logic used in the previous implementation.
            if np.isfinite(result.fun) and result.fun < best_f:
                best_x = result.x
                best_f = float(result.fun)

        if best_x is None:
            raise RuntimeError("CRM calibration failed for every optimisation start.")

        # Store the best physical parameter set and its scaled calibration objective.
        self.params_ = self.unpack(best_x, n_injectors, n_producers)
        self.objective_ = best_f
        return self

    def predict(
        self,
        time_values: np.ndarray,
        injection: np.ndarray,
        q0: np.ndarray,
    ) -> np.ndarray:
        """Predict production in the original physical rate units."""

        # A fitted parameter set is required before prediction is possible.
        if self.params_ is None:
            raise RuntimeError("CRM parameters are unavailable; fit() must succeed before predict().")

        # Apply the exact calibration scaling to the prediction inputs, run the CRM,
        # and then return the result to the original production units.
        return (
            crm_simulate(
                self.params_,
                time_values,
                np.asarray(injection, float) / self.scale_,
                np.asarray(q0, float) / self.scale_,
            )
            * self.scale_
        )


def make_logpost(
    time_values: np.ndarray,
    injection: np.ndarray,
    production: np.ndarray,
    tmin: float,
    tau_sd: float,
    lambda_sd: float,
):
    """Construct the differentiable Bayesian CRM log-posterior."""

    # Double precision is used because HMC gradients can be sensitive to numerical
    # rounding in a posterior with many interacting CRM parameters.
    dtype = torch.float64

    # Use the same production scaling concept as deterministic calibration.
    scale = max(float(production.mean()), 1e-12)
    tt = torch.tensor(time_values, dtype=dtype)
    ww = torch.tensor(injection / scale, dtype=dtype)
    q0 = torch.tensor(production[0] / scale, dtype=dtype)
    yy = torch.tensor(production[1:], dtype=dtype)

    # Determine the number of injectors and producers from the input matrices.
    n_injectors = injection.shape[1]
    n_producers = production.shape[1]

    # Only strictly positive production can enter the Lognormal likelihood because
    # log(0) is undefined. Zero rows remain in the original dataset and outputs.
    mask = torch.tensor(production[1:] > 0)
    n_observed = mask.sum(0).numpy().astype(float)

    # The minimum tau establishes a physically/numerically meaningful lower bound.
    log_min = math.log(tmin)

    # The reference tau is the geometric mean of the minimum time step and the
    # overall history span, providing a broad scale for the log-time priors.
    span = max(time_values[-1] - time_values[0], tmin * 1.01)
    log_reference = 0.5 * (math.log(tmin) + math.log(span))

    # These are the fixed hyperparameters of the Inverse-Gamma prior on sigma^2.
    # Integrating sigma^2 out reduces the HMC state dimension by one nuisance parameter.
    a0 = 2.0
    b0 = a0 * 0.2**2

    def log_posterior(theta: torch.Tensor) -> torch.Tensor:
        """Evaluate log posterior up to constants independent of CRM parameters."""

        # Split the HMC state into log(tau_j), connectivity logits, and log(tau_ij).
        log_tau_j = theta[:n_producers]
        z = theta[
            n_producers : n_producers + n_injectors * n_producers
        ].reshape(n_injectors, n_producers)
        log_tau_ij = theta[n_producers + n_injectors * n_producers :].reshape(
            n_injectors, n_producers
        )

        # Reject states below the minimum allowable response-time scale.
        if torch.any(log_tau_j < log_min) or torch.any(log_tau_ij < log_min):
            return torch.tensor(-float("inf"), dtype=dtype)

        # Transform logarithmic time constants back to positive physical values.
        tau_j = torch.exp(log_tau_j)
        tau_ij = torch.exp(log_tau_ij)

        # Append one zero-logit slack category to each injector row. The softmax is
        # therefore performed across that injector's producer connections, exactly
        # matching the deterministic CRM constraint sum_j lambda_ij <= 1.
        augmented = torch.cat(
            [z, torch.zeros((n_injectors, 1), dtype=dtype)],
            dim=1,
        )
        lambda_ij = torch.softmax(augmented, dim=1)[:, :n_producers]

        # Calculate the time intervals and elapsed time from the initial observation.
        dt = torch.diff(tt)
        elapsed = tt[1:] - tt[0]

        # Start each producer from its observed initial production state.
        q = q0[None, :] * torch.exp(
            -elapsed[:, None] / tau_j[None, :]
        )

        # Propagate each injector's first-order response into every producer.
        for injector in range(n_injectors):
            for producer in range(n_producers):
                coefficients = torch.exp(-dt / tau_ij[injector, producer])
                previous = torch.tensor(0.0, dtype=dtype)
                response = []

                # The recurrence is intentionally written explicitly so gradients
                # can flow through every response-time calculation.
                for k in range(dt.numel()):
                    previous = (
                        coefficients[k] * previous
                        + (1.0 - coefficients[k]) * ww[k + 1, injector]
                    )
                    response.append(previous)

                q[:, producer] = q[:, producer] + lambda_ij[injector, producer] * torch.stack(response)

        # Clamp tiny numerical values so log(q) remains defined.
        q = torch.clamp(q, min=1e-12)
        log_likelihood = torch.tensor(0.0, dtype=dtype)

        # Integrate sigma^2 out analytically under the Normal-Inverse-Gamma model.
        for producer in range(n_producers):
            residual = (
                torch.log(yy[:, producer][mask[:, producer]])
                - torch.log(q[:, producer][mask[:, producer]])
            )
            rss = torch.sum(residual * residual)
            log_likelihood = log_likelihood - (
                a0 + 0.5 * n_observed[producer]
            ) * torch.log(b0 + 0.5 * rss)

        # Apply independent Normal priors to the unconstrained log-time and logit
        # parameters.  The chosen SDs are modelling hyperparameters, not learned values.
        prior = (
            -0.5 * torch.sum(((log_tau_j - log_reference) / tau_sd) ** 2)
            -0.5 * torch.sum(((log_tau_ij - log_reference) / tau_sd) ** 2)
            -0.5 * torch.sum((z / lambda_sd) ** 2)
        )

        return log_likelihood + prior

    # Return the function itself plus useful scaling/reference values for diagnostics.
    return log_posterior, scale, math.exp(log_reference)


def hmc(
    log_posterior,
    start: np.ndarray,
    samples: int,
    burnin: int,
    thin: int,
    step_size: float,
    leapfrog_steps: int,
    seed: int,
):
    """Run a basic HMC chain with burn-in step-size adaptation and progress output."""

    # NumPy controls the accept/reject random numbers while PyTorch controls momentum
    # and posterior gradients, both seeded for reproducibility.
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Store the current HMC position as a differentiable float64 tensor.
    current = torch.tensor(start, dtype=torch.float64)
    dimension = current.numel()

    # Unit mass means momentum has a standard Normal distribution and simplifies the
    # kinetic-energy calculation.
    inverse_mass = torch.ones(dimension, dtype=torch.float64)
    mass = torch.ones(dimension, dtype=torch.float64)

    # The output contains only retained post-burn-in states.
    retained = []
    accepted_after_burnin = 0

    # Total iterations include both burn-in and retained sampling iterations.
    total = burnin + samples * thin

    # Define progress checkpoints explicitly.  This fixes the previous NameError:
    # the old function referenced `checkpoints` without creating it first.
    checkpoints = set(
        np.unique(
            np.linspace(1, total, min(10, total), dtype=int)
        ).tolist()
    )

    # HMC iteration loop.
    for iteration in range(1, total + 1):
        # Clone the current position so the gradient is calculated only for this proposal.
        position = current.detach().clone().requires_grad_(True)

        # Draw an auxiliary momentum from the Gaussian kinetic-energy distribution.
        momentum0 = torch.randn_like(position) * torch.sqrt(mass)

        # Evaluate the current log posterior and its gradient.
        logp0 = log_posterior(position)
        if not torch.isfinite(logp0):
            raise RuntimeError("HMC encountered a non-finite posterior at the current state.")
        gradient = torch.autograd.grad(logp0, position)[0]

        # Perform the first half-step of the leapfrog momentum update.
        momentum = momentum0 + 0.5 * step_size * gradient
        proposal = position
        proposal_logp = logp0
        proposal_gradient = gradient
        valid = True

        # Complete the requested number of leapfrog integration steps.
        for leapfrog in range(leapfrog_steps):
            # Update position using the current momentum.
            proposal = (
                proposal + step_size * inverse_mass * momentum
            ).detach().requires_grad_(True)

            # Evaluate the posterior at the new position.
            proposal_logp = log_posterior(proposal)

            # A non-finite posterior invalidates the whole HMC proposal.
            if not torch.isfinite(proposal_logp):
                valid = False
                break

            # Obtain the gradient required for the next momentum update.
            proposal_gradient = torch.autograd.grad(
                proposal_logp,
                proposal,
            )[0]

            # Full momentum updates occur between position updates; the final update
            # is handled separately as the required half-step below.
            if leapfrog < leapfrog_steps - 1:
                momentum = momentum + step_size * proposal_gradient

        accepted = False

        if valid:
            # Finish the final half-step and reverse momentum for detailed balance.
            momentum = momentum + 0.5 * step_size * proposal_gradient
            momentum = -momentum

            # Compute the Hamiltonian before and after the proposal.
            h0 = -logp0.detach() + 0.5 * torch.sum(momentum0 * momentum0 * inverse_mass)
            h1 = -proposal_logp.detach() + 0.5 * torch.sum(momentum * momentum * inverse_mass)

            # Accept according to the Metropolis-Hastings Hamiltonian criterion.
            log_acceptance = min(0.0, float(h0 - h1))
            if math.log(max(rng.random(), 1e-300)) < log_acceptance:
                current = proposal.detach()
                accepted = True

        # During burn-in, adapt the step size gently toward an approximate target
        # acceptance probability of 0.65.  After burn-in the step size stays fixed.
        if iteration <= burnin and iteration % 25 == 0:
            step_size = float(
                np.clip(
                    step_size * math.exp(0.05 * ((1 if accepted else 0) - 0.65)),
                    1e-4,
                    0.2,
                )
            )

        # Print progress at approximately ten evenly spaced checkpoints.  This is
        # deliberately outside the acceptance logic so progress output cannot alter
        # the numerical algorithm.
        if iteration in checkpoints:
            phase = "burn-in" if iteration <= burnin else "sampling"
            print(
                f"    {phase}: {iteration}/{total} "
                f"({100 * iteration / total:.0f}%) | step={step_size:.6g}",
                flush=True,
            )

        # Count acceptance only after burn-in because burn-in adaptation should not
        # be used as the reported sampling acceptance rate.
        if iteration > burnin:
            if accepted:
                accepted_after_burnin += 1

            # Retain one state every `thin` iterations after burn-in.
            if (iteration - burnin) % thin == 0:
                retained.append(current.numpy().copy())

    # The number of post-burn-in proposals is exactly samples*thin.
    acceptance_rate = accepted_after_burnin / max(samples * thin, 1)

    return np.asarray(retained), acceptance_rate, step_size


def rhat(chains: np.ndarray) -> np.ndarray:
    """Calculate a split-R-hat approximation for every sampled parameter."""

    # Expect shape [chains, draws, parameters].
    chains = np.asarray(chains)
    n_chains, n_draws, _ = chains.shape

    # Splitting each chain in half makes the diagnostic sensitive to within-chain
    # non-stationarity as well as between-chain differences.
    half = n_draws // 2
    if half < 2:
        return np.full(chains.shape[-1], np.nan)

    split = chains[:, : 2 * half].reshape(2 * n_chains, half, -1)

    # W measures average within-chain variance.
    within = split.var(axis=1, ddof=1).mean(axis=0)

    # B measures variation among split-chain means.
    between = half * split.mean(axis=1).var(axis=0, ddof=1)

    # Combine W and B using the standard split-R-hat structure.
    variance_hat = ((half - 1) / half) * within + between / half

    # Protect against division by zero when a parameter has effectively no variation.
    return np.sqrt(variance_hat / np.maximum(within, 1e-30))


def ess(chains: np.ndarray) -> np.ndarray:
    """Estimate bulk effective sample size using an autocorrelation truncation."""

    # Expect [chains, draws, parameters].
    chains = np.asarray(chains)
    n_chains, n_draws, n_parameters = chains.shape
    result = []

    # Calculate ESS separately for every parameter.
    for parameter in range(n_parameters):
        autocorrelation_sum = 0.0
        values = chains[:, :, parameter]

        # Estimate lag autocorrelation up to a moderate maximum lag.
        for lag in range(1, min(200, n_draws // 2)):
            correlations = []
            for chain in range(n_chains):
                series = values[chain]
                x = series[:-lag]
                y = series[lag:]

                # Skip pathological constant segments where correlation is undefined.
                if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
                    continue

                correlations.append(np.corrcoef(x, y)[0, 1])

            # Average the available chain correlations at this lag.
            if not correlations:
                break
            rho = float(np.nanmean(correlations))

            # Stop once autocorrelation becomes non-positive, a common truncation rule.
            if not np.isfinite(rho) or rho <= 0.0:
                break

            autocorrelation_sum += rho

        # Convert autocorrelation into an approximate effective sample count.
        result.append(
            max(1.0, n_chains * n_draws / (1.0 + 2.0 * autocorrelation_sum))
        )

    return np.asarray(result)

def par_to_theta(parameters: CRMParameters) -> np.ndarray:
    """Convert physical CRM parameters into the unconstrained HMC state."""
    log_tau_j = np.log(parameters.tau_j)

    z = lam_to_z(parameters.lambda_ij)

    log_tau_ij = np.log(parameters.tau_ij)

    return np.concatenate(
        [
            log_tau_j.ravel(),
            z.ravel(),
            log_tau_ij.ravel(),
        ]
    )


def main() -> None:
    """Parse arguments and execute the complete Bayesian CRM UQ pipeline."""

    # Define the command-line interface in one place so terminal usage remains explicit.
    parser = argparse.ArgumentParser(
        description="Bayesian CRM parameter-level UQ using HMC"
    )

    # Input workbook and forecast horizon are required because the model cannot infer
    # these reliably from a command-line invocation.
    parser.add_argument("--data", required=True)
    parser.add_argument("--forecast", type=int, required=True)

    # MCMC settings control the number of independent chains, retained samples,
    # burn-in iterations, and thinning interval.
    parser.add_argument("--mcmc-chains", type=int, default=4)
    parser.add_argument("--mcmc-samples", type=int, default=1000)
    parser.add_argument("--mcmc-burnin", type=int, default=1500)
    parser.add_argument("--mcmc-thin", type=int, default=1)

    # HMC settings control the leapfrog integration step and the number of leapfrog
    # steps taken for each proposal.
    parser.add_argument("--hmc-step-size", type=float, default=0.01)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=20)

    # Prior-width settings determine how broad the log-time and connectivity-logit
    # priors are around their reference values.
    parser.add_argument("--tau-log-sd", type=float, default=2.0)
    parser.add_argument("--lambda-logit-sd", type=float, default=1.5)

    # Deterministic calibration settings control optimizer restarts and iterations.
    parser.add_argument("--crm-starts", type=int, default=3)
    parser.add_argument("--crm-max-iter", type=int, default=500)

    # A fixed seed makes repeated experiments reproducible.
    parser.add_argument("--seed", type=int, default=2025)

    # Results are written to this directory rather than overwriting the input workbook.
    parser.add_argument("--out", default="bayesian_crm_uq_results")

    # Parse the user's actual command-line arguments.
    args = parser.parse_args()

    # Validate the most important settings before expensive computation begins.
    if args.forecast < 1:
        parser.error("--forecast must be at least 1")
    if args.mcmc_chains < 2:
        parser.error("--mcmc-chains must be at least 2")
    if args.mcmc_samples < 1:
        parser.error("--mcmc-samples must be at least 1")
    if args.mcmc_burnin < 0:
        parser.error("--mcmc-burnin must be non-negative")
    if args.mcmc_thin < 1:
        parser.error("--mcmc-thin must be at least 1")
    if args.hmc_step_size <= 0:
        parser.error("--hmc-step-size must be positive")
    if args.hmc_leapfrog_steps < 1:
        parser.error("--hmc-leapfrog-steps must be at least 1")
    if args.tau_log_sd <= 0:
        parser.error("--tau-log-sd must be positive")
    if args.lambda_logit_sd <= 0:
        parser.error("--lambda-logit-sd must be positive")

    # Print an unambiguous header so the terminal run is easy to identify later.
    print("\n" + "=" * 72, flush=True)
    print("BAYESIAN CRM PARAMETER-LEVEL UNCERTAINTY QUANTIFICATION", flush=True)
    print("CRM-ML: EXCLUDED", flush=True)
    print("=" * 72, flush=True)

    # Stage 1: load and validate the complete field dataset.
    stage_start = stage(1, "Loading and validating data...")
    data = load_excel(args.data)
    finished(stage_start, "Data loading completed")

    # Determine how many rows belong to historical calibration and how many are held
    # as the requested forecast horizon.
    n_total = len(data.time)
    n_history = n_total - args.forecast
    if n_history < 2:
        raise ValueError("Forecast is too long; at least two historical rows are required.")

    # Stage 2: make the historical/forecast split explicit to the user.
    stage_start = stage(2, "Preparing historical + forecast timeline...")
    progress(f"Historical rows: {n_history}")
    progress(f"Forecast rows: {args.forecast}")
    progress(f"Forecast boundary: {data.time[n_history - 1]:g}")
    finished(stage_start, "Timeline prepared")

    # Create the historical subset used for deterministic calibration and Bayesian
    # likelihood evaluation while keeping the complete dataset for final propagation.
    history = CRMData(
        data.time[:n_history],
        data.production[:n_history],
        data.injection[:n_history],
        data.producer_names,
        data.injector_names,
    )

    # Stage 3: obtain a sensible deterministic CRM parameter vector.
    stage_start = stage(3, "Deterministic CRM calibration...")
    progress(f"Optimisation starts: {args.crm_starts}")
    progress(f"Maximum iterations: {args.crm_max_iter}")
    calibrator = Calibrator(
        args.crm_starts,
        args.crm_max_iter,
        args.seed,
    ).fit(
        history.time,
        history.injection,
        history.production,
    )
    progress(f"Best calibration objective: {calibrator.objective_:.8g}")
    finished(stage_start, "Deterministic CRM calibration completed")

    # Stage 4: construct the Bayesian posterior around the calibrated CRM.
    stage_start = stage(4, "Building Bayesian posterior...")
    tmin = float(np.min(np.diff(history.time)))
    log_posterior, scale, tau_reference = make_logpost(
        history.time,
        history.injection,
        history.production,
        tmin,
        args.tau_log_sd,
        args.lambda_logit_sd,
    )

    # The deterministic solution is transformed into the unconstrained HMC coordinate
    # system and becomes the starting point for each chain, with a small perturbation.
    if calibrator.params_ is None:
        raise RuntimeError("Deterministic calibration did not produce CRM parameters.")
    theta0 = par_to_theta(calibrator.params_)
    progress(f"HMC parameters: {theta0.size}")
    progress(f"Minimum tau: {tmin:g}; reference tau: {tau_reference:g}")
    progress(f"log(tau) prior SD: {args.tau_log_sd:g}")
    progress(f"lambda-logit prior SD: {args.lambda_logit_sd:g}")
    progress("Likelihood: Lognormal production model")
    progress("sigma^2: Inverse-Gamma prior, analytically marginalised")

    # Check the starting point before launching potentially long HMC runs.
    initial_logp = log_posterior(torch.tensor(theta0, dtype=torch.float64))
    if not torch.isfinite(initial_logp):
        raise RuntimeError("Initial Bayesian CRM state has invalid posterior density.")
    finished(stage_start, "Bayesian posterior constructed")

    # Stage 5: sample the posterior independently with the requested number of chains.
    stage_start = stage(5, "Running Hamiltonian Monte Carlo...")
    progress(f"Chains: {args.mcmc_chains}")
    progress(f"Retained samples per chain: {args.mcmc_samples}")
    progress(f"Burn-in: {args.mcmc_burnin}")
    progress(f"Thinning: {args.mcmc_thin}")

    chains = []
    acceptance_rates = []

    # Recover the dimensions needed to convert each posterior state back to CRM parameters.
    n_producers = history.production.shape[1]
    n_injectors = history.injection.shape[1]

    # These indices identify all log-time parameters so their randomized chain starts
    # cannot fall below the minimum permitted log(tau).
    time_parameter_indices = list(range(n_producers)) + list(
        range(n_producers + n_injectors * n_producers, theta0.size)
    )

    # Run each chain with a deterministic but distinct seed.
    for chain_number in range(args.mcmc_chains):
        progress(f"Starting chain {chain_number + 1}/{args.mcmc_chains}")
        rng = np.random.default_rng(args.seed + 10000 * chain_number)

        # Perturb the deterministic solution slightly so chains do not start at exactly
        # the same point while remaining close enough to avoid obviously invalid states.
        start_theta = theta0 + rng.normal(0.0, 0.10, theta0.size)
        start_theta[time_parameter_indices] = np.maximum(
            start_theta[time_parameter_indices],
            math.log(tmin) + 1e-6,
        )

        # Run HMC for this chain and retain the sampled states and acceptance rate.
        draws, acceptance_rate, final_step_size = hmc(
            log_posterior,
            start_theta,
            args.mcmc_samples,
            args.mcmc_burnin,
            args.mcmc_thin,
            args.hmc_step_size,
            args.hmc_leapfrog_steps,
            args.seed + 10000 * chain_number,
        )

        # Store each chain separately until all chains have the same shape.
        chains.append(draws)
        acceptance_rates.append(acceptance_rate)
        progress(
            f"Chain {chain_number + 1} completed: "
            f"acceptance={acceptance_rate:.2%}, final step={final_step_size:.6g}"
        )

    # Stack chains into [chain, draw, parameter] format for diagnostics.
    chains = np.stack(chains)
    finished(stage_start, "HMC sampling completed")

    # Stage 6: evaluate convergence diagnostics before using the posterior for UQ.
    stage_start = stage(6, "Checking MCMC diagnostics...")
    rhat_values = rhat(chains)
    ess_values = ess(chains)
    progress(f"Maximum R-hat: {np.nanmax(rhat_values):.4f}")
    progress(f"Minimum bulk ESS: {np.nanmin(ess_values):.1f}")

    # Report the acceptance rate of every chain so unusually low/high values are visible.
    for chain_number, acceptance_rate in enumerate(acceptance_rates, start=1):
        progress(f"Chain {chain_number} acceptance: {acceptance_rate:.2%}")

    # These are operational checks, not a claim that the posterior is automatically valid.
    if np.nanmax(rhat_values) > 1.10 or np.nanmin(ess_values) < 100:
        progress("WARNING: R-hat/ESS does not meet the operational threshold.")
    else:
        progress("✓ R-hat and ESS meet the operational thresholds.")
    finished(stage_start, "MCMC diagnostics calculated")

    # Stage 7: propagate every posterior parameter draw through the complete timeline.
    stage_start = stage(7, "Propagating posterior CRM parameters...")
    flat = chains.reshape(-1, chains.shape[-1])
    total_draws = len(flat)
    progress(f"Posterior draws: {total_draws}")

    # Show approximately ten progress updates without assuming a particular draw count.
    propagation_checkpoints = set(
        np.unique(
            np.linspace(1, total_draws, min(10, total_draws), dtype=int)
        ).tolist()
    )

    predictions = []

    # Each posterior state is converted into physical parameters and run through the
    # CRM from the observed initial production state over historical + forecast time.
    for draw_number, theta in enumerate(flat, start=1):
        parameters = theta_to_par(theta, n_injectors, n_producers)
        predictions.append(
            crm_simulate(
                parameters,
                data.time,
                data.injection,
                data.production[0],
            )
        )

        # Periodic progress reporting makes long posterior propagation transparent.
        if draw_number in propagation_checkpoints:
            progress(
                f"Propagation: {draw_number}/{total_draws} "
                f"({100 * draw_number / total_draws:.0f}%)"
            )

    # Convert all posterior CRM trajectories into one [draw,time,producer] array.
    predictions = np.stack(predictions)

    # Compute the requested production quantiles from the posterior ensemble.
    p10, p50, p90 = np.quantile(predictions, [0.10, 0.50, 0.90], axis=0)
    lower_95, upper_95 = np.quantile(predictions, [0.025, 0.975], axis=0)
    finished(stage_start, "Posterior propagation completed")

    # Stage 8: write machine-readable tables, diagnostics, settings, and a summary plot.
    stage_start = stage(8, "Generating and saving outputs...")
    output_directory = Path(args.out)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Build a long-form table containing one row per time point and producer.
    rows = []
    for k, current_time in enumerate(data.time[1:]):
        original_index = k + 1
        period = "historical" if original_index < n_history else "forecast"

        for producer_index, producer_name in enumerate(data.producer_names):
            rows.append(
                {
                    "time": float(current_time),
                    "period": period,
                    "producer": producer_name,
                    "observed": float(data.production[original_index, producer_index]),
                    "P10_low": float(p10[k, producer_index]),
                    "P50": float(p50[k, producer_index]),
                    "P90_high": float(p90[k, producer_index]),
                    "95pct_lower": float(lower_95[k, producer_index]),
                    "95pct_upper": float(upper_95[k, producer_index]),
                    "interval_width_95": float(
                        upper_95[k, producer_index] - lower_95[k, producer_index]
                    ),
                }
            )

    # Save the complete posterior production time series.
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(
        output_directory / "crm_parameter_uq_timeseries.csv",
        index=False,
    )

    # Save the raw HMC states so diagnostics or alternative posterior summaries can be
    # reproduced without rerunning the expensive sampler.
    np.savez_compressed(
        output_directory / "crm_parameter_hmc_samples.npz",
        chains=chains,
    )

    # Create names matching the exact order of parameters in the HMC state vector.
    diagnostic_names = (
        [f"{producer}::log_tau_j" for producer in history.producer_names]
        + [
            f"{injector}->{producer}::lambda_logit"
            for injector in history.injector_names
            for producer in history.producer_names
        ]
        + [
            f"{injector}->{producer}::log_tau_ij"
            for injector in history.injector_names
            for producer in history.producer_names
        ]
    )

    # Save convergence diagnostics in a table suitable for inspection or reporting.
    pd.DataFrame(
        {
            "parameter": diagnostic_names,
            "R_hat": rhat_values,
            "ESS_bulk": ess_values,
        }
    ).to_csv(
        output_directory / "mcmc_diagnostics.csv",
        index=False,
    )

    # Save one acceptance-rate record for every HMC chain.
    pd.DataFrame(
        {
            "chain": np.arange(1, len(acceptance_rates) + 1),
            "acceptance_rate": acceptance_rates,
        }
    ).to_csv(
        output_directory / "hmc_acceptance.csv",
        index=False,
    )

    # Convert every sampled state into physical tau and lambda values for a readable
    # posterior parameter summary.
    physical_parameters = []
    for theta in flat:
        parameters = theta_to_par(theta, n_injectors, n_producers)
        physical_parameters.append(
            np.r_[
                parameters.tau_j,
                parameters.lambda_ij.ravel(),
                parameters.tau_ij.ravel(),
            ]
        )
    physical_parameters = np.asarray(physical_parameters)

    # Give the physical parameters human-readable names in the exported summary.
    physical_names = (
        [f"{producer}::tau_j" for producer in data.producer_names]
        + [
            f"{injector}->{producer}::lambda"
            for injector in data.injector_names
            for producer in data.producer_names
        ]
        + [
            f"{injector}->{producer}::tau_ij"
            for injector in data.injector_names
            for producer in data.producer_names
        ]
    )

    # Save posterior mean, median, standard deviation, and central 95% intervals.
    pd.DataFrame(
        {
            "parameter": physical_names,
            "mean": physical_parameters.mean(axis=0),
            "median": np.median(physical_parameters, axis=0),
            "sd": physical_parameters.std(axis=0, ddof=1),
            "q2.5": np.quantile(physical_parameters, 0.025, axis=0),
            "q97.5": np.quantile(physical_parameters, 0.975, axis=0),
        }
    ).to_csv(
        output_directory / "crm_parameter_posterior_summary.csv",
        index=False,
    )

    # Save the exact settings used for this run to make the experiment reproducible.
    pd.DataFrame(
        {
            "setting": [
                "forecast",
                "chains",
                "samples",
                "burnin",
                "thin",
                "step_size",
                "leapfrog",
                "tau_log_sd",
                "lambda_logit_sd",
                "tau_reference",
                "crm_starts",
                "crm_max_iter",
                "seed",
            ],
            "value": [
                args.forecast,
                args.mcmc_chains,
                args.mcmc_samples,
                args.mcmc_burnin,
                args.mcmc_thin,
                args.hmc_step_size,
                args.hmc_leapfrog_steps,
                args.tau_log_sd,
                args.lambda_logit_sd,
                tau_reference,
                args.crm_starts,
                args.crm_max_iter,
                args.seed,
            ],
        }
    ).to_csv(output_directory / "run_settings.csv", index=False)

    # Create one combined plot showing observed production, posterior median, and the
    # 95% parameter-induced interval for each producer.
    figure, axis = plt.subplots(figsize=(11, 6))
    plot_data = pd.DataFrame(rows)

    for producer_name in data.producer_names:
        producer_data = plot_data[plot_data.producer == producer_name]
        axis.plot(
            producer_data.time,
            producer_data.observed,
            label=f"{producer_name} observed",
            linewidth=0.8,
        )
        axis.plot(
            producer_data.time,
            producer_data.P50,
            label=f"{producer_name} P50",
            linewidth=1.4,
        )
        axis.fill_between(
            producer_data.time.to_numpy(),
            producer_data["95pct_lower"].to_numpy(),
            producer_data["95pct_upper"].to_numpy(),
            alpha=0.15,
        )

    # Mark the point at which the requested forecast portion begins.
    axis.axvline(
        float(data.time[n_history - 1]),
        linestyle="--",
        label="forecast start",
    )

    # Add labels and a compact legend so the figure can be used in a report.
    axis.set_xlabel("Time")
    axis.set_ylabel("Production rate")
    axis.set_title("Bayesian CRM parameter-level uncertainty")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(
        output_directory / "bayesian_crm_parameter_uq.png",
        dpi=200,
    )
    plt.close(figure)

    # Confirm that output generation has completed before printing the final summary.
    finished(stage_start, "Outputs generated")

    # Print the final run location and the most important MCMC diagnostics.
    print("\n" + "=" * 72, flush=True)
    print("RUN COMPLETE", flush=True)
    print("=" * 72, flush=True)
    print(f"Results: {output_directory.resolve()}", flush=True)
    print(f"Maximum R-hat: {np.nanmax(rhat_values):.4f}", flush=True)
    print(f"Minimum bulk ESS: {np.nanmin(ess_values):.1f}", flush=True)
    print(
        "CRM-ML is excluded. Intervals are induced by CRM parameter posterior samples.",
        flush=True,
    )

    # Do not present poor-convergence intervals as final validated UQ results.
    if np.nanmax(rhat_values) > 1.10 or np.nanmin(ess_values) < 100:
        print(
            "WARNING: Do not treat UQ as final until MCMC convergence is improved.",
            flush=True,
        )


# Execute the pipeline only when this file is run directly from the command line.
if __name__ == "__main__":
    main()
