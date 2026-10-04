"""CRM-ML hybrid production forecasting.

Implementation of the methodology in:
    Ogali, O.I.O. and Orodu, O.D. (2025). "Concatenating data-driven and
    reduced-physics models for smart production forecasting."
    Earth Science Informatics 18:246. https://doi.org/10.1007/s12145-025-01745-9

Overview of the workflow (paper Figs. 2-5):
    1. Calibrate a Capacitance-Resistance Model (CRM, Eq. 1) on the "historical"
       injection/production rates by constrained non-linear optimisation of the
       objective function in Eq. 2.
    2. Use the calibrated CRM parameters (tau_j, lambda_ij, tau_ij), the
       injector-producer distances X_ij, time t and the injection rates w_i(t)
       as input features (4I + 2 inputs) for an ML regressor whose target is the
       observed production rate of the producer.  This is the CRM-ML hybrid.
    3. Train stand-alone ML models on the injection rates only (I inputs).
    4. Compare CRM, 4 CRM-ML hybrids and 4 ML models using MAE (Eq. 3),
       RMSE (Eq. 4) and R^2 (Eq. 5) over several evaluations, then rank them.

Required packages:
    numpy, scipy, pandas, scikit-learn      (pip install numpy scipy pandas scikit-learn)
    xgboost (optional)                      (pip install xgboost)
        If xgboost is missing, scikit-learn's GradientBoostingRegressor is used
        as a stand-in for XGB and a warning is emitted.
    TensorFlow is NOT required: the MLP uses scikit-learn and the ELM is
    implemented directly in NumPy.

Notes / assumptions where the paper is silent:
    * The paper's CRM excludes the BHP term for the synthetic field (constant BHP);
      the BHP term is supported here via the optional ``bhp`` argument.
    * Hidden-layer sizes: MLP = 10 neurons (paper); ELM = 10 neurons (assumed).
    * NuSVR / XGB hyper-parameters are reasonable defaults, not paper values.
    * Deterministic models (NuSVM, XGB) use all historical data; stochastic
      models (ELM, MLP) use a random 75:25 train/validation split per evaluation,
      as described in the paper.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

try:  # optional dependency
    from xgboost import XGBRegressor

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor

    _HAS_XGBOOST = False
    warnings.warn("xgboost not installed; using GradientBoostingRegressor as stand-in for XGB.")

FloatArray = np.ndarray

# --------------------------------------------------------------------------- #
# 1. Error metrics (paper Eqs. 3-5)
# --------------------------------------------------------------------------- #


def mean_absolute_error(q_obs: FloatArray, q_est: FloatArray) -> float:
    """Mean absolute error, Eq. 3: MAE = (1/N) * sum |q_est - q_obs|.

    Args:
        q_obs: Observed production rates, shape (N,).
        q_est: Estimated production rates, shape (N,).

    Returns:
        The MAE in the units of the rates.
    """
    return float(np.mean(np.abs(q_est - q_obs)))


def root_mean_squared_error(q_obs: FloatArray, q_est: FloatArray) -> float:
    """Root-mean-square error, Eq. 4: RMSE = sqrt((1/N) * sum (q_est - q_obs)^2).

    Args:
        q_obs: Observed production rates, shape (N,).
        q_est: Estimated production rates, shape (N,).

    Returns:
        The RMSE in the units of the rates.
    """
    return float(np.sqrt(np.mean((q_est - q_obs) ** 2)))


def r_squared(q_obs: FloatArray, q_est: FloatArray) -> float:
    """Coefficient of determination as defined in Eq. 5 (squared correlation).

    R^2 = [ sum (q - q_bar)(q_hat - q_hat_bar) ]^2
          / ( sum (q - q_bar)^2 * sum (q_hat - q_hat_bar)^2 )

    Args:
        q_obs: Observed production rates, shape (N,).
        q_est: Estimated production rates, shape (N,).

    Returns:
        R^2 in [0, 1], or NaN if either series has zero variance.
    """
    d_obs = q_obs - q_obs.mean()
    d_est = q_est - q_est.mean()
    denom = np.sum(d_obs**2) * np.sum(d_est**2)
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(d_obs * d_est) ** 2 / denom)


# --------------------------------------------------------------------------- #
# 2. Capacitance-Resistance Model (paper Eq. 1, calibration Eq. 2)
# --------------------------------------------------------------------------- #


@dataclass
class CRMParameters:
    """Container for CRM parameters.

    Attributes:
        tau_j: Producer time constants, shape (K,).
        lambda_ij: Injector-producer connectivity indices, shape (I, K).
        tau_ij: Injector-producer time constants, shape (I, K).
        v_kj: Producer-producer connectivity coefficients (BHP term), shape (K, K),
            or None when the BHP term is not used.
        tau_kj: Producer-producer time constants (BHP term), shape (K, K), or None.
    """

    tau_j: FloatArray
    lambda_ij: FloatArray
    tau_ij: FloatArray
    v_kj: Optional[FloatArray] = None
    tau_kj: Optional[FloatArray] = None


def _exp_filter(signal: FloatArray, dt: FloatArray, tau: float) -> FloatArray:
    """Evaluate the CRM convolution term with a first-order recursive filter.

    Computes, for n = 1..M,
        S_n = sum_{m=1..n} [exp((t_m - t_n)/tau) - exp((t_{m-1} - t_n)/tau)] * s_m
    which satisfies the recursion  S_n = a_n * S_{n-1} + (1 - a_n) * s_n  with
    a_n = exp(-dt_n / tau).  This is O(M) rather than the O(M^2) of the explicit
    double sum, and is vectorised through ``scipy.signal.lfilter`` when the time
    step is uniform.

    Args:
        signal: Signal s_m (e.g. injection rate) for m = 1..M, shape (M,).
        dt: Time steps t_m - t_{m-1} for m = 1..M, shape (M,).
        tau: Time constant (same units as ``dt``), must be > 0.

    Returns:
        The filtered signal S_n, shape (M,).
    """
    if np.allclose(dt, dt[0]):
        alpha = np.exp(-dt[0] / tau)
        return lfilter([1.0 - alpha], [1.0, -alpha], signal)
    # Non-uniform sampling: explicit recursion (still O(M)).
    alpha_n = np.exp(-dt / tau)
    out = np.empty_like(signal, dtype=float)
    prev = 0.0
    for n in range(signal.shape[0]):
        prev = alpha_n[n] * prev + (1.0 - alpha_n[n]) * signal[n]
        out[n] = prev
    return out


def crm_simulate(
    params: CRMParameters,
    time: FloatArray,
    injection: FloatArray,
    q0: FloatArray,
    bhp: Optional[FloatArray] = None,
) -> FloatArray:
    """Evaluate the CRM, Eq. 1: q_hat = Production + Injection + BHP terms.

    Production term : q_j(t0) * exp(-(t_n - t0)/tau_j)
    Injection term  : sum_i lambda_ij * sum_m [e^{(t_m-t_n)/tau_ij}
                      - e^{(t_{m-1}-t_n)/tau_ij}] * w_i(t_m)
    BHP term        : sum_k v_kj * [ Pwf_j(t0) e^{-(t_n-t0)/tau_kj} - Pwf_k(t_n)
                      + sum_m (e^{(t_m-t_n)/tau_kj} - e^{(t_{m-1}-t_n)/tau_kj}) Pwf_k(t_m) ]

    Args:
        params: Calibrated or trial CRM parameters.
        time: Time stamps t_0..t_{N-1}, shape (N,).
        injection: Injection rates w_i(t), shape (N, I).
        q0: Initial production rate q_j(t0) of each producer, shape (K,).
        bhp: Producer bottom-hole pressures Pwf_k(t), shape (N, K), or None to
            drop the BHP term (constant-BHP assumption).

    Returns:
        Estimated production rates q_hat_j(t_n) for n = 1..N-1, shape (N-1, K).
    """
    t_rel = time[1:] - time[0]  # (t_n - t_0), n = 1..N-1
    dt = np.diff(time)  # t_n - t_{n-1}
    n_inj, n_prod = params.lambda_ij.shape

    # Production (natural depletion) term: (N-1, K)
    q_hat = q0[None, :] * np.exp(-t_rel[:, None] / params.tau_j[None, :])

    # Injection term
    for i in range(n_inj):
        w_i = injection[1:, i]
        for j in range(n_prod):
            q_hat[:, j] += params.lambda_ij[i, j] * _exp_filter(w_i, dt, params.tau_ij[i, j])

    # Optional BHP term
    if bhp is not None and params.v_kj is not None and params.tau_kj is not None:
        for k in range(n_prod):
            p_k = bhp[1:, k]
            for j in range(n_prod):
                tau = params.tau_kj[k, j]
                q_hat[:, j] += params.v_kj[k, j] * (
                    bhp[0, j] * np.exp(-t_rel / tau) - p_k + _exp_filter(p_k, dt, tau)
                )
    return q_hat


class CapacitanceResistanceModel:
    """Capacitance-Resistance Model calibrated by constrained optimisation.

    Calibration minimises Eq. 2,
        ObjFcn = 1/(N*K) * sum_j sum_n (q_j(n) - q_hat_j(n))^2
    subject to
        0 <= lambda_ij <= 1,
        sum_j lambda_ij <= 1 for every injector i,
        tau >= tau_min (the minimum is the sampling interval, per the paper).
    All producers are optimised concurrently.  Time constants are optimised in
    log-space and rates are internally normalised for numerical conditioning;
    both are transparent to the caller.

    Attributes:
        params_: Fitted :class:`CRMParameters` (after ``fit``).
        objective_: Final value of the (normalised) objective function.
    """

    def __init__(
        self,
        tau_min: Optional[float] = None,
        tau_max: Optional[float] = None,
        n_starts: int = 3,
        max_iter: int = 300,
        random_state: Optional[int] = 0,
    ) -> None:
        """Initialises the model.

        Args:
            tau_min: Lower bound for all time constants. Defaults to the smallest
                sampling interval of the data.
            tau_max: Upper bound for all time constants. Defaults to five times
                the length of the calibration window.
            n_starts: Number of optimisation starts (first is deterministic).
            max_iter: Maximum SLSQP iterations per start.
            random_state: Seed for random restarts.
        """
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_starts = n_starts
        self.max_iter = max_iter
        self.random_state = random_state
        self.params_: Optional[CRMParameters] = None
        self.objective_: float = float("nan")
        self._flow_scale: float = 1.0
        self._bhp_scale: float = 1.0
        self._use_bhp: bool = False

    # ---- parameter vector helpers ----------------------------------------- #
    def _unpack(self, x: FloatArray, n_inj: int, n_prod: int) -> CRMParameters:
        """Converts the optimiser vector into :class:`CRMParameters`."""
        k, i = n_prod, n_inj
        pos = 0
        tau_j = np.exp(x[pos : pos + k])
        pos += k
        lam = x[pos : pos + i * k].reshape(i, k)
        pos += i * k
        tau_ij = np.exp(x[pos : pos + i * k].reshape(i, k))
        pos += i * k
        v_kj = tau_kj = None
        if self._use_bhp:
            v_kj = x[pos : pos + k * k].reshape(k, k)
            pos += k * k
            tau_kj = np.exp(x[pos : pos + k * k].reshape(k, k))
        return CRMParameters(tau_j, lam, tau_ij, v_kj, tau_kj)

    # ---- public API -------------------------------------------------------- #
    def fit(
        self,
        time: FloatArray,
        injection: FloatArray,
        production: FloatArray,
        bhp: Optional[FloatArray] = None,
    ) -> "CapacitanceResistanceModel":
        """Calibrates the CRM on historical data.

        Args:
            time: Time stamps, shape (N,), strictly increasing.
            injection: Injection rates w_i(t), shape (N, I).
            production: Observed liquid production rates q_j(t), shape (N, K).
            bhp: Optional producer BHPs, shape (N, K). Adds the v_kj / tau_kj
                parameters (paper notes this greatly increases cost).

        Returns:
            self.
        """
        time = np.asarray(time, dtype=float)
        injection = np.asarray(injection, dtype=float)
        production = np.asarray(production, dtype=float)
        n_steps, n_inj = injection.shape
        n_prod = production.shape[1]
        if time.shape[0] != n_steps or production.shape[0] != n_steps:
            raise ValueError("time, injection and production must have equal length.")

        self._use_bhp = bhp is not None
        self._flow_scale = max(float(np.mean(np.abs(production))), 1e-12)
        q_s = production / self._flow_scale
        w_s = injection / self._flow_scale
        bhp_s = None
        if bhp is not None:
            self._bhp_scale = max(float(np.mean(np.abs(bhp))), 1e-12)
            bhp_s = np.asarray(bhp, dtype=float) / self._bhp_scale

        tau_min = self.tau_min if self.tau_min is not None else float(np.min(np.diff(time)))
        tau_max = self.tau_max if self.tau_max is not None else 5.0 * float(time[-1] - time[0])
        log_lo, log_hi = np.log(tau_min), np.log(tau_max)

        # Bounds and constraint matrix --------------------------------------
        bounds: List[Tuple[Optional[float], Optional[float]]] = []
        bounds += [(log_lo, log_hi)] * n_prod  # log tau_j
        bounds += [(0.0, 1.0)] * (n_inj * n_prod)  # lambda_ij
        bounds += [(log_lo, log_hi)] * (n_inj * n_prod)  # log tau_ij
        if self._use_bhp:
            bounds += [(None, None)] * (n_prod * n_prod)  # v_kj
            bounds += [(log_lo, log_hi)] * (n_prod * n_prod)  # log tau_kj
        n_par = len(bounds)

        lam_offset = n_prod
        A = np.zeros((n_inj, n_par))  # sum_j lambda_ij <= 1
        for i in range(n_inj):
            A[i, lam_offset + i * n_prod : lam_offset + (i + 1) * n_prod] = 1.0
        constraints = [{"type": "ineq", "fun": lambda x: 1.0 - A @ x, "jac": lambda x: -A}]

        q0 = q_s[0]

        def objective(x: FloatArray) -> float:
            p = self._unpack(x, n_inj, n_prod)
            q_hat = crm_simulate(p, time, w_s, q0, bhp_s)
            return float(np.mean((q_s[1:] - q_hat) ** 2))  # Eq. 2

        rng = np.random.default_rng(self.random_state)
        best_x, best_f = None, np.inf
        for start in range(max(1, self.n_starts)):
            x0 = np.empty(n_par)
            x0[:n_prod] = np.log(10.0 * tau_min) if start == 0 else rng.uniform(log_lo, log_hi, n_prod)
            lam0 = (0.8 / n_prod) * np.ones((n_inj, n_prod)) if start == 0 else \
                rng.dirichlet(np.ones(n_prod), size=n_inj) * 0.9
            x0[lam_offset : lam_offset + n_inj * n_prod] = lam0.ravel()
            tau_ij_slice = slice(lam_offset + n_inj * n_prod, lam_offset + 2 * n_inj * n_prod)
            x0[tau_ij_slice] = np.log(10.0 * tau_min) if start == 0 else \
                rng.uniform(log_lo, log_hi, n_inj * n_prod)
            if self._use_bhp:
                x0[tau_ij_slice.stop : tau_ij_slice.stop + n_prod * n_prod] = 0.0
                x0[tau_ij_slice.stop + n_prod * n_prod :] = np.log(10.0 * tau_min)
            res = minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": self.max_iter, "ftol": 1e-12},
            )
            if res.fun < best_f:
                best_x, best_f = res.x, float(res.fun)

        assert best_x is not None
        self.params_ = self._unpack(best_x, n_inj, n_prod)
        self.objective_ = best_f
        return self

    def predict(
        self,
        time: FloatArray,
        injection: FloatArray,
        q0: FloatArray,
        bhp: Optional[FloatArray] = None,
    ) -> FloatArray:
        """Estimates / forecasts production rates with the calibrated CRM.

        The forecast uses the full injection record (history + forecast period)
        anchored at ``t0`` and ``q0``, exactly as Eq. 1 is written.

        Args:
            time: Time stamps t_0..t_{N-1}, shape (N,).
            injection: Injection rates, shape (N, I).
            q0: Initial production rates q_j(t0), shape (K,).
            bhp: Optional BHPs, shape (N, K) (only if fitted with BHP).

        Returns:
            Estimated rates for n = 1..N-1 (physical units), shape (N-1, K).
        """
        if self.params_ is None:
            raise RuntimeError("Call fit() before predict().")
        bhp_s = None if (bhp is None or not self._use_bhp) else bhp / self._bhp_scale
        q_hat = crm_simulate(
            self.params_,
            np.asarray(time, float),
            np.asarray(injection, float) / self._flow_scale,
            np.asarray(q0, float) / self._flow_scale,
            bhp_s,
        )
        return q_hat * self._flow_scale


# --------------------------------------------------------------------------- #
# 3. Machine-learning models
# --------------------------------------------------------------------------- #

STOCHASTIC_MODELS = ("ELM", "MLP")  # results change with each evaluation
ML_MODELS = ("NuSVM", "XGB", "ELM", "MLP")


class ExtremeLearningMachine(BaseEstimator, RegressorMixin):
    """Single-hidden-layer Extreme Learning Machine (Liang et al., 2006).

    Hidden-layer weights and biases are drawn at random from U(-1, 1) and are
    never trained.  Output weights are the least-squares solution obtained with
    the Moore-Penrose pseudo-inverse:  beta = pinv(H) @ y,  H = sigmoid(XW + b).
    A tiny ridge term (``ridge``) is added by default,
    beta = (H'H + ridge*I)^-1 H'y, which tends to the pseudo-inverse solution as
    ridge -> 0 but prevents exploding weights on small/ill-conditioned data
    (observed on short real-field records). Set ``ridge=0`` for the pure version.

    Attributes:
        n_hidden: Number of hidden neurons.
        random_state: Seed for the random hidden layer.
        ridge: Ridge regularisation strength (0 = pure pseudo-inverse).
    """

    def __init__(self, n_hidden: int = 10, random_state: Optional[int] = None,
                 ridge: float = 1e-3) -> None:
        """Initialises the ELM.

        Args:
            n_hidden: Number of hidden neurons.
            random_state: Seed for reproducibility.
            ridge: Ridge regularisation strength (0 = pure pseudo-inverse).
        """
        self.n_hidden = n_hidden
        self.random_state = random_state
        self.ridge = ridge

    @staticmethod
    def _sigmoid(z: FloatArray) -> FloatArray:
        """Numerically stable logistic activation."""
        return 0.5 * (1.0 + np.tanh(0.5 * z))

    def fit(self, X: FloatArray, y: FloatArray) -> "ExtremeLearningMachine":
        """Fits the output weights.

        Args:
            X: Features, shape (n_samples, n_features).
            y: Targets, shape (n_samples,) or (n_samples, 1).

        Returns:
            self.
        """
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(len(X), -1)
        self.weights_ = rng.uniform(-1.0, 1.0, (X.shape[1], self.n_hidden))
        self.bias_ = rng.uniform(-1.0, 1.0, self.n_hidden)
        hidden = self._sigmoid(X @ self.weights_ + self.bias_)
        if self.ridge > 0:
            gram = hidden.T @ hidden + self.ridge * np.eye(self.n_hidden)
            self.beta_ = np.linalg.solve(gram, hidden.T @ y)
        else:
            self.beta_ = np.linalg.pinv(hidden) @ y
        return self

    def predict(self, X: FloatArray) -> FloatArray:
        """Predicts targets.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            Predictions, shape (n_samples,) (or (n_samples, n_targets) if >1).
        """
        out = self._sigmoid(np.asarray(X, float) @ self.weights_ + self.bias_) @ self.beta_
        return out.ravel() if out.shape[1] == 1 else out


def make_regressor(name: str, random_state: Optional[int] = 0, n_hidden: int = 10):
    """Builds one of the four ML regressors used in the paper.

    Inputs and targets are standardised internally so callers can pass raw data.

    Args:
        name: One of ``"NuSVM"``, ``"XGB"``, ``"ELM"``, ``"MLP"``.
        random_state: Seed (affects ELM, MLP, XGB).
        n_hidden: Hidden neurons for ELM / MLP (paper uses 10 for MLP).

    Returns:
        An unfitted scikit-learn compatible regressor.
    """
    if name == "NuSVM":
        base = NuSVR(kernel="rbf", nu=0.5, C=10.0, gamma="scale")
    elif name == "XGB":
        if _HAS_XGBOOST:
            base = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                random_state=random_state, n_jobs=1, verbosity=0)
        else:
            base = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                             random_state=random_state)
    elif name == "ELM":
        base = ExtremeLearningMachine(n_hidden=n_hidden, random_state=random_state)
    elif name == "MLP":
        base = MLPRegressor(hidden_layer_sizes=(n_hidden,), activation="tanh", solver="adam",
                            max_iter=2000, random_state=random_state)
    else:
        raise ValueError(f"Unknown ML model '{name}'. Choose from {ML_MODELS}.")
    pipe = Pipeline([("scale", StandardScaler()), ("model", base)])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


# --------------------------------------------------------------------------- #
# 4. Feature engineering for the hybrid and the stand-alone ML models
# --------------------------------------------------------------------------- #


def build_crm_ml_features(
    time: FloatArray,
    injection: FloatArray,
    params: CRMParameters,
    distances: FloatArray,
    producer: int,
) -> FloatArray:
    """Builds the 4I + 2 CRM-ML input features for one producer.

    Columns: [ t | X_ij (I) | lambda_ij (I) | tau_ij (I) | w_i(t) (I) | tau_j ].
    The CRM-derived columns are constant in time and broadcast down the rows.

    Args:
        time: Time stamps for the rows to featurise, shape (M,).
        injection: Injection rates aligned with ``time``, shape (M, I).
        params: Calibrated CRM parameters.
        distances: Injector-producer distances X_ij, shape (I, K).
        producer: Producer index j.

    Returns:
        Feature matrix of shape (M, 4I + 2).
    """
    m = time.shape[0]
    const = np.concatenate([
        distances[:, producer],
        params.lambda_ij[:, producer],
        params.tau_ij[:, producer],
    ])
    return np.column_stack([
        time,
        np.tile(const, (m, 1)),
        injection,
        np.full(m, params.tau_j[producer]),
    ])


def feature_names(n_injectors: int) -> List[str]:
    """Returns the names of the 4I + 2 CRM-ML features, in column order."""
    idx = range(1, n_injectors + 1)
    return (["t"] + [f"X_{i}j" for i in idx] + [f"lambda_{i}j" for i in idx]
            + [f"tau_{i}j" for i in idx] + [f"w_{i}" for i in idx] + ["tau_j"])


def _fit_predict_ml(
    name: str,
    features: FloatArray,
    target: FloatArray,
    n_train_rows: int,
    seed: int,
) -> Tuple[FloatArray, float]:
    """Trains an ML model on the historical rows and predicts every row.

    Deterministic models (NuSVM, XGB) train on all historical rows.  Stochastic
    models (ELM, MLP) train on a random 75 % of the historical rows and the
    remaining 25 % act as validation points, as in the paper.

    Args:
        name: Model name (see :data:`ML_MODELS`).
        features: Feature matrix for all rows, shape (M, F).
        target: Observed production rates, shape (M,).
        n_train_rows: Number of leading rows that form the "historical" data.
        seed: Random seed controlling the split and the model initialisation.

    Returns:
        Tuple ``(predictions for all M rows, validation MAE or NaN)``.
    """
    hist = np.arange(n_train_rows)
    val_mae = float("nan")
    if name in STOCHASTIC_MODELS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(hist)
        n_tr = int(round(0.75 * n_train_rows))
        train_idx, val_idx = perm[:n_tr], perm[n_tr:]
    else:
        train_idx, val_idx = hist, np.array([], dtype=int)

    model = make_regressor(name, random_state=seed)
    model.fit(features[train_idx], target[train_idx])
    if val_idx.size:
        val_mae = mean_absolute_error(target[val_idx], model.predict(features[val_idx]))
    return np.asarray(model.predict(features), float), val_mae


# --------------------------------------------------------------------------- #
# 5. Study driver: CRM vs CRM-ML vs ML, repeated evaluations and ranking
# --------------------------------------------------------------------------- #


def run_forecast_study(
    time: FloatArray,
    injection: FloatArray,
    production: FloatArray,
    distances: FloatArray,
    n_forecast: int,
    n_evaluations: int = 20,
    bhp: Optional[FloatArray] = None,
    crm_kwargs: Optional[Dict] = None,
    models: Sequence[str] = ML_MODELS,
    verbose: bool = True,
    active_start: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, CapacitanceResistanceModel, Dict[str, FloatArray]]:
    """Runs the full CRM / CRM-ML / ML comparison from the paper.

    Procedure:
        1. Calibrate the CRM on all but the last ``n_forecast`` samples.
        2. Forecast the whole record with the CRM.
        3. For each producer, ML model and evaluation, train a CRM-ML hybrid
           (4I+2 inputs) and a stand-alone ML model (I inputs: injection rates).
        4. Score every approach with MAE, RMSE and R^2 on the "entire" record
           (rows 1..N-1) and on the "forecast" period.

    Args:
        time: Time stamps, shape (N,).
        injection: Injection rates, shape (N, I).
        production: Observed production rates, shape (N, K).
        distances: Injector-producer distances, shape (I, K).
        n_forecast: Number of trailing samples reserved for forecasting.
        n_evaluations: Number of repeated evaluations (paper uses 20). Deterministic
            models are run once and replicated.
        bhp: Optional producer BHPs, shape (N, K).
        crm_kwargs: Extra keyword arguments for :class:`CapacitanceResistanceModel`.
        models: ML model names to combine with the CRM.
        verbose: Print progress.
        active_start: Optional per-producer index (into the original N-row record)
            of the first sample to train/score on, e.g. the first non-zero rate of
            a well drilled late. ``None`` uses every sample for every producer.

    Returns:
        Tuple ``(results, crm, predictions)`` where ``results`` is a long-format
        DataFrame with columns [evaluation, producer, approach, case, MAE, RMSE,
        R2, val_MAE]; ``crm`` is the calibrated model; ``predictions`` maps
        ``"CRM"``, ``"CRM-<ML>"`` and ``"<ML>"`` to the last evaluation's
        predictions of shape (N-1, K).
    """
    time = np.asarray(time, float)
    injection = np.asarray(injection, float)
    production = np.asarray(production, float)
    n_total, n_inj = injection.shape
    n_prod = production.shape[1]
    n_hist = n_total - n_forecast  # rows 0..n_hist-1 are "historical"
    if n_hist < 10:
        raise ValueError("Not enough historical data.")

    # --- Step 1-2: CRM -------------------------------------------------------
    crm = CapacitanceResistanceModel(**(crm_kwargs or {}))
    crm.fit(time[:n_hist], injection[:n_hist], production[:n_hist],
            None if bhp is None else bhp[:n_hist])
    q_crm = crm.predict(time, injection, production[0], bhp)  # rows 1..N-1
    if verbose:
        print(f"CRM calibrated (objective={crm.objective_:.3e}).")

    q_obs = production[1:]  # rows 1..N-1, aligned with q_crm
    t_rows, w_rows = time[1:], injection[1:]
    n_hist_rows = n_hist - 1  # historical rows among rows 1..N-1
    cases = {"entire": 0, "forecast": n_hist_rows}
    start_rows = (np.zeros(n_prod, dtype=int) if active_start is None
                  else np.maximum(np.asarray(active_start, dtype=int) - 1, 0))

    records: List[Dict] = []
    last_preds: Dict[str, FloatArray] = {"CRM": q_crm.copy()}

    def score(approach: str, evaluation: int, j: int, q_hat: FloatArray, val: float) -> None:
        for case, first in cases.items():
            first = max(first, int(start_rows[j]))
            if q_obs.shape[0] - first < 2:
                continue
            o, e = q_obs[first:, j], q_hat[first:]
            records.append(dict(evaluation=evaluation, producer=j, approach=approach, case=case,
                                MAE=mean_absolute_error(o, e), RMSE=root_mean_squared_error(o, e),
                                R2=r_squared(o, e), val_MAE=val))

    for e in range(n_evaluations):
        for j in range(n_prod):
            score("CRM", e, j, q_crm[:, j], float("nan"))

    # --- Step 3: hybrids and ML ----------------------------------------------
    for name in models:
        n_runs = n_evaluations if name in STOCHASTIC_MODELS else 1
        hybrid_pred = np.zeros_like(q_obs)
        ml_pred = np.zeros_like(q_obs)
        for e in range(n_runs):
            for j in range(n_prod):
                seed = 1000 * e + j
                s0 = int(start_rows[j])
                if n_hist_rows - s0 < 8:
                    if verbose and e == 0:
                        print(f"  skipping {name} for producer {j}: <8 active historical samples")
                    continue
                x_hyb = build_crm_ml_features(t_rows, w_rows, crm.params_, distances, j)[s0:]
                p_h, v_h = _fit_predict_ml(name, x_hyb, q_obs[s0:, j], n_hist_rows - s0, seed)
                p_m, v_m = _fit_predict_ml(name, w_rows[s0:], q_obs[s0:, j], n_hist_rows - s0, seed)
                p_h = np.concatenate([np.full(s0, np.nan), p_h])
                p_m = np.concatenate([np.full(s0, np.nan), p_m])
                hybrid_pred[:, j], ml_pred[:, j] = p_h, p_m
                # deterministic models: replicate the single run across evaluations
                for ee in (range(n_evaluations) if n_runs == 1 else [e]):
                    score(f"CRM-{name}", ee, j, p_h, v_h)
                    score(name, ee, j, p_m, v_m)
            if verbose:
                print(f"  {name}: evaluation {e + 1}/{n_runs} done")
        last_preds[f"CRM-{name}"], last_preds[name] = hybrid_pred.copy(), ml_pred.copy()

    results = pd.DataFrame.from_records(records)
    return results, crm, last_preds


def rank_approaches(results: pd.DataFrame, case: str = "forecast", metric: str = "MAE") -> pd.DataFrame:
    """Ranks approaches per evaluation, per producer (as in the paper's Fig. 23).

    For every (evaluation, producer) the approaches are ranked by ``metric``
    (1 = best). Ranks are *not* derived from averaged errors, to avoid a single
    bad evaluation skewing the outcome.

    Args:
        results: DataFrame from :func:`run_forecast_study`.
        case: ``"entire"`` or ``"forecast"``.
        metric: Column used for ranking (lower is better), e.g. ``"MAE"``.

    Returns:
        DataFrame indexed by approach with columns ``mean_rank``, ``pct_first``
        (share of rankings where the approach was best) and ``mean_<metric>``,
        sorted best to worst by mean rank.
    """
    sub = results[results["case"] == case].copy()
    sub["rank"] = sub.groupby(["evaluation", "producer"])[metric].rank(method="min")
    summary = sub.groupby("approach").agg(
        mean_rank=("rank", "mean"),
        pct_first=("rank", lambda r: 100.0 * float(np.mean(r == 1.0))),
        **{f"mean_{metric}": (metric, "mean")},
    )
    return summary.sort_values("mean_rank")


def run_precomputed_crm_study(
    time: FloatArray,
    production: FloatArray,
    q_crm: FloatArray,
    n_forecast: int,
    n_evaluations: int = 20,
    active_start: Optional[Sequence[int]] = None,
    models: Sequence[str] = ML_MODELS,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, FloatArray]]:
    """CRM / CRM-ML / ML comparison when CRM output is supplied, not calibrated.

    Use this when only observed rates and a pre-computed CRM series are available
    (no injection data), e.g. ``CRM_Volve3_Dataset.xlsx``.  This is an ADAPTATION
    of the paper: the CRM-ML hybrid cannot use lambda_ij / tau_ij, so it receives
    the CRM-estimated rate as its physics feature,
        CRM-ML inputs : [t, q_CRM_j(t)]
        ML inputs     : [t]
    The stand-alone ML baseline is therefore much weaker than in the paper (which
    gives it the injection rates), so gains of CRM-ML over ML are overstated.

    Args:
        time: Time stamps, shape (N,).
        production: Observed rates, shape (N, K).
        q_crm: Pre-computed CRM rates aligned with ``production``, shape (N, K).
        n_forecast: Trailing samples reserved as the forecast period.
        n_evaluations: Repeated evaluations for the stochastic models (ELM, MLP).
        active_start: Optional per-producer first sample index to train/score on.
        models: ML model names.
        verbose: Print progress.

    Returns:
        Tuple ``(results, predictions)``; ``results`` has the same columns as
        :func:`run_forecast_study`, and ``predictions`` maps approach names to
        arrays of shape (N, K) (NaN before a producer's active start).
    """
    time = np.asarray(time, float)
    production = np.asarray(production, float)
    q_crm = np.asarray(q_crm, float)
    n_total, n_prod = production.shape
    n_hist = n_total - n_forecast
    starts = np.zeros(n_prod, int) if active_start is None else np.asarray(active_start, int)
    cases = {"entire": 0, "forecast": n_hist}

    records: List[Dict] = []

    def score(approach: str, evaluation: int, j: int, q_hat: FloatArray, val: float) -> None:
        for case, first in cases.items():
            first = max(first, int(starts[j]))
            if n_total - first < 2:
                continue
            o, e = production[first:, j], q_hat[first:]
            records.append(dict(evaluation=evaluation, producer=j, approach=approach, case=case,
                                MAE=mean_absolute_error(o, e), RMSE=root_mean_squared_error(o, e),
                                R2=r_squared(o, e), val_MAE=val))

    preds: Dict[str, FloatArray] = {"CRM": q_crm.copy()}
    for e in range(n_evaluations):
        for j in range(n_prod):
            score("CRM", e, j, q_crm[:, j], float("nan"))

    for name in models:
        n_runs = n_evaluations if name in STOCHASTIC_MODELS else 1
        hyb = np.full((n_total, n_prod), np.nan)
        mlp = np.full((n_total, n_prod), np.nan)
        for e in range(n_runs):
            for j in range(n_prod):
                s0 = int(starts[j])
                n_train = n_hist - s0
                if n_train < 8:
                    if verbose and e == 0:
                        print(f"  skipping {name} for producer {j}: <8 active historical samples")
                    continue
                seed = 1000 * e + j
                x_h = np.column_stack([time, q_crm[:, j]])[s0:]
                x_m = time[s0:, None]
                p_h, v_h = _fit_predict_ml(name, x_h, production[s0:, j], n_train, seed)
                p_m, v_m = _fit_predict_ml(name, x_m, production[s0:, j], n_train, seed)
                hyb[s0:, j], mlp[s0:, j] = p_h, p_m
                for ee in (range(n_evaluations) if n_runs == 1 else [e]):
                    score(f"CRM-{name}", ee, j, hyb[:, j], v_h)
                    score(name, ee, j, mlp[:, j], v_m)
            if verbose:
                print(f"  {name}: evaluation {e + 1}/{n_runs} done")
        preds[f"CRM-{name}"], preds[name] = hyb, mlp
    return pd.DataFrame.from_records(records), preds


# --------------------------------------------------------------------------- #
# 6. Demonstration with mock (synthetic) data
# --------------------------------------------------------------------------- #


def make_mock_field(
    n_steps: int = 400, seed: int = 7, noise: float = 0.02
) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, CRMParameters]:
    """Generates a 5-injector / 4-producer mock field from a known CRM.

    The injector/producer layout mimics the paper's Fig. 6 five-spot-like pattern.
    Injection rates change randomly every 30 time steps, and observed production
    is the CRM response with multiplicative noise.

    Args:
        n_steps: Number of time samples (daily).
        seed: Random seed.
        noise: Relative standard deviation of Gaussian noise on production.

    Returns:
        Tuple ``(time, injection, production, distances, true_params)``.
    """
    rng = np.random.default_rng(seed)
    n_inj, n_prod = 5, 4
    time = np.arange(n_steps, dtype=float)

    inj_xy = np.array([[500, 2100], [2100, 2100], [1300, 1300], [500, 500], [2100, 500]], float)
    prod_xy = np.array([[1300, 2100], [500, 1300], [2100, 1300], [1300, 500]], float)
    distances = np.linalg.norm(inj_xy[:, None, :] - prod_xy[None, :, :], axis=2)  # (I, K)

    # Piecewise-constant random injection rates (mean pattern as in the paper)
    means = np.array([2700, 900, 1800, 900, 2700], float)
    n_blocks = int(np.ceil(n_steps / 30))
    blocks = means * rng.uniform(0.6, 1.4, size=(n_blocks, n_inj))
    injection = np.repeat(blocks, 30, axis=0)[:n_steps]

    # "True" CRM: connectivity decays with distance, rows sum to 1
    lam = 1.0 / distances
    lam /= lam.sum(axis=1, keepdims=True)
    true = CRMParameters(
        tau_j=rng.uniform(5, 10, n_prod),
        lambda_ij=lam,
        tau_ij=3.0 + distances / 150.0,
        )
    q0 = (lam * injection[0][:, None]).sum(axis=0)
    q_clean = crm_simulate(true, time, injection, q0)
    production = np.vstack([q0, q_clean])
    production *= 1.0 + noise * rng.standard_normal(production.shape)
    production[0] = q0
    return time, injection, production, distances, true


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option("display.width", 140)

    time_s, inj, prod, dist, true_par = make_mock_field(n_steps=400)
    n_forecast = 60  # last 60 days are forecast

    # NOTE: the paper uses 20 evaluations; 3 are used here to keep the demo quick.
    results, crm_model, preds = run_forecast_study(
        time_s, inj, prod, dist,
        n_forecast=n_forecast,
        n_evaluations=3,
        crm_kwargs=dict(n_starts=2, max_iter=150),
    )

    p = crm_model.params_
    print("\nCalibrated lambda_ij (rows = injectors, cols = producers):\n", p.lambda_ij)
    print("True lambda_ij:\n", true_par.lambda_ij)
    print("Calibrated tau_j:", p.tau_j, " | true tau_j:", true_par.tau_j)
    print("Sum of lambda per injector:", p.lambda_ij.sum(axis=1))

    for case in ("entire", "forecast"):
        print(f"\n=== Ranking on '{case}' data (by MAE, lower rank = better) ===")
        print(rank_approaches(results, case=case).round(3))

    print("\n=== Mean metrics, forecast period, all producers ===")
    print(results[results["case"] == "forecast"]
          .groupby("approach")[["MAE", "RMSE", "R2"]].mean().sort_values("MAE").round(3))
