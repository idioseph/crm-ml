#!/usr/bin/env python3
"""crm-ml-test.py: compare lambda-prior patterns for the Bayesian CRM and CRM-XGB.

Runs several prior configurations ("patterns") one after another on the same data,
then draws one multi-panel bar chart comparing History / Holdout MAE and RMSE.

Patterns (choose with --patterns; default runs all four):
  dirichlet          Dirichlet, with slack:    (lambda_i1..lambda_iJ, slack_i) ~ Dirichlet(alpha)
                                               -> sum_j lambda_ij <= 1, the rest goes to the slack term
  dirichlet-noslack  Dirichlet, without slack: (lambda_i1..lambda_iJ) ~ Dirichlet(alpha); no slack term
                                               -> sum_j lambda_ij = 1 exactly
  beta               Beta, with slack:         lambda_ij ~ Beta(a, b) independently, plus a Gaussian
                                               penalty when sum_j lambda_ij > 1; 1 - sum is the slack
  beta-noslack       Beta, without slack:      w_ij ~ Beta(a, b) independently, lambda_ij = w_ij / sum_j w_ij
                                               -> sum_j lambda_ij = 1 exactly; no slack term, no penalty

"Without slack" removes the slack term entirely: there is no slack variable in the model,
and every injector's connectivities sum to exactly 1, in BOTH the deterministic calibration
(SLSQP equality constraint) and the Bayesian sampling.  --no-slack applies this to every
selected pattern; duplicate configurations are run once.

Fixed in this test script:
  * Models: pure CRM and CRM-XGB only (ELM, MLP, NuSVM dropped for speed).
  * tau_max = 4.0 * H (H = number of history rows) for tau_j and tau_ij, everywhere.
  * Posterior parameter plots: plain linear-scale histograms of tau_j, lambda_ij, tau_ij.

Reuse between sessions (to avoid re-running expensive steps):
  * Deterministic fits are cached per constraint type (ineq / eq) in the output root,
    and shared by every pattern that uses the same constraint.
  * NUTS posteriors are cached per pattern, keyed by every setting that affects them.
    Re-running with the same settings reloads them instead of sampling; --no-cache forces a rerun.

This script imports the shared machinery (data loading, CRM forward model, NUTS runner,
metrics, residual ML, producer plots) from crm-ml.py in the same folder, or from --base.

Usage
  python crm-ml-test.py --data buffalo_field.xlsx
  python crm-ml-test.py --data buffalo_field.xlsx --patterns dirichlet beta --no-slack
  python crm-ml-test.py --data buffalo_field.xlsx --patterns dirichlet dirichlet-noslack beta beta-noslack
  python crm-ml-test.py --self-test
  python crm-ml-test.py --out results_prior_compare2 --compare-only \
      --patterns dirichlet dirichlet-noslack beta beta-noslack      # charts only, no fitting

Comparison outputs (written to --out after every run, or by --compare-only):
  comparison_runs.csv              convergence and lambda-sum checks for every run
  comparison_mae_rmse_pooled.png   all wells' rows pooled into one MAE/RMSE per run (rate-weighted)
  perwell_history_mae_rmse.png     MAE/RMSE for every well, history period
  perwell_holdout_mae_rmse.png     MAE/RMSE for every well, holdout period
  perwell_rank_heatmap.png         % worse than the best configuration in the same well
  perwell_rank_summary.csv         geo-mean ratio to per-well best, mean rank, wells won
  perwell_paired_effects.csv       same-well paired effect of model, prior and slack
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")


# =============================================================================
# 0. LOAD THE BASE MODULE (crm-ml.py)
# =============================================================================

def _load_base(path: Path):
    """Import crm-ml.py (hyphenated file name) as module 'crm_ml_base'.

    The module is registered in sys.modules before execution so its dataclasses
    and the pickled deterministic fits resolve correctly."""
    if not path.exists():
        sys.exit(f"base module not found: {path}  (put crm-ml.py next to this script or pass --base)")
    spec = importlib.util.spec_from_file_location("crm_ml_base", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crm_ml_base"] = mod
    spec.loader.exec_module(mod)
    return mod


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--base", default=None)
_known, _ = _pre.parse_known_args()
B = _load_base(Path(_known.base) if _known.base else Path(__file__).resolve().with_name("crm-ml.py"))

import jax                                   # noqa: E402  (after XLA_FLAGS and base import)
import jax.numpy as jnp                      # noqa: E402
import numpy as np                           # noqa: E402
import numpyro                               # noqa: E402
import numpyro.distributions as dist         # noqa: E402
import pandas as pd                          # noqa: E402
import torch                                 # noqa: E402
from scipy.optimize import minimize          # noqa: E402

log = B.log
DT = B.DT

TAU_MAX_FACTOR = 4.0          # tau_max = 4.0 * H, hardcoded (prevents "infinite memory")

PATTERNS = {
    "dirichlet":         dict(prior="dirichlet", slack=True),
    "dirichlet-noslack": dict(prior="dirichlet", slack=False),
    "beta":              dict(prior="beta", slack=True),
    "beta-noslack":      dict(prior="beta", slack=False),
}
DEFAULT_PATTERNS = ["dirichlet", "dirichlet-noslack", "beta", "beta-noslack"]
PATTERN_LABELS = {
    ("dirichlet", True): "Dirichlet, with slack",
    ("dirichlet", False): "Dirichlet, without slack",
    ("beta", True): "Beta, with slack",
    ("beta", False): "Beta, without slack",
}
NOT_CONVERGED = "not converged"      # marker used on chart labels for runs that miss the NUTS gate


def tau_max_for(H: int) -> float:
    """Upper bound for every tau parameter: 4.0 times the number of history rows."""
    return TAU_MAX_FACTOR * H


def default_beta_b(J: int, slack: bool) -> float:
    """Beta(1, b) that matches the marginal of a flat Dirichlet:
    with slack (J+1 components) b = J; without slack (J components) b = J - 1."""
    return float(J) if slack else float(max(J - 1, 1))


# =============================================================================
# 1. DETERMINISTIC CALIBRATION, WITH OR WITHOUT SLACK
# =============================================================================

def fit_deterministic_v(data, tau_min, tau_max, slack=True, n_starts=8, max_iter=1000, seed=2025):
    """Multi-start SLSQP fit of the CRM (same objective as crm-ml.py).

    slack=True : injector-wise inequality  sum_j lambda_ij <= 1
    slack=False: injector-wise equality    sum_j lambda_ij  = 1 (without slack: no slack term at all)
    """
    I, J, N = data.n_inj, data.n_prod, len(data.time)
    s = B.producer_starts(data.production)
    fwd = B.CRMForward(data.time, data.injection, s, data.production[s, np.arange(J)])
    y = torch.as_tensor(data.production, dtype=DT)
    mask = fwd.after
    denom = float(N * J)
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
    if slack:
        cons = {"type": "ineq", "fun": lambda x: 1.0 - A @ x, "jac": lambda x: -A}
    else:
        cons = {"type": "eq", "fun": lambda x: A @ x - 1.0, "jac": lambda x: A}
    bounds = [(lo, hi)] * J + [(0.0, 1.0)] * nl + [(lo, hi)] * nl

    def clean_lam(lam):
        lam = np.clip(lam, 0.0, 1.0)
        if not slack:
            lam = lam / np.clip(lam.sum(1, keepdims=True), 1e-12, None)
        return lam

    rng = np.random.default_rng(seed)
    rows, best, sols = [], None, []
    for k in range(n_starts):
        x0 = np.empty(J + 2 * nl)
        if k == 0:
            x0[:J] = np.log(np.clip(10.0, tau_min, tau_max))
            x0[J:J + nl] = (1.0 / J - 1e-5) if slack else 1.0 / J
            x0[J + nl:] = np.log(np.clip(10.0, tau_min, tau_max))
        else:
            x0[:J] = rng.uniform(np.log(max(tau_min, 0.1)), hi, J)
            if slack:
                x0[J:J + nl] = (rng.dirichlet(np.ones(J + 1), size=I)[:, :J] * 0.95).ravel()
            else:
                x0[J:J + nl] = rng.dirichlet(np.ones(J), size=I).ravel()
            x0[J + nl:] = rng.uniform(np.log(max(tau_min, 0.1)), hi, nl)
        res = minimize(f_and_g, x0, jac=True, method="SLSQP", bounds=bounds, constraints=[cons],
                       options={"maxiter": max_iter, "ftol": 1e-12})
        g = A @ res.x - 1.0
        viol = float(max(0.0, g.max())) if slack else float(np.abs(g).max())
        tj_k, lam_k, tij_k = (v.detach().numpy() for v in unpack(res.x))
        sols.append((float(res.fun), B.CRMParams(tj_k, clean_lam(lam_k), tij_k)))
        rows.append({"start": k, "objective": float(res.fun), "success": bool(res.success),
                     "status": int(res.status), "message": str(res.message), "nit": int(res.nit),
                     "max_constraint_violation": viol})
        if np.isfinite(res.fun) and (best is None or res.fun < best[1]):
            best = (res.x, float(res.fun))
    if best is None:
        raise RuntimeError("deterministic CRM fit failed on every start")
    tj, lam, tij = (v.detach().numpy() for v in unpack(best[0]))
    tbl = pd.DataFrame(rows)
    tbl["rel_gap_to_best"] = tbl.objective / best[1] - 1.0
    return B.FitResult(B.CRMParams(tj, clean_lam(lam), tij), best[1], tbl, (tau_min, tau_max),
                       sorted(sols, key=lambda z: z[0]))


# =============================================================================
# 2. BAYESIAN CRM WITH SELECTABLE LAMBDA PRIOR
# =============================================================================

def make_model_v(t, w, starts, q0, y, likelihood, tau_min, tau_max, prior="dirichlet", slack=True,
                 tau_mu=np.log(10.0), tau_sd=2.0, alpha=1.0, beta_a=1.0, beta_b=None, cap_sd=0.02,
                 sigma_scale=None):
    """NumPyro CRM model; identical to crm-ml.py except for the lambda prior block.

    prior="dirichlet", slack=True : site lam_full [I, J+1], Dirichlet(alpha); lambda = first J columns
    prior="dirichlet", slack=False: site lam_full [I, J],   Dirichlet(alpha); rows sum exactly to 1
    prior="beta",      slack=True : site lam [I, J] ~ Beta(a, b); factor -0.5 * (max(0, sum-1)/cap_sd)^2
    prior="beta",      slack=False: site lam_w [I, J] ~ Beta(a, b) (raw weights);
                                    deterministic lam = lam_w / row sum, so rows sum exactly to 1.
                                    No slack term and no penalty.
    """
    I, J = w.shape[1], y.shape[1]
    N = len(t)
    beta_b = default_beta_b(J, slack) if beta_b is None else beta_b
    st = np.asarray(starts)
    y_np = np.asarray(y)
    t, w, starts, q0, y = (jnp.asarray(a) for a in (t, w, starts, q0, y))
    after = np.arange(N)[:, None] > st[None, :]
    prev_after = np.zeros_like(after)
    prev_after[1:] = after[:-1]
    if likelihood in ("normal", "ar1"):
        mask = jnp.asarray(after)
        ytr = y
    elif likelihood == "lognormal":
        mask = jnp.asarray(after & (y_np > 0))
        ytr = jnp.log(jnp.where(mask, y, 1.0))
    else:
        raise ValueError(likelihood)
    if sigma_scale is None:
        sigma_scale = np.ones(J)
    lo, hi = np.log(tau_min), np.log(tau_max)

    def sample_lambda():
        if prior == "dirichlet":
            K = J + 1 if slack else J
            if K == 1:   # single producer without slack: lambda is fixed at 1
                return numpyro.deterministic("lam_full", jnp.ones((I, 1)))
            full = numpyro.sample("lam_full", dist.Dirichlet(alpha * jnp.ones(K)).expand([I]).to_event(1))
            return full[:, :J]
        if prior == "beta" and slack:
            lam = numpyro.sample("lam", dist.Beta(beta_a, beta_b).expand([I, J]).to_event(2))
            excess = jnp.maximum(lam.sum(axis=1) - 1.0, 0.0)
            numpyro.factor("mass_balance", -0.5 * jnp.sum((excess / cap_sd) ** 2))
            return lam
        if prior == "beta":
            wraw = numpyro.sample("lam_w", dist.Beta(beta_a, beta_b).expand([I, J]).to_event(2))
            return numpyro.deterministic("lam", wraw / wraw.sum(axis=1, keepdims=True))
        raise ValueError(prior)

    def model():
        tprior = dist.Normal(tau_mu, tau_sd)
        log_tau_j = numpyro.sample("log_tau_j", dist.Uniform(lo, hi).expand([J]).to_event(1))
        log_tau_ij = numpyro.sample("log_tau_ij", dist.Uniform(lo, hi).expand([I, J]).to_event(2))
        numpyro.factor("tau_prior", tprior.log_prob(log_tau_j).sum() + tprior.log_prob(log_tau_ij).sum())
        lam = sample_lambda()
        sigma = numpyro.sample("sigma", dist.HalfNormal(jnp.asarray(sigma_scale)).to_event(1))
        mu = B.crm_jax(jnp.exp(log_tau_j), lam, jnp.exp(log_tau_ij), t, w, starts, q0)
        if likelihood == "ar1":
            phi = numpyro.sample("phi", dist.Uniform(-0.99, 0.99).expand([J]).to_event(1))
            e = jnp.where(mask, ytr - mu, 0.0)
            e_prev = jnp.concatenate([jnp.zeros((1, J)), e[:-1]], axis=0)
            cond = jnp.asarray(prev_after)
            loc = mu + jnp.where(cond, phi[None, :] * e_prev, 0.0)
            scale = jnp.where(cond, sigma[None, :], sigma[None, :] / jnp.sqrt(1.0 - phi[None, :] ** 2))
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample("obs", dist.Normal(loc, scale), obs=ytr)
            return
        loc = mu if likelihood == "normal" else jnp.log(jnp.clip(mu, 1e-8))
        with numpyro.handlers.mask(mask=mask):
            numpyro.sample("obs", dist.Normal(loc, sigma[None, :]), obs=ytr)

    return model, int(np.asarray(mask).sum()), int((after & (y_np <= 0)).sum())


def init_values_v(p, J, sigma, tau_min, tau_max, prior, slack, phi=None):
    """Interior NUTS starting values from a deterministic solution, matching the prior's sites."""
    lo, hi = np.log(tau_min) + 1e-3, np.log(tau_max) - 1e-3
    tj = np.exp(np.clip(np.log(p.tau_j), lo, hi))
    tij = np.exp(np.clip(np.log(p.tau_ij), lo, hi))
    lam = np.clip(np.asarray(p.lam, float), 1e-4, 1 - 1e-4)
    out = {"log_tau_j": jnp.log(jnp.asarray(tj)), "log_tau_ij": jnp.log(jnp.asarray(tij)),
           "sigma": jnp.asarray(sigma)}
    if prior == "dirichlet":
        full = np.concatenate([lam, np.clip(1.0 - lam.sum(1, keepdims=True), 1e-3, None)], axis=1) if slack else lam
        full = full / full.sum(1, keepdims=True)
        if full.shape[1] > 1:
            out["lam_full"] = jnp.asarray(full)
    elif slack:
        rs = lam.sum(1, keepdims=True)
        out["lam"] = jnp.asarray(np.where(rs > 0.999, lam * 0.999 / rs, lam))
    else:
        # raw weights proportional to the deterministic lambdas (their row sums are rescaled away)
        out["lam_w"] = jnp.asarray(lam)
    if phi is not None:
        out["phi"] = jnp.asarray(np.clip(phi, -0.9, 0.9))
    return out


def physical_v(grouped, J):
    """Latent draws -> (tau_j [D,J], lambda [D,I,J], tau_ij [D,I,J], sigma [D,J]) for any prior."""
    flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in grouped.items()}
    lam = flat["lam_full"][:, :, :J] if "lam_full" in flat else flat["lam"]
    return np.exp(flat["log_tau_j"]), lam, np.exp(flat["log_tau_ij"]), flat["sigma"]


# =============================================================================
# 3. PLOTS
# =============================================================================

def plot_parameter_posteriors(path, tau_j, lam, tau_ij, pname, injectors, subtitle=""):
    """Clean linear-scale posterior histograms for one producer:
    tau_j on the top row, then lambda_ij (left) and tau_ij (right) for each injector.
    No log scale, no prior overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    I = len(injectors)
    fig, axes = plt.subplots(I + 1, 2, figsize=(10, 2.8 * (I + 1)), squeeze=False)
    gs = axes[0, 0].get_gridspec()
    for ax in axes[0]:
        ax.remove()
    ax0 = fig.add_subplot(gs[0, :])

    def hist(ax, x, color, title):
        ax.hist(np.asarray(x), bins=50, density=True, color=color, alpha=0.7, edgecolor="none")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.15)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    hist(ax0, tau_j, "#2a78d6", f"Producer time constant \u03c4_j of {pname} (time steps)")
    for i, iname in enumerate(injectors):
        hist(axes[i + 1, 0], lam[:, i], "#184f95", f"Connectivity \u03bb, injector {iname} \u2192 {pname} (fraction)")
        hist(axes[i + 1, 1], tau_ij[:, i], "#eb6834", f"Signal delay \u03c4_ij, injector {iname} \u2192 {pname} (time steps)")
    fig.suptitle(f"Bayesian CRM posterior parameter distributions for producer {pname}\n"
                 f"(\u03bb prior: {subtitle}; histograms of all NUTS posterior draws)" if subtitle else
                 f"Bayesian CRM posterior parameter distributions for producer {pname}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


PLOT_CTX = {"history_rows": None, "holdout_rows": None, "units": "MSTB/month"}


def period_desc(period, short=False):
    """Plain-language description of an evaluation window for chart titles."""
    H, F = PLOT_CTX["history_rows"], PLOT_CTX["holdout_rows"]
    if period == "history":
        span = f"first {H} months" if H else "calibration months"
        return f"History match ({span})" if short else f"History match ({span}, used for calibration)"
    span = f"last {F} months" if F else "final months"
    return f"Holdout forecast ({span})" if short else f"Holdout forecast ({span}, not used for calibration)"


def plot_comparison(path, comp):
    """2 x 2 grouped bar chart: History MAE, History RMSE, Holdout MAE, Holdout RMSE
    (field-pooled, P50 of the predictive ensemble), patterns on x, CRM vs CRM-XGB bars.
    Patterns whose NUTS run missed the convergence gate are marked 'not converged'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pats = list(dict.fromkeys(comp.pattern_label))
    nc = {p for p in pats if not bool(comp.loc[comp.pattern_label == p, "converged"].iloc[0])}
    ticks = [f"{p}\n({NOT_CONVERGED})" if p in nc else p for p in pats]
    models = list(dict.fromkeys(comp.model))
    colors = {"CRM": "#2a78d6", "CRM-XGB": "#eb6834"}
    x = np.arange(len(pats))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (period, metric) in zip(axes.flat, [("history", "MAE"), ("history", "RMSE"),
                                                ("holdout", "MAE"), ("holdout", "RMSE")]):
        sub = comp[comp.period == period]
        for k, m in enumerate(models):
            vals = []
            for p in pats:
                hit = sub[(sub.pattern_label == p) & (sub.model == m)]
                vals.append(float(hit[metric].iloc[0]) if len(hit) else np.nan)
            bars = ax.bar(x + (k - (len(models) - 1) / 2) * width, vals, width, label=m,
                          color=colors.get(m, None), edgecolor="none")
            ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=9)
        ax.set_title(f"{period_desc(period)}: {metric}", fontsize=10.5, loc="left")
        ax.set_ylabel(f"{metric} ({PLOT_CTX['units']})")
        ax.grid(axis="y", alpha=0.15)
        ax.margins(y=0.15)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, [{"CRM": "Bayesian CRM", "CRM-XGB": "Bayesian CRM + XGBoost residual (CRM-XGB)"}.get(v, v)
                   for v in l], loc="upper right", ncol=len(models), frameon=False)
    fig.suptitle("Field-level rate error of Bayesian CRM and CRM-XGB under four \u03bb priors\n"
                 "(all wells' monthly rates pooled into one error per run; P50 of the posterior predictive; "
                 "lower is better)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Per-well comparison (no averaging of raw errors across wells)
# -----------------------------------------------------------------------------

PATTERN_COLORS = {"Dirichlet, with slack": "#2a78d6", "Dirichlet, without slack": "#eb6834",
                  "Beta, with slack": "#1baf7a", "Beta, without slack": "#eda100"}
MODEL_NAMES = {"CRM": "Bayesian CRM", "CRM-XGB": "Bayesian CRM + XGBoost residual (CRM-XGB)"}
METRICS = ("MAE", "RMSE")
PERIODS = ("history", "holdout")


def collect_metrics(root, configs):
    """Read every pattern's saved metrics.csv (predictive P50 rows) into one long table.
    Returns (per_well, pooled); per_well excludes the FIELD(pooled) row."""
    frames = []
    for prior, slack in configs:
        tag = f"{prior}-{'slack' if slack else 'noslack'}"
        f = Path(root) / tag / "metrics.csv"
        if not f.exists():
            log(f"skipping {tag}: {f} not found")
            continue
        m = pd.read_csv(f)
        m = m[m.model.isin(["CRM predictive", "CRM-XGB predictive"])].copy()
        m["model"] = m.model.str.replace(" predictive", "", regex=False)
        sfile = Path(root) / tag / "summary.json"
        sj = json.load(open(sfile)) if sfile.exists() else {}
        conv = bool(sj.get("converged", False))
        PLOT_CTX["history_rows"] = sj.get("history_rows", PLOT_CTX["history_rows"])
        PLOT_CTX["holdout_rows"] = sj.get("holdout_rows", PLOT_CTX["holdout_rows"])
        frames.append(m.assign(pattern=tag, pattern_label=PATTERN_LABELS[(prior, slack)], prior=prior,
                               slack=slack, converged=conv))
    if not frames:
        raise FileNotFoundError(f"no metrics.csv found under {root}")
    allm = pd.concat(frames, ignore_index=True)
    pooled = allm.producer == "FIELD(pooled)"
    return allm[~pooled].reset_index(drop=True), allm[pooled].reset_index(drop=True)


def relative_to_best(pw):
    """Within each (well, period, metric): every configuration's error divided by the best
    configuration's error in that same well, plus its rank (1 = best).  Scale-free, so a
    high-rate well cannot dominate the comparison."""
    rows = []
    for (well, per), g in pw.groupby(["producer", "period"], sort=False):
        for met in METRICS:
            best = g[met].min()
            rk = g[met].rank(method="min")
            for k, r in g.iterrows():
                rows.append({"producer": well, "period": per, "metric": met, "pattern_label": r.pattern_label,
                             "model": r.model, "converged": r.converged, "value": r[met],
                             "rel_to_best": r[met] / best if best > 0 else np.nan, "rank": int(rk[k])})
    return pd.DataFrame(rows)


def rank_summary(rel):
    """Per configuration: geometric-mean error ratio to the per-well best, mean rank, number of
    wells won.  Sorted best first within each period/metric."""
    def gmean(x):
        x = np.asarray(x, float)
        return float(np.exp(np.log(x[x > 0]).mean()))
    g = (rel.groupby(["period", "metric", "pattern_label", "model", "converged"])
         .agg(wells=("producer", "nunique"), geomean_ratio_to_best=("rel_to_best", gmean),
              mean_rank=("rank", "mean"), wells_won=("rank", lambda r: int((r == 1).sum())))
         .reset_index())
    return g.sort_values(["period", "metric", "geomean_ratio_to_best"]).reset_index(drop=True)


def paired_effects(pw):
    """Paired, same-well effect of each design factor, holding the other two fixed:
       model : CRM-XGB vs CRM          prior : Beta vs Dirichlet          slack : without vs with slack.
    ratio = B / A per well (< 1 means B is better).  Reports geometric-mean ratio, median ratio
    and how many well-pairs B wins."""
    factors = {"model": ("model", "CRM", "CRM-XGB"), "prior": ("prior", "dirichlet", "beta"),
               "slack": ("slack", True, False)}
    names = {"model": "CRM-XGB vs CRM", "prior": "Beta vs Dirichlet", "slack": "without slack vs with slack"}
    out = []
    for fname, (col, a, b) in factors.items():
        others = [c for c in ("model", "prior", "slack") if c != col]
        for per in PERIODS:
            sub = pw[pw.period == per]
            A = sub[sub[col] == a].set_index(others + ["producer"])
            Bv = sub[sub[col] == b].set_index(others + ["producer"])
            common = A.index.intersection(Bv.index)
            if not len(common):
                continue
            for met in METRICS:
                r = (Bv.loc[common, met] / A.loc[common, met]).to_numpy(float)
                r = r[np.isfinite(r) & (r > 0)]
                out.append({"comparison": names[fname], "period": per, "metric": met, "pairs": len(r),
                            "B_wins": int((r < 1).sum()), "geomean_ratio_B_over_A": float(np.exp(np.log(r).mean())),
                            "median_ratio_B_over_A": float(np.median(r))})
    return pd.DataFrame(out)


def plot_perwell_bars(path, pw, period):
    """2 x 2: rows MAE / RMSE, columns CRM / CRM-XGB; wells on x, one bar per prior pattern.
    The y-axis is shared along each row so CRM and CRM-XGB panels read on the same scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = pw[pw.period == period]
    wells = list(dict.fromkeys(sub.producer))
    pats = [p for p in PATTERN_COLORS if p in set(sub.pattern_label)]
    nc = {p for p in pats if not bool(sub.loc[sub.pattern_label == p, "converged"].iloc[0])}
    models = [m for m in ("CRM", "CRM-XGB") if m in set(sub.model)]
    x = np.arange(len(wells))
    w = 0.8 / len(pats)
    fig, axes = plt.subplots(2, len(models), figsize=(8 * len(models), 9), sharey="row", squeeze=False)
    for r, met in enumerate(METRICS):
        for c, m in enumerate(models):
            ax = axes[r, c]
            for k, p in enumerate(pats):
                vals = [sub[(sub.producer == wl) & (sub.model == m) & (sub.pattern_label == p)][met].mean()
                        for wl in wells]
                ax.bar(x + (k - (len(pats) - 1) / 2) * w, vals, w * 0.92, color=PATTERN_COLORS[p],
                       label=p + (f" ({NOT_CONVERGED})" if p in nc else ""), edgecolor="none")
            ax.set_xticks(x)
            ax.set_xticklabels(wells, rotation=30, ha="right", fontsize=9)
            ax.set_title(f"{MODEL_NAMES.get(m, m)}: {met} for each producer well", fontsize=10.5, loc="left")
            if c == 0:
                ax.set_ylabel(f"{met} ({PLOT_CTX['units']})")
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9, title="\u03bb prior",
               title_fontsize=9)
    fig.suptitle(f"{period_desc(period)}:\nrate error of each producer well under four \u03bb priors "
                 f"(P50 of the posterior predictive; lower is better)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rank_heatmap(path, rel, summ):
    """2 x 2 heatmaps (rows history / holdout, columns MAE / RMSE).  Each row of a heatmap is one
    configuration (pattern x model), each column one well; the cell shows how much worse that
    configuration is than the best one in the same well (0% = best in that well).  The last column
    is the geometric mean across wells; configurations are sorted by it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5",
                                                          "#1c5cab", "#0d366b"])
    wells = list(dict.fromkeys(rel.producer))
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), squeeze=False)
    for r, per in enumerate(PERIODS):
        for c, met in enumerate(METRICS):
            ax = axes[r, c]
            s = summ[(summ.period == per) & (summ.metric == met)]
            sub = rel[(rel.period == per) & (rel.metric == met)]
            if s.empty:
                ax.axis("off")
                continue
            cfgs = list(zip(s.pattern_label, s.model))
            M = np.full((len(cfgs), len(wells) + 1), np.nan)
            for i, (p, m) in enumerate(cfgs):
                for j, wl in enumerate(wells):
                    v = sub[(sub.pattern_label == p) & (sub.model == m) & (sub.producer == wl)].rel_to_best
                    M[i, j] = v.iloc[0] - 1 if len(v) else np.nan
                M[i, -1] = s.geomean_ratio_to_best.iloc[i] - 1
            vmax = max(np.nanpercentile(M, 95), 0.02)
            ax.imshow(M * 100, cmap=cmap, vmin=0, vmax=vmax * 100, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    if np.isnan(v):
                        continue
                    txt = "best" if v < 5e-4 else f"+{v * 100:.1f}%"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if v > 0.55 * vmax else "#1a1a19",
                            fontweight="bold" if v < 5e-4 else "normal")
            ax.axvline(len(wells) - 0.5, color="#1a1a19", lw=1.2)
            ax.set_xticks(range(len(wells) + 1))
            ax.set_xticklabels(wells + ["Geo-mean\nacross wells"], rotation=30, ha="right", fontsize=8.5)
            conv = dict(zip(zip(s.pattern_label, s.model), s.converged))
            ax.set_yticks(range(len(cfgs)))
            ax.set_yticklabels([f"{m} | {p}" + ("" if conv[(p, m)] else f" ({NOT_CONVERGED})")
                                for p, m in cfgs], fontsize=8.5)
            ax.set_title(f"{period_desc(per, short=True)}: {met}, % above the best run in each well",
                         fontsize=10.5, loc="left")
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle("Well-by-well comparison of model and \u03bb prior: how far each run is from the best run "
                 "in the same well\n(\u2018best\u2019 = lowest error in that well; last column = geometric mean "
                 "across wells; rows sorted best first)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_comparisons(root, configs, units="MSTB/month"):
    """All cross-pattern outputs, rebuilt from the per-pattern folders already on disk."""
    root = Path(root)
    PLOT_CTX["units"] = units
    pw, pooled = collect_metrics(root, configs)
    keep = [c for c in ["pattern", "pattern_label", "converged", "model", "period", "n", "MAE", "RMSE",
                        "R2_det", "PICP_80", "PICP_95", "width_80", "CRPS"] if c in pooled]
    pooled[keep].to_csv(root / "comparison_field_metrics.csv", index=False)
    plot_comparison(root / "comparison_mae_rmse_pooled.png", pooled)

    pw_keep = [c for c in ["producer", "pattern", "pattern_label", "prior", "slack", "converged", "model",
                           "period", "n", "MAE", "RMSE", "PICP_80", "CRPS"] if c in pw]
    pw[pw_keep].to_csv(root / "perwell_metrics.csv", index=False)
    for per in PERIODS:
        plot_perwell_bars(root / f"perwell_{per}_mae_rmse.png", pw, per)
    rel = relative_to_best(pw)
    rel.to_csv(root / "perwell_relative_to_best.csv", index=False)
    summ = rank_summary(rel)
    summ.to_csv(root / "perwell_rank_summary.csv", index=False)
    plot_rank_heatmap(root / "perwell_rank_heatmap.png", rel, summ)
    eff = paired_effects(pw)
    eff.to_csv(root / "perwell_paired_effects.csv", index=False)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print("\nPer-well ranking, holdout (geo-mean ratio to the best config in each well; 1.000 = best everywhere):")
    print(summ[summ.period == "holdout"].drop(columns="period").to_string(index=False, float_format=fmt))
    print("\nPaired same-well effects (ratio B/A < 1 means B better; B_wins = wells where B is better):")
    print(eff.to_string(index=False, float_format=fmt), flush=True)
    log(f"comparison outputs written to {root}")


# =============================================================================
# 4. ONE PATTERN, END TO END
# =============================================================================

def _settings_key(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=float).encode()).hexdigest()[:10]


def get_det_fit(a, hist, H, slack, root, memo):
    """Deterministic fit shared by every pattern with the same constraint (cached on disk)."""
    if slack in memo:
        return memo[slack]
    tmax = tau_max_for(H)
    kind = "ineq" if slack else "eq"
    cache = root / f"det_fit_{kind}_H{H}_s{a.starts}_it{a.max_iter}_seed{a.seed}_tmin{a.tau_min:.6g}_tmax{tmax:g}.pkl"
    t0 = time.time()
    if cache.exists() and not a.no_cache:
        fit = pickle.load(open(cache, "rb"))
        log(f"deterministic fit ({kind}) loaded from {cache.name}")
    else:
        fit = fit_deterministic_v(hist, a.tau_min, tmax, slack, a.starts, a.max_iter, a.seed)
        pickle.dump(fit, open(cache, "wb"))
        log(f"deterministic fit ({kind}): objective {fit.objective:.5g} ({time.time() - t0:.0f}s); "
            f"converged starts {int(fit.starts_table.success.sum())}/{a.starts}")
    memo[slack] = fit
    return fit


def run_pattern(a, d, prior, slack, root, det_memo):
    """Deterministic fit -> NUTS -> propagation -> CRM-XGB for one prior pattern.
    Returns the FIELD(pooled) metric rows and a one-line summary."""
    tag = f"{prior}-{'slack' if slack else 'noslack'}"
    label = PATTERN_LABELS[(prior, slack)]
    out = root / tag
    (out / "plots").mkdir(parents=True, exist_ok=True)
    labels = B.p_labels(a.p_convention)

    N, I, J = len(d.time), d.n_inj, d.n_prod
    H = N - a.forecast
    tmax = tau_max_for(H)
    hist = d.head(H)
    s = B.producer_starts(hist.production)
    q0 = hist.production[s, np.arange(J)]
    y = d.production
    rows_hist = [np.arange(s[j] + 1, H) for j in range(J)]
    rows_fc = [np.arange(H, N) for _ in range(J)]
    beta_b = default_beta_b(J, slack) if a.beta_b is None else a.beta_b
    log(f"===== pattern {tag} ({label}); tau bounds [{a.tau_min:.4g}, {tmax:g}] =====")

    # ---- Step 1: deterministic CRM
    fit = get_det_fit(a, hist, H, slack, root, det_memo)
    fit.starts_table.to_csv(out / "deterministic_starts.csv", index=False)
    det = B.predict(d, hist, fit.params)

    # ---- Step 2: NUTS (cached per settings)
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
    model, n_obs, _ = make_model_v(hist.time, hist.injection, s, q0, hist.production, lik, a.tau_min, tmax,
                                   prior, slack, np.log(10.0), a.tau_prior_sd, a.alpha, a.beta_a, beta_b,
                                   a.cap_sd, scale)
    phi0 = None
    if lik == "ar1":
        phi0 = np.array([np.corrcoef((y[rows_hist[j], j] - det[rows_hist[j], j])[:-1],
                                     (y[rows_hist[j], j] - det[rows_hist[j], j])[1:])[0, 1] for j in range(J)])
        resid = resid * np.sqrt(1 - np.clip(phi0, -0.9, 0.9) ** 2)

    key = _settings_key({"H": H, "lik": lik, "chains": a.chains, "warmup": a.warmup, "samples": a.samples,
                         "seed": a.seed, "tau_sd": a.tau_prior_sd, "alpha": a.alpha, "beta_a": a.beta_a,
                         "beta_b": beta_b, "cap_sd": a.cap_sd, "tau_min": a.tau_min, "tau_max": tmax,
                         "prior": prior, "slack": slack, "det_obj": round(fit.objective, 12),
                         **({"beta_noslack_form": "normalised_weights"} if prior == "beta" and not slack else {})})
    post_file, chain_file = out / f"posterior_{key}.npz", out / f"chains_{key}.json"
    t0 = time.time()
    if post_file.exists() and chain_file.exists() and not a.no_cache:
        with np.load(post_file) as z:
            grouped = {k: z[k] for k in z.files}
        info = json.load(open(chain_file))
        log(f"NUTS posterior loaded from {post_file.name}")
    else:
        sols = [sp for _, sp in fit.solutions]
        inits = [init_values_v(sols[c % len(sols)], J, resid, a.tau_min, tmax, prior, slack, phi0)
                 for c in range(a.chains)]
        log(f"NUTS ({lik}, {label}): {a.chains} chains x ({a.warmup} warmup + {a.samples} draws), {n_obs} data rows")
        grouped, info = B.run_nuts(model, inits, a.warmup, a.samples, a.seed)
        np.savez_compressed(post_file, **grouped)
        json.dump(info, open(chain_file, "w"), indent=1)
    diag = pd.DataFrame(B.diagnostics({k: v for k, v in grouped.items() if k != "lam_w"}))
    n_div = sum(x["divergences"] for x in info)
    converged = bool(diag.r_hat.max() < 1.01 and diag.ess_bulk.min() > 400 and n_div == 0)
    log(f"NUTS ({time.time() - t0:.0f}s): max R-hat {diag.r_hat.max():.3f}, min ESS {diag.ess_bulk.min():.0f}, "
        f"divergences {n_div} -> {'CONVERGED' if converged else 'NOT CONVERGED (provisional)'}")
    diag.to_csv(out / "mcmc_diagnostics.csv", index=False)
    pd.DataFrame(info).to_csv(out / "mcmc_chains.csv", index=False)

    tau_j, lam, tau_ij, sig = physical_v(grouped, J)
    phi = grouped["phi"].reshape(-1, J) if lik == "ar1" else None
    D = len(tau_j)

    for j, pname in enumerate(d.producers):
        plot_parameter_posteriors(out / "plots" / f"Posterior_Params_{B.safe_name(pname)}.png",
                                  tau_j[:, j], lam[:, :, j], tau_ij[:, :, j], pname, d.injectors, label)

    names = ([f"tau_j[{n}]" for n in d.producers] + [f"lambda[{i}->{p}]" for i in d.injectors for p in d.producers]
             + [f"tau_ij[{i}->{p}]" for i in d.injectors for p in d.producers] + [f"sigma[{n}]" for n in d.producers]
             + ([f"phi[{n}]" for n in d.producers] if phi is not None else []))
    flat = np.column_stack([tau_j, lam.reshape(D, -1), tau_ij.reshape(D, -1), sig] + ([phi] if phi is not None else []))
    det_col = np.concatenate([fit.params.tau_j, fit.params.lam.ravel(), fit.params.tau_ij.ravel(),
                              np.full(J, np.nan)] + ([np.full(J, np.nan)] if phi is not None else []))
    qs = np.quantile(flat, [0.10, 0.50, 0.90], axis=0)
    pd.DataFrame({"parameter": names, labels["P10"]: qs[0], labels["P50"]: qs[1], labels["P90"]: qs[2],
                  "mean": flat.mean(0), "sd": flat.std(0), "deterministic": det_col}
                 ).to_csv(out / "posterior_parameters.csv", index=False)
    lam_sum = lam.sum(2)                                           # [D, I]
    mb = {"injector": d.injectors, "sum_lambda_P50": np.median(lam_sum, 0), "sum_lambda_min": lam_sum.min(0),
          "sum_lambda_max": lam_sum.max(0)}
    if slack:
        mb["slack_mean"] = (1 - lam_sum).mean(0)
    mb["deterministic_sum"] = fit.params.lam.sum(1)
    pd.DataFrame(mb).to_csv(out / "injector_mass_balance.csv", index=False)

    # ---- Step 3: propagate every draw through the CRM
    mu = B.propagate((tau_j, lam, tau_ij, sig), d.time, d.injection, s, q0)        # [D, N, J]
    if lik == "ar1":
        ypred, _ = B.predictive_ar1(mu, sig, phi, y, H, s, seed=a.seed)
    else:
        ypred, _ = B.predictive(mu, sig, lik, s, seed=a.seed)
    k_best, _ = B.best_fitting_draw(mu, y, s, H)
    best = mu[k_best]
    pp, pr = B.percentiles(ypred), B.percentiles(mu)

    tables = []
    for per, rows in (("history", rows_hist), ("holdout", rows_fc)):
        tables += [B.table(y, det, rows, d.producers, label=per).assign(model="CRM deterministic"),
                   B.table(y, pp["P50"], rows, d.producers, ypred, per).assign(model="CRM predictive"),
                   B.table(y, pr["P50"], rows, d.producers, mu, per).assign(model="CRM parameters only")]
    for j, pname in enumerate(d.producers):
        B.plot_producer(out / "plots" / f"CRM_{B.safe_name(pname)}.png", d.time, H, y[:, j], s[j],
                        {k: v[:, j] for k, v in pp.items()}, {k: v[:, j] for k, v in pr.items()}, best[:, j],
                        f"{pname}: Bayesian CRM rate forecast, \u03bb prior {label} ({lik} likelihood)", labels)

    # ---- Step 4: CRM-XGB, every (sub-sampled) draw through CRM + XGB
    t0 = time.time()
    Dm = min(a.ml_draws, D)
    idx = np.random.default_rng(a.seed).choice(D, Dm, replace=False)
    cfun = jax.jit(jax.vmap(lambda l, tij: B.crm_contrib_jax(l, tij, jnp.asarray(d.time), jnp.asarray(d.injection),
                                                             jnp.asarray(s))))
    contrib = np.asarray(cfun(jnp.asarray(lam[idx]), jnp.asarray(tau_ij[idx])))            # [Dm, N, J, I]
    contrib_ref = np.asarray(B.crm_contrib_jax(jnp.asarray(fit.params.lam), jnp.asarray(fit.params.tau_ij),
                                               jnp.asarray(d.time), jnp.asarray(d.injection), jnp.asarray(s)))
    hy = B.ResidualML("xgb", a.seed).fit(y, det, contrib_ref, d.injection, s, H)
    hyb = hy.predict_draws(mu[idx], contrib, d.injection, s)                                # [Dm, N, J]
    sh = np.array([hy.sigma[j] for j in range(J)])
    before = np.arange(N)[:, None] <= s[None, :]
    rng = np.random.default_rng(a.seed + 7)
    hyb_pred = np.where(before[None], hyb, np.maximum(hyb + sh[None, None, :] * rng.standard_normal(hyb.shape), 0.0))
    ph, phr = B.percentiles(hyb_pred), B.percentiles(hyb)
    k_hb, _ = B.best_fitting_draw(hyb, y, s, H)
    for per, rows in (("history", rows_hist), ("holdout", rows_fc)):
        tables += [B.table(y, ph["P50"], rows, d.producers, hyb_pred, per).assign(model="CRM-XGB predictive"),
                   B.table(y, phr["P50"], rows, d.producers, hyb, per).assign(model="CRM-XGB parameters only")]
    for j, pname in enumerate(d.producers):
        B.plot_producer(out / "plots" / f"CRM-XGB_{B.safe_name(pname)}.png", d.time, H, y[:, j], s[j],
                        {k: v[:, j] for k, v in ph.items()}, {k: v[:, j] for k, v in phr.items()}, hyb[k_hb][:, j],
                        f"{pname}: CRM-XGB rate forecast, \u03bb prior {label} ({lik} likelihood)", labels,
                        ref=pp["P50"][:, j], ref_label=f"CRM-only {labels['P50']}")
    log(f"CRM-XGB: {Dm} draws through CRM + XGB ({time.time() - t0:.0f}s)")

    # ---- outputs
    ts = []
    for j, pname in enumerate(d.producers):
        ts.append(pd.DataFrame({"producer": pname, "time": d.time,
                                "period": np.where(np.arange(N) < H, "history", "holdout"),
                                "observed": y[:, j], "crm_deterministic": det[:, j],
                                **{f"crm_{labels[k]}": pp[k][:, j] for k in ("P10", "P50", "P90")},
                                **{f"xgb_{labels[k]}": ph[k][:, j] for k in ("P10", "P50", "P90")}}))
    pd.concat(ts, ignore_index=True).to_csv(out / "percentile_timeseries.csv", index=False)
    res = pd.concat(tables, ignore_index=True)
    res.to_csv(out / "metrics.csv", index=False)
    summary = {"pattern": tag, "label": label, "prior": prior, "slack": slack, "likelihood": lik,
               "history_rows": H, "holdout_rows": N - H, "tau_max": tmax, "beta_a": a.beta_a if prior == "beta" else None,
               "beta_b": beta_b if prior == "beta" else None, "cap_sd": a.cap_sd if prior == "beta" and slack else None,
               "posterior_draws": D, "deterministic_objective": fit.objective,
               "max_rhat": float(diag.r_hat.max()), "min_ess_bulk": float(diag.ess_bulk.min()),
               "divergences": n_div, "converged": converged, "max_injector_lambda_sum": float(lam_sum.max()),
               "mean_slack": float((1 - lam_sum).mean()) if slack else None, "xgb_hyperparameters": {d.producers[j]: hy.chosen[j] for j in range(J)}}
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)

    field = res[(res.producer == "FIELD(pooled)") & res.model.isin(["CRM predictive", "CRM-XGB predictive"])].copy()
    field["model"] = field.model.str.replace(" predictive", "", regex=False)
    field = field.assign(pattern=tag, pattern_label=label, converged=converged)
    cols = [c for c in ["model", "period", "MAE", "RMSE", "PICP_80", "CRPS"] if c in field]
    print(field[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    return field, summary


# =============================================================================
# 5. SELF-TESTS (quick; python crm-ml-test.py --self-test)
# =============================================================================

def _t_tau_max_is_4H():
    assert tau_max_for(120) == 480.0


def _t_noslack_dirichlet_rows_sum_to_one():
    from numpyro.infer import Predictive
    t, w, q, _ = B._toy_case()
    s = B.producer_starts(q)
    model, *_ = make_model_v(t, w, s, q[s, np.arange(4)], q, "normal", 0.1, 120.0, "dirichlet", False)
    lam = np.asarray(Predictive(model, num_samples=100)(jax.random.PRNGKey(0))["lam_full"])
    assert lam.shape[-1] == 4 and np.allclose(lam.sum(-1), 1.0)


def _t_beta_with_slack_penalises_only_excess():
    from numpyro import handlers
    t, w, q, p = B._toy_case()
    s = B.producer_starts(q)
    model, *_ = make_model_v(t, w, s, q[s, np.arange(4)], q, "normal", 0.1, 120.0, "beta", True, cap_sd=0.02)
    vals = []
    for tot in (0.8, 1.2):
        init = init_values_v(p, 4, np.ones(4), 0.1, 120.0, "beta", True)
        init["lam"] = jnp.full((3, 4), tot / 4)
        tr = handlers.trace(handlers.substitute(model, data=init)).get_trace()
        vals.append(float(tr["mass_balance"]["fn"].log_factor))
    np.testing.assert_allclose(vals, [0.0, -0.5 * 3 * (0.2 / 0.02) ** 2], rtol=1e-9)


def _t_beta_without_slack_rows_sum_to_one_no_penalty():
    from numpyro import handlers
    from numpyro.infer import Predictive
    t, w, q, _ = B._toy_case()
    s = B.producer_starts(q)
    model, *_ = make_model_v(t, w, s, q[s, np.arange(4)], q, "normal", 0.1, 120.0, "beta", False)
    lam = np.asarray(Predictive(model, num_samples=100)(jax.random.PRNGKey(0))["lam"])
    assert lam.shape[-2:] == (3, 4) and np.allclose(lam.sum(-1), 1.0)
    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    assert "mass_balance" not in tr, "Beta without slack must not carry a penalty term"


def _t_det_fit_without_slack_sums_to_one():
    t, w, q, _ = B._toy_case()
    data = B.CRMData(t, q, w, [f"P{j}" for j in range(4)], [f"I{i}" for i in range(3)])
    fit = fit_deterministic_v(data, 0.1, tau_max_for(len(t)), slack=False, n_starts=1, max_iter=60)
    assert np.allclose(fit.params.lam.sum(1), 1.0, atol=1e-6)


def run_self_tests():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("_t_") and callable(v)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name[3:]}  ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name[3:]}: {type(e).__name__}: {e}", flush=True)
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed", flush=True)
    return failed == 0


# =============================================================================
# 6. COMMAND LINE / PATTERN RUNNER
# =============================================================================

def main():
    jax.config.update("jax_enable_x64", True)
    ap = B.build_parser("Compare lambda priors (Dirichlet / Beta, with and without slack) for CRM and CRM-XGB.")
    ap.add_argument("--base", default=None, help="path to crm-ml.py (default: next to this script)")
    ap.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS, choices=list(PATTERNS),
                    help="prior patterns to run in sequence")
    ap.add_argument("--no-slack", action="store_true",
                    help="run every selected pattern without slack (no slack term; sum_j lambda_ij = 1 exactly) "
                         "in calibration and sampling")
    ap.add_argument("--beta-a", type=float, default=1.0, help="Beta(a, b) prior a for each lambda_ij")
    ap.add_argument("--beta-b", type=float, default=None,
                    help="Beta(a, b) prior b; default J with slack, J-1 without slack (flat-Dirichlet marginal)")
    ap.add_argument("--cap-sd", type=float, default=0.02,
                    help="sd of the Gaussian penalty on sum_j lambda_ij > 1 (Beta, with slack only)")
    ap.add_argument("--units", default="MSTB/month", help="rate units shown on the comparison chart axes")
    ap.add_argument("--ml-draws", type=int, default=1000, help="posterior draws pushed through CRM + XGB")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached deterministic fits and posteriors")
    ap.add_argument("--compare-only", action="store_true",
                    help="skip all fitting; rebuild the comparison charts/tables from the pattern folders in --out")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_tests() else 1)

    configs = []
    for p in a.patterns:
        cfg = (PATTERNS[p]["prior"], PATTERNS[p]["slack"] and not a.no_slack)
        if cfg not in configs:
            configs.append(cfg)
    root = Path(a.out or "results_crm_ml_test")

    if a.compare_only:
        build_comparisons(root, configs, a.units)
        return
    if not a.data:
        ap.error("--data is required (or use --compare-only)")
    root.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(root / "run_settings.json", "w"), indent=1)

    d = B.load_excel(a.data)
    N, H = len(d.time), len(d.time) - a.forecast
    if H < 10:
        raise ValueError("forecast too long: fewer than 10 history rows left")
    log(f"N={N} history={H} holdout={N - H}; injectors={d.n_inj} producers={d.n_prod}; "
        f"tau_max = {TAU_MAX_FACTOR} x H = {tau_max_for(H):g}")

    log("patterns to run: " + ", ".join(PATTERN_LABELS[c] for c in configs))

    det_memo, summaries = {}, []
    for prior, slack in configs:
        try:
            _, sm = run_pattern(a, d, prior, slack, root, det_memo)
            summaries.append(sm)
        except Exception as e:  # noqa: BLE001 - keep going so the other patterns still finish
            log(f"pattern {prior}-{'slack' if slack else 'noslack'} FAILED: {type(e).__name__}: {e}")

    if not summaries:
        sys.exit("no pattern finished; nothing to compare")
    pd.DataFrame([{k: v for k, v in sm.items() if k != "xgb_hyperparameters"} for sm in summaries]
                 ).to_csv(root / "comparison_runs.csv", index=False)
    build_comparisons(root, configs, a.units)


if __name__ == "__main__":
    main()
