# **CRM-ML Hybrid Forecasting**

Bayesian production forecasting for CRM-based reservoir analytics, with an optional machine-learning residual layer for hybrid forecasting. The project calibrates CRM parameters from historical production and injection data, samples the posterior with NUTS, evaluates uncertainty bands, and can layer residual models such as XGBoost, ELM, MLP, and NuSVM on top of the CRM forecast.

## Features

- Deterministic CRM calibration using constrained least-squares on historical data
- Bayesian posterior sampling over CRM parameters with JAX + NumPyro
- Posterior predictive simulation with P10 / P50 / P90 uncertainty bands
- Metric reporting for MAE, RMSE, R2, interval coverage, interval score, and CRPS
- Optional CRM-ML hybrid forecasting using residual learners
- Plot generation and structured output files saved per run
- Built-in self-test suite for numerical and regression checks

## Prerequisites

Before installing the project, ensure the following are available:

- Python 3.10 or newer
- uv (Python package manager and runner)
- Git
- A workbook with at least the required sheets for time, observed production, and injection data

Recommended toolchain:

- Python: 3.10+ (tested with the project’s dependency set)
- uv: latest stable release
- Optional: Excel workbook with `time`, `q_observed`, and `w_injection` sheets

## Installation

1. Install Python and uv.

   On Windows (PowerShell):

   ```powershell
   winget install Python.Python.3.12
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Verify the installation:

   ```bash
   python --version
   uv --version
   ```

2. Clone the repository.

   ```bash
   git clone https://github.com/idioseph/crm-ml
   cd crm-ml
   ```
3. Install the project dependencies.

   ```bash
   uv sync
   ```

   This installs the Python dependencies declared in `pyproject.toml`, including NumPy, SciPy, pandas, JAX, NumPyro, scikit-learn, XGBoost, torch, and matplotlib.

4. Prepare your data file.

   Put your Excel workbook in the project directory or another accessible path, and ensure it contains the expected worksheets (for example, `time`, `q_observed`, and `w_injection`), you can check `buffalo_field.xlsx` as a refrence.

5. Run the project.

   ```bash
   uv run python crm.py --data buffalo_field.xlsx --forecast 12 --out results_crm
   uv run python crm-ml.py --data buffalo_field.xlsx --ml xgb elm mlp nusvm --out results_crm_ml
   ```

## Code Overview

This repository has two main analysis entry points:

- `crm.py`: CRM-only Bayesian UQ workflow
- `crm-ml.py`: CRM + ML Bayesian UQ workflow

Core project structure:

```text
crm-ml/
├── crm.py                 # CRM-only deterministic + Bayesian uncertainty workflow
├── crm-ml.py              # CRM + residual ML hybrid workflow
├── pyproject.toml         # Python dependencies and project metadata
├── README.md              # Project documentation
├── results_crm/            # CRM outputs (plots, metrics, posterior draws)
├── results_crm_ml/         # CRM-ML outputs (plots, metrics, posterior draws)
├── redundant/             # Extra or legacy scripts
└── DOC.md                 # Additional project notes and design documentation
```

The core flow is straightforward:

1. Load and validate time, production, and injection data
2. Fit CRM parameters deterministically
3. Sample the Bayesian posterior over CRM parameters
4. Propagate posterior draws through the CRM forward model
5. Compute predictive uncertainty and evaluation metrics
6. Optionally train a residual ML model and blend it with the CRM forecast

### Critical entry point: CRM-only runner

```python
# crm.py

def main():
    ap = build_parser("Bayesian UQ for CRM calibration (CRM only).")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if run_self_tests() else 1)
    if not a.data:
        ap.error("--data is required")
    a.out = a.out or "results_crm"
    if a.synthetic_check:
        return synthetic_check(a)
    run_pipeline(a)
```

### Critical entry point: CRM-ML runner

```python
# crm-ml.py

def run_pipeline(a, ml_models=("xgb",)):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_excel(a.data)
    fit = fit_deterministic(hist, a.tau_min, None, a.starts, a.max_iter, a.seed)
    grouped, info = run_nuts(model, inits, a.warmup, a.samples, a.seed)
    mu = propagate((tau_j, lam, tau_ij, sig), d.time, d.injection, s, q0)
    # Optional ML residual layer
    for name in ml_models:
        hy = ResidualML(name, a.seed).fit(y, mu_ref, contrib_ref, d.injection, s, H)
```

## Usage

Run the project locally with a workbook and either the CRM-only or CRM-ML workflow:

```bash
# CRM-only
uv run python crm.py --data buffalo_field.xlsx --forecast 12 --out results_crm

# CRM + ML hybrid
uv run python crm-ml.py --data buffalo_field.xlsx --ml xgb elm mlp nusvm --out results_crm_ml
```

### Shared command-line options for both scripts

These flags are available in both `crm.py` and `crm-ml.py`:

- `--data PATH`: Excel workbook containing the required sheets such as `time`, `q_observed`, and `w_injection`.
- `--forecast N`: Number of final rows held out from the fit as a validation/forecast window. These rows are not used for fitting or posterior calibration.
- `--tau-min VALUE`: Lower bound for the time-constant parameter `tau`, expressed in the workbook time units. The default is approximately 1 day.
- `--starts N`: Number of multi-start deterministic optimizations used to initialize the MCMC sampler.
- `--max-iter N`: Maximum SLSQP iterations per deterministic optimization run.
- `--likelihood MODE`: Observation model used in the Bayesian layer. Supported values are `normal`, `ar1`, and `lognormal`.
- `--chains N`: Number of MCMC chains to run.
- `--warmup N`: Number of warmup iterations to discard before sampling begins.
- `--samples N`: Number of posterior draws retained per chain.
- `--tau-prior-sd VALUE`: Standard deviation used in the prior for the log-`tau` parameters.
- `--alpha VALUE`: Dirichlet concentration parameter controlling the injector weighting structure in the CRM prior.
- `--p-convention MODE`: How uncertainty quantiles are labeled. Use `percentile` for P10/P50/P90 or `reserves` for reserve-style conventions.
- `--seed N`: Random seed controlling initialization, sampling, and the ML-only stochastic components.
- `--out DIR`: Output folder where CSVs, JSON summary files, and plots are saved.
- `--self-test`: Runs the built-in regression checks and exits without performing the main forecast run.

### CRM-only script: `crm.py`

This script runs the Bayesian CRM workflow without any machine-learning augmentation.

Example commands:

```bash
# Standard CRM run
uv run python crm.py --data buffalo_field.xlsx --forecast 12 --out results_crm

# Use AR(1) residual structure
uv run python crm.py --data buffalo_field.xlsx --forecast 12 --likelihood ar1 --warmup 2000 --samples 2000 --out results_crm

# Run synthetic recovery check
uv run python crm.py --data buffalo_field.xlsx --synthetic-check --reps 3 --noise 1.0

# Run internal self-tests only
uv run python crm.py --self-test
```

Additional CRM-only options:

- `--synthetic-check`: Generates synthetic CRM data from a known parameterization and attempts to recover those parameters.
- `--reps N`: Number of synthetic-recovery repetitions to run in the synthetic check.
- `--noise VALUE`: Observation noise level used in the synthetic-generation check.

These options are useful for validating the CRM model’s recovery behavior and checking that parameter estimation remains numerically stable.

### CRM + ML script: `crm-ml.py`

This script runs the same Bayesian CRM layer but adds a residual machine-learning model on top of the CRM forecast.

Example commands:

```bash
# Single ML model
uv run python crm-ml.py --data buffalo_field.xlsx --forecast 12 --ml xgb --out results_crm_ml

# Multiple ML models
uv run python crm-ml.py --data buffalo_field.xlsx --forecast 12 --ml xgb elm mlp nusvm --out results_crm_ml

# Heavier sampling run with custom MCMC settings
uv run python crm-ml.py --data buffalo_field.xlsx --forecast 12 --likelihood ar1 --chains 8 --warmup 2000 --samples 2000 --ml xgb elm --out results_crm_ml

# Self-test
uv run python crm-ml.py --self-test
```

Additional CRM-ML options:

- `--ml MODEL [MODEL ...]`: One or more residual models to stack above the CRM forecast. Supported options are `xgb`, `elm`, `mlp`, and `nusvm`.
- `--ml-draws N`: Number of posterior draws forwarded through the CRM + ML combination. If fewer draws are available than requested, a random subset is used.

Each selected model produces its own predictive results, plots, and summary entries in the output folder.

Outputs are stored under the run-specific result folders and include posterior samples, diagnostics, summary metrics, and producer-level plot files.

---

This project is intended for scientific and forecasting workflows where uncertainty quantification is as important as the point estimate. It is best suited for operators working with historical injection/production time series and wanting interpretable probabilistic forecasts rather than opaque black-box predictions.