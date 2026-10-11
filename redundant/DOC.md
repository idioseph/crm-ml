# BAYESIAN UQ CRM-ML

## Production Forecasting with Uncertainty Quantification

**Python Implementation: Technical Documentation (CLI and Streamlit GUI)**

**Scope:** Capacitance Resistance Model (CRM), machine-learning CRM (CRM-ML), Hamiltonian Monte Carlo (HMC) inference, posterior-predictive forecasts, validation tools, command-line interface and web GUI.

---

# 1. Purpose of the Document

This document explains the supplied Python Code implementation of Bayesian uncertainty-quantified CRM-ML Model for production forecasting. With emphasis on the flow of the program: what enters the program, what each component does, why it is there, and how the final production forecasts and uncertainty intervals are produced. A dedicated chapter describes the Streamlit graphical interface, which wraps the same computational core as the command line.

The whole implementation lives in one script. It can be run three ways: as a command-line tool (`main()`), as a browser GUI (`run_gui()` under Streamlit), or imported as a library (for example `load_field_data` followed by `run_bayesian_uq`). The script header uses inline script metadata, so it can be launched directly with `uv run` and its dependencies are resolved automatically.

# 2. The Problem and How the Program Addresses It

In a waterflood, injectors push fluid into the reservoir and producers respond with a delay that depends on the connectivity and storage between each injector-producer pair. The Capacitance Resistance Model (CRM) is a compact physics-based model of that response: each producer rate is the sum of a decaying contribution from its initial rate and contributions from every injector, weighted by connectivity coefficients (λ) and time constants (τ).

A pure CRM is easy to interpret but can miss behaviour it was not designed for. THe CRM-ML therefore feeds the CRM response (together with time and injection rates) into a machine-learning regressor trained to predict the observed production. This implementation goes one step further. It treats the CRM parameters as uncertain, samples their posterior with HMC, pushes every posterior draw through the CRM-ML pipeline, and adds a calibrated residual-noise term. The result is a forecast with a median (P50), P10/P90 bounds, a 95% interval, and a split of the uncertainty into epistemic and aleatoric parts.

In summary, the program: (1) reads a workbook of historical rates, (2) holds out the last N time steps as a forecast window, (3) calibrates a deterministic CRM, (4) samples a Bayesian CRM, (5) trains and applies a per-producer ML model for each posterior draw, (6) calibrates observation noise on a leak-free chronological hold-out, (7) forms posterior-predictive distributions, and (8) scores them against the held-out observations.


# 3. Complete Processing Flow

| Stage | What happens | Main functions / classes |
|---|---|---|
| 1. Read data | Workbook sheets are located by alias, parsed into numeric blocks, units are extracted and the result is validated. | `load_field_data()`, `_parse_block()`, `validate_field()` |
| 2. Split | The last `n_forecast` rows are held out. Everything before is history. | `run_bayesian_uq()` |
| 3. Deterministic CRM | A constrained least-squares CRM is fitted to history with SLSQP and several restarts. It provides the starting point and the baseline forecast. | `CapacitanceResistanceModel.fit()` |
| 4. Residual correlation | Lag-1 autocorrelation of CRM log-residuals sets a likelihood tempering weight per producer. | `_lag1_autocorr()` |
| 5. Bayesian CRM | HMC samples log τ, softmax connectivity logits and observation sigma. Several chains are run from dispersed starts. | `_make_crm_posterior()`, `_hmc_sample()` |
| 6. Diagnostics | Split R-hat and bulk ESS are computed for every parameter. | `_split_rhat()`, `_ess_bulk()`, `_diagnostics()` |
| 7. Baseline forecasts | CRM-ML is run once with the deterministic CRM and once with posterior-mean parameters. | `_crm_ml_forecast()` |
| 8. Noise calibration | The CRM is refitted on the first 75% of history; ML is trained there and scored on the rest. RMS log residual becomes aleatoric sigma. | `_oos_ml_calibration()` |
| 9. Propagation | Posterior CRM draws are passed through CRM-ML (ML refitted per draw by default) to give forecast draws. | loop in `run_bayesian_uq()` |
| 10. Predictive UQ | Lognormal noise is added to each draw; quantiles, interval metrics and uncertainty split are computed. | `_forecast_tables()` |
| 11. Report and export | Tables are printed and written to CSV/NPZ/PNG. | `_print_report()`, `export_result()`, `plot_*()` |
| 12. Optional validation | Rolling-origin backtest and a known-truth synthetic recovery test. | `rolling_origin_backtest()`, `synthetic_recovery_check()` |

# 4. Execution Order

## 4.1 Command-line run

`main()` acts as the coordinator. It does not compute anything itself; it calls the functions responsible for loading, inference, reporting and export.

1. `argparse` reads the settings. `--write-template`, `--gui` and `--synthetic-check` are handled first and exit early.
2. If `--data` is missing a `ValueError` is raised with instructions to use `--write-template`.
3. `load_field_data()` reads and validates the workbook; any importer notes are printed.
4. The shared keyword arguments (`common`) are assembled from the command-line options.
5. `run_bayesian_uq()` performs the full inference pipeline described in Sections 9 to 14.
6. `_print_report()` prints baseline metrics, Bayesian metrics, HMC acceptance and R-hat/ESS.
7. `export_result()` writes tables, MCMC samples and four figures to the output folder.
8. If `--backtest N` is greater than zero, `rolling_origin_backtest()` repeats the pipeline at N origins and writes per-origin and pooled tables.
9. The number of files and the output location are printed.

## 4.2 Inside run_bayesian_uq()

1. Validate arguments (model name, forecast length, MCMC sizes, sigma growth, draw count).
2. If the workbook carries a `q_CRM` sheet, branch to `_run_precomputed_crm_uq()` and return.
3. Split into history and forecast; require at least 20 history rows.
4. Build a per-producer mask of strictly positive rows; require at least 8 positive rows per producer.
5. Fit the deterministic CRM and record the production scale.
6. Compute lag-1 tempering weights from the deterministic residuals.
7. Build the log-posterior, find a finite-density start for each chain, and run HMC.
8. Compute posterior-mean CRM parameters and the two baseline CRM-ML forecasts.
9. Calibrate aleatoric sigma with the leak-free hold-out procedure.
10. Subsample posterior draws, propagate each through CRM-ML, add noise, and tabulate results.
11. Assemble the `BayesianUQResult` and emit warnings if R-hat > 1.10 or minimum ESS < 100.

## 4.3 GUI run

Under `streamlit run` the same file is executed as a Streamlit script. The module-level dispatcher detects the Streamlit runtime and calls `run_gui()` instead of `main()`. The GUI loads the uploaded workbook, shows the production overview, and only runs inference when the Run button is pressed. Results are stored in session state and displayed across seven tabs.

# 5. Imported Libraries

| Library | Purpose in this implementation |
|---|---|
| argparse, sys, io, tempfile, warnings, math | Command-line parsing, Streamlit launch, in-memory workbooks, temporary export folders, runtime warnings, basic maths. |
| dataclasses, typing, pathlib | `FieldData`, `CRMParameters` and `BayesianUQResult` containers; type hints; file paths. |
| NumPy | Array maths, random generators, quantiles, FFT-based autocorrelation, parameter reshaping. |
| Pandas | Excel parsing, result tables and CSV export. |
| SciPy | `minimize` (SLSQP) for deterministic CRM calibration; `lfilter` for the exponential filter. |
| scikit-learn | `NuSVR`, `MLPRegressor`, `Pipeline`, `StandardScaler`, `TransformedTargetRegressor`, and the base classes used by the custom ELM. |
| XGBoost | `XGBRegressor`, the default CRM-ML model. The import fails with an explicit message if it is missing. |
| PyTorch | Differentiable CRM simulator and automatic gradients for HMC (float64). |
| Matplotlib (Agg backend) | All figures. The non-interactive backend lets figures be saved on servers and inside Streamlit. |
| openpyxl | Excel engine used by pandas for reading and writing the template. |
| Streamlit | Browser GUI. It is imported lazily inside `run_gui()` and `main()` so the library and CLI work without it being active. |

# 6. Configuration and Constants

## 6.1 Module-level constants

| Constant | Value | Meaning |
|---|---|---|
| `CRM_ML_MODELS` | CRM-XGB, CRM-NuSVM, CRM-ELM, CRM-MLP | Selectable ML corrections. The text after the hyphen selects the regressor. |
| `DEFAULT_CRM_ML_MODEL` | CRM-XGB | Default method. |
| `PAPER_MLP_HIDDEN_NEURONS` | 10 | Hidden size of the MLP (and of the ELM). |
| `PAPER_TRAIN_FRACTION` / `PAPER_VALIDATION_FRACTION` | 0.75 / 0.25 | Reference split fractions. The UQ calibration uses `UQ_CALIBRATION_FRACTION`. |
| `PAPER_MIN_TAU_SAMPLING_INTERVAL` | True | Minimum τ equals the smallest sampling interval. |
| `MCMC_DEFAULT_CHAINS` | 3 | Number of chains. |
| `MCMC_DEFAULT_SAMPLES` | 1500 | Retained draws per chain. |
| `MCMC_DEFAULT_BURN_IN` | 1000 | Burn-in iterations (also where step size adapts). |
| `MCMC_DEFAULT_THIN` | 2 | Thinning interval. |
| `MCMC_DEFAULT_SEED` | 2025 | Base random seed. |
| `MCMC_TARGET_ACCEPTANCE` | 0.65 | Target acceptance rate for step-size adaptation. |
| `UQ_CI_LEVEL` | 0.95 | Interval level (the 2.5% and 97.5% quantiles are used in code). |
| `UQ_RHAT_WARNING` | 1.10 | R-hat threshold for the warning. |
| `UQ_MIN_ESS_WARNING` | 100 | Minimum bulk ESS before a warning. |
| `UQ_CALIBRATION_FRACTION` | 0.75 | Share of history used to train before scoring the hold-out block. |
| `UQ_MAX_CRM_PREDICTIVE_DRAWS` | 200 | Maximum posterior draws propagated through CRM-ML. |
| `UQ_TAU_LOG_SD` | 2.0 | Prior sd of log τ around a data-scale reference. |
| `UQ_LAMBDA_LOGIT_SD` | 1.5 | Prior sd of the softmax-with-slack logits. |
| `DEFAULT_RATE_UNIT` / `DEFAULT_TIME_UNIT` | MSTB/day / days | Used when the workbook has no unit row. |

## 6.2 ML_SETTINGS (implementation choices)

| Model | Settings |
|---|---|
| NuSVM | RBF kernel, nu = 0.5, C = 10, gamma = "scale". |
| XGB | 300 trees, max depth 4, learning rate 0.05, n_jobs = 1, silent. Single-threaded so repeated refits stay deterministic and cheap. |
| ELM | Ridge penalty 1e-3, 10 hidden units. |
| MLP | One hidden layer of 10 neurons, tanh activation, L-BFGS solver, max 500 iterations. |

Every regressor is wrapped by `make_ml_regressor()` in a `TransformedTargetRegressor` containing a `StandardScaler` + model pipeline, with the target also standardised. The target passed to it is always `log(q)` (Section 10).

# 7. Data Structures and Metrics

## 7.1 FieldData

`FieldData` is the container for everything the importer produces. Rows are time steps; columns are wells.

| Field | Meaning |
|---|---|
| `time` | 1-D strictly increasing time vector. |
| `production` | 2-D observed producer rates [time × producers]. |
| `injection` | 2-D injector rates [time × injectors]. May have zero columns. |
| `distances` | Injector-producer distance matrix [injectors × producers]. Carried and validated, but not used by the current Bayesian pipeline. |
| `producer_names`, `injector_names` | Well labels from the workbook headers. |
| `bhp` | Optional bottom-hole pressure block. Validated for finiteness; not used by the model. |
| `time_unit`, `rate_unit` | Units read from the unit row; used on plot axes. |
| `notes` | Importer messages (missing injectors, default distances, detected q_CRM). |
| `crm_baseline` | Optional pre-computed CRM response. Its presence switches the pipeline to the pre-computed mode. |

`first_active_index()` returns the first positive-rate row per producer, and `summary()` turns that into a table (first and last positive step, number of positive steps, peak rate). The GUI shows this table as the input data summary. `_truncate_field()` returns a copy limited to the first n rows; the rolling backtest uses it to move the forecast origin backwards.

## 7.2 BayesianUQResult

| Field | Meaning |
|---|---|
| `model` | Method label. Pre-computed runs append " [precomputed CRM]". |
| `forecast` | Long table of forecast rows. |
| `metrics`, `baseline_metrics` | Per-producer Bayesian and deterministic metric tables. |
| `posterior_summary` | Mean, median, sd and 2.5%/97.5% quantiles of every parameter, plus the ML out-of-sample sigma. |
| `chain_samples` | Array [chains × draws × parameters]. |
| `rhat` | Table of parameter, R_hat and ESS_bulk. |
| `acceptance_rates` | Post burn-in HMC acceptance per chain. |
| `posterior_mean_params` | Posterior-mean `CRMParameters` (placeholder values in pre-computed mode). |
| `ml_sigma`, `ml_bias` | Out-of-sample RMS log residual and mean log residual per producer. |
| `calibration_predictions` | The hold-out predictions behind sigma. |
| `historical_production` | Full time and production table for plotting. |
| `settings` | Run options recorded for reproducibility. |
| `calibration_fraction`, `mcmc_chains`, `mcmc_samples`, `burn_in`, `thin`, `seed` | The sampler configuration. |

## 7.3 Point metrics

`mean_absolute_error()`, `root_mean_squared_error()` and `r_squared()` validate that inputs have equal non-empty shapes. R² is 1 - SSE/SST and returns NaN when the observed values are constant.

# 8. Excel Import and Validation

## 8.1 Workbook layout

| Sheet (accepted aliases) | Required | Layout |
|---|---|---|
| Time (`time`, `t`, `date`, `days`, `months`) | Yes | Header cell, optional unit row such as [days], then one numeric value per row. |
| q_Observed (`q_observed`, `q_obs`, `production`, `observed`, `q`) | Yes | Row of producer names, optional unit row [MSTB/day], then rates. |
| w_Injection (`w_injection`, `injection`, `w_inj`, `w`, `inj`) | No | Row of injector names, optional unit row, then rates. If absent the model has no injectors. |
| Distances (`distances`, `distance`, `x_ij`, `dist`) | No | Injectors as rows, producers as columns. An optional leading label column and header rows are tolerated. |
| BHP (`bhp`, `pwf`, `p_wf`) | No | Same block style as production. |
| q_CRM (`q_crm`, `crm`, `crm_prediction`, `q_crm_prediction`) | No | Pre-computed CRM response with the same shape and producer columns as q_Observed. |


## 8.2 _parse_block()

Drops empty rows, treats the first row as names, converts the body to numbers, and finds the first row where every cell is numeric. Rows between the header and that row (for example the unit row) are kept as notes. The numeric part becomes the data and the first row supplies the column names.

## 8.3 _extract_unit()

Scans the notes for a cell wrapped in square brackets, such as [MSTB/day], and returns the text inside. If none is found, the defaults apply.

## 8.4 load_field_data()

1. Seek to the start when a file-like object is passed, open with `pd.ExcelFile`, and resolve each sheet by alias.
2. Raise a clear error if Time or q_Observed is missing, listing the sheets that are present.
3. Parse Time, q_Observed and (if present) w_Injection. With no injection sheet, a zero-column injection block and an empty distance matrix are created and a note is added.
4. Parse Distances separately: take the first row with any numeric value as the start, drop a label column when the width is producers + 1, otherwise keep the last producers-many columns.
5. If injectors exist but no Distances sheet is supplied, distances default to 1 and a note records it. A wrong distance shape raises an error that explains the expected orientation.
6. If q_CRM is present, its shape and producer names must match q_Observed exactly; a note records that the precomputed baseline will be used.
7. Build `FieldData`, read units, and call `validate_field()`.

## 8.5 validate_field()

Checks dimensions (time 1-D; others 2-D), equal row counts, distances shape [injectors × producers], name counts, finite and strictly increasing time, and absence of NaN/inf in production, injection, distances and BHP. Negative injection only triggers a warning, because the physical meaning should be reviewed rather than rejected.

## 8.6 load_from_bytes() and write_template()

`load_from_bytes()` wraps uploaded bytes in `BytesIO` for the GUI. `write_template()` writes a workbook that exactly matches the importer layout (default 36 steps, 3 injectors, 2 producers, random placeholder values, units [days], [MSTB/day] and [ft]). It is used by `--write-template` and by the GUI download button.

# 9. Capacitance Resistance Model

## 9.1 Model equation

```
q_j(t) = q_j(0) · exp(-(t - t0) / τ_j)  +  Σ_i λ_ij · F_ij(t)
F_ij(t_n) = a_n · F_ij(t_{n-1}) + (1 - a_n) · w_i(t_n),   a_n = exp(-Δt_n / τ_ij)
```

The first term is the primary-depletion decay of the initial rate with time constant τ_j. The second term sums exponentially filtered injection from each injector, scaled by the connectivity λ_ij. Predictions are produced for `time[1:]`.

## 9.2 CRMParameters

A small container for `tau_j` (per producer), `lambda_ij` and `tau_ij` (both injectors × producers).

## 9.3 _exp_filter()

Applies the recursive filter above. For a uniform time step it uses SciPy `lfilter` (fast); otherwise it loops with a per-step coefficient. It rejects non-positive τ or step sizes.

## 9.4 crm_simulate()

Validates dimensions, forms the decaying initial-rate term, then loops over every injector-producer pair adding λ_ij times the filtered injection. Because it accepts the whole time vector, the same function simulates history and forecast in one pass, so the CRM state at the forecast boundary is conditioned on the observed history.

## 9.5 CapacitanceResistanceModel

Deterministic calibration by constrained least squares.

- `tau_min` defaults to the smallest sampling interval; `tau_max` is optional (none by default).
- Production and injection are divided by the mean absolute production so the optimiser works near unit scale. The scale is stored in `_scale` and reused as the production scale for the Bayesian stage.
- Parameters are packed as [log τ_j, λ_ij, log τ_ij]. λ is bounded in [0, 1] and each injector row must satisfy Σ_j λ_ij ≤ 1 through an SLSQP inequality constraint (skipped when there are no injectors).
- The objective is the mean squared error of scaled rates. The first start uses τ = 10 × minimum interval and λ = 0.8 / number of producers; further starts use random log τ and Dirichlet-sampled λ (scaled by 0.9).
- The best of `n_starts` results is kept. Fitting needs at least 10 time steps. `predict()` rescales back to field units.

# 10. CRM-ML Models

## 10.1 ExtremeLearningMachine

A single-hidden-layer ELM: random input weights and biases in [-1, 1], sigmoid activation, and output weights solved in closed form with ridge regularisation. It follows the scikit-learn estimator interface so it fits the same pipeline as the other models.

## 10.2 make_ml_regressor()

Creates the selected base model (XGB, NuSVM, ELM or MLP) with the settings from Section 6.2 and the given seed, and wraps it in the scaled pipeline and scaled target.

## 10.3 build_crm_ml_features()

Returns columns [time (optional), every injection rate, CRM response]. The docstring explains an important design decision: per-producer constants (distances, λ, τ) are deliberately excluded. They have zero variance within a producer-specific model so carry no information, and substituting posterior draws into them produced spurious shifts in kernel and neural models. The time-varying CRM response carries the physics and changes with every posterior draw.

## 10.4 fit_positive_crm_ml() and positive_prediction()

Models are trained on `log(q)`, which requires strictly positive targets, and predictions are exponentiated after clipping the log value to [-40, 40]. This guarantees positive, finite forecasts and matches the Lognormal likelihood used elsewhere.

> **Note.** One ML model is trained per producer. Zero-rate (shut-in) rows are excluded from training because their logarithm is undefined; they remain in the data and in point metrics.

# 11. Bayesian CRM

## 11.1 Parameters

| Quantity | Sampled as | Why |
|---|---|---|
| τ_j, τ_ij | log τ, hard lower bound at log(τ_min) | Keeps time constants positive and at least one sampling interval. |
| λ_ij | Softmax logits with an implicit slack category | Each injector's connectivities are non-negative and sum to at most 1, with the leftover going to slack (loss). |
| σ_j | Marginalised analytically, then drawn from its conditional | Removes a difficult parameter from HMC. |

`_logit_slack_from_lambda()` and `_lambda_from_logit_slack()` convert between λ and logits; `_crm_theta_from_params()` and `_crm_params_from_theta()` do the same for the full parameter vector. The sampled vector has length P + 2·I·P (P producers, I injectors), and the stored chain appends P sigma columns.

## 11.2 Differentiable simulator

`_crm_simulate_torch()` reproduces the CRM in PyTorch so gradients are available to HMC. The injection response is computed in log space: response_n = Σ_{k≤n} (1 - a_k)·s_k·exp(L_n - L_k) with L = -cumsum(Δt/τ). Every exponent is non-positive, so nothing overflows. When N²·I·P is at most 3×10⁷ the all-pairs matrix form is used; longer histories fall back to a memory-safe sequential recursion.

## 11.3 Likelihood and priors

`_make_crm_posterior()` returns the log-posterior and a sigma sampler.

```
log y_j ∼ Normal(log q_CRM,j , σ_j)       on strictly positive rows
σ_j² ∼ InvGamma(a0 = 2, b0 = 2·0.20²)    (marginalised)
log-lik_j = -(a0 + ½·w_j·n_j) · log(b0 + ½·w_j·SSE_j)
log τ ∼ Normal(log τ_ref, 2.0²),   τ_ref = sqrt(τ_min · span)
logits ∼ Normal(0, 1.5²)
```

The weight w_j = (1 - ρ_j)/(1 + ρ_j) tempers the likelihood to account for serial correlation in CRM residuals (an effective-sample-size correction). ρ_j is the lag-1 autocorrelation of the deterministic CRM log-residuals, clipped to [0, 0.9], and w_j is floored at 0.05. A parameter set that violates τ ≥ τ_min or gives non-finite rates returns -∞.

Given a parameter draw, `sigma_draw()` samples σ_j from its inverse-gamma conditional, so each retained draw contains the full parameter vector plus the noise levels.

## 11.4 _hmc_sample()

A generic HMC sampler with an identity mass matrix.

- Total iterations are burn-in + samples × thin. Inputs below 100 samples or burn-in raise an error.
- Each iteration draws momentum, runs the configured number of leapfrog steps, and rejects immediately if the density becomes non-finite. Acceptance uses the Hamiltonian difference with a momentum flip.
- During burn-in the step size is adjusted every 50 iterations: ε ← ε·exp(2·(rate - 0.65)), clipped to [0.002, 0.5]. After burn-in it is fixed.
- Every `thin`-th post-burn-in state is stored; `extra_fn` appends the sigma draw. The function returns draws, the post burn-in acceptance rate and the final step size.

## 11.5 Chain initialisation

Each chain starts from the deterministic CRM solution plus Gaussian jitter (sd 0.5, shrinking by a factor 0.7 per retry, up to 20 attempts) until the log-posterior is finite. τ coordinates are kept above log(τ_min). Chain c uses seed + 100000·c. The results are stacked into [chains × draws × parameters].

## 11.6 Posterior summaries

`_posterior_crm_arrays()` converts raw chains back to τ_j, λ, τ_ij and σ. `_posterior_mean_crm_params()` averages them, and `_crm_posterior_summary()` builds the table of mean, median, sd and 95% quantiles (labels such as `I-01->P-01::lambda`). `_theta_names()` supplies the matching names for diagnostics.

# 12. MCMC Diagnostics

- `_split_rhat()`: each chain is split in half, giving 2m sequences, and the standard potential-scale-reduction statistic is computed. Needs at least two chains and four draws, otherwise NaN.
- `_ess_bulk()`: multi-chain effective sample size from FFT autocorrelation with Geyer's initial positive sequence truncation. Needs at least 8 draws; floors at 1.
- `_diagnostics()` joins the two into a table with parameter, R_hat and ESS_bulk.
- `_lag1_autocorr()`: lag-1 autocorrelation of residuals, used for the tempering weight (not an MCMC diagnostic).
At the end of `run_bayesian_uq()`, a `RuntimeWarning` is issued if the largest R-hat exceeds 1.10 or the smallest ESS is below 100. The GUI shows the same thresholds in a banner.

# 13. Forecasting and Uncertainty Quantification

## 13.1 _crm_ml_forecast()

Given a CRM parameter set, simulates the CRM over the whole time axis, then for each producer builds training features from history rows (CRM response aligned to `time[1:n_hist]`) and forecast features from the forecast rows. It trains the ML model (or reuses supplied models) and predicts the forecast window. At least 8 positive history rows per producer are required. Returns predictions and the fitted models.

## 13.2 Baseline forecasts

- Deterministic baseline: CRM-ML using the SLSQP parameters. This is the reference the Bayesian result is compared against.
- Posterior-mean forecast: CRM-ML using averaged posterior parameters; its models can be reused when per-draw refitting is turned off.

## 13.3 Aleatoric calibration

`_oos_ml_calibration()` implements a leak-free chronological hold-out.

1. Take the first 75% of history (at least 10 rows) as the training block.
2. Refit the CRM on that block only, then simulate it across all history.
3. For each producer train ML on positive training-block rows and predict the remaining positive rows.
4. Compute log residuals log(observed) - log(prediction); the RMS (which includes bias) is the aleatoric sigma, and the mean is reported as bias.
5. Each producer needs at least 8 positive rows in both blocks, otherwise an informative error is raised.
Because neither the CRM nor the ML model has seen the scoring rows, the sigma reflects genuine out-of-sample error rather than in-sample fit.

## 13.4 Propagation and predictive distribution

- Up to `predictive_draws` posterior draws (default 200) are chosen at random from the pooled chains.
- Each draw is converted to CRM parameters and pushed through `_crm_ml_forecast()`. With `refit_ml_per_draw=True` (default) the ML model is retrained for every draw, which captures how parameter uncertainty changes the learned correction. `--no-refit` reuses the posterior-mean models, which is faster but understates epistemic uncertainty.
- Epistemic sd = standard deviation across draws of log forecasts.
- Aleatoric sd = ml_sigma × sqrt(1 + sigma_growth × lead). The default growth is 0; a positive value widens intervals with lead time as a sensitivity option.
- Predictive samples = draw × exp(ε·aleatoric sd), ε ∼ N(0, 1). The code checks that all samples are finite and positive.

## 13.5 Forecast table and metrics

`_forecast_tables()` computes the quantiles and builds two tables. The forecast table has one row per time step and producer.

| Column | Meaning |
|---|---|
| `time`, `producer`, `observed`, `observed_is_zero` | Identification, held-out value, and shut-in flag. |
| `crm_ml_deterministic`, `crm_ml_posterior_mean` | Baseline and posterior-mean forecasts. |
| `P90_low`, `P50`, `P10_high` | 10th, 50th and 90th percentiles. Naming follows the petroleum convention where P90 is the low case and P10 the high case. |
| `95pct_lower`, `95pct_upper` | 2.5% and 97.5% predictive quantiles. |
| `deterministic_error` / `_abs_error` | Baseline minus observed. |
| `bayesian_error` / `_abs_error` | P50 minus observed. |
| `interval_width_95`, `covered_95` | Interval width and whether the observation lies inside. |
| `epistemic_log_sd`, `aleatoric_log_sd` | The two uncertainty components on the log scale. |

| Metric | Meaning |
|---|---|
| `MAE_P50`, `RMSE_P50`, `R2_P50` | Point accuracy of the posterior median. |
| `PICP_95` | Share of positive observations inside the 95% interval. |
| `MPIW_95` | Mean 95% prediction-interval width. |
| `interval_score_95` | Interval width plus a (2/α) penalty for misses, α = 0.05; lower is better. Positive observations only. |
| `n_zero_observed` | Count of zero-rate observations excluded from coverage metrics. |
| `mean_epistemic_log_sd`, `mean_aleatoric_log_sd` | Average uncertainty components. |
| `MAE_improvement_vs_deterministic` | Deterministic MAE minus Bayesian P50 MAE. Positive means the Bayesian median is better. |

`_deterministic_forecast_metrics()` produces the equivalent MAE, RMSE and R² table for the baseline. `run_bayesian_uq()` also appends the ML out-of-sample sigma to the posterior summary as `::sigma_ML_oos`.

# 14. Pre-computed q_CRM Mode

When the workbook contains a q_CRM sheet, `run_bayesian_uq()` delegates to `_run_precomputed_crm_uq()`. The CRM is then a fixed, externally supplied baseline, and the code does not invent injector or connectivity parameters.

- Requirements: at least 20 history rows and at least 16 rows per producer with positive observed and positive q_CRM values; q_CRM may not be negative or non-finite.
- The ML model is trained on [time (optional), q_CRM] per producer to give the deterministic baseline.
- A chronological hold-out (75% of active rows for training, at least 8 for scoring) yields out-of-sample log predictions, sigma and bias.
- HMC then calibrates log q_obs ∼ Normal(α + β·log q_ml, σ_j), with priors α ∼ N(0, 1), β ∼ N(1, 0.5²) and log σ_j ∼ N(log 0.2, 1.5²). It uses step size 0.02 and 8 leapfrog steps; chains start near (0, 1, log σ).
- Predictive draws use posterior (α, β, σ_j). Epistemic uncertainty therefore reflects calibration-parameter uncertainty only, since the CRM itself is fixed.
The result is labelled "[precomputed CRM]", the posterior summary lists alpha, beta and log_sigma entries, and `posterior_mean_params` holds placeholders. The GUI adapts accordingly: MCMC trace plots show α, β and a log-sigma term, and CRM parameter matrices are hidden when no λ values exist.

# 15. Validation Utilities

## 15.1 rolling_origin_backtest()

Repeats the whole pipeline at several forecast origins by truncating the data in steps of the forecast length (each origin needs at least 20 history rows). Returns per-origin metrics and pooled results per producer: number of observations, PICP_95, MPIW_95 and MAE_P50 over every positive held-out point. Pooled coverage is far more informative than the few points in one split. Enabled with `--backtest N`.

## 15.2 synthetic_recovery_check()

Simulates a two-injector, two-producer field from known parameters with Lognormal noise, runs the pipeline with a short chain configuration, and reports whether the true λ and τ_ij fall inside the 95% posterior intervals, together with forecast metrics and diagnostics. The docstring cautions that λ and τ can be weakly identified, so a miss is informative rather than necessarily a bug. Enabled with `--synthetic-check`.

# 16. Figures and Exports

## 16.1 Plot functions

| Function | Shows |
|---|---|
| `plot_production_history()` | Observed rates for all producers with the forecast boundary (used in the GUI overview). |
| `plot_producer_forecast()` | One producer: observed, deterministic CRM-ML, Bayesian P50 and the 95% band. |
| `plot_forecast_test()` | All producers on one axis for the held-out test. |
| `plot_ml_calibration()` | Chronological out-of-sample log residuals used for aleatoric uncertainty. |
| `plot_uq_forecast()` | Observed, P50 and 95% interval for all producers. |
| `plot_mcmc_diagnostics()` | Trace plots. Standard mode: first τ_j, first λ logit and the last sigma column. Pre-computed mode: α, β and the first log-sigma. |

## 16.2 export_result()

Creates the folder and writes the files below. The CLI calls it once; the GUI calls it inside a temporary folder when building the ZIP.

| File | Content |
|---|---|
| uq_forecast.csv | Full forecast table. |
| uq_metrics.csv / deterministic_baseline_metrics.csv | Bayesian and baseline metrics. |
| uq_posterior_summary.csv | Posterior summaries. |
| uq_rhat_ess.csv / uq_acceptance_rates.csv | MCMC diagnostics. |
| uq_ml_aleatoric_sigma.csv | Out-of-sample sigma and bias per producer. |
| uq_ml_calibration_predictions.csv | Hold-out predictions and log residuals. |
| historical_production.csv | Time and production history. |
| run_settings.csv | Recorded run options. |
| posterior_mean_lambda.csv, posterior_mean_tau_ij.csv, posterior_mean_tau_j.csv | Posterior-mean CRM parameters. |
| uq_mcmc_samples.npz | Compressed chain samples. |
| uq_forecast.png, uq_mcmc_diagnostics.png, uq_forecast_test_with_interval.png, uq_ml_calibration_residuals.png | Figures saved at 200 dpi. |
| backtest_per_origin.csv, backtest_pooled.csv | Only with `--backtest`. |

`_print_report()` prints the method, MCMC setup, settings, baseline and Bayesian metrics, acceptance rates and R-hat/ESS tables to the console.

# 17. Command-Line Arguments

| Argument | Purpose (default) |
|---|---|
| `--data` | Excel workbook. Required unless using the template, GUI or synthetic check. |
| `--forecast` | Final time steps treated as forecast (12). |
| `--model` | One of CRM-XGB, CRM-NuSVM, CRM-ELM, CRM-MLP (CRM-XGB). |
| `--crm-starts` / `--crm-max-iter` | SLSQP restarts and iteration cap (3 / 300). |
| `--mcmc-chains`, `--mcmc-samples`, `--mcmc-burnin`, `--mcmc-thin` | Sampler size (3, 1500, 1000, 2). |
| `--hmc-step-size` / `--hmc-leapfrog-steps` | Initial HMC step and leapfrog count (0.05 / 5). |
| `--predictive-draws` | Posterior draws propagated through CRM-ML (200). |
| `--no-refit` | Reuse posterior-mean ML models. Faster; understates epistemic UQ. |
| `--no-time-feature` | Drop raw time from ML features to avoid flat extrapolation by tree models. |
| `--sigma-growth` | Widen sigma with lead k as sigma·sqrt(1 + g·k) (0). |
| `--backtest N` | Also run an N-origin rolling backtest (0 = off). |
| `--synthetic-check` | Run the known-truth recovery check and exit. |
| `--seed` | Base random seed (2025). |
| `--trim-leading-inactive` | Deprecated, no effect (shut-in rows are always excluded from log modelling). |
| `--out` | Output folder (uq_results). |
| `--quiet` | Suppress progress messages. |
| `--write-template FILE` | Write a template workbook and exit. |
| `--gui` | Launch the Streamlit interface. |

```
uv run crm_ml_uq.py --write-template template.xlsx
uv run crm_ml_uq.py --data field.xlsx --forecast 12 --model CRM-XGB --out uq_results
uv run crm_ml_uq.py --data field.xlsx --backtest 3
uv run crm_ml_uq.py --gui        # or: streamlit run crm_ml_uq.py
```

The file name above is a placeholder; use the name of your script.

# 18. The Streamlit GUI

The GUI is a thin presentation layer. It collects options in a sidebar, calls the same `load_from_bytes()` and `run_bayesian_uq()` used by the CLI, and renders the returned `BayesianUQResult` with the same plot and export functions. No inference logic is duplicated.

## 18.1 Launching and dispatch

- `--gui` imports `streamlit.web.cli`, sets `sys.argv` to ["streamlit", "run", <this file>] and calls `stcli.main()`, so Streamlit re-executes the script.
- The bottom of the file decides what to run. If `streamlit` is already in `sys.modules` it asks `get_script_run_ctx()` for a run context: a valid context calls `run_gui()`; otherwise, when run as a script, `main()` is called. If the Streamlit import fails, only `main()` is used. Without Streamlit loaded, the file simply calls `main()` when executed.
- Because Streamlit is imported inside functions, importing the module as a library never starts a UI.

## 18.2 Page layout

`run_gui()` sets a wide layout, the title "Bayesian CRM-ML Production Forecasting" and a caption summarising the method (posterior over CRM parameters → CRM-ML refit per draw → Lognormal predictive uncertainty). A sidebar holds three groups of controls, and the main area shows data previews and results.

## 18.3 Sidebar controls

| Control | Widget and default | Maps to |
|---|---|---|
| Upload Excel workbook | File uploader, .xlsx only | `load_from_bytes()` |
| Download workbook template | Download button; template built on the fly in a temp folder | `write_template()` |
| CRM-ML method | Select box, default CRM-XGB | `model` |
| Forecast length | Number input, min 1, default 12 | `n_forecast` |
| CRM optimisation restarts | Slider 1 to 8, default 3 | `crm_starts` |
| Use raw time as an ML feature (paper) | Checkbox, on | `use_time_feature` |
| Refit ML for every posterior draw | Checkbox, on | `refit_ml_per_draw` |
| Sigma growth per lead step | Number input 0 to 1, default 0, step 0.01 | `sigma_growth` |
| Posterior draws propagated | Number input 20 to 1000, default 200, step 20 | `predictive_draws` |
| MCMC chains | Slider 2 to 20, default 3 | `mcmc_chains` |
| Retained samples per chain | Number input 100 to 10000, default 1500 | `mcmc_samples` |
| Burn-in iterations | Number input 100 to 20000, default 1000 | `burn_in` |
| Thinning interval | Number input 1 to 20, default 2 | `thin` |
| Initial HMC step size | Number input 0.005 to 0.50, default 0.05 | `hmc_step_size` |
| HMC leapfrog steps | Number input 1 to 20, default 5 | `hmc_leapfrog_steps` |
| Random seed | Number input 0 to 999999, default 2025 | `seed` |
| Run Bayesian UQ | Primary button | Triggers the run |

Defaults come from the same constants as the CLI, so both interfaces behave identically unless changed. The GUI does not expose `crm_max_iter`, the rolling backtest or the synthetic check; those are CLI-only.

## 18.4 Data loading and preview

1. Without an upload, an info message asks for a workbook and the function returns.
2. The upload is parsed with `load_from_bytes()`. A parsing or validation error is shown in a red message and processing stops.
3. A "Production overview" figure is drawn with `plot_production_history()`, placing the forecast boundary at the current forecast length. Figures are closed after display to avoid leaking memory.
4. An expanded "Input data summary" shows `FieldData.summary()`. An "Input notes" expander lists importer notes such as the missing injectors or default distances.

## 18.5 Running inference

Pressing Run starts `run_bayesian_uq()` inside a spinner with all sidebar values cast to the right types and `verbose=True`. On success the result is stored in `st.session_state["uq_result"]`; on failure the exception message is shown and the run stops. Because the result is kept in session state, adjusting widgets or switching tabs does not trigger another run. If no result exists yet, an info message prompts the user to run.

## 18.6 Result summary

A header shows the method name, followed by six metric cards: maximum R-hat, minimum bulk ESS, mean 95% PICP, mean 95% MPIW, mean Bayesian P50 MAE and mean MAE improvement (signed). If R-hat exceeds 1.10 or ESS falls below 100 a warning asks for a longer run; otherwise a success banner confirms the checks passed. A caption reminds users that PICP covers positive observations only and that, for a short window, the CLI backtest gives more reliable coverage.

## 18.7 Result tabs

| Tab | Content |
|---|---|
| Overview | Deterministic vs Bayesian comparison (baseline metrics merged with Bayesian MAE, RMSE, R², PICP, MPIW, interval score and MAE improvement); uncertainty decomposition on the log scale; ML out-of-sample sigma and bias. |
| Production & Forecast | One bordered panel per producer with its forecast plot, then the complete forecast table. |
| Detailed Comparison | Selected forecast columns (observed, deterministic and Bayesian errors, P10/P50/P90 bounds, 95% interval, width, coverage, uncertainty split) and the metrics table. |
| UQ / Calibration | Out-of-sample residual plot and the calibration predictions table. |
| MCMC Diagnostics | Trace plots, R-hat/ESS table and per-chain acceptance rates. |
| Posterior Parameters | Full posterior summary. When injectors exist: posterior-mean λ matrix and τ_ij matrix. Always: posterior-mean τ_j table. |
| Downloads | Single button that exports every result file into a temporary folder, zips them in memory and offers `bayesian_crm_uq_results.zip`. |

## 18.8 Behaviour worth knowing

- Streamlit executes every tab body on each rerun, including the Downloads tab. The ZIP (with its four figures) is therefore regenerated whenever the page refreshes. This is correct but adds a little overhead for large runs.
- The stored result is not cleared when a different workbook is uploaded. After changing the file, press Run again so that tables, plots and the loaded data agree.
- Long runs block the page while the spinner is shown. Runtime grows with chains × samples × thinning and with posterior draws × producers when the ML model is refitted for every draw.
- The checkbox label "(paper)" on the raw-time option records that including time follows the reference method. Turning it off is the GUI equivalent of `--no-time-feature`.

# 19. Practical Notes and Limitations

- Reproducibility: chain seeds are seed + 100000·c, ML seeds are seed + producer index, calibration uses seed + 700000 and draw subsampling uses seed + 77. XGBoost runs single-threaded for the same reason.
- Minimum data: at least 20 history rows overall, 8 positive history rows per producer, a calibration block of at least 10 rows with 8 positive rows on each side, and 16 positive rows per producer in pre-computed mode.
- Zeros and negatives: zero (shut-in) production is retained for point metrics but excluded from log-space modelling and coverage metrics; negative production raises an error.
- Known future injection: forecasts use the injection values in the forecast window as inputs, so they are conditional on that schedule.
- Unused inputs: distances and BHP are loaded and validated but do not enter the current model.
- Cost: the per-draw ML refit multiplies model fits by the number of draws and producers. Use `--no-refit` or fewer draws for quick exploration, accepting narrower epistemic bands.
- Weak identifiability: λ and τ can be poorly identified, so wide posteriors or elevated R-hat are properties of the problem, not necessarily defects. Check diagnostics and consider longer chains.
- Coverage: single-split PICP is a rough indicator; use the rolling backtest for a reliable estimate.

# 20. Function and Class Reference

| Name | Responsibility |
|---|---|
| `FieldData`, `_truncate_field()` | Input container; truncated copy for backtests. |
| `mean_absolute_error()`, `root_mean_squared_error()`, `r_squared()` | Point metrics. |
| `_find_sheet()`, `_parse_block()`, `_extract_unit()` | Workbook parsing helpers. |
| `validate_field()`, `load_field_data()`, `load_from_bytes()` | Validation and loading. |
| `write_template()` | Template workbook. |
| `CRMParameters`, `_exp_filter()`, `crm_simulate()` | CRM parameters and NumPy simulator. |
| `CapacitanceResistanceModel` | Deterministic SLSQP calibration and prediction. |
| `ExtremeLearningMachine`, `make_ml_regressor()` | ELM and regressor factory. |
| `build_crm_ml_features()`, `fit_positive_crm_ml()`, `positive_prediction()` | Features, log-target fitting, positive prediction. |
| `BayesianUQResult` | Result container. |
| `_positive_active_mask()` | Positive-row mask; rejects NaN and negatives. |
| `_logit_slack_from_lambda()`, `_lambda_from_logit_slack()` | λ and softmax-logit conversion. |
| `_crm_theta_from_params()`, `_crm_params_from_theta()` | Parameter vector conversion. |
| `_split_rhat()`, `_ess_bulk()`, `_diagnostics()`, `_lag1_autocorr()` | Diagnostics and residual correlation. |
| `_crm_simulate_torch()` | Differentiable CRM. |
| `_sample_inverse_gamma()`, `_hmc_sample()` | Sigma sampling and HMC. |
| `_make_crm_posterior()` | Log-posterior and sigma sampler. |
| `_posterior_crm_arrays()`, `_posterior_mean_crm_params()`, `_summ()`, `_crm_posterior_summary()`, `_theta_names()` | Posterior post-processing. |
| `_crm_ml_forecast()`, `_oos_ml_calibration()` | CRM-ML forecasting and noise calibration. |
| `_deterministic_forecast_metrics()`, `_forecast_tables()` | Metric and forecast tables. |
| `_run_precomputed_crm_uq()` | Pre-computed q_CRM workflow. |
| `run_bayesian_uq()` | Main inference pipeline. |
| `rolling_origin_backtest()`, `synthetic_recovery_check()` | Validation tools. |
| `plot_*()` (six functions) | Figures. |
| `export_result()`, `_print_report()` | Export and console report. |
| `main()` | CLI coordinator. |
| `run_gui()` | Streamlit interface. |

# 21. Conclusion

The implementation combines a physics-based CRM, a data-driven ML correction and Hamiltonian Monte Carlo inference into one forecasting workflow. Parameter uncertainty is sampled rather than assumed, each posterior draw is carried through the ML correction, and observation noise is calibrated on a chronological hold-out that neither stage has seen. The outputs are a posterior median, P10/P90 bounds, a 95% interval and a decomposition of uncertainty, all scored against held-out observations.

The same core is reachable through a scriptable command-line interface, a Streamlit GUI that exposes the main controls and presents results in structured tabs, and a library API. Optional validation tools (rolling-origin backtest and synthetic recovery) help judge whether the intervals can be trusted for a given field before the forecasts are used in decisions.