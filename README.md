# CRM-ML Hybrid Production Forecasting

Python implementation of the workflow in

> Ogali, O.I.O. & Orodu, O.D. (2025). *Concatenating data-driven and reduced-physics models for smart production forecasting.* Earth Science Informatics 18:246. <https://doi.org/10.1007/s12145-025-01745-9>

It combines the **Capacitance-Resistance Model (CRM)** with four machine-learning models
(**NuSVM, XGB, ELM, MLP**) into "CRM-ML" hybrids, compares nine approaches
(CRM, 4 CRM-ML hybrids, 4 stand-alone ML models), and reproduces the paper's comparison charts,
through a **command line**, a **Streamlit GUI**, or a **Python API**.

---

## 1. Contents

| File | Purpose |
|---|---|
| `crm_ml_hybrid.py` | Core: CRM (Eq. 1), calibration (Eq. 2), ML models, features, metrics (Eqs. 3-5), studies, ranking |
| `data_io.py` | Excel/array import, validation, blank template writer |
| `plots.py` | Paper-style charts (matplotlib) |
| `pipeline.py` | `run_pipeline`, `make_figures`, `export_outputs` shared by CLI and GUI |
| `run_study.py` | Command-line runner (exports PNG + CSV) |
| `app.py` | Streamlit GUI |
| `examples/make_synthetic_example.py` | Creates `examples/synthetic_field.xlsx` (Layout A demo) |
| `pyproject.toml` | Dependencies for `uv` |

---

## 2. Installation (with uv)

```bash
# 1) install uv  (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# 2) in the project folder
uv python pin 3.12
uv sync --extra gui --extra xgb        # core + Streamlit GUI + real XGBoost
```

* `--extra xgb` is optional. Without it a scikit-learn `GradientBoostingRegressor` stands in
  for XGB (a warning is printed once).
* `--extra gui` is only needed for the GUI.
* Plain pip works too:
  `pip install numpy scipy pandas scikit-learn matplotlib openpyxl streamlit xgboost`.

---

## 3. Running it

### 3.1 GUI

```bash
uv run streamlit run app.py
```

1. Upload your `.xlsx` workbook (or download the blank template from the sidebar).
2. Check the **Data preview** tab: detected mode, per-producer active window, rate plot.
3. Set the forecast length, evaluations, models; press **Run study**.
4. Browse the **Results** tab: rankings, nine chart tabs, and a **ZIP download** of all charts/tables.

### 3.2 Command line

```bash
uv run python examples/make_synthetic_example.py
uv run python run_study.py examples/synthetic_field.xlsx --forecast 60 --evals 20 --out results_syn
uv run python run_study.py CRM_Volve3_Dataset.xlsx      --forecast 12 --evals 10 --out results_volve
```

Options: `--models NuSVM XGB ELM MLP`, `--no-trim`, `--quiet`.

### 3.3 Python API

```python
from data_io import load_field_data
from pipeline import run_pipeline, make_figures, export_outputs

data = load_field_data("my_field.xlsx")
out  = run_pipeline(data, n_forecast=12, n_evaluations=20)

print(out.ranking("forecast"))          # mean rank, % first place, mean MAE
print(out.summary_table("forecast"))    # mean MAE / RMSE / R2 per approach
figs = make_figures(out)                # {name: matplotlib Figure}
export_outputs(out, "results")          # PNGs + CSVs (+ CRM parameters if available)
```

---

## 4. Charts reproduced from the paper

| Paper figure | Chart | `plots.py` function | GUI tab |
|---|---|---|---|
| Fig. 8, 17 | Heat-maps of λij and τij | `plot_crm_parameters` | CRM parameters (full mode only) |
| Fig. 9, 18 | Observed vs CRM, history (blue) / forecast (red), with MAE, RMSE, R² | `plot_crm_fit` | CRM fit and forecast |
| Fig. 10, 19 | All nine approaches vs observed + MAE table, whole record | `plot_all_approaches` | All approaches, entire record |
| Fig. 11, 14 | Same, forecast window only | `plot_all_approaches(only_forecast=True)` | All approaches, forecast |
| Fig. 12, 15, 20, 21 | MAE of every approach per evaluation | `plot_mae_by_evaluation` | MAE per evaluation |
| Fig. 13, 16, 22 | Grouped bars of average MAE per producer + all producers | `plot_average_mae_bars` | Average MAE |
| Fig. 23 | Ranking ladder (best on top) | `plot_ranking` | Ranking ladder |

Tables written by `export_outputs`: `metrics_all_evaluations.csv`, `ranking_{forecast,entire}.csv`,
`summary_{forecast,entire}.csv`, `predictions.csv`, and `crm_parameters.xlsx` (full mode).

---

## 5. How the method is implemented

1. **CRM (Eq. 1)** - production term + injection term (+ optional BHP term). The convolution
   sums are evaluated with an exact first-order recursion (`scipy.signal.lfilter`), O(N) instead of O(N²).
2. **Calibration (Eq. 2)** - SLSQP minimises the mean squared error for all producers at once, subject
   to 0 ≤ λij ≤ 1, Σⱼ λij ≤ 1 per injector, τ ≥ sampling interval. Multiple restarts (`n_starts`).
3. **CRM-ML features** - `[t, Xij, λij, τij, wi(t), τj]` = 4I + 2 inputs; target = observed rate.
   **ML-only features** - the I injection rates.
4. **Training** - NuSVM/XGB are deterministic and use all historical data. ELM/MLP use a random
   75:25 train/validation split per evaluation (the paper's "erratic ANN" behaviour), repeated
   `n_evaluations` times (paper: 20).
5. **Scoring** - MAE, RMSE, R² (Eq. 5, squared correlation) on the **entire** record and on the
   **forecast** window.
6. **Ranking** - per evaluation, per producer, per case; ranks are not derived from averaged errors.

---

## 6. Importing your own data

### 6.1 Which layout do you have?

| You have... | Use | What happens |
|---|---|---|
| Observed rates **and injection rates** (+ ideally well distances) | **Layout A (full)** | The CRM is calibrated; the full paper workflow runs |
| Observed rates and an **already-computed CRM series** but no injection data | **Layout B (pre-computed CRM)** | The supplied `q_CRM` is used as the CRM |

### 6.2 Workbook format

One `.xlsx` workbook, one sheet per quantity. Sheet names are case-insensitive
(aliases such as `q_obs`, `injection`, `bhp`, `pwf` also work).

| Sheet | Layout A | Layout B | Shape |
|---|---|---|---|
| `Time` | required | required | N × 1 |
| `q_Observed` | required | required | N × K (one column per producer) |
| `w_Injection` | **required** | - | N × I (one column per injector) |
| `Distances` | recommended | - | I × K injector-producer distances |
| `BHP` | optional | - | N × K producer bottom-hole pressures |
| `q_CRM` | - | **required** | N × K |

**Header rules** (this is the format of `CRM_Volve3_Dataset.xlsx`):

* Row 1 = well names (column headers). `Time` sheet: row 1 = `Time`.
* Following rows may be descriptive text (`Oil + Water`, `[bbl/mth]`). They are skipped
  automatically; a bracketed unit like `[bbl/mth]` is picked up and used in axis labels.
* From the first fully numeric row onward everything must be numeric - **no blanks/NaN**.
* Time must be strictly increasing, at a (preferably) uniform step. Use rates, not cumulative volumes.
* Column order in `q_Observed`, `q_CRM`, `BHP` and the `Distances` columns must be the same producer order.
* `Distances` is I rows × K columns: rows follow the injector order of `w_Injection`.

Example (`q_Observed`):

| 15/9-F-1 C | 15/9-F-11 | 15/9-F-12 |
|---|---|---|
| Oil + Water | Oil + Water | Oil + Water |
| [bbl/mth] | [bbl/mth] | [bbl/mth] |
| 0 | 0 | 34764 |
| ... | ... | ... |

### 6.3 Step by step

1. **Get the template:** GUI sidebar button *Download blank template*, or
   `python -c "from data_io import write_template; write_template('template.xlsx')"`.
2. Paste your series into the sheets, keeping the header rows.
3. Check consistency: same number of rows in `Time`, `q_Observed`, `w_Injection`, `BHP`, `q_CRM`.
4. Load it (GUI upload, or `load_field_data("file.xlsx")`). Read the **notes** the loader prints,
   for example "No distances provided".
5. Choose a forecast length (samples at the end of the record). For monthly data 6-12 is typical;
   keep at least ~10-20 historical samples per well.
6. Run and review the charts.

If your sheets have other names:

```python
data = load_field_data("file.xlsx",
        sheet_map={"time": "Months", "production": "Liquid", "injection": "WaterInj"})
```

### 6.4 CSV / pandas / database users

```python
import pandas as pd
from data_io import from_arrays
from pipeline import run_pipeline

prod = pd.read_csv("producers.csv", index_col=0)      # columns = producer names
inj  = pd.read_csv("injectors.csv", index_col=0)      # columns = injector names
dist = pd.read_csv("distances.csv", index_col=0)      # injectors x producers
data = from_arrays(prod.index.values, prod, inj, dist, time_unit="mth", rate_unit="bbl/mth")
out  = run_pipeline(data, n_forecast=12)
```

### 6.5 Your two attached files

**`CRM_Volve3_Dataset.xlsx` → Layout B (works now).** Sheets `Time` (112 months), `q_Observed`
and `q_CRM` for five Volve wells (15/9-F-1 C, F-11, F-12, F-14, F-15 D). It contains **no injection
data**, so the CRM cannot be recalibrated and the loader runs in *pre-computed CRM mode*:

```bash
uv run python run_study.py CRM_Volve3_Dataset.xlsx --forecast 12 --evals 10 --out results_volve
```

Things to know about this file:

* Three wells start late (first non-zero rate at months ~71-80); leading zeros are skipped by default
  (*trim* option) so they do not distort training and scoring.
* Several wells shut in during the last months, so the observed forecast window contains zeros while
  the CRM series keeps producing. Choose the forecast length with this in mind.
* In this mode the hybrids receive `[t, q_CRM(t)]` and the ML baseline receives `[t]` only. This is an
  **adaptation of the paper** (see 7). To follow the paper exactly, add a `w_Injection` sheet
  (monthly rates of your field's injection wells) and a `Distances` sheet; the file then
  switches automatically to Layout A, and `q_CRM` is ignored (the CRM is recalibrated).

**`test_FD001_txt_file.xlsx` → not usable here.** It is the NASA C-MAPSS turbofan-engine degradation
set (26 unlabeled columns: unit, cycle, 3 operating settings, 21 sensors; 100 engines).
It has no wells, injectors or production rates, so the CRM (a reservoir material-balance model)
does not apply. The ML half of this code could in principle be adapted to remaining-useful-life
prediction, but that is a different problem and not implemented.

---

## 7. Limitations and deviations from the paper

* **Hyperparameters** for NuSVR/XGB/ELM and the ELM hidden size (10) are my assumptions; the paper does not list them.
* **ELM ridge term** (`ridge=1e-3`) stabilises output weights on short records; set `ridge=0` in
  `ExtremeLearningMachine` for the pure pseudo-inverse.
* **Pre-computed CRM mode** (Layout B) is not in the paper; it overstates the benefit of the hybrid
  because the ML baseline lacks injection data.
* **CRM calibration** uses the whole historical window of all producers; zero-rate periods of late wells
  are not excluded from calibration (only from ML training and scoring).
* Without `xgboost` installed, "XGB" is scikit-learn gradient boosting.
* Results on synthetic data come from a CRM-generated field, so CRM naturally wins there.
* The GUI is a thin layer over `pipeline.py`; the same results are available headless via `run_study.py`.

## 8. Troubleshooting

| Message | Cause / fix |
|---|---|
| `Could not find a 'time' sheet` | Rename the sheet to `Time` or pass `sheet_map` |
| `q_Observed has N rows but Time has M` | Series lengths differ; trim or pad |
| `contains NaN/inf` | Fill gaps (0 for shut-in, or interpolate) |
| `Time must be strictly increasing` | Sort the rows / remove duplicates |
| `Distances must be I x K` | Rows = injectors, columns = producers |
| `skipping ... <8 active historical samples` | The well has too little history before the forecast window |
| Slow CRM calibration | Lower `n_starts`, or reduce injectors/producers/BHP term |
# crm-ml
