# CRM--ML Hybrid Production Forecasting

## Academic Documentation and User Guide

### Based on the work of Ogali and Orodu (2025)

**Reference:**\
Ogali, O.I.O. and Orodu, O.D. (2025). _Concatenating data-driven and
reduced-physics models for smart production forecasting._ Earth Science
Informatics, 18:246.\
DOI: 10.1007/s12145-025-01745-9

---

## 1. Purpose of the Program

This program implements the production-forecasting workflow described by
Ogali and Orodu (2025).

The main idea is to combine:

1.  a **Capacitance--Resistance Model (CRM)**, which represents the
    physical relationship between injection and production wells; and
2.  **machine-learning models**, which learn additional relationships
    from the available production and injection data.

The purpose of combining these two approaches is to produce a
forecasting system that uses both **reservoir-response information** and
**data-driven information**.

The program compares nine approaches:

- CRM alone
- CRM + NuSVM
- CRM + XGBoost
- CRM + Extreme Learning Machine (ELM)
- CRM + Multilayer Perceptron (MLP)
- NuSVM alone
- XGBoost alone
- ELM alone
- MLP alone

The program evaluates these approaches using:

- **MAE --- Mean Absolute Error**
- **RMSE --- Root Mean Squared Error**
- **R² --- Coefficient of Determination**

The lower the MAE and RMSE, the closer the predictions are to the
observed production values. For R², a value closer to 1 generally
indicates a stronger agreement between predicted and observed
production.

---

# 2. How the Model Works

The complete workflow can be understood without programming knowledge.

### Step 1 --- Production and injection data are provided

The program receives historical information about:

- time
- producer production rates
- injector injection rates
- distances between injectors and producers

Optional bottom-hole-pressure information can also be supplied where
appropriate.

### Step 2 --- The CRM is calibrated

The Capacitance--Resistance Model estimates how strongly each injector
influences each producer and how quickly that influence is transmitted.

The main CRM quantities are:

**Connectivity index (λᵢⱼ)**\
This represents the strength of the relationship between injector _i_
and producer _j_.

**Interwell response time (τᵢⱼ)**\
This represents the characteristic time required for an injection change
to influence the corresponding producer.

**Producer time constant (τⱼ)**\
This represents the production-response time associated with the
producer itself.

The program estimates these parameters from the historical data while
applying the constraints described in the paper.

### Step 3 --- CRM predictions are generated

The calibrated CRM produces an estimate of the production response.

The historical portion is used for calibration, while the later portion
can be treated as the forecast period.

### Step 4 --- Machine-learning models are trained

The program also trains four machine-learning approaches:

- NuSVM
- XGBoost
- Extreme Learning Machine
- Multilayer Perceptron

The CRM-ML models receive the CRM information together with the
data-driven variables.

The CRM-ML feature structure follows the paper's formulation and
contains **4I + 2 inputs**, where _I_ is the number of injectors.

The standalone machine-learning models use the historical injection
information without the CRM feature set.

### Step 5 --- Predictions are compared

The predicted production from each approach is compared with the
observed production.

The program calculates MAE, RMSE and R² and uses these values to compare
the approaches.

### Step 6 --- The approaches are ranked

The program can rank the approaches according to their performance.

The paper protocol uses **20 evaluations** for the stochastic ANN-based
approaches. The revised implementation therefore uses 20 evaluations by
default.

---

# 3. What Is Included in This Implementation

The implementation has been deliberately separated into two categories.

### Settings directly supported by the paper

The program explicitly follows the paper for the following items:

- nine model approaches
- CRM formulation
- CRM constraints
- CRM-ML feature structure
- standalone ML input structure
- MLP hidden-layer size of 10 neurons
- 75% historical training / 25% validation split for the ANN-based
  models
- 20 evaluations
- CRM minimum response time based on the sampling interval
- exclusion of the BHP term for the Buffalo case
- the five-case overall ranking structure

### Settings that are implementation choices

The paper does not specify every numerical software setting required to
reproduce a computer implementation.

Therefore, the program does **not** present the following as if they
were values reported by the paper:

- optimiser restart count
- exact XGBoost hyperparameters
- exact NuSVM hyperparameters
- ELM regularisation settings
- some numerical initialisation choices
- numerical scaling used internally for conditioning
- software-library-specific solver settings

This distinction is important for academic reproducibility.

---

# 4. Important Note About Exact Reproduction of the Published Figures

The paper contains published figures, but the paper does not provide
every underlying numerical array required to recreate every curve
point-for-point from scratch.

For this reason, this project distinguishes between:

### A. Published figure reference

The project includes the supplied article PDF and a reference document
containing the actual published pages on which Figures 8--23 appear.

These are the appropriate references when the objective is to show
**exactly what the published figures look like**.

### B. Data-driven reproduction

The Python program generates new figures from the data supplied to the
program.

These figures use descriptive labels such as:

- Observed liquid production
- CRM estimate
- CRM + XGBoost hybrid
- Mean absolute error
- Injector-to-producer connectivity strength
- Injector-to-producer response time

The generated plots are therefore intended to reproduce the **analysis
and figure concepts**, rather than claim pixel-for-pixel reproduction
where the original underlying numerical data are unavailable.

The program intentionally avoids manufacturing numerical values that are
not supplied by the article.

---

# 5. Demonstration Dataset Versus the Article Dataset

The program contains a small built-in synthetic field so that the
software can be tested without first obtaining a reservoir dataset.

This demonstration field is **not the Synfield used in the paper**.

The demonstration dataset is generated by the program for software
testing and contains:

- 5 injectors
- 4 producers
- a configurable number of time samples
- synthetic injection rates
- synthetic production rates
- synthetic injector-producer distances

The article's Synfield is different. The article describes a 67 × 67 × 5
reservoir grid with 2,922 days of production/injection history.

Therefore:

> Running the program with the built-in demonstration field demonstrates
> that the implementation works; it does not constitute a numerical
> reproduction of the article's Synfield results.

---

# 6. System Requirements

The program requires:

- Windows, Linux or macOS
- Python 3.10 or newer
- Internet access during installation of the required Python packages
- approximately 1 GB or more of free storage for Python packages,
  depending on the existing environment

The required Python packages are:

- NumPy
- SciPy
- pandas
- scikit-learn
- Matplotlib
- openpyxl
- XGBoost
- Streamlit
- PyMuPDF

---

# 7. Installation

There are two recommended installation methods.

## Method A --- Using `uv` (recommended)

`uv` is a Python environment and package manager that can install and
run the dependencies specified at the beginning of the Python file.

### Step 1 --- Install Python

Install Python 3.10 or a newer version.

Verify the installation:

```bash
python --version
```

A result similar to this is expected:

```text
Python 3.11.x
```

### Step 2 --- Install `uv`

If `uv` is already installed, this step can be skipped.

On Windows PowerShell:

```powershell
pip install uv
```

Verify:

```powershell
uv --version
```

### Step 3 --- Place the program in its project folder

For example:

```text
crm-ml/
│
├── main.py
├── README.md
├── CRM_ML_Figure_Documentation.pdf
└── Published_Figure_Reference_Pages.pdf
```

The Python implementation should be saved as:

```text
main.py
```

### Step 4 --- Run the program

From the project folder:

```powershell
uv run main.py
```

`uv` will use the dependency information specified in the Python file.

---

# 8. Installation Using Standard Python and `pip`

If `uv` is not preferred, the required packages can be installed
directly.

From the project folder:

```powershell
python -m pip install --upgrade pip
```

Then:

```powershell
python -m pip install numpy scipy pandas scikit-learn matplotlib openpyxl xgboost streamlit pymupdf
```

After installation, the program can be run with:

```powershell
python main.py
```

---

# 9. Running the Program Without the GUI

The command-line version is useful when a complete study needs to be run
automatically.

## Run the built-in demonstration

```powershell
python main.py
```

or:

```powershell
uv run main.py
```

The program will use the built-in demonstration field.

The results are written to:

```text
results/
```

The program reports:

- number of time samples
- number of producers
- operating mode
- model results
- CRM parameters
- model rankings
- summary performance metrics

---

# 10. Running the Program With Your Own Excel Data

The program can read an Excel workbook containing the field data.

A normal full CRM study requires the following information:

### `Time`

The time values for the production history.

### `q_Observed`

Observed production rates for the producers.

### `w_Injection`

Injection rates for the injectors.

### `Distances`

Injector-to-producer distances.

The workbook should therefore contain these sheets:

```text
Time
q_Observed
w_Injection
Distances
```

The program also recognises optional information such as:

```text
BHP
q_CRM
```

The `q_CRM` sheet is used when the supplied data already contain a
pre-computed CRM production series.

---

# 11. Recommended Workbook Structure

A simple workbook can be organised as follows.

### Sheet 1 --- `Time`

```text
Time
[days]
1
2
3
4
...
```

### Sheet 2 --- `q_Observed`

```text
P-01    P-02    P-03    ...
[rate]  [rate]  [rate]
...
```

### Sheet 3 --- `w_Injection`

```text
I-01    I-02    I-03    ...
[rate]  [rate]  [rate]
...
```

### Sheet 4 --- `Distances`

The distances are arranged as:

```text
             P-01    P-02    P-03
I-01         ...     ...     ...
I-02         ...     ...     ...
I-03         ...     ...     ...
```

In other words, the rows represent injectors and the columns represent
producers.

---

# 12. Creating a Blank Workbook Template

The program can create a workbook showing the expected structure.

Run:

```powershell
python main.py --write-template crm_ml_template.xlsx
```

This creates:

```text
crm_ml_template.xlsx
```

The template contains example values that should be replaced with the
actual field data.

This is the easiest way to prepare a workbook for a new user.

---

# 13. Creating the Demonstration Workbook

A demonstration workbook can also be generated:

```powershell
python main.py --write-example example_field.xlsx
```

This produces a workbook containing the program's synthetic
demonstration field.

The demonstration workbook is useful for confirming that the
installation is working correctly.

It should not be presented as the article's Synfield dataset.

---

# 14. Choosing the Forecast Period

The forecast period is specified by the number of samples.

For example:

```powershell
python main.py --data my_field.xlsx --forecast 12
```

means that the final 12 samples are treated as the forecast period.

For a dataset where each row represents one month, `12` represents
approximately one year.

For a dataset where each row represents one day, `12` represents 12
days.

The meaning therefore depends on the time unit in the supplied data.

---

# 15. Running the Graphical User Interface (GUI)

The program includes a Streamlit graphical interface so that the model
can be used without entering many command-line options.

There are two ways to launch it.

## Option 1 --- From the Python program

```powershell
python main.py --gui
```

or:

```powershell
uv run main.py --gui
```

## Option 2 --- Directly through Streamlit

```powershell
python -m streamlit run main.py
```

After starting, Streamlit normally displays a local web address in the
terminal.

Open that address in a web browser.

---

# 16. How to Use the GUI

The GUI is organised into a simple sequence.

## Step 1 --- Data

The first section allows the user to:

- upload an Excel workbook; or
- use the built-in demonstration field.

There is also a **Download blank template** button.

If actual field data are available, the Excel workbook should be
uploaded instead of using the demonstration field.

---

## Step 2 --- Settings

The GUI provides controls for:

### Forecast length

Specifies how many final observations should be treated as the forecast
period.

### Evaluations

Specifies the number of evaluations performed for the stochastic
ANN-based models.

The paper protocol uses:

```text
20 evaluations
```

The GUI therefore uses 20 as its default.

Lower values can be used during preliminary testing to reduce
computation time, but a paper-protocol run should use 20.

### Machine-learning models

The user can select:

- NuSVM
- XGBoost
- ELM
- MLP

### Optional preprocessing

The program allows leading zero-production periods to be skipped.

This option is deliberately disabled by default because it is an
optional preprocessing choice rather than a requirement of the paper.

### CRM optimiser restarts

This controls how many optimisation starting points are attempted when
calibrating the CRM.

It is a numerical implementation setting and is not presented as a value
specified by the paper.

---

# 17. Running a Study in the GUI

After selecting the data and settings:

1.  Confirm the forecast length.
2.  Leave **Evaluations** at 20 for the paper protocol.
3.  Select the required machine-learning models.
4.  Keep optional preprocessing disabled unless there is a specific
    reason to use it.
5.  Click **Run study**.

The program then:

1.  validates the input data;
2.  calibrates the CRM;
3.  generates CRM predictions;
4.  trains the selected machine-learning models;
5.  creates CRM-ML hybrid models;
6.  calculates MAE, RMSE and R²;
7.  ranks the approaches;
8.  generates the result figures.

---

# 18. What the GUI Results Mean

The results page contains two main rankings.

### Ranking --- Forecast Period

This compares the approaches using only the designated forecast period.

This is particularly important when the purpose is to determine which
method predicts unseen production most effectively.

### Ranking --- Entire Record

This compares the approaches using the complete available record.

This gives a broader view of how well each approach represents both
historical and forecast-period behaviour.

The ranking should be interpreted together with MAE, RMSE and R² rather
than using the ranking alone.

---

# 19. Downloading Results From the GUI

After a successful study, the GUI provides:

**Download all charts + tables (ZIP)**

The ZIP contains the generated figures and result tables.

The exported results include:

```text
metrics_all_evaluations.csv
ranking_forecast.csv
ranking_entire.csv
summary_forecast.csv
summary_entire.csv
predictions.csv
```

If CRM calibration was performed, the program also creates:

```text
crm_parameters.xlsx
```

This workbook contains:

- connectivity indices
- injector-producer response times
- producer time constants

---

# 20. Understanding the Main Output Tables

## `metrics_all_evaluations.csv`

Contains the performance measurements for the individual evaluations.

Important columns include:

- evaluation
- producer
- approach
- case
- MAE
- RMSE
- R²

This is the detailed performance record.

## `ranking_forecast.csv`

Ranks the approaches using the forecast-period results.

## `ranking_entire.csv`

Ranks the approaches using the entire record.

## `summary_forecast.csv`

Provides average performance values for the forecast period.

## `summary_entire.csv`

Provides average performance values for the entire record.

## `predictions.csv`

Contains observed and predicted production values in a long-format
table.

This file is useful for examining the prediction of each approach for
each producer.

---

# 21. Understanding the CRM Parameter Workbook

When CRM calibration is performed, `crm_parameters.xlsx` contains three
main sheets.

### `connectivity_indices`

These are the estimated λᵢⱼ values.

They describe the relative strength of the modeled injector-to-producer
relationships.

### `response_times`

These are the estimated τᵢⱼ values.

They describe the characteristic response time between each injector and
producer.

### `producer_time_constants`

These contain τⱼ for each producer.

They describe the production response time used by the CRM.

These parameters are useful for interpreting the physical meaning of the
calibrated CRM, not only for evaluating prediction accuracy.

---

# 22. The Nine Approaches

## 22.1 CRM

The Capacitance--Resistance Model is the reduced-physics model.

It uses the relationship between injection and production to estimate
future production.

Its major advantage is that it explicitly represents interwell
relationships.

---

## 22.2 CRM + NuSVM

The CRM provides physics-informed information, while NuSVM supplies a
data-driven correction or prediction component.

---

## 22.3 CRM + XGBoost

The CRM information is combined with XGBoost, a tree-based
machine-learning method.

---

## 22.4 CRM + ELM

The CRM information is combined with an Extreme Learning Machine.

---

## 22.5 CRM + MLP

The CRM information is combined with a Multilayer Perceptron.

The implementation uses one hidden layer with 10 neurons, consistent
with the paper's stated configuration.

---

## 22.6 Standalone NuSVM

NuSVM is used without the CRM feature information.

---

## 22.7 Standalone XGBoost

XGBoost is used directly with the available injection information.

---

## 22.8 Standalone ELM

ELM is used directly with the injection information.

---

## 22.9 Standalone MLP

MLP is used directly with the injection information.

---

# 23. Meaning of the Evaluation Metrics

### MAE --- Mean Absolute Error

MAE measures the average absolute difference between observed and
predicted production.

A smaller MAE means that, on average, the predictions are closer to the
observed production values.

### RMSE --- Root Mean Squared Error

RMSE also measures prediction error, but larger errors have a stronger
influence on the final value.

A smaller RMSE therefore indicates better predictive agreement,
particularly when large prediction errors are important.

### R² --- Coefficient of Determination

R² indicates how well the predicted values explain the variation in the
observed values.

A value closer to 1 generally indicates stronger agreement.

These metrics should be considered together because a single metric does
not describe every aspect of model behaviour.

---

# 24. Understanding the Generated Figures

The program uses descriptive engineering labels rather than unexplained
symbols.

Typical figures include:

### Observed versus predicted production

Shows how closely each model follows the actual production history.

### CRM calibration figure

Shows observed production alongside the CRM estimate.

This allows the quality of the reduced-physics calibration to be
inspected visually.

### CRM parameter figure

Shows the estimated injector-to-producer connectivity and response-time
parameters.

### Evaluation error figure

Shows MAE across repeated evaluations.

This is useful for seeing whether the stochastic models produce stable
results or whether their performance changes between evaluations.

### Average error figure

Shows average model error by producer and across the available
producers.

### Overall ranking figure

Shows the relative performance of the approaches according to the
selected ranking procedure.

A separate figure documentation file is included with the project for
the detailed interpretation of Figures 8--23.

---

# 25. The Paper's Five-Case Overall Ranking

The paper considers five evaluation cases:

1.  Synfield --- 8-year production history
2.  Synfield --- 1-year forecast period
3.  Synfield --- 5-month forecast period
4.  Buffalo --- 189-month production history
5.  Buffalo --- 12-month forecast period

The revised implementation preserves this structure for the overall
ranking procedure.

The paper uses 20 evaluations.

For the stated producer counts, the combined ranking represents:

```text
3 Synfield cases × 20 evaluations × 4 producers
+
2 Buffalo cases × 20 evaluations × 8 producers
=
560 producer-evaluation groups
```

This is the level at which the overall comparison is formed.

The five-case ranking requires the corresponding five case result sets.
A single demonstration-field run is not equivalent to the paper's
complete five-case ranking.

---

# 26. Buffalo BHP Treatment

For the Buffalo case, the implementation does not include the BHP term.

This follows the treatment described for the Buffalo dataset in the
paper.

The program therefore does not silently add a BHP contribution to a
Buffalo paper-protocol run.

---

# 27. Recommended Procedure for an Academic Demonstration

For a lecturer who wants to inspect the implementation, the following
sequence is recommended.

### First demonstration --- Confirm that the software works

Run:

```powershell
python main.py
```

This uses the built-in demonstration field.

The purpose is only to demonstrate the operation of the software.

### Second demonstration --- Inspect the GUI

Run:

```powershell
python main.py --gui
```

Then use the built-in demonstration field and click **Run study**.

### Third demonstration --- Use actual field data

Prepare an Excel workbook using the supplied template:

```powershell
python main.py --write-template crm_ml_template.xlsx
```

Replace the example values with the appropriate field data.

Then run:

```powershell
python main.py --data crm_ml_template.xlsx --forecast 12
```

The exact forecast length should be changed according to the time
resolution of the dataset.

---

# 28. Recommended Paper-Protocol Settings

For a study intended to follow the paper's stated evaluation protocol,
use:

---

Setting Recommended value

---

Number of evaluations 20

MLP hidden neurons 10

Historical training fraction 75%

Validation fraction 25%

CRM minimum response time Sampling interval

Buffalo BHP term Disabled

Optional leading-zero trimming Disabled unless specifically justified

---

The optimiser restart count and detailed hyperparameters should not be
described as paper-reported values because the article does not specify
every software-level setting.

---

# 29. Common Questions

## Does the program reproduce the paper exactly?

It implements the paper's stated modelling workflow and protocol where
the paper provides enough information to do so.

It does not claim point-for-point reproduction of every published curve
when the underlying numerical arrays are not supplied.

The published figure reference pages are included separately for exact
visual reference.

---

## Is the built-in dataset the Synfield?

No.

It is a synthetic demonstration dataset created solely to test the
program.

---

## Why are there both CRM-ML and standalone ML models?

The comparison is intended to determine whether adding reduced-physics
CRM information improves the performance of machine-learning models.

This allows three broad questions to be examined:

1.  How well does CRM work by itself?
2.  How well do standalone machine-learning models work?
3.  Does combining CRM with machine learning improve the result?

---

## Why are 20 evaluations used?

The paper reports repeated evaluations for the ANN-based approaches.

The revised implementation uses 20 evaluations as the default paper
protocol.

During software testing, fewer evaluations may be used to reduce
execution time.

For final academic results intended for comparison with the paper, 20
should be used.

---

## Why does the program use an Excel workbook?

The workbook provides a simple way to keep the time series and field
relationships organised.

It also makes the input data easier for a researcher or lecturer to
inspect independently of the Python program.

---

# 30. Troubleshooting

### `ModuleNotFoundError`

Install the required packages:

```powershell
python -m pip install numpy scipy pandas scikit-learn matplotlib openpyxl xgboost streamlit pymupdf
```

Or use:

```powershell
uv run main.py
```

if `uv` is being used.

### XGBoost installation error

The article-faithful implementation requires XGBoost.

Install it with:

```powershell
python -m pip install xgboost
```

### The Excel workbook cannot be read

Check that:

- the workbook is an `.xlsx` file;
- the `Time` sheet exists;
- the `q_Observed` sheet exists;
- either `w_Injection` or `q_CRM` is provided;
- the number of time rows matches the production rows;
- the time values increase continuously;
- there are no NaN or infinite values.

### The program is taking a long time

The main factors affecting computation time include:

- number of time samples
- number of producers and injectors
- CRM optimisation restarts
- number of evaluations
- number of machine-learning models selected

For testing, the evaluation count can be reduced.

For the final paper-protocol run, restore it to:

```text
20
```

### The GUI does not open

Try:

```powershell
python -m streamlit run main.py
```

If Streamlit is missing:

```powershell
python -m pip install streamlit
```

---

# 31. Project Files

A recommended project directory is:

```text
crm-ml/
│
├── main.py
│   └── Complete CRM-ML implementation
│
├── README.md
│   └── This academic/user documentation
│
├── CRM_ML_Figure_Documentation.pdf
│   └── Explanation of the generated figures
│
├── CRM_ML_Figure_Documentation.md
│   └── Editable version of the figure documentation
│
├── Published_Figure_Reference_Pages.pdf
│   └── Published article pages containing Figures 8–23
│
└── results/
    └── Generated charts and tables after a study run
```

---

# 32. Academic Interpretation

The purpose of this implementation is not simply to produce a numerical
forecast.

It provides a framework for comparing:

**Reduced physics**

→ CRM

against

**Data-driven modelling**

→ NuSVM, XGBoost, ELM and MLP

and against

**Hybrid modelling**

→ CRM + NuSVM, CRM + XGBoost, CRM + ELM and CRM + MLP.

This makes it possible to investigate whether incorporating
reservoir-response information into machine-learning models improves
production forecasting.

The most important outputs are therefore not only the predicted
production curves, but also:

- the calibrated CRM parameters;
- the prediction errors;
- the stability of repeated evaluations;
- the comparison between standalone and hybrid approaches;
- and the overall ranking of the approaches.

---

# 33. Reproducibility Statement

For a reproducible academic run, the following should be recorded:

1.  Input dataset used.
2.  Number of forecast samples.
3.  Number of evaluations.
4.  Machine-learning models selected.
5.  CRM optimisation settings.
6.  Whether optional leading-zero trimming was enabled.
7.  Software/Python version.
8.  Generated result tables.
9.  Generated figures.
10. The exact version of the source code used.

This information allows another researcher to repeat the experiment and
determine whether differences are caused by the data, model
configuration, or software implementation.

---

# 34. Final Note

This implementation should be interpreted as a **paper-based research
implementation**, not as a claim that every undocumented software choice
made by the original authors has been recovered.

Where the article provides a modelling rule or evaluation protocol, the
implementation follows it.

Where the article does not provide enough information to determine an
exact software setting or underlying numerical dataset, the
implementation identifies that limitation rather than presenting an
assumption as an established fact.

This distinction is maintained throughout the project to make the
implementation suitable for academic review and further research.
