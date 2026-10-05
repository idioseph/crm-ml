# Figure Documentation -- CRM-ML Production Forecasting

## Important distinction about "exact reproduction"

The supplied Ogali & Orodu (2025) paper contains the published figures, but it does not provide all of the underlying numerical arrays required to recreate every published curve point-for-point.

Therefore this project uses **two different figure modes**:

1. **Published-figure reference mode:** renders the exact pages of the supplied article PDF containing Figures 8–23. These are direct renderings of the publication and are the correct choice when the requirement is an exact visual copy of the published figure.
2. **Data-driven reproduction mode:** the revised Python program generates article-mapped figures from the user's actual Synfield/Buffalo data. These use descriptive labels and the article's methodological structure, but they can only be numerically identical to the paper when the same source data and implementation details are available.

It would be misleading to manufacture "exact" curves from values that are not supplied by the paper.

---

## Figure 8 -- Optimized interwell CRM parameters for the Synfield

**What it shows:** Two parameter maps from CRM calibration.

- **Connectivity strength:** how strongly each injector is connected to each producer according to the calibrated CRM.
- **Response time:** how quickly the effect of an injector is transferred to a producer.

**How to read it:** Rows represent injectors and columns represent producers. A cell is therefore one injector–producer pair.

**Why it matters:** These parameters are the reduced-physics information subsequently supplied to the CRM-ML models.

**Important article interpretation:** The paper reports that the sum of connectivity indices for each Synfield injector equals 1.0.

---

## Figure 9 -- CRM estimates and forecasts for the Synfield

**What it shows:** Observed liquid production compared with CRM-generated production.

- Historical data are the period used to calibrate CRM.
- The final year is held out as the forecast period.
- The figure reports MAE, RMSE and R² to show numerical agreement.

**How to read it:** The closer the CRM curve is to the observed production curve, the smaller the forecasting error.

**Why it matters:** This establishes the performance of CRM before it is combined with machine learning.

---

## Figure 10 -- Full 8-year Synfield comparison

**What it shows:** Observed production and predictions from:

- CRM
- CRM-NuSVM
- CRM-XGB
- CRM-ELM
- CRM-MLP
- NuSVM
- XGB
- ELM
- MLP

The comparison uses the full 8-year record: 7 years historical plus 1 year forecast.

**How to read it:** Each producer panel compares the same observed production series against the nine approaches. The accompanying MAE table allows direct numerical comparison.

**Why it matters:** It tests performance when both the historical calibration period and the forecast period are considered.

---

## Figure 11 -- One-year Synfield forecast comparison

**What it shows:** The same nine approaches, but evaluated only on the held-out 1-year forecast period.

**How to read it:** A smaller MAE means that the model followed the unseen production history more accurately.

**Why it matters:** This is the more important test of genuine forecasting ability because the models are being judged on data not used for the final forecast training/calibration stage.

---

## Figure 12 -- MAE across 20 Synfield evaluations

**What it shows:** MAE for each producer and approach over 20 evaluations.

**How to read it:**

- A flat line indicates a model whose result is unchanged across evaluations.
- A fluctuating line indicates a model affected by the randomized ANN training/validation assignment.

The paper specifically reports this behaviour for CRM-ELM, CRM-MLP, ELM and MLP.

**Why it matters:** One evaluation is not enough to characterize the stochastic ANN-based approaches.

---

## Figure 13 -- Average MAE for the Synfield

**What it shows:** The average MAE over 20 evaluations for each producer and for all producers.

There are two scopes:

- full 8-year production data;
- 1-year forecast data.

**How to read it:** Shorter bars represent smaller average production-rate error.

**Why it matters:** Averaging the 20 evaluations summarizes the typical performance rather than one random ANN realization.

---

## Figure 14 -- Five-month short-term Synfield forecast

**What it shows:** Observed production and forecasts from CRM, CRM-ML hybrids and standalone ML models over the 5-month forecast window.

**Why it matters:** The paper uses this as a short-term forecasting test and compares whether the ranking of approaches changes when the forecast horizon is shortened.

---

## Figure 15 -- MAE across 20 five-month evaluations

**What it shows:** How MAE changes over the 20 evaluations for the short-term Synfield forecast.

**Why it matters:** It tests whether the stochastic behaviour observed in the one-year experiment remains when the forecast window is shorter.

---

## Figure 16 -- Average MAE for the five-month forecast

**What it shows:** Mean MAE for each producer and for all producers over the 20 five-month evaluations.

**Why it matters:** It provides the short-term counterpart to Figure 13.

---

## Figure 17 -- Optimized CRM parameters for Buffalo

**What it shows:** Calibrated injector-to-producer connectivity and response-time parameters for the selected Buffalo sector.

**How to read it:** Each cell corresponds to one of the five selected injectors and one of the eight producers.

**Important paper detail:** The Buffalo study has 5 injectors and 8 producers, producing 22 CRM-ML inputs per producer:

`4 × 5 + 2 = 22`.

The standalone ML models have 5 injection-rate inputs.

---

## Figure 18 -- CRM estimates and forecasts for Buffalo

**What it shows:** Observed Buffalo production compared with CRM estimates during the 177-month historical period and the 12-month forecast period.

**Why it matters:** It establishes the baseline reduced-physics performance on a real field rather than a reservoir simulation.

**Important:** The Buffalo CRM implementation excludes the BHP term according to the paper's stated methodology.

---

## Figure 19 -- Full Buffalo model comparison

**What it shows:** Observed production and all nine approaches for the selected Buffalo sector using the 189-month production record.

**Why it matters:** This evaluates the models on a real field and includes both historical and forecast observations.

---

## Figure 20 -- MAE across 20 Buffalo evaluations using 189 months

**What it shows:** MAE variation for each approach over the 20 evaluations using the entire Buffalo record.

**Why it matters:** It tests whether the performance patterns seen in the Synfield also occur in a real reservoir.

---

## Figure 21 -- MAE across 20 Buffalo forecast evaluations

**What it shows:** The same evaluation analysis, restricted to the final 12-month forecast period.

**Why it matters:** It isolates true forecast performance instead of allowing the much longer historical period to dominate the error calculation.

---

## Figure 22 -- Average RMSE for Buffalo

**What it shows:** Average RMSE for each producer and all producers for:

- the full 189-month record;
- the 12-month forecast period.

**Why RMSE matters:** RMSE penalizes larger errors more strongly than MAE. The paper therefore uses it as a complementary measure of forecast-error spread.

---

## Figure 23 -- Overall ranking

**What it shows:** The paper's ranking of the nine approaches across the Synfield and Buffalo cases.

The paper considers five cases:

1. Synfield -- 8-year production data.
2. Synfield -- 1-year forecast.
3. Synfield -- 5-month forecast.
4. Buffalo -- 189-month production data.
5. Buffalo -- 12-month forecast.

Each case uses 20 evaluations.

That produces:

`5 cases × 20 evaluations = 100 evaluation groups`

with multiple producers within each case. The paper reports the combined ranking as being based on **560 evaluations** in its overall aggregation.

**How to read it:** Rank 1 is the strongest overall performance according to the paper's MAE-based ranking procedure.

**Why it matters:** This is the final comparison used to assess whether adding CRM information to ML improves production forecasting relative to using ML alone.

---

# Figure-label policy used in the rewritten code

The revised program deliberately uses labels such as:

- **Observed liquid production**
- **CRM estimate during historical calibration**
- **CRM forecast during held-out period**
- **Injector-to-producer connectivity strength**
- **Injector-to-producer response time**
- **Liquid production rate [MSTB/day]**
- **Evaluation number (1–20)**
- **Mean absolute error [MSTB/day]**
- **CRM + XGBoost hybrid**
- **Extreme Learning Machine**
- **Multilayer Perceptron**

The mathematical terminology is retained in the technical documentation, but the figure itself tells the reader what the quantity means.
