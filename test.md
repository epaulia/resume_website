# Forecasting + Anomaly Detection in Food Cost Per Manday

I built a full data pipeline to forecast and detect anomalies in food cost per person per day ("per manday") across various work camps. The goal was to proactively flag spikes or unusual behavior in food costs, normalized by occupancy levels.

This solution integrates:
- **Invoice Data** (daily food costs per camp)
- **Loading Data** (daily occupancy per camp)

From these, I computed a normalized per-manday cost trend and applied a combination of statistical checks, anomaly logic, and ARIMA forecasting to flag past and future cost outliers.

---

## 📈 Architecture Overview

The pipeline I developed runs per camp and follows this sequence:

1. **Data Loading**
2. **Manday Normalization** (`cost / occupancy`)
3. **Stationarity Enforcement** (ADF test + first differencing)
4. **Anomaly Detection** on historical trends
5. **ARIMA Forecasting** (5-day horizon)
6. **Anomaly Detection** on predicted values

Each module is fully modular and encapsulated, allowing for clean testing and flexible reuse.

---

## 🧪 Stationarity Enforcement via Recursive Differencing

I began each time series by checking for stationarity using the **Augmented Dickey-Fuller (ADF)** test. This statistical test checks for the presence of a unit root in the series. The test statistic (0th element) and p-value (1st element) were central to my logic:
- If the **p-value < 0.05**, I rejected the null hypothesis and considered the data stationary.
- I also validated that the test statistic fell below the critical values at standard confidence thresholds (10%, 5%, 1%).

If the series failed this check, I applied **first differencing** to stabilize the mean and remove trends:
```python
df_stationary = df.diff().dropna()
```
This operation was applied recursively: after each differencing pass, I reran the ADF test. This allowed me to ensure that the differencing degree `d` in the ARIMA model was not chosen arbitrarily — I backed it up with empirical tests.

This recursive enforcement of stationarity was essential. Feeding a non-stationary series into ARIMA can lead to biased coefficients and unstable forecasts. By rigorously checking and transforming the input, I preserved model integrity.

---

## 📉 Historical Anomaly Detection Logic

With a stationary series in place, I ran anomaly detection logic on historical trends.

The idea was to compare **recent values** against a **stable historical baseline**. To compute this baseline, I:
- Calculated a **10-day rolling mean**
- Calculated a **10-day rolling standard deviation**
- Took the **overall average** of those rolling means and standard deviations

This gave me two stable reference points. From there, I defined the anomaly bounds as:
```python
upper_bound = overall_mean + threshold * overall_std
lower_bound = overall_mean - threshold * overall_std
```
where `threshold = 2` by default (i.e., 95.45% of values in a normal distribution).

But I didn’t blindly apply that threshold. If the average rolling std exceeded the average mean — indicating high volatility or skew — I interpreted the series as too noisy for std-based flagging and skipped anomaly detection. This protected the system from over-flagging in turbulent data regimes.

If the thresholding condition was satisfied, I evaluated the **last N days** (configurable via `how_far`) and flagged any outliers beyond the defined range. These were surfaced as anomalies.

The final output was a filtered DataFrame of anomalies, enriched with contextual columns like the computed mean and std multiplier.

**Code snippet:**
```python
rolling_means = historical_data['per_manday'].rolling(window).mean()
rolling_stds = historical_data['per_manday'].rolling(window).std()
overall_mean = rolling_means.mean()
overall_std = rolling_stds.mean()

if overall_std < overall_mean:
    upper = overall_mean + threshold * overall_std
    lower = overall_mean - threshold * overall_std
    recent = df_after_check.iloc[-how_far:]
    anomalies = recent[(recent['per_manday'] < lower) | (recent['per_manday'] > upper)]
```

---

## 🔮 Forward Forecasting with Auto-ARIMA

I extended the pipeline by predicting future behavior using ARIMA. I specifically used **Auto-ARIMA** (`pmdarima.auto_arima`) to avoid manual tuning.

The configuration was:
- `start_p=0`, `max_p=5`, `max_q=5`, `max_d=2`
- `seasonal=False` (since no strong weekly or monthly patterns were observed)
- `stepwise=True` for faster convergence

I didn’t just fit the model blindly. I split the time series into **training and validation folds** using `TimeSeriesSplit(n_splits=2)`. For each fold, I computed the **mean squared error (MSE)** between predicted and actual values. This provided tangible feedback on model accuracy.

After tuning, I forecasted the **next 5 days**, along with **confidence intervals**. I generated synthetic timestamps for the next 5 days and merged the predictions into the historical timeline for a unified view.

**Code snippet:**
```python
ts = TimeSeriesSplit(n_splits=2)
for train_idx, test_idx in ts.split(daily_data['per_manday']):
    train, test = daily_data['per_manday'][train_idx], daily_data['per_manday'][test_idx]
    model = pm.auto_arima(train, seasonal=False, stepwise=True)
    forecast = model.predict(n_periods=len(test))

forecasts = model.predict(n_periods=days_predict)
```

This gave me a projected view of future costs — ready for downstream analysis.

---

## 🔍 Future Anomaly Detection

Once the forecast was appended to the timeline, I re-applied the anomaly detection logic to the forecasted segment only.

This reused the same global bounds (mean ± `X × std`) calculated earlier. The idea here was consistency — the same criteria used to judge past anomalies would be used to judge future ones.

By doing this, I enabled **forward-looking anomaly detection**. If the model predicted a spike outside the acceptable range, the system flagged it as a **future anomaly**, which could then trigger alerts or follow-ups.

This part of the architecture transforms the system from a retrospective dashboard into a proactive risk detection tool.

---

## 🧠 Design Highlights

- **Recursive ADF-Driven Differencing**: I didn’t guess the differencing parameter. I tested for it, recursively enforced it, and ensured statistical validity before modeling.
- **Dynamic Anomaly Thresholding**: I computed rolling means/stdevs, then averaged them to build stable reference points. I adapted the detection logic dynamically based on volatility.
- **Forecast Integration**: I carefully aligned the forecast with the historical timeline and treated predicted values as first-class citizens for anomaly detection.
- **Cross-Validation for Time Series**: I used time-aware splits to validate model accuracy and avoid overfitting.
- **Camp-Level Isolation**: Every camp had its own pipeline instance, allowing for independent forecasting, anomaly bounds, and exception management.
- **Configurable Thresholding and Lookahead**: Thresholds (`X`) and prediction horizons (`N`) were modular and easily tunable for experimentation.

---

This project was a deep dive into applied time series forecasting and anomaly detection. It showcases my ability to:
- Design end-to-end analytical systems
- Implement statistical tests and transformation logic
- Tune, evaluate, and deploy classical models
- Build pipelines that are explainable, modular, and production-ready

If you’d like to discuss the system architecture or see example outputs, feel free to reach out.

