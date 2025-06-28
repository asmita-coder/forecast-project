Hi! I’m Asmita — and in this dev notebook series, we’ll walk through how to build a real-world, production-style **forecasting pipeline**. We’ll use dummy data, but the patterns, principles, and pain points will all mirror what real forecasting systems look like.

But before we dive into code or Docker or MLflow, I want to lay the groundwork — by walking you through the **key concepts behind forecasting**. If you're an ML practitioner who has only done classification or regression work, this will help you leap into time series forecasting.

---

## 🚀 Why Forecasting? (Very Briefly)

Forecasting is one of the most practical, high-impact areas in applied machine learning.

It’s used to answer questions like:
- How many units should we stock next month?
- What will user engagement look like this quarter?
- How will demand shift across regions next season?

While ML folks often start with classification or regression, forecasting introduces a unique twist: **time**. And that changes everything — from how you split your data to how you train and validate models.

---

## 🧠 Forecasting Concepts You Should Understand

Let’s walk through the foundations — each explained in plain terms with practical examples.

---

### ⏱️ Forecast Horizon

This is **how far into the future** you want to predict.

**Example:**  
If your data is daily and today is June 30, and you want to forecast until July 31,  
your **forecast horizon = 31 days**.

_If your time series is daily, the horizon is measured in days. If it’s weekly, it’s in weeks — and so on_

| Frequency | Horizon Value | What it Means           |
| --------- | ------------- | ----------------------- |
| Daily     | 30            | Forecast 30 days ahead  |
| Weekly    | 12            | Forecast 12 weeks ahead |
| Monthly   | 6             | Forecast 6 months ahead |


The length of your horizon impacts:
- How much past data is needed
- Which models are appropriate (e.g. some DL models struggle with long horizons)
- How error compounds over time

In our project, we’ll set the horizon upfront and design everything else to respect it.

---

### 🧮 Cutoff Point (a.k.a. Forecast Origin)

The **cutoff** is the last date in the training data.  
We train the model on everything before this point, then forecast **after** it.

**Think of it like a snapshot in time**:  
You freeze the data on day `T`, and the model must predict the future from that point on.

This helps:
- Separate training from evaluation
- Simulate real-world forecasting (where you don’t know the future)

We'll simulate multiple cutoff points later to evaluate our models more robustly.

---

### 📆 Data Frequency

Forecasting is very sensitive to **time frequency** — is your data daily, weekly, monthly?

- Daily: can capture short-term trends, but can be noisy
- Weekly: often used for retail or operations
- Monthly: common for economic indicators

The **frequency must be consistent** — we’ll resample or impute if we find irregular timestamps.

---

### 🧩 Covariates (a.k.a. Features That Help)

Covariates are **additional variables** that help models understand context.

#### 🔹 Static Covariates
- Don’t change over time
- Examples: product category, store type, region

#### 🔹 Dynamic Covariates
- Change at every time step
- Examples: day of week, holiday flag, temperature, promotion status

These are especially helpful in:
- Deep learning models (which learn complex interactions)
- Situations with many related series (products, locations, etc.)

We'll engineer both types from our dummy data.

---

### 🔁 Lag Features

Sometimes the best predictor of the future is... the past.

Lag features are previous values of the target — we shift them backward in time.

**Example:**
- `y(t-1)`: value one step before
- `y(t-7)`: value a week ago

Useful when there are **seasonal or autocorrelated patterns** — like demand spikes every Monday.

```

Time →     t-3     t-2     t-1     t     t+1     t+2

Target y →  100     105     108     ?      ?       ?

Lag-1  →          100     105    108      
Lag-2  →          105     108
Lag-3  →          108
```

We train a model like:

```
y(t) = f(y(t-1), y(t-2), y(t-3))

```
---

### ♻️ Seasonality

**Seasonality** is a repeating pattern over fixed time intervals.

Types:
- **Daily**: More sales at lunch and dinner time
- **Weekly**: Weekday vs weekend patterns
- **Yearly**: Holiday seasons, school schedules

We’ll simulate weekly and monthly seasonality in our dummy data, and use models that handle it well.

---

### 🧱 Exogenous Variables

Sometimes you'll hear "exogenous variables" — that’s just a fancy word for **covariates that aren’t the target**.

They help give the model external context:
- Weather, pricing, calendar effects
- Macroeconomic indicators

Some models (like AutoARIMA) support these directly. Others (like DL models) can take them as inputs.

---

### 🧹 Missing Values & Gaps

Time series data often comes with:
- Missing timestamps (e.g., no data on holidays)
- NaNs in the target or covariates
- Irregular intervals

You can't ignore these in forecasting — models assume regular time steps.

We’ll:
- Use imputation (forward fill, interpolation)
- Drop incomplete series
- Ensure regular date ranges per ID

---

### 🧪 Backtesting / Rolling Forecasts

Unlike typical machine learning, where train/test splits are often random, forecasting relies on temporal splits — because time matters.

To evaluate performance realistically, we simulate real-world conditions using **backtesting** (also called **rolling forecasts**). This involves:

- Choosing a cutoff point in time
- Training the model on data before the cutoff
- Forecasting into the future for the next N steps
- Repeating the process by shifting the cutoff forward


This technique helps us assess:

- Whether the model performs consistently across different time periods
- If it overfits to one specific point in time
- How well it adapts to
 evolving patterns in the data

We’ll implement backtesting in our pipeline.

---

### 📊 Forecast Evaluation Metrics

Choosing the right metric matters. Common ones:

| Metric | What It Measures | When to Use |
|--------|------------------|-------------|
| **MAE** | Average absolute error | Simple, interpretable |
| **RMSE** | Root mean squared error | Penalizes large errors more |
| **MAPE** | Mean absolute percentage error | Good for scale-invariant evaluation |
| **sMAPE** | Symmetric MAPE | Handles zeros better |

We’ll use a mix of these to compare models.

---

### 🔍 Common Forecasting Models (We'll Use Some of These Later)

Here are some widely-used forecasting algorithms, both classical and deep learning:

| Model | Type | Works Well For |
|-------|------|----------------|
| **Naive Forecast** | Baseline | Just repeats the last value |
| **AutoARIMA** | Statistical | Trend/seasonal series |
| **ETS** | Statistical | Captures level + trend + seasonality |
| **Prophet** | Hybrid | User-friendly, calendar-aware |
| **LightGBM / XGBoost** | ML | Forecasting as regression |
| **NBEATSx / NHITS / TFT** | Deep Learning | Complex, high-variance series |

We'll experiment with both classical and deep models in this series.

---

## 🛠️ Coming Next: Data Setup

In the next notebook, we’ll:
- Generate a dummy dataset with multiple series
- Create group labels and covariates
- Set a forecast horizon and cutoff
- Visualize and inspect our training + forecasting splits

This will be the foundation for building the full pipeline — from model training to logging to SHAP explainability.

Let’s build something real — and let’s make it solid.  
See you in **Dev Notebook – Part 1**.

— Asmita ✨
