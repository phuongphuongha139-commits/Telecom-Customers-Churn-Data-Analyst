# Telecom Customer Churn Prediction

A complete, end-to-end machine learning project that predicts which telecom customers are likely to **churn** (stop using the service), so the business can target them with retention efforts before they leave.

> Churn matters because keeping an existing customer is far cheaper than acquiring a new one. The telecom industry sees an annual churn rate of roughly 15–25%, so flagging "high-risk" customers in advance has real financial value.

---

## Project Overview

The work lives in a single notebook, `_TEAM_2__Telecom_Customers_Churn.ipynb`, organized into five sections that take the data from raw CSV all the way to a model recommendation:

1. **Import, Data & Diagnosis** — load the data and find data-quality problems.
2. **Data Visualization (EDA)** — explore who churns and why.
3. **Data Preprocessing** — clean, encode, scale, and split the data.
4. **Machine Learning Modeling** — train and evaluate five classifiers.
5. **Model Comparison & Recommendation** — pick the best model for the business goal.

---

## Dataset

- **Source:** IBM Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Size:** 7,043 rows × 21 columns
- **Target:** `Churn` (`Yes` / `No`), converted to `1` / `0` for modeling
- **Class balance:** about **26.5% churn** vs **73.5% retained** — an imbalanced problem

Feature groups:

| Group | Columns |
|-------|---------|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Contract & Billing | `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `tenure` |
| Identifier (dropped) | `customerID` |

---

## Key Steps

### 1. Data Diagnosis
- Inspected structure with `head()`, `info()`, and null checks.
- Found that `TotalCharges` was stored as text (`object`) because **11 rows contained a blank string `' '`** instead of a number. These all belonged to brand-new customers (`tenure = 0`).
- Coerced the column to numeric and imputed those 11 values as `MonthlyCharges × tenure` (effectively `0`).

### 2. Exploratory Data Analysis
Using pie, bar, KDE, box, scatter plots and a correlation heatmap, the analysis surfaced clear churn drivers:
- **Contract:** month-to-month customers churn heavily; two-year contracts are very loyal.
- **Internet service:** fiber-optic users churn unusually often (a likely price/quality issue).
- **Payment method:** electronic-check users churn the most.
- **Tenure:** churn spikes in the first 0–6 months.
- **Monthly charges:** churners tend to pay more per month.
- **Demographics:** gender has no effect; senior citizens churn more.

This produced two customer profiles — a **High-Risk** persona (new, high-bill, fiber, month-to-month, e-check) and a **Loyal** persona (long-tenure, long contract).

### 3. Preprocessing
- Dropped `customerID`; finalized `TotalCharges` cleaning.
- Built scikit-learn pipelines:
  - **Numeric:** median imputation → `StandardScaler`
  - **Categorical:** most-frequent imputation → `OneHotEncoder`
- Combined them with a `ColumnTransformer`.
- Split 80/20 with `stratify=y` and `random_state=42` to preserve class balance.

### 4. Modeling
Five classifiers were trained, all configured to handle class imbalance (`class_weight='balanced'` or `scale_pos_weight`):

- Random Forest
- XGBoost
- LightGBM
- Logistic Regression
- CatBoost

Each was evaluated with a classification report, confusion matrix, and ROC AUC. Because missing a real churner is costly, **recall on the churn class** was prioritized over raw accuracy.

### 5. Comparison & Recommendation
Metrics were consolidated into a table and visualized with bar charts.

| Model | Accuracy | Churn Recall | ROC AUC |
|-------|----------|--------------|---------|
| Random Forest | ~0.77 | 0.80 | 0.831 |
| **XGBoost** | 0.75 | **0.81** | **0.846** |
| LightGBM | ~0.77 | 0.76 | 0.833 |
| Logistic Regression | 0.73 | 0.82 | 0.837 |
| CatBoost | 0.75 | 0.79 | 0.846 |

**Recommendation:** **XGBoost** (with **Logistic Regression** as a strong, simpler alternative) — both deliver the best churn recall and ROC AUC, which best supports proactive retention.

### Bonus Experiments
The notebook also explores a **random under-sampling** variant and a **hybrid rule-based override** (force a "churn" prediction for month-to-month customers with `tenure < 6`) layered on top of the XGBoost output.

---

## Tech Stack

- **Python**, **Jupyter Notebook**
- **pandas**, **numpy** — data handling
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — pipelines, preprocessing, Logistic Regression, Random Forest, metrics
- **xgboost**, **lightgbm**, **catboost** — gradient-boosting models
- **imbalanced-learn (imblearn)** — imbalance handling / under-sampling
- **joblib** — model persistence

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn \
            xgboost lightgbm catboost imbalanced-learn jupyter

# 2. Place the dataset next to the notebook
#    WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. Launch and run all cells
jupyter notebook _TEAM_2__Telecom_Customers_Churn.ipynb
```

> **Note:** The data-loading cell uses a placeholder path. Update it to point to your local CSV, e.g.
> ```python
> df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
> ```

---

## Business Takeaways

- Focus retention on **new, month-to-month, high-bill, fiber-optic, electronic-check** customers — they churn the most.
- Encourage longer contracts and automatic payment methods to improve loyalty.
- Use the XGBoost model to score customers and trigger early outreach within their first few months.

---

*Built by **Team 2** as a telecom customer churn analysis and prediction project.*
