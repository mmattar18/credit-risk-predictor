# Credit Default Predictor

Predict whether a credit card client will default on their next payment using machine learning.

## Dataset

UCI Default of Credit Card Clients — 30,000 clients from Taiwan (2005).

**Features:** credit limit, gender, education, marital status, age, 6 months of repayment history, 6 months of bill amounts, 6 months of payment amounts + 14 engineered features (payment ratio, late count, utilization, etc.).

**Target:** binary (1 = default, 0 = no default).

## Results

| Model | ROC-AUC | Accuracy |
|-------|---------|----------|
| Logistic Regression | 0.7713 | 81.5% |
| Random Forest | 0.7780 | 81.8% |
| XGBoost | 0.7781 | 82.0% |

## Feature Engineering

- `payment_ratio` — average amount paid / average bill
- `late_count` — number of months with late payment
- `utilization` — average bill / credit limit
- `max_late` — worst repayment status across 6 months
- `pay_trend` — change in repayment status (Sep vs Apr)
- `ever_2months_late` / `ever_3months_late` — binary flags
- `bill_std` / `paid_std` — volatility in bills and payments
- `credit_per_age` — credit limit relative to age

## Project Structure

```
├── data/                  # Raw dataset (not tracked)
├── notebooks/             # EDA and modeling notebook
├── src/                   # Training and prediction scripts
├── figures/               # Generated plots (not tracked)
├── models/                # Saved models (not tracked)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py
jupyter notebook notebooks/credit_default_analysis.ipynb
```

## Source

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
