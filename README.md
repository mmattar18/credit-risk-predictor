# Credit Default Predictor

Predict whether a credit card client will default on their next payment using machine learning.

## Dataset

UCI Default of Credit Card Clients — 30,000 clients from Taiwan (2005).

**Features:** credit limit, gender, education, marital status, age, 6 months of repayment history, 6 months of bill amounts, 6 months of payment amounts.

**Target:** binary (1 = default, 0 = no default).

## Results

| Model | ROC-AUC | F1 (default) |
|-------|---------|--------------|
| Logistic Regression | — | — |
| Random Forest | — | — |
| XGBoost | — | — |

*(filled after running the notebook)*

## Project Structure

```
├── data/                  # Raw dataset
├── notebooks/             # EDA and modeling notebook
├── src/                   # Training and prediction scripts
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/credit_default_analysis.ipynb
```

## Source

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
