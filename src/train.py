import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

df = pd.read_csv("data/credit_default.csv")
df.columns = [
    "id", "credit_limit", "sex", "education", "marriage", "age",
    "pay_sep", "pay_aug", "pay_jul", "pay_jun", "pay_may", "pay_apr",
    "bill_sep", "bill_aug", "bill_jul", "bill_jun", "bill_may", "bill_apr",
    "paid_sep", "paid_aug", "paid_jul", "paid_jun", "paid_may", "paid_apr",
    "default",
]
df = df.drop(columns=["id"])

pay_cols = ["pay_sep", "pay_aug", "pay_jul", "pay_jun", "pay_may", "pay_apr"]
bill_cols = ["bill_sep", "bill_aug", "bill_jul", "bill_jun", "bill_may", "bill_apr"]
paid_cols = ["paid_sep", "paid_aug", "paid_jul", "paid_jun", "paid_may", "paid_apr"]

df["avg_bill"] = df[bill_cols].mean(axis=1)
df["avg_paid"] = df[paid_cols].mean(axis=1)
df["payment_ratio"] = df["avg_paid"] / (df["avg_bill"] + 1)
df["late_count"] = (df[pay_cols] > 0).sum(axis=1)
df["utilization"] = df["avg_bill"] / (df["credit_limit"] + 1)
df["max_late"] = df[pay_cols].max(axis=1)
df["pay_trend"] = df["pay_sep"] - df["pay_apr"]
df["bill_std"] = df[bill_cols].std(axis=1)
df["paid_std"] = df[paid_cols].std(axis=1)
df["total_paid"] = df[paid_cols].sum(axis=1)
df["total_bill"] = df[bill_cols].sum(axis=1)
df["total_payment_ratio"] = df["total_paid"] / (df["total_bill"] + 1)
df["ever_2months_late"] = (df[pay_cols] >= 2).any(axis=1).astype(int)
df["credit_per_age"] = df["credit_limit"] / (df["age"] + 1)
df = df.fillna(0)

X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=scale_pos, eval_metric="logloss", random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_credit_default.pkl")
print("Model saved to models/xgb_credit_default.pkl")
