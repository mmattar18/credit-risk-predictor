import os
from ucimlrepo import fetch_ucirepo

os.makedirs("data", exist_ok=True)
dataset = fetch_ucirepo(id=350)
df = dataset.data.original
df.to_csv("data/credit_default.csv", index=False)
print(f"Downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
