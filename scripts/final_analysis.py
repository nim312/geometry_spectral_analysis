import numpy as np
import pandas as pd
from scipy.stats import linregress

df = pd.read_csv(r"descriptors_25.csv")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df = df[df["delta_lambda1"] > 0]
EPS_MIN = 0.02
EPS_MAX = 0.1
df = df[(df["amplitude"] >= EPS_MIN) & (df["amplitude"] <= EPS_MAX)]
results = {}

for shape in df["shape"].unique():
    sub = df[df["shape"] == shape]
    if len(sub) < 3:
        print(f"{shape}: skipped (not enough points)")
        continue
    x = np.log(sub["amplitude"].values)
    y = np.log(sub["delta_lambda1"].values)
    slope, intercept, r, _, _ = linregress(x, y)
    p = slope
    C = np.exp(intercept)
    results[shape] = (p, r**2)

for shape, (p, r2) in results.items():
    print(f"{shape}")
    print(f"  p  = {p:.3f}")
    print(f"  R² = {r2:.4f}\n")
