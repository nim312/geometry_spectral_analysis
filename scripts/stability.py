import numpy as np
import pandas as pd
from scipy.stats import linregress

csv_path = r"C:\Users\nihal\Desktop\Research\results\descriptors.csv"
df = pd.read_csv(csv_path)

df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df["delta_lambda1"] > 0]

EPS_MIN = 0.02
EPS_MAX = 0.10

print("\n=== GEOMETRY CLASS STABILITY ANALYSIS ===\n")

for shape in df["shape"].unique():

    sub = df[(df["shape"] == shape)]
    sub = sub[(sub["amplitude"] >= EPS_MIN) & (sub["amplitude"] <= EPS_MAX)]

    eps = sub["amplitude"].values
    dl1 = sub["delta_lambda1"].values

    if len(eps) < 3:
        print(f"{shape}: insufficient data ({len(eps)})")
        continue

    x = np.log(eps)
    y = np.log(dl1)

    # full fit
    p_full, _, r, _, _ = linregress(x, y)

    # sub-sampling stability (remove one point at a time)
    p_vals = []

    for i in range(len(eps)):
        mask = np.ones(len(eps), dtype=bool)
        mask[i] = False

        try:
            p, _, r_sub, _, _ = linregress(np.log(eps[mask]), np.log(dl1[mask]))
            p_vals.append(p)
        except:
            continue

    p_vals = np.array(p_vals)

    print(f"\n{shape}")
    print(f"  n = {len(eps)}")
    print(f"  p (full fit) = {p_full:.3f}")
    print(f"  R² = {r**2:.4f}")

    if len(p_vals) > 0:
        print(f"  p range = [{np.min(p_vals):.3f}, {np.max(p_vals):.3f}]")
        print(f"  p std   = {np.std(p_vals):.3f}")
