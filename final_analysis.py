import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(r"C:\Users\nihal\Desktop\Research\results\descriptors.csv")

# =========================
# GLOBAL CLEANING (ONLY ONCE)
# =========================
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# CRITICAL: enforce positivity for log stability
df = df[df["delta_lambda1"] > 0]

# =========================
# FIXED PHYSICAL REGIME (LOCKED)
# =========================
EPS_MIN = 0.02
EPS_MAX = 0.1

df = df[(df["amplitude"] >= EPS_MIN) & (df["amplitude"] <= EPS_MAX)]

print("\nTOTAL POINTS IN LOCKED REGIME:", len(df))

# =========================
# GLOBAL FIT (PRIMARY RESULT)
# =========================
results = {}

plt.figure()

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

    results[shape] = (p, C, r**2)

    # plot raw + fit
    eps = np.linspace(sub["amplitude"].min(), sub["amplitude"].max(), 100)
    plt.scatter(sub["amplitude"], sub["delta_lambda1"])
    plt.plot(eps, C * eps**p, linestyle="--")

plt.xscale("log")
plt.yscale("log")
plt.title("Unified Scaling Law (Locked Regime)")
plt.grid(True, which="both")
plt.savefig(r"C:\Users\nihal\Desktop\Research\figs\final_plot.png", dpi=300)
plt.close()

# =========================
# OUTPUT RESULTS
# =========================
print("\n=== FINAL CONSISTENT RESULTS ===\n")

for shape, (p, C, r2) in results.items():
    print(f"{shape}")
    print(f"  p  = {p:.3f}")
    print(f"  C  = {C:.3f}")
    print(f"  R² = {r2:.4f}\n")