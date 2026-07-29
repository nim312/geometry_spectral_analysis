import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

CSV_N20 = r"C:\Users\nihal\Desktop\Research\results\descriptors_20.csv"
CSV_N25 = r"C:\Users\nihal\Desktop\Research\results\descriptors.csv"

OUT_FIG = r"C:\Users\nihal\Desktop\Research\figs\res_compare.png"
OUT_SUMMARY = r"C:\Users\nihal\Desktop\Research\results\res_compare.csv"

SHAPES = ["sphere", "cube"]
FIT_MIN = 0.05
FIT_MAX = 0.10

def load_clean(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["delta_lambda1"] > 0].copy()
    return df

def fit_powerlaw(sub):
    sub = sub[(sub["amplitude"] >= FIT_MIN) & (sub["amplitude"] <= FIT_MAX)].copy()
    if len(sub) < 2:
        return None

    x = np.log(sub["amplitude"].values)
    y = np.log(sub["delta_lambda1"].values)
    p, b, r, _, _ = linregress(x, y)
    return {
        "p": float(p),
        "C": float(np.exp(b)),
        "R2": float(r**2),
        "n": int(len(sub)),
    }

df20 = load_clean(CSV_N20)
df25 = load_clean(CSV_N25)

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)

summary_rows = []

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, shape in zip(axes, SHAPES):
    for label, df in [("N=20", df20), ("N=25", df25)]:
        sub = df[df["shape"] == shape].sort_values("amplitude")
        if len(sub) == 0:
            continue

        ax.plot(
            sub["amplitude"].values,
            sub["delta_lambda1"].values,
            marker="o",
            linewidth=2,
            label=label
        )

        fit = fit_powerlaw(sub)
        if fit is not None:
            summary_rows.append({
                "shape": shape,
                "resolution": label,
                **fit
            })

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(shape.capitalize())
    ax.set_xlabel("amplitude")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

axes[0].set_ylabel("delta_lambda1")
fig.suptitle("Resolution Robustness Check: N=20 vs N=25")
fig.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.show()

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUMMARY, index=False)

print("\n=== RESOLUTION COMPARISON SUMMARY ===\n")
print(summary_df.to_string(index=False))
print(f"\nSaved figure -> {OUT_FIG}")
print(f"Saved summary -> {OUT_SUMMARY}")
