import numpy as np
import pandas as pd
from scipy.stats import linregress

# =========================
# PATH
# =========================
csv_path = r"C:\Users\nihal\Desktop\Research\results\descriptors.csv"
df = pd.read_csv(csv_path)

df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df["delta_lambda1"] > 0]

# =========================
# CONFIG
# =========================
EPS_MIN = 0.02
EPS_MAX = 0.10

smooth_shapes = ["sphere", "cylinder", "ellipse"]
faceted_shapes = ["cube", "rectangle"]

# =========================
# FUNCTION: compute p
# =========================
def compute_p(sub):

    sub = sub[(sub["amplitude"] >= EPS_MIN) & (sub["amplitude"] <= EPS_MAX)]

    if len(sub) < 3:
        return None

    x = np.log(sub["amplitude"].values)
    y = np.log(sub["delta_lambda1"].values)

    try:
        p, _, r, _, _ = linregress(x, y)
        return p, r**2
    except:
        return None

# =========================
# MAIN TEST
# =========================
def analyze_group(name, shapes):

    print(f"\n=== {name.upper()} GROUP ===")

    p_vals = []
    r_vals = []

    for s in shapes:
        sub = df[df["shape"] == s]

        result = compute_p(sub)

        if result is None:
            print(f"{s}: insufficient data")
            continue

        p, r2 = result

        p_vals.append(p)
        r_vals.append(r2)

        print(f"{s}: p = {p:.3f}, R² = {r2:.3f}")

    if len(p_vals) > 0:
        print("\n--- GROUP SUMMARY ---")
        print(f"mean p = {np.mean(p_vals):.3f}")
        print(f"std p  = {np.std(p_vals):.3f}")
        print(f"mean R² = {np.mean(r_vals):.3f}")


# =========================
# RUN BOTH GROUPS
# =========================
analyze_group("smooth", smooth_shapes)
analyze_group("faceted", faceted_shapes)