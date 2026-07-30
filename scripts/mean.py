import numpy as np
import pandas as pd
from scipy.stats import linregress

csv_path = r"C:\Users\nihal\Desktop\Research\results\descriptors_25.csv"
df = pd.read_csv(csv_path)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df["delta_lambda1"] > 0]

EPS_MIN = 0.02
EPS_MAX = 0.10
smooth = ["sphere", "cylinder", "ellipse"]
faceted = ["cube", "rectangle"]

def compute_p(sub):
    eps = sub["amplitude"].values
    dl1 = sub["delta_lambda1"].values
    if len(eps) < 3:
        return None
    x = np.log(eps)
    y = np.log(dl1)
    p, _, r, _, _ = linregress(x, y)
    return p

def analyze(group, name):
    p_vals = []
    for shape in group:
        sub = df[df["shape"] == shape]
        sub = sub[(sub["amplitude"] >= EPS_MIN) & (sub["amplitude"] <= EPS_MAX)]
        p = compute_p(sub)
        if p is not None:
            p_vals.append(p)
    p_vals = np.array(p_vals)

    print(f"\n{name} CLASS")
    print(f"mean p = {np.mean(p_vals):.3f}")
    print(f"standard dev p  = {np.std(p_vals):.3f}")
    print(f"shapes = {len(p_vals)}")
    
analyze(smooth, "SMOOTH")
analyze(faceted, "FACETED")
