import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"C:\Users\nihal\Desktop\Research\results\descriptors.csv"
df = pd.read_csv(csv_path)

df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df["delta_lambda1"] > 0]

EPS_MIN = 0.02
EPS_MAX = 0.10

plt.figure(figsize=(8,6))

for shape in df["shape"].unique():

    sub = df[(df["shape"] == shape)]
    sub = sub[(sub["amplitude"] >= EPS_MIN) & (sub["amplitude"] <= EPS_MAX)]

    if len(sub) < 3:
        continue

    eps = sub["amplitude"].values
    dl1 = sub["delta_lambda1"].values

    # normalize (removes scale differences)
    eps_n = eps / np.max(eps)
    dl1_n = dl1 / np.max(dl1)

    plt.plot(eps_n, dl1_n, marker='o', label=shape)

plt.xlabel("normalized ε")
plt.ylabel("normalized Δλ₁")
plt.title("Geometry-dependent scaling collapse test")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()