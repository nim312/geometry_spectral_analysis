import numpy as np
import matplotlib.pyplot as plt

data = {
    "Sphere": {
        "eps": np.array([0.05, 0.08, 0.10]),
        "delta": np.array([0.018901066117777532,
                           0.08923284464102466,
                           0.19445406197938553])
    },
    "Cylinder": {
        "eps": np.array([0.01, 0.02, 0.05, 0.08, 0.10]),
        "delta": np.array([0.5287696415195597,
                           1.6956757701381093,
                           5.98962789873368,
                           9.946106390297942,
                           13.523777959214584])
    },
    "Elliptic Prism": {
        "eps": np.array([0.01, 0.02, 0.05, 0.08, 0.10]),
        "delta": np.array([1.0617937046193546,
                           2.9499401016361375,
                           12.372693700159537,
                           20.323629238449165,
                           27.007734314699952])
    },
    "Cube": {
        "eps": np.array([0.05, 0.08, 0.10]),
        "delta": np.array([0.21992583637670648,
                           1.1029084444544708,
                           2.1997098537663184])
    },
    "Rectangular Prism": {
        "eps": np.array([0.05, 0.08, 0.10]),
        "delta": np.array([0.29628355618601177,
                           1.2069677522072766,
                           2.105105915983124])
    }
}
plt.figure(figsize=(8,6))
for name, d in data.items():
    eps = d["eps"]
    delta = d["delta"]
    p_local = np.diff(np.log(delta)) / np.diff(np.log(eps))
    eps_mid = (eps[:-1] + eps[1:]) / 2
    plt.plot(eps_mid, p_local, marker='o', linewidth=2, label=name)

plt.xlabel("Amplitude ε")
plt.ylabel("Local exponent p")
plt.title("Local Scaling Exponent for Δλ₁")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(r"scaling_exponent.png", dpi=300)
plt.show()
