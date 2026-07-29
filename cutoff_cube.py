import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import glob

# =========================
# PARAMETERS
# =========================
K_values = [30, 50, 80, 120]
TOL = 0.05  # 5% plateau criterion

# =========================
# LOAD FILES (EDIT IF NEEDED)
# =========================
base_file = sorted(glob.glob(r"C:\Users\nihal\Desktop\Research\results\cube.npz"))[0]
wavy_file = sorted(glob.glob(r"C:\Users\nihal\Desktop\Research\results\cube_1.npz"))[0]

print("Base file:", base_file)
print("Wavy file:", wavy_file)

# =========================
# LOAD MASK
# =========================
def load_mask(path):
    data = np.load(path)
    for key in data.files:
        arr = data[key]
        if arr.ndim == 3:
            return arr.astype(bool)
    raise ValueError("No valid 3D mask found")

# =========================
# BUILD LAPLACIAN
# =========================
def build_laplacian(mask):
    N = mask.shape[0]
    h = 1.0 / (N - 1)

    coords = np.argwhere(mask)
    index_map = -np.ones(mask.shape, dtype=int)

    for idx, (i,j,k) in enumerate(coords):
        index_map[i,j,k] = idx

    rows, cols, data_vals = [], [], []

    diag = 6.0 / h**2
    off = -1.0 / h**2

    neighbors = [
        (1,0,0),(-1,0,0),
        (0,1,0),(0,-1,0),
        (0,0,1),(0,0,-1)
    ]

    for idx, (i,j,k) in enumerate(coords):
        rows.append(idx)
        cols.append(idx)
        data_vals.append(diag)

        for di,dj,dk in neighbors:
            ni,nj,nk = i+di, j+dj, k+dk
            if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                if mask[ni,nj,nk]:
                    rows.append(idx)
                    cols.append(index_map[ni,nj,nk])
                    data_vals.append(off)

    return sp.csr_matrix((data_vals, (rows, cols)))

# =========================
# DELTA ENERGY (STABLE)
# =========================
def compute_delta_energy(L_base, L_wavy, K):
    vals_base, _ = spla.eigsh(L_base, k=K, which='SM', tol=1e-8)
    vals_wavy, _ = spla.eigsh(L_wavy, k=K, which='SM', tol=1e-8)

    vals_base = np.sort(vals_base)
    vals_wavy = np.sort(vals_wavy)

    # Mode-by-mode subtraction
    delta_E = 0.5 * np.sum(np.sqrt(vals_wavy) - np.sqrt(vals_base))
    return delta_E

# =========================
# MAIN
# =========================
mask_base = load_mask(base_file)
mask_wavy = load_mask(wavy_file)

print("\nBuilding Laplacians...")
L_base = build_laplacian(mask_base)
L_wavy = build_laplacian(mask_wavy)

results = []

# =========================
# RUN STUDY
# =========================
for K in K_values:
    print(f"\nK = {K}")

    start = time.time()
    dE = compute_delta_energy(L_base, L_wavy, K)
    end = time.time()

    print(f"ΔE = {dE:.8f}")
    print(f"Time = {end - start:.2f}s")

    results.append((K, dE))

# =========================
# ANALYSIS
# =========================
Ks = np.array([r[0] for r in results])
dEs = np.array([r[1] for r in results])

print("\nRelative changes:")

plateau_K = None

for i in range(1, len(dEs)):
    if abs(dEs[i-1]) > 1e-12:
        rel = abs(dEs[i] - dEs[i-1]) / abs(dEs[i-1])
        print(f"K {Ks[i-1]} → {Ks[i]}: {rel:.2%}")

        if rel < TOL and plateau_K is None:
            plateau_K = Ks[i]
    else:
        print(f"K {Ks[i-1]} → {Ks[i]}: undefined (too small)")

# =========================
# DECISION
# =========================
if plateau_K is not None:
    print(f"\n Plateau detected at K ≈ {plateau_K}")
else:
    print("\n No clean plateau detected — inspect manually")

# Warn about instability
if dEs[-1] > 1.5 * dEs[-2]:
    print("\n High-K instability detected (mode mismatch likely)")

# =========================
# PLOT
# =========================
plt.figure()
plt.plot(Ks, dEs, 'o-', linewidth=2)
plt.xlabel("K (number of modes)")
plt.ylabel("ΔE")
plt.title("Mode Cutoff Independence")
plt.grid()

plt.savefig(r"C:\Users\nihal\Desktop\Research\figs", dpi=300)
plt.show()