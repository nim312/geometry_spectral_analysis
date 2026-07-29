import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import math
import time

# =========================
# LOAD BASE MASK (N=25)
# =========================
data = np.load(r"C:\Users\nihal\Desktop\Research\results\cube.npz")
mask25 = data[list(data.files)[0]].astype(bool)

# =========================
# RESAMPLING FUNCTION
# =========================
def resample_mask(mask, N_new):
    oldN = mask.shape[0]
    factor = (N_new - 1) / (oldN - 1)
    new_mask = ndi.zoom(mask.astype(float), factor, order=0)
    return new_mask.astype(bool)

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

    L = sp.csr_matrix((data_vals, (rows, cols)))
    return L

# =========================
# COMPUTE LAMBDA1
# =========================
def compute_lambda1(mask):
    L = build_laplacian(mask)
    vals, _ = spla.eigsh(L, k=3, which='SM', tol=1e-8)
    vals = np.sort(vals)
    return vals[0]

# =========================
# RUN STUDY
# =========================
Ns = [20, 25, 30]

hs = []
errors = []
lambdas = []

expected = 3 * math.pi**2

for N in Ns:
    print(f"\nRunning N = {N}")

    if N == 25:
        mask = mask25
    else:
        mask = resample_mask(mask25, N)

    h = 1.0 / (N - 1)

    start = time.time()
    lam = compute_lambda1(mask)
    end = time.time()

    err = abs(lam - expected)

    print(f"λ1 = {lam:.6f}")
    print(f"Error = {err:.6e}")
    print(f"Time = {end-start:.2f}s")

    hs.append(h)
    errors.append(err)
    lambdas.append(lam)

# =========================
# FIT CONVERGENCE RATE
# =========================
hs = np.array(hs)
errors = np.array(errors)

coeffs = np.polyfit(np.log(hs), np.log(errors), 1)
p = coeffs[0]

print("\nEstimated convergence order p =", p)

# =========================
# PLOT
# =========================
plt.figure()
plt.loglog(hs, errors, 'o-', label='Error')
plt.loglog(hs, np.exp(coeffs[1]) * hs**p, '--', label=f'Fit (p={p:.2f})')

plt.xlabel("h")
plt.ylabel("Error")
plt.legend()
plt.title("Grid Convergence (Cube λ₁)")

plt.savefig(r"C:\Users\nihal\Desktop\Research\figs\convergence.png", dpi=300)
plt.show()
