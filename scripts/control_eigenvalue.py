import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import math
import os
import glob

# =========================
# AUTO-FIND CUBE FILE
# =========================
files = glob.glob(r"C:\Users\nihal\Desktop\Research\results\cube.npz")

if len(files) == 0:
    raise FileNotFoundError("No cube .npz file found in results/")

print("Available cube files:")
for f in files:
    print(" -", f)

# Pick the first one (you can change this if needed)
MASK_PATH = files[0]
print("\nUsing file:", MASK_PATH)

# =========================
# LOAD VOXEL MASK
# =========================
data = np.load(MASK_PATH)

print("\nKeys inside file:", data.files)

# Automatically pick the correct array
mask = None
for key in data.files:
    arr = data[key]
    if arr.ndim == 3:
        mask = arr
        print("Using key:", key)
        break

if mask is None:
    raise ValueError("No 3D array found in npz file.")

mask = mask.astype(bool)

N = mask.shape[0]
h = 1.0 / (N - 1)

print("\nGrid size N:", N)
print("Voxel spacing h:", h)

# =========================
# BUILD INDEX MAP
# =========================
coords = np.argwhere(mask)
num_voxels = len(coords)

print("Interior voxel count:", num_voxels)

index_map = -np.ones(mask.shape, dtype=int)

for idx, (i, j, k) in enumerate(coords):
    index_map[i, j, k] = idx

# =========================
# BUILD LAPLACIAN
# =========================
print("\nBuilding Laplacian...")
t0 = time.time()

rows = []
cols = []
data_vals = []

diag = 6.0 / h**2
off = -1.0 / h**2

neighbors = [
    (1,0,0),(-1,0,0),
    (0,1,0),(0,-1,0),
    (0,0,1),(0,0,-1)
]

for idx, (i, j, k) in enumerate(coords):

    # Diagonal
    rows.append(idx)
    cols.append(idx)
    data_vals.append(diag)

    # Neighbors
    for di, dj, dk in neighbors:
        ni, nj, nk = i+di, j+dj, k+dk

        if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
            if mask[ni, nj, nk]:
                neighbor_idx = index_map[ni, nj, nk]
                rows.append(idx)
                cols.append(neighbor_idx)
                data_vals.append(off)
        # else → Dirichlet BC

L = sp.csr_matrix((data_vals, (rows, cols)), shape=(num_voxels, num_voxels))

t1 = time.time()
print("Laplacian built in %.2f seconds" % (t1 - t0))
print("Matrix size:", L.shape)
print("Nonzeros:", L.nnz)

# =========================
# COMPUTE EIGENVALUES
# =========================
K = 50
print("\nComputing eigenvalues (K =", K, ")...")

t0 = time.time()

vals, _ = spla.eigsh(L, k=K, which='SM', tol=1e-8)

t1 = time.time()

vals = np.sort(vals)

print("Eigenvalues computed in %.2f seconds" % (t1 - t0))

# =========================
# RESULTS
# =========================
print("\nFirst 5 eigenvalues:")
for i in range(5):
    print(f"λ_{i+1} = {vals[i]:.6f}")

# =========================
# VALIDATION
# =========================
lambda1 = vals[0]
expected = 3 * math.pi**2
error = (lambda1 - expected) / expected

print("\n===== VALIDATION =====")
print(f"λ1 (numerical) = {lambda1:.6f}")
print(f"λ1 (expected)  = {expected:.6f}")
print(f"Relative error = {error:.4%}")

