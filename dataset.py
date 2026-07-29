import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
import glob
import csv
import time

# =========================
# PARAMETERS
# =========================
K = 80  # number of modes to compute
RESULTS_DIR = r"C:\Users\nihal\Desktop\Research\results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "descriptors.csv")

# amplitude mapping for files
amplitude_map = {1: 0.01, 2: 0.02, 3: 0.05, 4: 0.08, 5: 0.1}

# =========================
# LOAD VOXEL MASK
# =========================
def load_mask(path):
    data = np.load(path)
    for key in data.files:
        arr = data[key]
        if arr.ndim == 3:
            return arr.astype(bool)
    raise ValueError(f"No valid mask in {path}")

# =========================
# BUILD MASKED LAPLACIAN
# =========================
def build_laplacian(mask):
    N = mask.shape[0]
    h = 1.0 / (N - 1)

    coords = np.argwhere(mask)
    index_map = -np.ones(mask.shape, dtype=int)
    for idx, (i, j, k) in enumerate(coords):
        index_map[i, j, k] = idx

    rows, cols, data_vals = [], [], []

    diag = 6.0 / h**2
    off = -1.0 / h**2
    neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

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
# COMPUTE EIGENVALUES
# =========================
def compute_eigs(L, K):
    vals, _ = spla.eigsh(L, k=K, which='SM', tol=1e-8)
    return np.sort(vals)

# =========================
# COMPUTE DELTA ENERGY
# =========================
def delta_energy(vals_base, vals):
    return 0.5 * np.sum(np.sqrt(vals) - np.sqrt(vals_base))

# =========================
# FIND FILES
# =========================
files = glob.glob(os.path.join(RESULTS_DIR, "*.npz"))

# Group files by shape
shape_groups = {}
for f in files:
    name = os.path.basename(f)
    parts = name.replace(".npz","").split("_")
    shape = parts[0]  # cube, sphere, etc.
    shape_groups.setdefault(shape, []).append(f)

# =========================
# PREP OUTPUT
# =========================
os.makedirs(RESULTS_DIR, exist_ok=True)
header = ["shape", "amplitude", "lambda1", "delta_lambda1", "delta_E"]
rows = []

# =========================
# MAIN LOOP
# =========================
for shape, flist in shape_groups.items():
    print(f"\nProcessing shape: {shape}")

    # Base file: no number
    base_file = None
    for f in flist:
        if f.endswith(f"{shape}.npz"):
            base_file = f
            break
    if base_file is None:
        print(f"No base file for {shape}, skipping")
        continue

    # Load base
    mask_base = load_mask(base_file)
    L_base = build_laplacian(mask_base)
    vals_base = compute_eigs(L_base, K)
    lambda1_base = vals_base[0]

    # Process each file
    for f in flist:
        name = os.path.basename(f)
        if f == base_file:
            amp = 0.0
        else:
            try:
                idx = int(name.split("_")[-1].replace(".npz",""))
                amp = amplitude_map.get(idx, 0.0)
            except:
                amp = 0.0

        print(f"  Amplitude: {amp}")
        mask = load_mask(f)
        L = build_laplacian(mask)
        vals = compute_eigs(L, K)

        lambda1 = vals[0]
        d_lambda1 = lambda1 - lambda1_base
        dE = delta_energy(vals_base, vals)

        rows.append([shape, amp, lambda1, d_lambda1, dE])

# =========================
# SAVE CSV
# =========================
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\nDataset saved to: {OUTPUT_FILE}")