import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import csv
import gc
import os

# Voxel files with resolution 20
FILES = [
    ("sphere", 0.00, r"results_20\sphere.npz"),
    ("sphere", 0.01, r"results_20\sphere_1.npz"),
    ("sphere", 0.02, r"results_20\sphere_2.npz"),
    ("sphere", 0.05, r"results_20\sphere_3.npz"),
    ("sphere", 0.08, r"results_20\sphere_4.npz"),
    ("sphere", 0.10, r"results_20\sphere_5.npz"),

    ("cube", 0.00, r"results_20\cube.npz"),
    ("cube", 0.01, r"results_20\cube_1.npz"),
    ("cube", 0.02, r"results_20\cube_2.npz"),
    ("cube", 0.05, r"results_20\cube_3.npz"),
    ("cube", 0.08, r"results_20\cube_4.npz"),
    ("cube", 0.10, r"results_20\cube_5.npz"),
]

OUT_CSV = r"descriptors_20.csv"

# Discrete Laplacian
def build_laplacian(mask: np.ndarray, h: float) -> sp.csr_matrix:
    nx, ny, nz = mask.shape
    index_map = -np.ones(mask.shape, dtype=np.int32)
    coords = np.argwhere(mask)

    for idx, (i, j, k) in enumerate(coords):
        index_map[i, j, k] = idx

    n = len(coords)
    A = sp.lil_matrix((n, n), dtype=np.float64)

    for idx, (i, j, k) in enumerate(coords):
        A[idx, idx] = 6.0 / (h * h)

        for di, dj, dk in [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1)
        ]:
            ni, nj, nk = i + di, j + dj, k + dk

            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                if mask[ni, nj, nk]:
                    jdx = index_map[ni, nj, nk]
                    A[idx, jdx] = -1.0 / (h * h)

    return A.tocsr()

def compute_lambda1(mask: np.ndarray) -> float:
    N = mask.shape[0]
    h = 1 / (N - 1)

    A = build_laplacian(mask, h)
    vals = spla.eigsh(A, k=1, which="SM", return_eigenvectors=False, tol=1e-10)
    return float(vals[0])


rows = []
baseline = {}

for shape, amp, path in FILES:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    data = np.load(path)
    if "mask" not in data:
        raise ValueError(f"{path} does not contain a 'mask' array")

    mask = data["mask"].astype(bool)

    print(f"Processing {shape}, amplitude={amp:.2f} ...")
    lambda1 = compute_lambda1(mask)
    print(f"  lambda1 = {lambda1:.12f}")

    rows.append({
        "shape": shape,
        "amplitude": amp,
        "lambda1": lambda1,
    })

    if amp == 0.0:
        baseline[shape] = lambda1

    del data, mask
    gc.collect()


for r in rows:
    shape = r["shape"]
    if shape not in baseline:
        raise ValueError(f"No baseline amplitude 0.0 found for shape={shape}")

    r["delta_lambda1"] = r["lambda1"] - baseline[shape]
    


# Save CSV
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

rows = sorted(rows, key=lambda x: (x["shape"], x["amplitude"]))

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["shape", "amplitude", "lambda1", "delta_lambda1"])
    for r in rows:
        writer.writerow([
            r["shape"],
            r["amplitude"],
            f'{r["lambda1"]:.15f}',
            f'{r["delta_lambda1"]:.15f}'
        ])

print(f"\nSaved -> {OUT_CSV}")
