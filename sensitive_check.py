import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import random

# =========================
# LOAD MASK
# =========================
data = np.load(r"C:\Users\nihal\Desktop\Research\results\cube.npz")
mask = data["mask"].copy()

N = mask.shape[0]
h = 1.0 / (N - 1)

# =========================
# FIND BOUNDARY VOXELS
# =========================
boundary = []

for i in range(1, N-1):
    for j in range(1, N-1):
        for k in range(1, N-1):

            if not mask[i,j,k]:
                continue

            # if any neighbor is outside → boundary
            for di, dj, dk in [
                (-1,0,0),(1,0,0),
                (0,-1,0),(0,1,0),
                (0,0,-1),(0,0,1)
            ]:
                if not mask[i+di, j+dj, k+dk]:
                    boundary.append((i,j,k))
                    break

print("Boundary voxels:", len(boundary))

# =========================
# BUILD LAPLACIAN
# =========================
def build_laplacian(mask):
    index_map = -np.ones_like(mask, dtype=int)
    coords = np.argwhere(mask)

    for idx, (i,j,k) in enumerate(coords):
        index_map[i,j,k] = idx

    n = len(coords)
    A = sp.lil_matrix((n,n))

    for idx, (i,j,k) in enumerate(coords):

        A[idx, idx] = 6.0 / h**2

        for di, dj, dk in [
            (-1,0,0),(1,0,0),
            (0,-1,0),(0,1,0),
            (0,0,-1),(0,0,1)
        ]:
            ni, nj, nk = i+di, j+dj, k+dk

            if mask[ni,nj,nk]:
                jdx = index_map[ni,nj,nk]
                A[idx, jdx] = -1.0 / h**2

    return A.tocsr()

# =========================
# BASE λ1
# =========================
A = build_laplacian(mask)
λ1_base = np.sort(spla.eigsh(A, k=1, which='SM')[0])[0]

print("λ1 base:", λ1_base)

# =========================
# PERTURB BOUNDARY
# =========================
num_flips = int(0.01 * len(boundary))  # 1% perturbation

perturbed_mask = mask.copy()

for (i,j,k) in random.sample(boundary, num_flips):
    perturbed_mask[i,j,k] = False

A2 = build_laplacian(perturbed_mask)
λ1_pert = np.sort(spla.eigsh(A2, k=1, which='SM')[0])[0]

print("λ1 perturbed:", λ1_pert)
print("Δλ1:", λ1_pert - λ1_base)