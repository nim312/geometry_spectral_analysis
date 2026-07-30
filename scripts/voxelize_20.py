import os
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STL_FOLDER = os.path.join(BASE_DIR, "..", "stl")
RESULTS_FOLDER = os.path.join(BASE_DIR, "..", "results")
FIGS_FOLDER = os.path.join(BASE_DIR, "..", "figs")

N = 20  #resolution
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIGS_FOLDER, exist_ok=True)

def voxelize_stl(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\nProcessing: {name}")
    mesh = trimesh.load_mesh(filepath)
    if not mesh.is_watertight:
        print(f"Skipping {name} (not watertight)")
        return
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounds[0])
    scale = mesh.bounds[1] - mesh.bounds[0]
    mesh.apply_scale(1 / np.max(scale))
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    z = np.linspace(0, 1, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    try:
        inside = mesh.contains(points)
    except Exception as e:
        print(f"Error during contains() for {name}: {e}")
        return

    mask = inside.reshape((N, N, N))
    voxel_path = os.path.join(RESULTS_FOLDER, f"{name}.npz")
    np.savez_compressed(voxel_path, mask=mask)
    metadata = {
        "shape_name": name,
        "resolution": N,
        "voxel_size": float(1 / (N - 1)),
        "normalized_bounds": [0.0, 1.0]
    }
    meta_path = os.path.join(RESULTS_FOLDER, f"{name}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"{name} voxelization complete.")
def main():
    if not os.path.exists(STL_FOLDER):
        print("STL folder not found:", STL_FOLDER)
        return
    files = [f for f in os.listdir(STL_FOLDER) if f.lower().endswith(".stl")]
    if len(files) == 0:
        print("No STL files found in:", STL_FOLDER)
        return
    print(f"Found {len(files)} STL files.")

    for filename in files:
        filepath = os.path.join(STL_FOLDER, filename)
        voxelize_stl(filepath)

    print("\nAll voxelizations completed successfully.")

if __name__ == "__main__":
    main()
