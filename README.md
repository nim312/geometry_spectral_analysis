# Geometry-Dependent Spectral Sensitivity in Voxelized Laplacian Domains Under Wave-Based Boundary Perturbations

This repository contains the code, geometry files, and data for the paper.

The study examines how the first Dirichlet Laplacian eigenvalue λ₁ responds to controlled sinusoidal boundary perturbations across five three-dimensional geometries, a sphere, cylinder, elliptic prism, cube, and rectangular prism, voxelized on uniform Cartesian grids.


## Dependencies

All code is written in Python 3. The following libraries are required:

```
numpy
scipy
trimesh
matplotlib
pandas
math
time
csv
gc
os
json
```

Install them with:

```bash
pip install numpy scipy trimesh matplotlib pandas math time csv gc os json
```
---

## Reproducing the Results

### Step 1 — Voxelize the STL meshes

```bash
python voxelize_25.py
python voxelize_20.py
```

This reads each STL file, voxelizes it on a uniform N×N×N Cartesian grid using center-based occupancy (trimesh.contains), and saves the binary occupancy masks. The main dataset uses resolution N=25.

### Step 2 — Compute eigenvalues

```bash
python dataset_25.py
python dataset_20.py
```

This builds the Laplacian for each voxelized domain and computes the smallest eigenvalue λ₁ using scipy.sparse.linalg.eigsh. 

### Step 3 — Convergence study

```bash
python convergence_cube.py
```

This runs the unperturbed cube at N = 20, 25, and 30 and computes the error relative to the exact value 3π² ≈ 29.609, confirming second-order convergence.

### Step 5 — Scaling and fit analysis

```bash
python scaling_plot.py
python fit_analysis.py
```
---

## Citation

If you use this code or data in your own work, please cite it:

```
@software{Mallipudi_2026,
    title={Geometry-Dependent Spectral Sensitivity in Voxelized Laplacian Domains Under Wave-Based Boundary Perturbations},
    url={https://github.com/nim312/geometry_spectral_analysis},
    author={Mallipudi, Nihal},
    year={2026},
    month={Aug}
}
```

## License
MIT License. See `LICENSE` for details.
