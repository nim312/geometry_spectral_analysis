import numpy as np
import matplotlib.pyplot as plt

# replace with your computed values
shapes = ["cube","cylinder","ellipse","rectangle","sphere"]

mean_p = [17.8, 1.28, 1.37, 18.1, 3.35]
std_p  = [6.86, 0.056, 0.122, 7.35, 0.079]

plt.figure(figsize=(7,6))

plt.scatter(mean_p, std_p)

for i, s in enumerate(shapes):
    plt.text(mean_p[i], std_p[i], s)

plt.xlabel("Mean p (scaling exponent)")
plt.ylabel("Std p (stability)")
plt.title("Geometry-dependent scaling regime structure")

plt.grid(True, alpha=0.3)
plt.show()
