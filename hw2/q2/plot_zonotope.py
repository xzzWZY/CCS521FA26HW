import numpy as np
import matplotlib.pyplot as plt

# Zonotope: x = c + eps1 * g1 + eps2 * g2, eps1, eps2 in [-1, 1]
c = np.array([4, 3])
g1 = np.array([2, 1])
g2 = np.array([1, 2])

# Four corner points
vertices = np.array([
    c - g1 - g2,  # (1, 0)
    c + g1 - g2,  # (5, 2)
    c + g1 + g2,  # (7, 6)
    c - g1 + g2,  # (3, 4)
])

# Order vertices around the center
center = vertices.mean(axis=0)
angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
vertices = vertices[np.argsort(angles)]

# Close polygon
polygon = np.vstack([vertices, vertices[0]])

fig, ax = plt.subplots(figsize=(7, 5.5), dpi=180)

# Draw zonotope
ax.plot(polygon[:, 0], polygon[:, 1], linewidth=2)
ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.15)

# Draw vertices
ax.scatter(vertices[:, 0], vertices[:, 1], s=40)
for x, y in vertices:
    ax.text(x + 0.08, y + 0.08, f"({int(x)}, {int(y)})", fontsize=10)

# Draw center
ax.scatter([c[0]], [c[1]], marker='x', s=80)
ax.text(c[0] + 0.1, c[1] - 0.35, "center (4, 3)", fontsize=10)

# Axes
ax.set_title("Input zonotope $\\phi$ for Problem 2(a)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim(0, 8)
ax.set_ylim(0, 7)
ax.set_xticks(range(0, 9))
ax.set_yticks(range(0, 8))
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig("simple_zonotope_problem2a.png")
