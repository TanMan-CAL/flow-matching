import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Setup
# ============================================================

os.makedirs("photos", exist_ok=True)

# ============================================================
# Create 2D grid (toy "image space")
# ============================================================

x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)

# ============================================================
# Clean data distribution (sharp, structured)
# ============================================================

centers = [(-1.5, 1.0), (1.2, -1.0), (0.5, 1.5)]

p_clean = np.zeros_like(X)
for cx, cy in centers:
    dx = X - cx
    dy = Y - cy
    p_clean += np.exp(-(dx**2 + dy**2) / 0.05)  # very sharp peaks

# normalize (optional but cleaner visuals)
p_clean /= p_clean.max()

# ============================================================
# Gaussian blur (manual, no scipy dependency)
# ============================================================

def gaussian_kernel(size=21, sigma=3.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def convolve2d(image, kernel):
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode='reflect')
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            out[i, j] = np.sum(region * kernel)

    return out

# blur distributions (simulate adding noise)
kernel_low = gaussian_kernel(size=21, sigma=2.0)
kernel_high = gaussian_kernel(size=41, sigma=6.0)

p_low_noise = convolve2d(p_clean, kernel_low)
p_high_noise = convolve2d(p_clean, kernel_high)

# normalize again
p_low_noise /= p_low_noise.max()
p_high_noise /= p_high_noise.max()

# ============================================================
# Plot: clean → noisy distributions
# ============================================================

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

axs[0].contourf(X, Y, p_clean, levels=50)
axs[0].set_title("Clean Data Distribution\n(sharp, complex)")

axs[1].contourf(X, Y, p_low_noise, levels=50)
axs[1].set_title("Small Noise\n(slightly smoothed)")

axs[2].contourf(X, Y, p_high_noise, levels=50)
axs[2].set_title("Large Noise\n(very smooth)")

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("photos/distribution_smoothing.png", dpi=200)
plt.show()

# ============================================================
# Compute score field: ∇ log p(x)
# ============================================================

log_p = np.log(p_high_noise + 1e-8)

grad_y, grad_x = np.gradient(log_p)

norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
grad_x /= norm
grad_y /= norm

# ============================================================
# Plot: score field on smoothed distribution
# ============================================================

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, p_high_noise, levels=40)

plt.quiver(
    X[::10, ::10],
    Y[::10, ::10],
    grad_x[::10, ::10],
    grad_y[::10, ::10]
)

plt.title("Score Field on Smoothed Distribution")
plt.xticks([])
plt.yticks([])
plt.tight_layout()

plt.savefig("photos/score_on_smooth.png", dpi=200)
plt.show()