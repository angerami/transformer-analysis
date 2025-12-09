import numpy as np
import matplotlib.pyplot as plt
import umap


def damped_oscillator(t, omega, zeta):
    """
    Solve x'' + 2*zeta*omega*x' + omega^2*x = 0
    Initial conditions: x(0)=1, x'(0)=0
    """
    if zeta < 1:
        # Underdamped
        omega_d = omega * np.sqrt(1 - zeta**2)
        x = np.exp(-zeta * omega * t) * (
            np.cos(omega_d * t) + (zeta * omega / omega_d) * np.sin(omega_d * t)
        )
    elif zeta == 1:
        # Critically damped
        x = np.exp(-omega * t) * (1 + omega * t)
    else:
        # Overdamped
        s1 = -omega * (zeta + np.sqrt(zeta**2 - 1))
        s2 = -omega * (zeta - np.sqrt(zeta**2 - 1))
        c1 = -s2 / (s1 - s2)
        c2 = s1 / (s1 - s2)
        x = c1 * np.exp(s1 * t) + c2 * np.exp(s2 * t)

    return x


# Parameters
np.random.seed(456)
omega_min, omega_max = 0.75, 2.0
zeta_min, zeta_max = 0.5, 1.5
n_samples = 100
n_points = 200
t = np.linspace(0, 5, n_points)
my_noise = 0.005
# Generate data
time_series = []
labels = []

# Class 0: Underdamped (oscillatory)
for i in range(n_samples):
    omega = np.random.uniform(omega_min, omega_max)
    zeta = np.random.uniform(zeta_min, zeta_max)
    y = damped_oscillator(t, omega, zeta)
    y += np.random.normal(0, my_noise, len(t))
    time_series.append(y)
    if zeta < 1:
        labels.append(0)
    else:
        labels.append(1)

# Convert to arrays
X = np.array(time_series)  # Shape: (100, 200)
y_true = np.array(labels)  # Shape: (100,)

print(f"Generated {len(X)} time series")
print(f"Shape: {X.shape}")
print(f"Class distribution: {np.bincount(y_true)}")

# Visualize some examples
fig, axes = plt.subplots(4, 6, figsize=(16, 8))
fig.suptitle("Example Time Series from Each Class")

axes = axes.flatten()
for i, ax in enumerate(axes):
    # Underdamped
    ax.plot(t, X[i])
    ax.set_xlabel("Time")
    ax.set_ylabel("x(t)")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/test_data_examples.png", dpi=150, bbox_inches="tight")


# Stack time series (N_s × N)
Y = X

# UMAP to 2D
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(Y)

# Visualize
fig, axes = plt.subplots()
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6)
# plt.show()
plt.savefig("outputs/umap_decomp.png", dpi=150, bbox_inches="tight")
