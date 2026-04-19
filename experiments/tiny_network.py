import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def model(x, w1, w2):
    """A tiny 2-weight network: y = sigmoid(w1 * x + w2)"""
    return sigmoid(w1 * x + w2)

def compute_loss(w1, w2, x_train, y_train):
    """Mean Squared Error Loss"""
    y_pred = model(x_train, w1, w2)
    return np.mean((y_pred - y_train)**2)

# 1. Generate Synthetic Data (a simple sine-like pattern)
x_train = np.linspace(-5, 5, 20)
y_train = 0.5 * (1 + np.sin(x_train)) # Target pattern

# 2. Map the Landscape
w1_range = np.linspace(-10, 10, 100)
w2_range = np.linspace(-10, 10, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)
Z = np.array([[compute_loss(w1, w2, x_train, y_train) for w1 in w1_range] for w2 in w2_range])

# 3. Radar Logic: Local Probe (Gradient + Hessian)
def radar_probe(w1, w2, x_train, y_train):
    # Numerical Gradient
    h = 1e-5
    grad_w1 = (compute_loss(w1 + h, w2, x_train, y_train) - compute_loss(w1 - h, w2, x_train, y_train)) / (2 * h)
    grad_w2 = (compute_loss(w1, w2 + h, x_train, y_train) - compute_loss(w1, w2 - h, x_train, y_train)) / (2 * h)
    
    # Numerical Hessian (Curvature)
    f_ww1 = (compute_loss(w1 + h, w2, x_train, y_train) - 2*compute_loss(w1, w2, x_train, y_train) + compute_loss(w1 - h, w2, x_train, y_train)) / (h**2)
    f_ww2 = (compute_loss(w1, w2 + h, x_train, y_train) - 2*compute_loss(w1, w2, x_train, y_train) + compute_loss(w1, w2 - h, x_train, y_train)) / (h**2)
    
    return (grad_w1, grad_w2), (f_ww1, f_ww2)

# 4. Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W1, W2, Z, cmap='terrain', alpha=0.8, edgecolor='none')

# Mark Global Minimum (approx)
min_idx = np.unravel_index(np.argmin(Z), Z.shape)
w1_min, w2_min, z_min = W1[min_idx], W2[min_idx], Z[min_idx]
ax.scatter([w1_min], [w2_min], [z_min], color='red', s=100, label='Global Minimum Target')

ax.set_title("TerraLens: 3D Loss Landscape (2 Weights)")
ax.set_xlabel("Weight 1 (w1)")
ax.set_ylabel("Weight 2 (w2)")
ax.set_zlabel("Loss")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.legend()

plt.savefig("experiments/landscape.png")
print(f"Landscape map generated. Global Min Target at: w1={w1_min:.2f}, w2={w2_min:.2f}, Loss={z_min:.4f}")

# Example Radar Probe at a 'Mountain' point
w_test = (2.0, 8.0)
grad, curv = radar_probe(*w_test, x_train, y_train)
print(f"\nRADAR PROBE at {w_test}:")
print(f"Gradient: {grad}")
print(f"Curvature (Hessian diag): {curv}")
if any(c < 0 for c in curv):
    print("--- MOUNTAIN DETECTED: SKIP SIGNAL ACTIVE ---")
else:
    print("--- VALLEY DETECTED: STAY AND OPTIMIZE ---")
