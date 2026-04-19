import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satellite.scanner import SatelliteScanner
from radar.probe import RadarProbe
from skip.engine import SkipEngine

# --- SETUP LANDSCAPE ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_train = np.linspace(-5, 5, 20)
y_train = 0.5 * (1 + np.sin(x_train))

def compute_loss(w1, w2):
    y_pred = sigmoid(w1 * x_train + w2)
    return np.mean((y_pred - y_train)**2)

# --- OPTIMIZERS ---

def run_sgd(start_w, lr=5.0, steps=100):
    w = np.array(start_w)
    path = [w.copy()]
    losses = [compute_loss(*w)]
    
    probe = RadarProbe(compute_loss)
    for _ in range(steps):
        grad, _ = probe.probe(*w)
        w -= lr * np.array(grad)
        path.append(w.copy())
        losses.append(compute_loss(*w))
    return np.array(path), losses

def run_terralens(start_w, lr=5.0, steps=100):
    w = np.array(start_w)
    path = [w.copy()]
    losses = [compute_loss(*w)]
    
    probe = RadarProbe(compute_loss)
    skipper = SkipEngine(base_jump_factor=2.0)
    
    for _ in range(steps):
        grad, curv = probe.probe(*w)
        
        # Check for Mountain (Negative Curvature)
        if any(c < 0 for c in curv):
            # SKIP TRIGGERED
            old_w = w.copy()
            w = skipper.calculate_skip(w, grad, curv)
            # Boundary check
            w = np.clip(w, -10, 10)
            print(f"TerraLens: SKIP detected at {old_w} -> Jumped to {w}")
        else:
            # Normal descent in valleys
            w -= lr * np.array(grad)
        
        path.append(w.copy())
        losses.append(compute_loss(*w))
        
        # Early stop if converged
        if np.abs(losses[-1] - losses[-2]) < 1e-6 if len(losses) > 1 else False:
            break
            
    return np.array(path), losses

# --- EXECUTION ---
start_pos = (2.0, 8.0) # Start on a known mountain peak
print(f"Starting comparison from {start_pos}...")

sgd_path, sgd_losses = run_sgd(start_pos)
tl_path, tl_losses = run_terralens(start_pos)

# --- VISUALIZATION ---
w1_range = np.linspace(-10, 10, 100)
w2_range = np.linspace(-10, 10, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)
Z = np.array([[compute_loss(w1, w2) for w1 in w1_range] for w2 in w2_range])

plt.figure(figsize=(10, 6))
plt.contourf(W1, W2, Z, levels=50, cmap='terrain')
plt.colorbar(label='Loss')

plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'r.-', label=f'Standard SGD (Final Loss: {sgd_losses[-1]:.4f})')
plt.plot(tl_path[:, 0], tl_path[:, 1], 'b.-', label=f'TerraLens (Final Loss: {tl_losses[-1]:.4f})')
plt.scatter(*start_pos, color='yellow', s=100, edgecolors='black', label='Start', zorder=5)

plt.title("TerraLens vs Standard SGD: Mountain Escape Test")
plt.xlabel("w1")
plt.ylabel("w2")
plt.legend()
plt.savefig("experiments/comparison_result.png")
print("\nComparison complete. Results saved to experiments/comparison_result.png")
print(f"SGD Final Loss: {sgd_losses[-1]:.6f}")
print(f"TerraLens Final Loss: {tl_losses[-1]:.6f}")
