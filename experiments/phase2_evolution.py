import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terralens_core.radar_wrapper import FastRadar, EvolutionaryMap

# --- Setup Fast Radar ---
dll_path = os.path.join("terralens_core", "radar_core.dll")
radar = FastRadar(dll_path)

# --- Simulation Data ---
x_train = np.linspace(-5, 5, 50).astype(np.float64)
y_train = 0.5 * (1 + np.sin(x_train)).astype(np.float64)

def scan_landscape(w1_range, w2_range):
    grid = np.zeros((len(w2_range), len(w1_range)))
    for i, w2 in enumerate(w2_range):
        for j, w1 in enumerate(w1_range):
            # We simulate architecture change by adding a bias/frequency shift 
            # based on "neuron count" elsewhere. For this scan, we just map.
            grad, curv = radar.probe(w1, w2, x_train, y_train)
            # For simplicity, let's just get the loss here (re-using probe for now or adding a loss-only fast func)
            # Actually, let's just use the loss value from the C++ core directly if we had a wrapper.
            # I'll just use a simple Python loss for the scan to keep it moving, 
            # but using C++ for the Radar probe later.
            y_pred = 1 / (1 + np.exp(-(w1 * x_train + w2)))
            grid[i, j] = np.mean((y_pred - y_train)**2)
    return grid

# --- Phase 2: Evolution Simulation ---
ev_map = EvolutionaryMap()
w1_range = np.linspace(-5, 5, 50)
w2_range = np.linspace(-5, 5, 50)

print("Starting Evolutionary Mapping...")

# Stage 1: "1 Neuron"
grid1 = scan_landscape(w1_range, w2_range)
ev_map.record_state(1, w1_range, w2_range, grid1)

# Stage 2: "2 Neurons" (Simulated by adding more complexity to the target/model)
# We simulate a "Morph" by shifting the target slightly
y_train_new = 0.5 * (1 + np.sin(x_train * 1.5)) 
def scan_landscape_v2(w1_range, w2_range):
    grid = np.zeros((len(w2_range), len(w1_range)))
    for i, w2 in enumerate(w2_range):
        for j, w1 in enumerate(w1_range):
            y_pred = 1 / (1 + np.exp(-(w1 * x_train + w2)))
            grid[i, j] = np.mean((y_pred - y_train_new)**2)
    return grid

grid2 = scan_landscape_v2(w1_range, w2_range)
ev_map.record_state(2, w1_range, w2_range, grid2)

# Calculate Delta
delta = ev_map.calculate_delta()

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].contourf(w1_range, w2_range, grid1, levels=20, cmap='viridis')
axes[0].set_title("Landscape: 1 Neuron")
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].contourf(w1_range, w2_range, grid2, levels=20, cmap='viridis')
axes[1].set_title("Landscape: 2 Neurons")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].contourf(w1_range, w2_range, delta, levels=20, cmap='RdBu')
axes[2].set_title("THE DELTA (The Morph)")
fig.colorbar(im2, ax=axes[2])

plt.savefig("experiments/evolutionary_morph.png")
print("\nEvolutionary Map complete. 'The Morph' saved to experiments/evolutionary_morph.png")
