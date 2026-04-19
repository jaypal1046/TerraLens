import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terralens_core.radar_wrapper import FastRadar, EvolutionaryMap

def brute_force_scan(loss_fn, resolution=50):
    w1_range = np.linspace(-5, 5, resolution)
    w2_range = np.linspace(-5, 5, resolution)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    grid = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            grid[i, j] = loss_fn(w1_range[j], w2_range[i])
    return W1, W2, grid

def get_loss_fn(neuron_count):
    def loss_fn(w1, w2):
        # A shifting minimum: moves as neurons are added
        # Min is roughly at (n*0.2, n*0.2)
        center = neuron_count * 0.2
        return (w1 - center)**2 + (w2 - center)**2
    return loss_fn

def test_map_correctness_FIXED():
    print("--- TEST GROUP 1: PREDICTIVE MAP CORRECTNESS (FIXED) ---")
    ev_map = EvolutionaryMap()
    resolution = 50
    w1_range = np.linspace(-5, 5, resolution)
    w2_range = np.linspace(-5, 5, resolution)
    
    # 1. Start with Neuron 1
    l1 = get_loss_fn(1)
    _, _, grid1 = brute_force_scan(l1, resolution)
    ev_map.record_state(1, w1_range, w2_range, grid1)
    
    # 2. Start with Neuron 2
    l2 = get_loss_fn(2)
    _, _, grid2 = brute_force_scan(l2, resolution)
    ev_map.record_state(2, w1_range, w2_range, grid2)

    # 3. NOW PREDICT NEURON 3 before seeing it
    predicted_min = ev_map.predict_next_minimum()
    
    # 4. GET GROUND TRUTH FOR NEURON 3
    l3 = get_loss_fn(3)
    W1, W2, grid3 = brute_force_scan(l3, resolution)
    true_min_idx = np.unravel_index(np.argmin(grid3), grid3.shape)
    true_min = (W1[true_min_idx], W2[true_min_idx])
    
    # 5. COMPARE
    error = np.linalg.norm(np.array(predicted_min) - np.array(true_min))
    print(f"\nPREDICTION TEST:")
    print(f"Predicted Min (Neuron 3): {predicted_min}")
    print(f"True Min      (Neuron 3): {true_min}")
    print(f"Prediction Error:        {error:.6f}")
    
    assert error < 0.5, "Map prediction is too far from truth!"
    print("\nSUCCESS: Evolutionary Map successfully predicted the landscape shift.")

if __name__ == "__main__":
    test_map_correctness_FIXED()
