import numpy as np
import time
import os
import sys
import ctypes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terralens_core.coordinator import BlockCoordinator

# --- Dummy High-Dim Loss Function ---
@ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int)
def dummy_loss(weights_ptr, n):
    # Simulate a complex non-convex landscape
    # Sum of sin(w) * cos(w) for all weights
    weights = np.ctypeslib.as_array(weights_ptr, shape=(n,))
    return np.mean(np.sin(weights) * np.cos(weights))

# --- Scaling Test ---
if __name__ == "__main__":
    dll_path = os.path.join("terralens_core", "radar_parallel.dll")
    coordinator = BlockCoordinator(dll_path, block_size=1000)
    
    # Scale to 10,000 parameters
    param_count = 10000
    weights = np.random.uniform(-5, 5, param_count)
    
    print(f"Scaling Test: Mapping {param_count} parameters...")
    
    # Test 1: Single-Threaded (Simulated)
    start = time.time()
    curvatures_1 = coordinator.coordinate_scan(weights, dummy_loss, num_threads=1)
    end = time.time()
    print(f"Single-Threaded Scan Time: {end - start:.4f}s")
    
    # Test 2: Multi-Threaded (4 Threads)
    start = time.time()
    curvatures_4 = coordinator.coordinate_scan(weights, dummy_loss, num_threads=4)
    end = time.time()
    print(f"4-Threaded Parallel Scan Time: {end - start:.4f}s")
    
    # Verify
    skip_points = coordinator.get_skip_indices(curvatures_4)
    print(f"Detected {len(skip_points)} points requiring a SKIP JUMP.")
    print(f"Scaling Efficiency: {((time.time()-start) / (end-start)):.2f}x (estimated)")
