import numpy as np
import time
import os
import sys
import ctypes

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from terralens_core.coordinator import BlockCoordinator

@ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int)
def complex_landscape_fn(weights_ptr, n):
    # Highly non-convex mock landscape
    weights = np.ctypeslib.as_array(weights_ptr, shape=(n,))
    return np.mean(np.sin(weights**2) + np.cos(weights))

if __name__ == "__main__":
    dll_path = os.path.join("terralens_core", "radar_parallel.dll")
    coordinator = BlockCoordinator(dll_path, block_size=2000)
    
    # SCALE TO 100,000 PARAMETERS
    param_count = 100000
    weights = np.random.uniform(-5, 5, param_count)
    
    print(f"--- MASSIVE SCALE TEST: {param_count} Parameters ---")
    
    start = time.time()
    curvatures = coordinator.coordinate_scan(weights, complex_landscape_fn, num_threads=8)
    end = time.time()
    
    print(f"100,000 Parameter Scan Time: {end - start:.2f} seconds")
    print(f"Throughput: {param_count / (end - start):.0f} parameters/second")
    
    skip_points = len(np.where(curvatures < -0.1)[0])
    print(f"Radar successfully identified {skip_points} non-convex regions in the 100k space.")
