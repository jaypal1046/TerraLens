import numpy as np
import os
import sys
import ctypes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terralens_core.coordinator import BlockCoordinator

# --- MATHEMATICAL BENCHMARKS ---

@ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int)
def rosenbrock_fn(w_ptr, n):
    w = np.ctypeslib.as_array(w_ptr, shape=(n,))
    return np.sum(100.0 * (w[1:] - w[:-1]**2)**2 + (1.0 - w[:-1])**2)

@ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int)
def rastrigin_fn(w_ptr, n):
    w = np.ctypeslib.as_array(w_ptr, shape=(n,))
    return 10 * n + np.sum(w**2 - 10 * np.cos(2 * np.pi * w))

@ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int)
def ackley_fn(w_ptr, n):
    w = np.ctypeslib.as_array(w_ptr, shape=(n,))
    a, b, c = 20, 0.2, 2 * np.pi
    sum1 = np.sum(w**2)
    sum2 = np.sum(np.cos(c * w))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + a + np.exp(1)

# --- TEST RUNNER ---

def run_test(name, loss_fn, start_w, true_min, steps=1000, lr=0.001):
    print(f"\n[TEST] Running {name}...")
    dll_path = os.path.join("terralens_core", "radar_parallel.dll")
    coord = BlockCoordinator(dll_path, block_size=len(start_w))
    
    w = start_w.copy().astype(np.float64)
    
    for i in range(steps):
        # 1. Scan for Curvature
        curvatures = coord.coordinate_scan(w, loss_fn, num_threads=1)
        
        # 2. Numerical Gradient
        h = 1e-5
        grad = np.zeros_like(w)
        for d in range(len(w)):
            w_h = w.copy(); w_h[d] += h
            l_plus = loss_fn(w_h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(w))
            w_h[d] -= 2*h
            l_minus = loss_fn(w_h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(w))
            grad[d] = (l_plus - l_minus) / (2*h)
            
        # 3. Skip Logic
        if any(c < 0 for c in curvatures):
            # MOUNTAIN: Execute Skip
            skip_dir = -np.sign(grad)
            w += skip_dir * 0.1 
        else:
            # VALLEY: Descent
            w -= lr * grad
            
    error = np.linalg.norm(w - true_min)
    if error < 0.2:
        print(f"PASSED: Found {name} minimum at {w}, error={error:.6f}")
    else:
        print(f"FAILED: {name} error={error:.6f} at {w}")

if __name__ == "__main__":
    run_test("Rosenbrock", rosenbrock_fn, np.array([0.0, 0.0]), np.array([1.0, 1.0]), steps=2000, lr=0.002)
    run_test("Rastrigin", rastrigin_fn, np.array([1.0, 1.0]), np.array([0.0, 0.0]), steps=1000, lr=0.001)
    run_test("Ackley", ackley_fn, np.array([2.0, 2.0]), np.array([0.0, 0.0]), steps=1000, lr=0.001)
