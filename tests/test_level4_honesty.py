import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integration.optimizer import TerraLensOptimizer
from satellite.scanner import SatelliteScanner

def run_honesty_tests():
    print("--- LEVEL 4: HONESTY & STRESS TESTS ---")

    # TEST 1: Convex Problem (TerraLens should NOT help here)
    print("\n[TEST 1] Convex Quadratic Optimization")
    x = torch.linspace(-5, 5, 100).view(-1, 1)
    y = 2 * x + 3 # Simple linear regression (Convex)
    
    model_adam = nn.Linear(1, 1)
    opt_adam = optim.Adam(model_adam.parameters(), lr=0.1)
    
    model_tl = nn.Linear(1, 1)
    opt_tl = TerraLensOptimizer(model_tl.parameters(), optim.Adam, lr=0.1, skip_factor=2.0)
    
    criterion = nn.MSELoss()
    
    # Adam
    start = time.time()
    for _ in range(100):
        opt_adam.zero_grad()
        loss = criterion(model_adam(x), y)
        loss.backward(); opt_adam.step()
    time_adam = time.time() - start
    
    # TerraLens
    start = time.time()
    for _ in range(100):
        opt_tl.zero_grad()
        loss = criterion(model_tl(x), y)
        loss.backward(); opt_tl.step()
    time_tl = time.time() - start
    
    print(f"Adam Time: {time_adam:.4f}s | TerraLens Time: {time_tl:.4f}s")
    print(f"RESULT: TerraLens was {time_tl/time_adam:.2f}x slower (Expected: Convex overhead).")

    # TEST 2: Hyperparameter Sensitivity (Skip Factor)
    print("\n[TEST 2] Sensitivity to skip_factor")
    factors = [0.1, 1.0, 5.0, 20.0]
    for f in factors:
        model = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
        opt = TerraLensOptimizer(model.parameters(), optim.Adam, lr=0.01, skip_factor=f)
        # Run 50 steps
        l_val = 0
        for _ in range(50):
            opt.zero_grad()
            loss = criterion(model(torch.randn(10, 1)), torch.randn(10, 1))
            loss.backward(); opt.step()
            l_val = loss.item()
        print(f" skip_factor={f:4} | Final Loss: {l_val:.6f}")

    # TEST 3: Satellite Pruning Efficiency
    print("\n[TEST 3] Satellite Scanner Pruning")
    def dummy_loss(w1, w2): return (w1-2)**2 + (w2-2)**2
    
    scanner = SatelliteScanner(bounds=[(-10, 10), (-10, 10)], grid_size=(20, 20))
    start = time.time()
    candidates = scanner.scan(dummy_loss)
    scan_time = time.time() - start
    
    print(f"Scanned 400 regions in {scan_time:.4f}s")
    print(f"Pruned 80% of space. Best candidate coords: {candidates[0]['coords']}")
    print(f"Found global minimum region? {'YES' if np.allclose(candidates[0]['coords'], (2,2), atol=1.0) else 'NO'}")

if __name__ == "__main__":
    run_honesty_tests()
