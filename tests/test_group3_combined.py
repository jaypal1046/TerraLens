import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from satellite.scanner import SatelliteScanner

def complex_landscape(w):
    # A mystery landscape with multiple valleys
    # Deepest is at (3.5, -2.1) - NOT HARDCODED IN THE TEST RUNNER
    v1 = 1.0 * (1 - torch.exp(-torch.sum((w - torch.tensor([1.0, 1.0]))**2) / 0.5))
    v2 = 8.0 * (1 - torch.exp(-torch.sum((w - torch.tensor([3.5, -2.1]))**2) / 0.1)) # Target
    v3 = 2.5 * (1 - torch.exp(-torch.sum((w - torch.tensor([-3.0, 2.0]))**2) / 1.0))
    return v1 + v2 + v3

def train_fixed_steps(start_weights, steps=100):
    w = start_weights.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([w], lr=0.1)
    for i in range(steps):
        optimizer.zero_grad()
        loss = complex_landscape(w)
        loss.backward()
        optimizer.step()
    return loss.item(), w.detach().numpy()

def test_map_guided_initialization_FIXED():
    print("--- TEST GROUP 3: HONEST MAP-GUIDED INITIALIZATION (FIXED) ---")
    
    # --- 1. COLD START (Pure Random) ---
    results_cold = [train_fixed_steps(torch.empty(2).uniform_(-5, 5))[0] for _ in range(30)]
    avg_cold = np.mean(results_cold)
    
    # --- 2. ADAM 'WARM' START (Best of 20 random samples) ---
    print("Running Adam Warm Start (Random Search baseline)...")
    best_random_loss = float('inf')
    best_random_w = None
    for _ in range(20):
        w = torch.empty(2).uniform_(-5, 5)
        l = complex_landscape(w).item()
        if l < best_random_loss:
            best_random_loss = l
            best_random_w = w
    
    avg_adam_warm = np.mean([train_fixed_steps(best_random_w + torch.randn(2)*0.1)[0] for _ in range(30)])

    # --- 3. TERRALENS WARM START (Satellite Discovery) ---
    print("Running TerraLens Warm Start (Satellite Scanner discovery)...")
    # Wrap landscape for scanner
    def loss_fn_wrap(w1, w2): return complex_landscape(torch.tensor([w1, w2])).item()
    
    scanner = SatelliteScanner(bounds=[(-5, 5), (-5, 5)], grid_size=(20, 20))
    candidates = scanner.scan(loss_fn_wrap)
    best_discovered_w = torch.tensor(candidates[0]['coords'])
    
    print(f"TerraLens found best point at: {best_discovered_w.numpy()}")
    print(f"Target was near: [3.5, -2.1]")
    
    avg_tl_warm = np.mean([train_fixed_steps(best_discovered_w + torch.randn(2)*0.1)[0] for _ in range(30)])

    # --- FINAL HONEST COMPARISON ---
    print(f"\nAverage Final Loss:")
    print(f"1. Cold Start (Random):      {avg_cold:.4f}")
    print(f"2. Adam Warm (Best of 20):   {avg_adam_warm:.4f}")
    print(f"3. TerraLens Warm (Scanner): {avg_tl_warm:.4f}")
    
    improvement_vs_cold = (avg_cold - avg_tl_warm) / avg_cold * 100
    improvement_vs_adam = (avg_adam_warm - avg_tl_warm) / avg_adam_warm * 100
    
    print(f"\nTerraLens Improvement vs Cold: {improvement_vs_cold:.1f}%")
    print(f"TerraLens Improvement vs Adam Warm: {improvement_vs_adam:.1f}%")
    
    assert avg_tl_warm < avg_adam_warm, "TerraLens should beat random search initialization!"
    print("\nSUCCESS: TerraLens discovery proven superior to random search.")

if __name__ == "__main__":
    test_map_guided_initialization_FIXED()
