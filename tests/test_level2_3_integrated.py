import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integration.optimizer import TerraLensOptimizer

# --- DYNAMIC MODEL ---
class EvolvingNet(nn.Module):
    def __init__(self, initial_neurons=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, initial_neurons),
            nn.ReLU(),
            nn.Linear(initial_neurons, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def add_neuron(self):
        """Phase 2: Simulate adding a neuron during construction"""
        with torch.no_grad():
            old_linear = self.layers[0]
            new_neurons = old_linear.out_features + 1
            new_linear = nn.Linear(1, new_neurons)
            
            # Copy old weights, init new one randomly
            new_linear.weight[:old_linear.out_features] = old_linear.weight
            new_linear.bias[:old_linear.out_features] = old_linear.bias
            
            # Update output layer too
            old_out = self.layers[2]
            new_out = nn.Linear(new_neurons, 1)
            new_out.weight[:, :old_linear.out_features] = old_out.weight
            
            self.layers[0] = new_linear
            self.layers[2] = new_out
        print(f"DEBUG: Neuron added. Total hidden neurons: {new_neurons}")

# --- INTEGRATED TEST RUNNER ---

def run_integrated_test():
    print("--- LEVEL 2 & 3 INTEGRATED EVALUATION ---")
    
    # 1. Setup Data (Sine Wave fitting - Non-convex task)
    x = torch.linspace(-np.pi, np.pi, 100).view(-1, 1)
    y = torch.sin(x)
    
    # 2. Compare Adam vs TerraLens vs TerraLens(No Skip)
    configs = [
        ("Standard Adam", False),
        ("TerraLens (Full)", True),
        ("TerraLens (No Skip)", False) # Ablation
    ]
    
    results = {}

    for name, use_skip in configs:
        print(f"\n[RUNNING] {name}")
        model = EvolvingNet(initial_neurons=2)
        
        # Start Training
        criterion = nn.MSELoss()
        
        # For the ablation 'No Skip', we use a high skip_factor=0 or just standard Adam
        if "TerraLens" in name:
            skip_val = 2.0 if use_skip else 0.0
            optimizer = TerraLensOptimizer(model.parameters(), optim.Adam, lr=1e-2, skip_factor=skip_val)
        else:
            optimizer = optim.Adam(model.parameters(), lr=1e-2)

        start_time = time.time()
        
        # Training loop with "Map-while-building" moments
        for i in range(300):
            # Simulate adding neurons at step 100 and 200
            if i == 100 or i == 200:
                model.add_neuron()
                # Re-init optimizer with new parameters
                if "TerraLens" in name:
                    optimizer = TerraLensOptimizer(model.parameters(), optim.Adam, lr=1e-2, skip_factor=skip_val)
                else:
                    optimizer = optim.Adam(model.parameters(), lr=1e-2)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f" Step {i}: Loss = {loss.item():.6f}")

        results[name] = {"loss": loss.item(), "time": time.time() - start_time}

    # --- PRINT FINAL COMPARISON ---
    print("\n--- FINAL EVALUATION RESULTS ---")
    for name, data in results.items():
        print(f"{name:20} | Final Loss: {data['loss']:.6f} | Time: {data['time']:.2f}s")

if __name__ == "__main__":
    run_integrated_test()
