import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.optimizer import TerraLensOptimizer
import time

# --- Simple MLP Architecture ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# --- Training Loop ---
def train_model(name, optimizer_type, model, train_loader, epochs=2):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_type == "terralens":
        optimizer = TerraLensOptimizer(model.parameters(), optim.Adam, lr=1e-3)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nTraining {name}...")
    start_time = time.time()
    
    for epoch in range(epochs):
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Track accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 100 == 0:
                acc = 100 * correct / total
                print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f} Acc: {acc:.2f}%")
    
    end_time = time.time()
    print(f"{name} finished in {end_time - start_time:.2f} seconds.")
    return end_time - start_time

# --- Execution ---
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # 1. Baseline Adam
    model_adam = SimpleMLP()
    time_adam = train_model("Standard Adam", "adam", model_adam, train_loader)

    # 2. TerraLens Adam
    model_tl = SimpleMLP()
    time_tl = train_model("TerraLens (Adam)", "terralens", model_tl, train_loader)

    print("\n--- PHASE 3 RESULTS ---")
    print(f"Standard Adam Time: {time_adam:.2f}s")
    print(f"TerraLens Time: {time_tl:.2f}s")
