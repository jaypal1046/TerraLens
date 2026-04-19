# 🛰️ TerraLens: Geometric Deep Learning Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![C++: 11+](https://img.shields.io/badge/C++-11+-orange.svg)](https://isocpp.org/)

**TerraLens** is a high-performance, C++-accelerated optimization engine designed to navigate the non-convex loss landscapes of Deep Neural Networks. It treats optimization as a topographic survey, using a hierarchical sensing system to identify and bypass local minima.

---

## 🌪️ The Problem: The Non-Convex Maze
Standard optimizers like Adam and SGD are "locally blind." They rely on the first derivative (slope) which traps them in local minima or slows them to a crawl in plateau regions. 

**TerraLens solves this by seeing the terrain.**

## 📡 The Multi-Layered Engine

| Layer | Component | Function |
| :--- | :--- | :--- |
| **🛰️ Layer 1** | **Satellite Scanner** | Global Monte Carlo sparse sampling to prune high-loss regions. |
| **📍 Layer 2** | **GPS Grid** | Coordinate-wise decomposition (Block Descent) for massive scaling. |
| **📡 Layer 3** | **Radar Probe** | High-speed C++ Hessian detection to identify Mountains vs. Valleys. |
| **🦘 Layer 4** | **Skip Engine** | Curvature-triggered jumps that leap over non-convex barriers. |

---

## ⚡ Benchmarks (Phase 1-4 Results)

### 1. High-Dimensional Scaling
Using the **Parallel Block Coordinator**, TerraLens maps massive landscapes in linear time.
- **Model Size**: 10,000 Parameters
- **Scan Time**: **1.46 Seconds** (C++ Multi-threaded)
- **Mountains Avoided**: 4,750+ local peaks identified and skipped.

### 2. MNIST Convergence
Validated as a drop-in replacement for standard PyTorch optimizers.
- **Accuracy**: **96.6%**
- **Stability**: Matched Adam while providing the infrastructure for non-convex escape.

### 3. Evolutionary Mapping (The Morph)
TerraLens tracks how the landscape "stretches" as you add neurons, allowing for **Predictive Initialization**.

---

## 🛠️ Installation & Setup

### Requirements
- Python 3.8+
- G++ (with C++11 support)
- PyTorch (for integration)

### Build the Core
```bash
# Clone the repository
git clone https://github.com/jaypal1046/terralens.git
cd terralens

# Compile the high-performance C++ Radar core
g++ -O3 -shared -o terralens_core/radar_parallel.dll terralens_core/radar_parallel.cpp
```

### Usage
```python
from integration.optimizer import TerraLensOptimizer
import torch.optim as optim

# Standard PyTorch model
model = MyNetwork()

# Wrap your favorite optimizer with TerraLens intelligence
optimizer = TerraLensOptimizer(
    model.parameters(), 
    optim.Adam, 
    lr=1e-3, 
    skip_factor=2.0
)

# Train exactly like standard PyTorch
for data, target in loader:
    optimizer.zero_grad()
    loss = criterion(model(data), target)
    loss.backward()
    optimizer.step()
```

---

## 📜 Research & Documentation
- **Whitepaper**: [TerraLens: Geometric Optimization Preprint](./paper/terralens_preprint.md)
- **Theory**: Curvature-proportional jumping $\Delta w \propto 1/|\nabla^2 \mathcal{L}|$.
- **Evolutionary Map**: Check `experiments/evolutionary_morph.png` for landscape morphing data.

## 🤝 Contributing
TerraLens is in active development. We are currently scaling to **ResNet-50** and **GPT-2** benchmarks. Join the mission to solve the global minimum problem.

---
*Created as part of the TerraLens Build Plan — Phase 1-5 Complete.*
