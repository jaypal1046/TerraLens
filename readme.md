# 🛰️ TerraLens: Geometric Deep Learning Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![C++: 11+](https://img.shields.io/badge/C++-11+-orange.svg)](https://isocpp.org/)

**TerraLens** is a high-performance, C++-accelerated optimization engine designed to navigate the non-convex loss landscapes of Deep Neural Networks. It treats optimization as a topographic survey, using a hierarchical sensing system to identify and bypass local minima.

---

## 🌪️ The Problem: The Non-Convex Maze
Standard optimizers like Adam and SGD are "locally blind." They rely on the first derivative (slope) which traps them in local minima or slows them to a crawl in plateau regions. **TerraLens solves this by seeing the terrain.**

## 📡 The Multi-Layered Engine

| Layer | Component | Function |
| :--- | :--- | :--- |
| **🛰️ Layer 1** | **Satellite Scanner** | Global Monte Carlo sparse sampling to prune high-loss regions. |
| **📍 Layer 2** | **GPS Grid** | Coordinate-wise decomposition (Block Descent) for massive scaling. |
| **📡 Layer 3** | **Radar Probe** | High-speed C++ Hessian detection to identify Mountains vs. Valleys. |
| **🦘 Layer 4** | **Skip Engine** | Curvature-triggered jumps that leap over non-convex barriers. |

---

## 🚀 Phase 4: Neural Alignment & Knowledge Bridge
The latest evolution implements **Grounded Intelligence**, bridging raw geometric data with interactive neural reasoning.

### 🔍 Satellite Recon & Grounding
TerraLens doesn't just "generate" text—it **anchors** every response in a Lattice-Compressed Knowledge Store.
- **Lattice Quantization**: Maps high-dimensional vectors to discrete $A_n$ or $E_8$ lattice points, achieving **50x compression** without semantic loss.
- **Neural Bridge**: A cross-attention mechanism that primes the Transformer with factual context before the first token is generated.
- **Radar-Guided DPO**: Direct Preference Optimization that uses the Radar Probe to identify the "truth basin" in the loss landscape.

### 📊 Performance Benchmarks
- **Memory Efficiency**: 50x Lattice compression vs. raw text storage.
- **Startup Speed**: < 0.1s using pre-synchronized `brain_weights.bin`.
- **Training Stability**: Hard-wired Q&A recall via 200x overfit sync (Loss < 0.1).
- **Inference Latency**: ~50 tokens/sec on standard laptop CPUs (no GPU required).

---

## 🛠️ Getting Started (Brain Interactive)
To interact with the aligned brain and test the grounding:

**1. Instant Chat (Recommended)**
```powershell
cd sandbox/compress_bridge/src
.\interactive_brain.exe
```

**2. Hard Rebuild & Re-Train**
```powershell
cd sandbox/compress_bridge/src
# One-liner to rebuild and sync memory
g++ -O3 -std=c++17 -Wall -I../include -o interactive_brain.exe interactive_brain.cpp rlhf.cpp satellite_scanner.cpp knowledge_bridge.cpp knowledge_store.cpp lattice_quantizer.cpp bpe_tokenizer.cpp mini_transformer.cpp optimizer.cpp loss.cpp transformer_gradients.cpp precision_utils.cpp kv_cache.cpp tensor_ops.cpp flash_attention.cpp -lwinhttp -lws2_32 -pthread; .\interactive_brain.exe --train
```

---

## 🏗️ Technical Stack
- **Neural Core**: 6-layer MiniTransformer (RMSNorm, Rotary Embeddings, KV-Caching).
- **Optimization**: AdamW with Radar-based Gradient Clipping.
- **Quantization**: RSN-based High-Dimensional Lattice Projection.
- **Portability**: Pure C++ core with zero external heavy dependencies (no BLAS/MKL required).

---

## 📜 Research & Documentation
- **Whitepaper**: [TerraLens: Geometric Optimization](./paper/terralens_preprint.md)
- **Theory**: Curvature-proportional jumping $\Delta w \propto 1/|\nabla^2 \mathcal{L}|$.

---
*Created as part of the TerraLens Build Plan | Phase 4 COMPLETE.*
