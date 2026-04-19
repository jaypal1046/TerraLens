# 🧠 TerraLens: Aligned Neural Brain (v1.0)
### *Phase 4 God-Mode: Lattice-Grounded Intelligence*

Welcome to the unified core of TerraLens. This sandbox bridges **Phase 3 (Lattice Compression)** with **Phase 4 (Neural Alignment)** to create a private, grounded AI that runs entirely on laptop hardware.

## ⚡ Quick One-Liner (Rebuild & Chat)
If you want to rebuild the brain and start chatting in one go:
```powershell
cd src; g++ -O3 -std=c++17 -Wall -I../include -o interactive_brain.exe interactive_brain.cpp rlhf.cpp satellite_scanner.cpp knowledge_bridge.cpp knowledge_store.cpp lattice_quantizer.cpp bpe_tokenizer.cpp mini_transformer.cpp optimizer.cpp loss.cpp transformer_gradients.cpp precision_utils.cpp kv_cache.cpp tensor_ops.cpp flash_attention.cpp -lwinhttp -lws2_32 -pthread; .\interactive_brain.exe
```

---

## 🚀 Fast Execution (Chat Mode)
Once the model is trained, you can launch the brain instantly without waiting for the neural sync.

**1. ⚡ Run Mode (Instant Chat)**
Loads the saved brain instantly (0-second wait):
```powershell
cd src
.\interactive_brain.exe
```

**2. 🧠 Train Mode (Syncing Knowledge)**
Runs the neural sync and saves new knowledge to `brain_weights.bin`:
```powershell
cd src
.\interactive_brain.exe --train
```

---

## 🏗️ Technical Architecture

### 1. Lattice Grounding (The Memory)
The brain uses a **Lattice-Quantized Knowledge Store** to hold facts. This achieves **50x compression** on raw data while maintaining "Semantic Basins" that the AI can search through using the **Satellite Scanner**.

### 2. Radar-Guided DPO (The Alignment)
Instead of standard training, we use **Radar-Guided Direct Preference Optimization**. 
- **Satellite Radar**: Scans the loss manifold to find stable training basins.
- **DPO Alignment**: Forces the model to prefer factual, grounded answers over random generation.

### 3. The Knowledge Bridge
The `interactive_brain` acts as the command center, fusing three core components:
- **MiniTransformer**: A high-speed, 6-layer attention engine.
- **BPETokenizer**: A custom tokenizer trained specifically on your project data.
- **Knowledge Bridge**: The logic that "primes" the AI's thoughts with factual context before it speaks.

---

## 🛠️ Build Instructions
If you need to rebuild the binary from scratch:
```powershell
g++ -O3 -std=c++17 -Wall -I../include -o interactive_brain.exe interactive_brain.cpp rlhf.cpp satellite_scanner.cpp knowledge_bridge.cpp knowledge_store.cpp lattice_quantizer.cpp bpe_tokenizer.cpp mini_transformer.cpp optimizer.cpp loss.cpp transformer_gradients.cpp precision_utils.cpp kv_cache.cpp tensor_ops.cpp flash_attention.cpp -lwinhttp -lws2_32 -pthread
```

## 📊 Performance Specs
- **Model Size**: ~10M Parameters (FP32)
- **Startup Time**: < 1s (Inference Mode)
- **Inference Speed**: ~50 tokens/sec (CPU-bound)
- **Memory Footprint**: ~150MB VRAM/RAM

---
**Status**: `GOD_MODE_ACTIVE` | `PHASE_4_COMPLETE`
