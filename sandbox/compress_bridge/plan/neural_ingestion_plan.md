# TerraLens: Unified Knowledge Ingestion Protocol (UKIP)

## 1. Overview
The goal of UKIP is to provide a scalable, multi-tier data ingestion pipeline for the TerraLens micro-transformer. It allows the model to learn general language fluency from foundational data while maintaining perfect recall for technical facts.

## 2. Data Classifications
The system divides all incoming data into five weighted categories:

| Category | Folder | Weight | Purpose |
| :--- | :--- | :--- | :--- |
| **FACT** | `brain_data/facts/` | 1.0 | Critical Q&A pairs (JSON format). |
| **PROSE** | `brain_data/prose/` | 0.7 | Technical documentation and READMEs. |
| **CODE** | `brain_data/code/` | 0.8 | Source code and API header definitions. |
| **CHAT** | `brain_data/personality/` | 0.4 | Dialogue logs for tone and alignment. |
| **BASE** | `brain_data/foundation/` | 0.2 | General language data (Wikipedia). |

## 3. The Ingestion Pipeline
### Stage 1: Parsing
- **Fact Parser**: Extracts "Q:" and "A:" from JSON arrays.
- **Prose Parser**: Cleans Markdown and Plaintext into sentence-level tokens.
- **Code Parser**: Tokenizes function signatures and comments.

### Stage 2: Weighted Rehearsal (The "Sync" Logic)
To prevent "Catastrophic Forgetting" when adding new facts:
1. Load **New Delta Data** (The file just added).
2. Sample **10% random facts** from the existing vault.
3. Sample **5% random prose** for grammatical stability.
4. Train on this "Weighted Batch" for 50-100 iterations.

## 4. Automation Commands
- `.\manage_brain.ps1 --add <file>`: Automatically classifies and moves file to `brain_data/`.
- `.\manage_brain.ps1 --sync`: Runs the incremental training loop.
- `.\manage_brain.ps1 --full-sync`: Performs a full re-alignment of all tiers.

## 5. Success Metrics
- **Fact Recall**: 99% accuracy on `facts/` tier.
- **Linguistic Fluency**: Perplexity < 5.0 on `prose/` tier.
- **Zero-Looping**: Verified by Anti-Loop v2.0 decoder.
