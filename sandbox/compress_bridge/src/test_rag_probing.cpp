#include "mini_transformer.h"
#include "knowledge_store.h"
#include "knowledge_bridge.h"
#include <iostream>

using namespace TerraLens;

int main() {
    std::cout << "--- 🛰️ TerraLens: Phase 3 Satellite Probing RAG Test ---" << std::endl;
    
    // 1. Setup Model
    TransformerConfig config;
    config.vocab_size = 1000;
    config.embedding_dim = 128;
    config.num_layers = 2;
    MiniTransformer model(config);
    
    BPETokenizer tokenizer;
    std::vector<std::string> dummy_corpus = {"The future of AI is local.", "Higman-Sims lattices optimize memory."};
    tokenizer.train(dummy_corpus, 1000);
    
    // 2. Setup Knowledge Store (V19 Lattice)
    LatticeQuantizer::Config l_config;
    l_config.dim = 128;
    l_config.max_stages = 6;
    KnowledgeStore store(l_config);
    
    // Ingest specific facts
    std::cout << "[STORE] Ingesting knowledge into Lattice-Pulse store..." << std::endl;
    
    auto add_fact_with_emb = [&](const std::string& text) {
        // Average embedding of text tokens
        auto tokens = tokenizer.encode(text);
        std::vector<float> emb(128, 0.0f);
        for(int t : tokens) {
            auto t_emb = model.get_embedding(t);
            for(int i=0; i<128; i++) emb[i] += t_emb[i];
        }
        if (!tokens.empty()) {
            for(float& e : emb) e /= tokens.size();
        }
        store.add_fact(text, emb);
    };
    
    add_fact_with_emb("TerraLens uses Radar-guided skip jumps to accelerate training.");
    add_fact_with_emb("Standard Adam optimizers often get stuck in local minima on laptop CPUs.");
    add_fact_with_emb("Satellite Scanner performs global basin mapping before training starts.");
    
    // 3. Setup Knowledge Bridge
    KnowledgeBridge bridge(model, store, tokenizer);
    
    // 4. Test Probed Answer
    std::cout << "\n[TEST] Question: 'How does TerraLens accelerate training?'" << std::endl;
    std::string answer = bridge.answer("How does TerraLens accelerate training?", true);
    
    std::cout << "\n------------------------------------------------" << std::endl;
    std::cout << "🤖 AI Answer: " << answer << std::endl;
    
    if (answer.find("Radar") != std::string::npos || answer.find("skip jumps") != std::string::npos) {
        std::cout << "✅ SUCCESS: Probing Bridge correctly identified and used the fact!" << std::endl;
    } else {
        std::cout << "⚠️ NOTE: Model generation is untrained, but check if fact was retrieved above." << std::endl;
    }
    
    return 0;
}
