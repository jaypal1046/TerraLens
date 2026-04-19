#include "knowledge_store.h"
#include <iostream>
#include <vector>
#include <random>

// Mock Embedding Generator (In production, this would be the Transformer encoder)
std::vector<float> generate_mock_embedding(int seed, int dim = 256) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> emb(dim);
    for (int i = 0; i < dim; i++) emb[i] = dist(gen);
    return emb;
}

int main() {
    std::cout << "--- 🛰️ TerraLens: Phase 3 Knowledge Bridge Test ---" << std::endl;
    
    TerraLens::LatticeQuantizer::Config config;
    config.dim = 256;
    config.max_stages = 6;
    
    TerraLens::KnowledgeStore store(config);
    
    // 1. Ingest Facts
    std::cout << "[STORE] Ingesting facts into V19 Lattice Store..." << std::endl;
    store.add_fact("The Higman-Sims graph has 100 vertices.", generate_mock_embedding(1));
    store.add_fact("The E8 Gosset Lattice is the optimal 8D packing.", generate_mock_embedding(2));
    store.add_fact("TerraLens reduces convergence time by 55%.", generate_mock_embedding(3));
    store.add_fact("Laptop-native AI requires extreme quantization.", generate_mock_embedding(4));
    
    std::cout << "[STORE] Memory Usage (Compressed): " << store.memory_usage() << " bytes" << std::endl;
    
    // 2. Semantic Search (Search for "TerraLens efficiency")
    std::cout << "[SEARCH] Querying: 'How much does TerraLens improve training?'" << std::endl;
    auto query = generate_mock_embedding(3); // Mock query matching the TerraLens fact
    
    auto results = store.search(query, 1);
    
    std::cout << "------------------------------------------------" << std::endl;
    if (!results.empty()) {
        std::cout << "🎯 Top Result: " << results[0] << std::endl;
        if (results[0].find("TerraLens") != std::string::npos) {
            std::cout << "✅ SUCCESS: Fact retrieved correctly from compressed memory!" << std::endl;
        }
    } else {
        std::cout << "❌ FAILED: No results found." << std::endl;
    }
    
    return 0;
}
