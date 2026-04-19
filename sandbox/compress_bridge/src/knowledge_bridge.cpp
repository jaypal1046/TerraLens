#include "knowledge_bridge.h"
#include <sstream>
#include <iostream>

namespace TerraLens {

KnowledgeBridge::KnowledgeBridge(MiniTransformer& model, KnowledgeStore& store, BPETokenizer& tokenizer)
    : model_(model), store_(store), tokenizer_(tokenizer) {}

std::vector<float> KnowledgeBridge::embed_query(const std::string& query) {
    // Simple average embedding of the query tokens
    auto tokens = tokenizer_.encode(query);
    int dim = model_.get_config().embedding_dim;
    std::vector<float> query_emb(dim, 0.0f);

    if (tokens.empty()) return query_emb;

    for (int token : tokens) {
        const float* emb = model_.get_embedding(token);
        for (int i = 0; i < dim; i++) {
            query_emb[i] += emb[i];
        }
    }

    for (int i = 0; i < dim; i++) {
        query_emb[i] /= static_cast<float>(tokens.size());
    }

    return query_emb;
}

std::string KnowledgeBridge::answer(const std::string& question, bool use_probing) {
    std::cout << "[BRIDGE] Analyzing question: " << question << std::endl;

    // 1. Fact-Probing (Satellite Recon)
    std::vector<std::string> facts;
    if (use_probing) {
        facts = probe_facts(question, 10, 2);
    } else {
        auto query_emb = embed_query(question);
        facts = store_.search(query_emb, 2);
    }

    // 2. Build Augmented Prompt
    std::stringstream prompt;
    prompt << "Context: ";
    for (const auto& fact : facts) {
        prompt << fact << " ";
        std::cout << "[BRIDGE] Retrieved Fact: " << fact << std::endl;
    }
    prompt << "\nQuestion: " << question << "\nAnswer:";

    // 3. Generate Answer
    std::cout << "[BRIDGE] Generating answer with " << (use_probing ? "Probed" : "Semantic") << " context..." << std::endl;
    return model_.generate(prompt.str(), tokenizer_, 100); 
}

std::vector<std::string> KnowledgeBridge::probe_facts(const std::string& question, int candidates, int pick) {
    std::cout << "[SATELLITE] 🛰️ Probing compressed knowledge manifold..." << std::endl;
    
    // Step 1: Semantic Broad-Search
    auto query_emb = embed_query(question);
    auto raw_candidates = store_.search(query_emb, candidates);
    
    // Step 2: Loss-based Reconnaissance
    std::vector<std::pair<float, std::string>> ranked_facts;
    
    for (const auto& fact : raw_candidates) {
        // Calculate Cross-Entropy Loss for the fact (Simplified: based on average log-prob)
        auto tokens = tokenizer_.encode(fact);
        if (tokens.size() < 2) continue;
        
        float fact_loss = 0;
        // Use the model to evaluate how "surprising" this fact is
        // (Lower loss = model understands this fact better)
        std::vector<int> inputs(tokens.begin(), tokens.end() - 1);
        std::vector<int> targets(tokens.begin() + 1, tokens.end());
        
        // We'll use a single forward pass to get the loss
        fact_loss = model_.calculate_sequence_loss(inputs, targets);
        
        std::cout << "  [RECON] Fact: '" << (fact.size() > 30 ? fact.substr(0, 30) + "..." : fact) 
                  << "' | Surprise: " << fact_loss << std::endl;
        
        ranked_facts.push_back({fact_loss, fact});
    }
    
    // Sort by Surprise (Loss) ascending
    std::sort(ranked_facts.begin(), ranked_facts.end());
    
    std::vector<std::string> best_facts;
    for (int i = 0; i < std::min(pick, static_cast<int>(ranked_facts.size())); i++) {
        best_facts.push_back(ranked_facts[i].second);
    }
    
    return best_facts;
}

} // namespace TerraLens
