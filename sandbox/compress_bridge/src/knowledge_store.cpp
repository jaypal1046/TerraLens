#include "knowledge_store.h"
#include <cmath>
#include <algorithm>

namespace TerraLens {

KnowledgeStore::KnowledgeStore(const LatticeQuantizer::Config& config) 
    : quantizer_(config) {}

void KnowledgeStore::add_fact(const std::string& text, const std::vector<float>& embedding) {
    Fact fact;
    fact.text = text;
    fact.compressed_embedding = quantizer_.encode(embedding.data(), embedding.size());
    facts_.push_back(fact);
}

std::vector<std::string> KnowledgeStore::search(const std::vector<float>& query_embedding, int k) {
    if (facts_.empty()) return {};

    std::vector<std::pair<float, int>> scores;
    std::vector<float> reconstructed(query_embedding.size());

    for (int i = 0; i < static_cast<int>(facts_.size()); i++) {
        // Decompress on-the-fly (O(1) per block)
        quantizer_.decode(facts_[i].compressed_embedding, reconstructed.data(), reconstructed.size());
        
        // Calculate Cosine Similarity or Dot Product
        float dot = 0;
        float mag1 = 0, mag2 = 0;
        for (size_t j = 0; j < query_embedding.size(); j++) {
            dot += query_embedding[j] * reconstructed[j];
            mag1 += query_embedding[j] * query_embedding[j];
            mag2 += reconstructed[j] * reconstructed[j];
        }
        float similarity = dot / (std::sqrt(mag1) * std::sqrt(mag2) + 1e-9f);
        scores.push_back({similarity, i});
    }

    // Sort by similarity descending
    std::sort(scores.rbegin(), scores.rend());

    std::vector<std::string> results;
    for (int i = 0; i < std::min(k, static_cast<int>(scores.size())); i++) {
        results.push_back(facts_[scores[i].second].text);
    }
    return results;
}

size_t KnowledgeStore::memory_usage() const {
    size_t total = 0;
    for (const auto& f : facts_) {
        total += f.text.size();
        for (const auto& cv : f.compressed_embedding) {
            total += cv.qs.size() * 8 * 2; // int16 indices
            total += cv.scales.size() * 4; // float scales
            total += 8; // gm, gs, ls
        }
    }
    return total;
}

} // namespace TerraLens
