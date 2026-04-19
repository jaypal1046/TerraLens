#ifndef KNOWLEDGE_STORE_H
#define KNOWLEDGE_STORE_H

#include "lattice_quantizer.h"
#include <string>
#include <vector>
#include <map>

namespace TerraLens {

// HigmanCompressedKnowledge: High-density knowledge storage for Laptop-AI
class KnowledgeStore {
public:
    KnowledgeStore(const LatticeQuantizer::Config& config);

    // Add a "Fact" (sentence + its embedding) to the store
    void add_fact(const std::string& text, const std::vector<float>& embedding);

    // Search for the most relevant facts (using compressed distance)
    std::vector<std::string> search(const std::vector<float>& query_embedding, int k = 3);

    size_t size() const { return facts_.size(); }
    size_t memory_usage() const;

private:
    LatticeQuantizer quantizer_;
    
    struct Fact {
        std::string text;
        std::vector<LatticeQuantizer::CompressedVector> compressed_embedding;
    };

    std::vector<Fact> facts_;
};

} // namespace TerraLens

#endif // KNOWLEDGE_STORE_H
