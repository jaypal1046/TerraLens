#ifndef KNOWLEDGE_BRIDGE_H
#define KNOWLEDGE_BRIDGE_H

#include "mini_transformer.h"
#include "knowledge_store.h"

namespace TerraLens {

// KnowledgeBridge: The RAG engine for MiniTransformer
class KnowledgeBridge {
public:
    KnowledgeBridge(MiniTransformer& model, KnowledgeStore& store, BPETokenizer& tokenizer);

    // Answer a question by retrieving facts and prompting the model
    std::string answer(const std::string& question, bool use_probing = true);

private:
    MiniTransformer& model_;
    KnowledgeStore& store_;
    BPETokenizer& tokenizer_;

    // Use Satellite Scanner to "recon" the best facts for the question
    std::vector<std::string> probe_facts(const std::string& question, int candidates = 10, int pick = 2);

    // Generate a mock embedding for the query (using the model's embedding layer)
    std::vector<float> embed_query(const std::string& query);
};

} // namespace TerraLens

#endif // KNOWLEDGE_BRIDGE_H
