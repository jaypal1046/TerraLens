#include "rlhf.h"
#include "mini_transformer.h"
#include "satellite_scanner.h"
#include "knowledge_bridge.h"
#include "knowledge_store.h"
#include "lattice_quantizer.h"
#include "bpe_tokenizer.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <map>
#include <filesystem>

namespace fs = std::filesystem;

struct BrainConfig {
    int vocab_size;
    int embedding_dim;
    int num_layers;
    int num_heads;
    float best_alignment;

    void save(const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        f.write((char*)this, sizeof(BrainConfig));
    }
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.good()) return false;
        f.read((char*)this, sizeof(BrainConfig));
        return true;
    }
};

void ingest_directory(const std::string& path, std::vector<std::string>& corpus) {
    if (!fs::exists(path)) return;
    std::cout << "[SYSTEM] Scanning: " << path << "..." << std::endl;
    int count = 0;
    for (const auto& entry : fs::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) {
            count++;
            if (count % 100 == 0) std::cout << "   [SCAN] Found " << count << " files..." << std::flush << "\r";
            
            std::string ext = entry.path().extension().string();
            if (ext == ".json") {
                auto facts = rlhf::load_json_array(entry.path().string());
                for (const auto& f : facts) corpus.push_back(f);
            } else if (ext == ".txt" || ext == ".md" || ext == ".cpp" || ext == ".h") {
                std::ifstream f(entry.path());
                std::string line;
                while (std::getline(f, line)) if (!line.empty()) corpus.push_back(line);
            }
        }
    }
    std::cout << "\n[SYSTEM] Ingestion complete. Total files: " << count << std::endl;
}

bool does_file_exist(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char* argv[]) {
    bool force_train = (argc > 1 && std::string(argv[1]) == "--train");

    std::cout << "===============================================" << std::endl;
    std::cout << " TERRALENS: WEIGHTED TIER ENGINE (v6.5)" << std::endl;
    std::cout << " Grammar Glue | Priority Sync Active" << std::endl;
    std::cout << "===============================================\n" << std::endl;

    std::vector<std::string> corpus;
    ingest_directory("../brain_data", corpus);
    
    std::vector<size_t> fact_indices;
    std::vector<size_t> grammar_indices;
    std::vector<size_t> foundation_indices;
    
    std::cout << "[SYSTEM] Sorting corpus via Zero-Copy Indexing..." << std::endl;
    for(size_t i = 0; i < corpus.size(); i++) {
        if (i % 100000 == 0) std::cout << "   [PROCESS] Mapping " << i << " lines..." << std::flush << "\r";
        const auto& s = corpus[i];
        if (s.find("Q:") != std::string::npos) fact_indices.push_back(i);
        else if (s.find(".") != std::string::npos && s.length() < 50) grammar_indices.push_back(i);
        else foundation_indices.push_back(i);
    }
    std::cout << "\n[SYSTEM] Indexing complete. Facts: " << fact_indices.size() << " | Grammar: " << grammar_indices.size() << std::endl;

    std::cout << "[SYSTEM] Initializing BPE Tokenizer..." << std::endl;
    BPETokenizer tokenizer(32000);
    if (fs::exists("semantic_vocab.bin")) {
        std::cout << "[SYSTEM] Loading existing vocabulary..." << std::endl;
        tokenizer.load("semantic_vocab.bin");
    } else {
        std::cout << "[SYSTEM] Training BPE Tokenizer (5000 merges)..." << std::endl;
        tokenizer.train(corpus, 5000); 
        tokenizer.save("semantic_vocab.bin");
    }

    BrainConfig b_config;
    b_config.vocab_size = tokenizer.vocab_size() + 10;
    b_config.embedding_dim = 256;
    b_config.num_layers = 4;
    b_config.num_heads = 8;
    b_config.best_alignment = 0.0f;

    std::string weight_path = "brain_weights.bin";
    std::string config_path = "brain_config.bin";
    
    std::cout << "[SYSTEM] Synchronizing Brain Config..." << std::endl;
    if (fs::exists(config_path)) {
        if (b_config.load(config_path)) {
            std::cout << "[SUCCESS] Config synchronized. Vocabulary: " << b_config.vocab_size << std::endl;
        } else {
            std::cout << "[WARNING] Config mismatch. Re-calibrating for BPE..." << std::endl;
        }
    }

    TransformerConfig config;
    config.vocab_size = b_config.vocab_size;
    config.embedding_dim = b_config.embedding_dim;
    config.num_layers = b_config.num_layers;
    config.num_heads = b_config.num_heads;
    MiniTransformer model(config);
    
    if (fs::exists("brain_weights.bin")) {
        std::cout << "[SYSTEM] Loading existing memory..." << std::endl;
        try {
            model.load("brain_weights.bin");
            std::cout << "[SUCCESS] 13M parameters synchronized." << std::endl;
        } catch (...) {
            std::cout << "[WARNING] Memory corrupted. Attempting backup..." << std::endl;
            try { model.load("brain_weights.bin.bak"); } catch(...) { std::cout << "[CRITICAL] Memory lost." << std::endl; }
        }
    }

    if (force_train) {
        std::cout << "[SYSTEM] Starting 10,000-Step Deep Lock Sync..." << std::endl;
        float current_best = b_config.best_alignment;
        for (int iter = 0; iter < 10000; iter++) {
            float total_loss = 0; int samples = 0;
            
            for (int i = 0; i < 70; i++) {
                if (grammar_indices.empty()) break;
                auto tokens = tokenizer.encode(corpus[grammar_indices[rand() % grammar_indices.size()]]);
                if (tokens.size() < 2) continue;
                total_loss += model.training_step(tokens, tokens, 5e-3f);
                samples++;
            }
            
            for (int i = 0; i < 25; i++) {
                if (fact_indices.empty()) break;
                auto tokens = tokenizer.encode(corpus[fact_indices[rand() % fact_indices.size()]]);
                if (tokens.size() < 2) continue;
                total_loss += model.training_step(tokens, tokens, 8e-3f);
                samples++;
            }

            for (int i = 0; i < 5; i++) {
                if (foundation_indices.empty()) break;
                auto tokens = tokenizer.encode(corpus[foundation_indices[rand() % foundation_indices.size()]]);
                if (tokens.size() < 2) continue;
                total_loss += model.training_step(tokens, tokens, 1e-3f);
                samples++;
            }

            if (samples == 0) continue;
            float alignment = (1.0f - (total_loss / samples / 10.0f)) * 100.0f;
            if (alignment > current_best && !std::isnan(alignment)) {
                current_best = alignment;
                model.save(weight_path);
                b_config.best_alignment = current_best;
                b_config.save(config_path);
            }
            if (iter % 100 == 0) std::cout << "   [Priority Sync] Alignment: " << alignment << "%" << std::endl;
        }
    }

    while (true) {
        std::string user_input;
        std::cout << "\n[USER]: ";
        std::getline(std::cin, user_input);
        if (user_input == "exit" || user_input == "quit") break;

        std::cout << "[ASSISTANT]: ";
        std::string response = model.generate(user_input, tokenizer, 50, 0.7f, 40);
        std::cout << response << std::endl;
    }

    return 0;
}
