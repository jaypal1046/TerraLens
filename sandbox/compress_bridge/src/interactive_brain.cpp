#include "rlhf.h"
#include "mini_transformer.h"
#include "satellite_scanner.h"
#include "knowledge_bridge.h"
#include "knowledge_store.h"
#include "lattice_quantizer.h"
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

class WordTokenizer {
public:
    std::map<std::string, int> word_to_id;
    std::map<int, std::string> id_to_word;
    int vocab_size = 0;
    int min_freq = 5; // Increase min_freq to filter dictionary noise

    void train(const std::vector<std::string>& corpus) {
        if (load("semantic_vocab.bin")) return;
        std::map<std::string, int> freq;
        for (const auto& text : corpus) {
            std::string cleaned = "";
            for(char c : text) cleaned += (std::isalnum(c) || c == '\'' ? (char)std::tolower(c) : ' ');
            std::stringstream ss(cleaned);
            std::string word;
            while (ss >> word) freq[word]++;
        }
        word_to_id["<pad>"] = 0; id_to_word[0] = "<pad>";
        word_to_id["<eos>"] = 1; id_to_word[1] = "<eos>";
        vocab_size = 2;
        for (auto const& [word, count] : freq) {
            if (count >= min_freq) { // Filter out rare words
                word_to_id[word] = vocab_size;
                id_to_word[vocab_size] = word;
                vocab_size++;
            }
        }
        save("semantic_vocab.bin");
    }
    void save(const std::string& path) {
        std::ofstream f(path);
        for (auto const& [word, id] : word_to_id) f << word << " " << id << "\n";
    }
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.good()) return false;
        word_to_id.clear(); id_to_word.clear();
        std::string word; int id;
        while (f >> word >> id) {
            word_to_id[word] = id;
            id_to_word[id] = word;
        }
        vocab_size = id_to_word.size();
        return true;
    }
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        std::string cleaned = "";
        for(char c : text) cleaned += (std::isalnum(c) || c == '\'' ? (char)std::tolower(c) : ' ');
        std::stringstream ss(cleaned);
        std::string word;
        while (ss >> word) if (word_to_id.count(word)) tokens.push_back(word_to_id[word]);
        return tokens;
    }
    std::string decode(const std::vector<int>& tokens) {
        std::string text = "";
        for (int id : tokens) if (id_to_word.count(id)) text += id_to_word[id] + " ";
        return text;
    }
};

void ingest_directory(const std::string& path, std::vector<std::string>& corpus) {
    if (!fs::exists(path)) return;
    for (const auto& entry : fs::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) {
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
}

bool does_file_exist(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char* argv[]) {
    bool force_train = (argc > 1 && std::string(argv[1]) == "--train");

    std::cout << "===============================================" << std::endl;
    std::cout << "🧠 TERRALENS: WEIGHTED TIER ENGINE (v6.5)" << std::endl;
    std::cout << "Grammar Glue | Priority Sync Active" << std::endl;
    std::cout << "===============================================\n" << std::endl;

    std::vector<std::string> corpus;
    ingest_directory("../brain_data", corpus);
    
    std::vector<std::string> facts;
    std::vector<std::string> grammar;
    std::vector<std::string> foundation;
    for(const auto& s : corpus) {
        if (s.find("Q:") != std::string::npos) facts.push_back(s);
        else if (s.find(".") != std::string::npos && s.length() < 50) grammar.push_back(s);
        else foundation.push_back(s);
    }

    WordTokenizer tokenizer;
    tokenizer.train(corpus);

    BrainConfig b_config;
    b_config.vocab_size = tokenizer.vocab_size + 10;
    b_config.embedding_dim = 256;
    b_config.num_layers = 4;
    b_config.num_heads = 8;
    b_config.best_alignment = 0.0f;

    std::string weight_path = "brain_weights.bin";
    std::string config_path = "brain_config.bin";
    if (does_file_exist(config_path)) b_config.load(config_path);

    TransformerConfig config;
    config.vocab_size = b_config.vocab_size;
    config.embedding_dim = b_config.embedding_dim;
    config.num_layers = b_config.num_layers;
    config.num_heads = b_config.num_heads;
    MiniTransformer model(config);
    if (does_file_exist(weight_path)) model.load(weight_path);

    if (force_train) {
        std::cout << "[SYSTEM] Starting 10,000-Step Deep Lock Sync..." << std::endl;
        float current_best = b_config.best_alignment;
        for (int iter = 0; iter < 10000; iter++) {
            float total_loss = 0; int samples = 0;
            
            // 70% Grammar (The Structural Glue)
            for (int i = 0; i < 70; i++) {
                if (grammar.empty()) break;
                auto tokens = tokenizer.encode(grammar[rand() % grammar.size()]);
                if (tokens.size() < 2) continue;
                total_loss += model.training_step(tokens, tokens, 5e-3f);
                samples++;
            }
            
            // 25% Facts (The Knowledge)
            for (int i = 0; i < 25; i++) {
                if (facts.empty()) break;
                auto tokens = tokenizer.encode(facts[rand() % facts.size()]);
                if (tokens.size() < 2) continue;
                total_loss += model.training_step(tokens, tokens, 8e-3f);
                samples++;
            }

            // 5% Foundation (The Vocabulary)
            for (int i = 0; i < 5; i++) {
                if (foundation.empty()) break;
                auto tokens = tokenizer.encode(foundation[rand() % foundation.size()]);
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
            if (iter % 1000 == 0) {
                std::cout << "   [Snapshot] Saving iteration " << iter << "..." << std::endl;
                model.save(weight_path);
            }
        }
    }

    while (true) {
        std::string user_input;
        std::cout << "\n👤 You: ";
        std::getline(std::cin, user_input);
        if (user_input == "exit" || user_input == "quit") break;

        std::cout << "🤖 Assistant: ";
        auto tokens = tokenizer.encode(user_input);
        if (tokens.empty()) { std::cout << "..." << std::endl; continue; }

        for (int i = 0; i < 40; i++) {
            auto logits = model.test_forward(tokens).back();
            for (int j = std::max(0, (int)tokens.size() - 20); j < (int)tokens.size(); j++) logits[tokens[j]] -= 15.0f;
            
            std::vector<std::pair<float, int>> top_k;
            for (int k = 0; k < (int)logits.size(); k++) top_k.push_back({logits[k], k});
            std::sort(top_k.rbegin(), top_k.rend());
            
            int next_id = top_k[0].second;
            if (next_id == 1 || next_id == 0) break; 
            std::string word = tokenizer.id_to_word[next_id];
            std::cout << word << " " << std::flush;
            tokens.push_back(next_id);
            if (word.back() == '.' || word.back() == '!' || word.back() == '?') break;
        }
        std::cout << std::endl;
    }
    return 0;
}
