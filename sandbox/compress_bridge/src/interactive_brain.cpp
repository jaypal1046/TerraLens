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

bool file_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char* argv[]) {
    bool force_train = false;
    if (argc > 1 && std::string(argv[1]) == "--train") {
        force_train = true;
    }

    std::cout << "===============================================" << std::endl;
    std::cout << "🧠 TERRALENS: UNIFIED ALIGNED BRAIN (v1.0)" << std::endl;
    std::cout << "God-Mode Activated | Lattice Store v19 Linked" << std::endl;
    std::cout << "===============================================\n" << std::endl;

    TransformerConfig config;
    config.vocab_size = 512; 
    config.embedding_dim = 256;
    MiniTransformer model(config);
    
    BPETokenizer tokenizer;
    std::vector<std::string> training_data = {
        "Q: What is TerraLens? A: TerraLens is a C++ optimization engine for laptop AI.",
        "Q: What is Lattice quantization? A: Lattice quantization achieves 50x compression on RSN metrics.",
        "Q: What is Radar-Guided DPO? A: Radar-Guided DPO ensures human-aligned behavior.",
        "Q: Hi A: Hi! How can I help you today?",
        "Q: what your name A: My name is TerraLens, your AI optimization partner."
    };
    
    tokenizer.train(training_data, 512);

    // 📂 SAVE/LOAD LOGIC
    std::string weight_path = "brain_weights.bin";
    
    if (force_train || !file_exists(weight_path)) {
        std::cout << "[SYSTEM] Hard-Wiring Instruction Set (200x)..." << std::endl;
        for (int epoch = 0; epoch < 200; epoch++) {
            float epoch_loss = 0;
            int steps = 0;
            for (const auto& line : training_data) {
                auto tokens = tokenizer.encode(line);
                if (tokens.size() < 2) continue;
                std::vector<int> inputs(tokens.begin(), tokens.end() - 1);
                std::vector<int> targets(tokens.begin() + 1, tokens.end());
                epoch_loss += model.training_step(inputs, targets, 1e-3f); 
                steps++;
            }
            if (epoch % 50 == 0) std::cout << "   [Epoch " << epoch << "] Loss: " << (epoch_loss / steps) << std::endl;
        }
        std::cout << "[SYSTEM] Saving weights to " << weight_path << "..." << std::endl;
        model.save(weight_path);
        std::cout << "[SYSTEM] Training complete.\n" << std::endl;
    } else {
        std::cout << "[SYSTEM] Loading pre-trained weights from " << weight_path << "..." << std::endl;
        model.load(weight_path);
        std::cout << "[SYSTEM] Brain loaded and ready.\n" << std::endl;
    }

    // 2. Interactive Loop
    std::string user_input;
    while (true) {
        std::cout << "\n👤 You: ";
        std::getline(std::cin, user_input);
        if (user_input == "exit" || user_input == "quit") break;

        std::cout << "🧠 AI [Thinking]: Generating aligned response..." << std::endl;
        
        std::string prompt = "Q: " + user_input + " A:";
        auto current_tokens = tokenizer.encode(prompt);
        std::vector<int> generated;
        
        for (int i = 0; i < 50; i++) { 
            auto logits = model.test_forward(current_tokens);
            if (logits.empty()) break;
            
            int next_id = 0;
            float max_l = -1e9;
            for(int j=0; j<config.vocab_size; j++) {
                if(logits.back()[j] > max_l) {
                    max_l = logits.back()[j];
                    next_id = j;
                }
            }
            
            generated.push_back(next_id);
            current_tokens.push_back(next_id);
            std::string word = tokenizer.decode({next_id});
            if (word == "." || word == "!" || word == "?") break;
        }

        std::cout << "🤖 Assistant: " << tokenizer.decode(generated) << std::endl;
    }

    return 0;
}
