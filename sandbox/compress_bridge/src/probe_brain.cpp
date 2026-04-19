#include "mini_transformer.h"
#include "satellite_scanner.h"
#include "bpe_tokenizer.h"
#include <iostream>
#include <vector>
#include <iomanip>

void check_topic(const std::string& topic, MiniTransformer& model, BPETokenizer& tokenizer, TerraLens::SatelliteScanner& radar) {
    auto tokens = tokenizer.encode(topic);
    std::vector<int> targets = tokens; // Self-supervised probe
    
    // 1. Scan for the basin
    float min_loss = radar.scan(tokens, targets, 50);
    
    // 2. Interpret Understanding
    std::string level;
    if (min_loss < 0.5f) level = "🧠 PROFOUND (Native Concept)";
    else if (min_loss < 1.5f) level = "✅ STRONG (Well-Trained)";
    else if (min_loss < 3.0f) level = "⚖️ AVERAGE (General Awareness)";
    else level = "❓ WEAK (Unknown/Noisy)";

    std::cout << "| " << std::left << std::setw(30) << topic 
              << " | " << std::setw(10) << std::fixed << std::setprecision(4) << min_loss 
              << " | " << level << " |" << std::endl;
}

int main() {
    std::cout << "\n--- 🛰️ TerraLens: Brain Understanding Diagnostic ---" << std::endl;
    std::cout << "Measuring internal concept stability via Satellite Recon...\n" << std::endl;
    
    TransformerConfig config;
    MiniTransformer model(config);
    BPETokenizer tokenizer;
    tokenizer.train({"C++ programming is powerful.", "The capital of France is Paris.", "Neural networks learn from data."}, 1000);
    TerraLens::SatelliteScanner radar(model);

    std::cout << "----------------------------------------------------------------------------" << std::endl;
    std::cout << "| Topic                          | Loss       | Understanding Level          |" << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;

    check_topic("C++ programming", model, tokenizer, radar);
    check_topic("Neural networks", model, tokenizer, radar);
    check_topic("Quantum physics", model, tokenizer, radar);
    check_topic("Cooking pasta", model, tokenizer, radar);

    std::cout << "----------------------------------------------------------------------------" << std::endl;
    
    return 0;
}
