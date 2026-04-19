#include "rlhf.h"
#include "mini_transformer.h"
#include "satellite_scanner.h"
#include "bpe_tokenizer.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "--- 🛰️ TerraLens: Phase 4 Radar-Guided Alignment Test ---" << std::endl;
    
    // 1. Setup Model, Tokenizer & Radar
    TransformerConfig config;
    config.vocab_size = 1000;
    config.embedding_dim = 128;
    MiniTransformer model(config);
    
    BPETokenizer tokenizer;
    tokenizer.train({"Helpful.", "Harmful."}, 1000);
    
    TerraLens::SatelliteScanner radar(model);
    
    rlhf::DPOTrainer trainer(model, tokenizer, 0.1f);
    
    // 2. Prepare Preference Data
    rlhf::DPOPair pair;
    pair.prompt = "User: How should I help others? Assistant:";
    pair.chosen = " I should provide helpful and accurate information.";
    pair.rejected = " I should give incorrect or harmful advice.";
    
    // 3. Measure Log-Probs BEFORE Alignment
    float logp_c_before = trainer.calculate_log_prob(pair.prompt, pair.chosen);
    float logp_r_before = trainer.calculate_log_prob(pair.prompt, pair.rejected);
    
    std::cout << "[BEFORE] Chosen Log-Prob: " << logp_c_before << std::endl;
    std::cout << "[BEFORE] Rejected Log-Prob: " << logp_r_before << std::endl;
    
    // 4. Run Radar-Guided Alignment
    std::cout << "\n[ALIGN] Running Radar-Guided DPO steps..." << std::endl;
    for (int i = 0; i < 3; i++) {
        float loss = trainer.train_step_radar(pair, 1e-3f, radar);
        std::cout << "  [STEP " << i+1 << "] DPO Loss: " << loss << std::endl;
    }
    
    // 5. Measure Log-Probs AFTER Alignment
    float logp_c_after = trainer.calculate_log_prob(pair.prompt, pair.chosen);
    float logp_r_after = trainer.calculate_log_prob(pair.prompt, pair.rejected);
    
    std::cout << "\n[AFTER] Chosen Log-Prob: " << logp_c_after << std::endl;
    std::cout << "[AFTER] Rejected Log-Prob: " << logp_r_after << std::endl;
    
    float margin_before = logp_c_before - logp_r_before;
    float margin_after = logp_c_after - logp_r_after;
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Preference Margin (Before): " << margin_before << std::endl;
    std::cout << "Preference Margin (After):  " << margin_after << std::endl;
    
    if (margin_after > margin_before) {
        std::cout << "✅ SUCCESS: Phase 4 Alignment verified! Model shifted towards helpfulness." << std::endl;
    } else {
        std::cout << "⚠️ NOTE: Margin improvement was " << (margin_after - margin_before) << ". Check learning rate if 0." << std::endl;
    }
    
    return 0;
}
