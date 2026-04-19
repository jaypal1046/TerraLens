#include "rlhf.h"
#include "mini_transformer.h"
#include "satellite_scanner.h"
#include "bpe_tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>

namespace rlhf {

// =============================================================================
// Utility Functions
// =============================================================================

static std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\": \"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    size_t end = json.find("\"", pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

static std::vector<std::string> load_json_array(const std::string& file_path) {
    std::vector<std::string> result;
    std::ifstream file(file_path);
    if (!file) return result;
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t pos = 0;
    int brace_depth = 0;
    size_t object_start = std::string::npos;
    for (size_t i = 0; i < content.size(); i++) {
        if (content[i] == '{') {
            if (brace_depth == 0) object_start = i;
            brace_depth++;
        } else if (content[i] == '}') {
            brace_depth--;
            if (brace_depth == 0 && object_start != std::string::npos) {
                result.push_back(content.substr(object_start, i - object_start + 1));
                object_start = std::string::npos;
            }
        }
    }
    return result;
}

// =============================================================================
// DPO Implementation
// =============================================================================

DPOTrainer::DPOTrainer(MiniTransformer& model, BPETokenizer& tokenizer, float beta)
    : model_(model), tokenizer_(tokenizer), beta_(beta) {}

float DPOTrainer::calculate_log_prob(const std::string& prompt, const std::string& completion) {
    auto tokens = tokenizer_.encode(prompt + completion);
    auto prompt_tokens = tokenizer_.encode(prompt);
    
    if (tokens.size() <= prompt_tokens.size()) return -100.0f;
    
    std::vector<int> inputs(tokens.begin(), tokens.end() - 1);
    std::vector<int> targets(tokens.begin() + 1, tokens.end());
    
    int start_idx = static_cast<int>(prompt_tokens.size()) - 1;
    auto logits = model_.test_forward(inputs);
    float log_prob = 0;
    
    for (size_t i = std::max(0, start_idx); i < targets.size(); i++) {
        float sum_exp = 1e-9f;
        for (float l : logits[i]) sum_exp += std::exp(l);
        log_prob += logits[i][targets[i]] - std::log(sum_exp);
    }
    return log_prob;
}

float DPOTrainer::train_step(const DPOPair& pair, float lr) {
    float log_p_chosen = calculate_log_prob(pair.prompt, pair.chosen);
    float log_p_rejected = calculate_log_prob(pair.prompt, pair.rejected);
    
    float margin = beta_ * (log_p_chosen - log_p_rejected);
    float sigmoid = 1.0f / (1.0f + std::exp(-std::clamp(margin, -20.0f, 20.0f)));
    float loss = -std::log(sigmoid + 1e-9f);
    
    float grad_scale = beta_ * (1.0f - sigmoid);
    
    auto tokens_c = tokenizer_.encode(pair.prompt + pair.chosen);
    if (tokens_c.size() > 1) {
        std::vector<int> in_c(tokens_c.begin(), tokens_c.end() - 1);
        std::vector<int> tg_c(tokens_c.begin() + 1, tokens_c.end());
        model_.training_step(in_c, tg_c, lr * grad_scale);
    }
    
    auto tokens_r = tokenizer_.encode(pair.prompt + pair.rejected);
    if (tokens_r.size() > 1) {
        std::vector<int> in_r(tokens_r.begin(), tokens_r.end() - 1);
        std::vector<int> tg_r(tokens_r.begin() + 1, tokens_r.end());
        model_.training_step(in_r, tg_r, -lr * grad_scale);
    }
    
    return loss;
}

float DPOTrainer::train_step_radar(const DPOPair& pair, float lr, TerraLens::SatelliteScanner& radar) {
    auto tokens_c = tokenizer_.encode(pair.prompt + pair.chosen);
    if (tokens_c.size() < 2) return 0;
    std::vector<int> in_c(tokens_c.begin(), tokens_c.end() - 1);
    std::vector<int> tg_c(tokens_c.begin() + 1, tokens_c.end());

    float min_loss = radar.scan(in_c, tg_c, 10); 
    float adaptive_lr = lr;
    if (min_loss < 2.0f) {
        adaptive_lr *= 5.0f;
        std::cout << "  [DPO-RADAR] 🛰️ Stable basin detected. Executing Skip-Alignment." << std::endl;
    }
    return train_step(pair, adaptive_lr);
}

void DPOTrainer::run_alignment(const std::string& preference_file, int epochs, float lr) {
    std::vector<DPOPair> dataset;
    auto json_objects = load_json_array(preference_file);
    for (const auto& json : json_objects) {
        DPOPair p;
        p.prompt = extract_json_string(json, "prompt");
        p.chosen = extract_json_string(json, "chosen");
        p.rejected = extract_json_string(json, "rejected");
        dataset.push_back(p);
    }
    if (dataset.empty()) return;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0;
        for (const auto& pair : dataset) epoch_loss += train_step(pair, lr);
        std::cout << "[DPO] Epoch " << (epoch + 1) << " | Loss: " << (epoch_loss / dataset.size()) << std::endl;
    }
}

// Stubs for neural_engine.cpp compatibility
float run_sft(const std::string&, int, float, int) { return 0; }
float run_reward_model_training(const std::string&, const std::string&, int, float) { return 0; }
float bradley_terry_loss(float, float) { return 0; }
float run_ppo(const std::string&, const std::string&, int, float, float, float) { return 0; }
void create_sample_sft_data(const std::string&) {}
void create_sample_comparisons(const std::string&) {}
void create_sample_prompts(const std::string&) {}

} // namespace rlhf
