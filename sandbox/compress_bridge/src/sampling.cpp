#include "mini_transformer.h"
#include <algorithm>
#include <cmath>
#include <random>

int MiniTransformer::sample_top_p_top_k(const std::vector<float>& logits, float temperature, int k, float p, const std::vector<int>& history) {
    std::vector<std::pair<float, int>> probs;
    float sum_exp = 0.0f;
    float max_logit = *std::max_element(logits.begin(), logits.end());

    // 1. Apply Temperature and Repetition Penalty
    for (int i = 0; i < (int)logits.size(); i++) {
        float l = logits[i] / std::max(temperature, 1e-6f);
        
        // Repetition Penalty: Lower logit if word was used recently
        for (int h : history) {
            if (i == h) {
                l -= 2.0f; // Aggressive penalty to break word salad
                break;
            }
        }
        
        float e = std::exp(l - max_logit);
        probs.push_back({e, i});
        sum_exp += e;
    }

    // Normalize
    for (auto& pair : probs) pair.first /= sum_exp;

    // 2. Top-K Filter
    std::sort(probs.rbegin(), probs.rend());
    if (k > 0 && k < (int)probs.size()) probs.erase(probs.begin() + k, probs.end());

    // 3. Top-P (Nucleus) Filter
    float cumulative_p = 0.0f;
    int last_idx = 0;
    for (int i = 0; i < (int)probs.size(); i++) {
        cumulative_p += probs[i].first;
        last_idx = i;
        if (cumulative_p >= p) break;
    }
    probs.erase(probs.begin() + last_idx + 1, probs.end());

    // 4. Sample from remaining
    float r = (float)rand() / RAND_MAX;
    float current_p = 0.0f;
    for (const auto& pair : probs) {
        current_p += (pair.first / cumulative_p);
        if (r <= current_p) return pair.second;
    }

    return probs[0].second;
}
