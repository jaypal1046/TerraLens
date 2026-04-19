#ifndef RLHF_H
#define RLHF_H

#include <vector>
#include <string>
#include <memory>

// Forward declarations
class MiniTransformer; // Global namespace
class BPETokenizer;   // Global namespace
namespace TerraLens {
    class SatelliteScanner;
}

namespace rlhf {

// SFT (Supervised Fine-Tuning) structures
struct SFTPair {
    std::string prompt;
    std::string completion;

    static SFTPair from_json(const std::string& json);
    std::string to_json() const;
};

// Reward Model structures
struct RewardComparison {
    std::string prompt;
    std::string response_a;
    std::string response_b;
    char preferred; // 'a' or 'b'

    static RewardComparison from_json(const std::string& json);
    std::string to_json() const;
};

// =============================================================================
// Reward Model Implementation
// =============================================================================

class RewardModel {
public:
    RewardModel(int embedding_dim);

    float score(const std::string& prompt, const std::string& response);
    float train(const std::vector<RewardComparison>& comparisons, int epochs = 5, float lr = 0.001f);

    void save(const std::string& path);
    void load(const std::string& path);

private:
    int embedding_dim_;
    int hidden_dim_;

    std::vector<std::vector<float>> fc1_weight;
    std::vector<float> fc1_bias;
    std::vector<std::vector<float>> fc2_weight;
    std::vector<float> fc2_bias;

    float relu(float x) { return x > 0 ? x : 0; }
    std::vector<float> get_embedding(const std::string& text);
    float forward(const std::vector<float>& embedding);
    void backward(const std::vector<float>& embedding, float grad_output,
                  std::vector<std::vector<float>>& fc1_grad_w, std::vector<float>& fc1_grad_b,
                  std::vector<std::vector<float>>& fc2_grad_w, std::vector<float>& fc2_grad_b);
};

// =============================================================================
// Phase F4: DPO Implementation (God-Mode Alignment)
// =============================================================================

struct DPOPair {
    std::string prompt;
    std::string chosen;
    std::string rejected;
};

class DPOTrainer {
public:
    DPOTrainer(MiniTransformer& model, BPETokenizer& tokenizer, float beta = 0.1f);

    // Train the model using Direct Preference Optimization
    float train_step(const DPOPair& pair, float lr);

    // Radar-Guided Alignment: Uses SatelliteScanner to adjust LR and skip terrain
    float train_step_radar(const DPOPair& pair, float lr, TerraLens::SatelliteScanner& radar);

    // Run full DPO alignment loop
    void run_alignment(const std::string& preference_file, int epochs, float lr);

    // Calculate log probabilities for a sequence
    float calculate_log_prob(const std::string& prompt, const std::string& completion);

private:
    MiniTransformer& model_;
    BPETokenizer& tokenizer_;
    float beta_; // DPO temperature (implicit reward scale)
};

// =============================================================================
// Phase F1: SFT (Supervised Fine-Tuning)
// =============================================================================

float run_sft(const std::string& training_file, int epochs = 3, float lr = 1e-5f, int batch_size = 4);

// =============================================================================
// Phase F2: Reward Model Training
// =============================================================================

float run_reward_model_training(const std::string& comparisons_file,
                                 const std::string& output_model_path,
                                 int epochs = 10,
                                 float lr = 0.001f);

float bradley_terry_loss(float score_preferred, float score_other);

// =============================================================================
// Phase F3: PPO (Proximal Policy Optimization) - Legacy Stubs
// =============================================================================

float run_ppo(const std::string& prompts_file,
              const std::string& reward_model_path,
              int num_iterations = 10,
              float lr = 1e-5f,
              float clip_epsilon = 0.2f,
              float kl_penalty = 0.01f);

// Sample Data Creation
void create_sample_sft_data(const std::string& output_file);
void create_sample_comparisons(const std::string& output_file);
void create_sample_prompts(const std::string& output_file);

} // namespace rlhf

#endif // RLHF_H
