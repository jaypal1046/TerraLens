#ifndef SATELLITE_SCANNER_H
#define SATELLITE_SCANNER_H

#include "mini_transformer.h"
#include <vector>
#include <string>

namespace TerraLens {

// Satellite Scanner: Performs sparse global optimization to find the best initialization "Basin"
class SatelliteScanner {
public:
    SatelliteScanner(MiniTransformer& model);

    // Perform a global sweep with N candidates
    // Returns the best loss found
    float scan(const std::vector<int>& tokens, const std::vector<int>& targets, int num_candidates = 100);

    // Apply the best found basin to the model
    void apply_best_basin();

private:
    MiniTransformer& model_;
    
    struct BasinCandidate {
        std::vector<std::vector<float>> token_emb;
        float loss;
    };

    std::vector<BasinCandidate> candidates_;
    int best_candidate_idx_ = -1;

    float evaluate_candidate(const std::vector<int>& tokens, const std::vector<int>& targets);
};

} // namespace TerraLens

#endif // SATELLITE_SCANNER_H
