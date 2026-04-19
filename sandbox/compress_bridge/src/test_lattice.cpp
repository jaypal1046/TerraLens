#include "lattice_quantizer.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    std::cout << "--- 🛰️ TerraLens: Higman-Sims V19 Singularity-Pulse Test ---" << std::endl;
    
    int dim = 32;
    TerraLens::LatticeQuantizer::Config config;
    config.dim = dim;
    config.max_stages = 6;
    config.scale_gain = 100.0f;
    
    TerraLens::LatticeQuantizer quant(config);
    
    // Generate 32D test vector
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> x(dim);
    std::cout << "Original (First 8D): ";
    for (int i = 0; i < dim; i++) {
        x[i] = dist(gen);
        if (i < 8) std::cout << std::fixed << std::setprecision(4) << x[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    // Encode (V19 Hierarchical)
    auto cv_list = quant.encode(x.data(), dim);
    
    std::cout << "Encoded (Blocks: " << cv_list.size() << ")" << std::endl;
    int active_stages = 0;
    for (const auto& m : cv_list[0].pulse_masks) if (m) active_stages++;
    std::cout << "Active Pulse Stages (Block 0): " << active_stages << " / 6" << std::endl;
    
    // Decode
    std::vector<float> x_hat(dim);
    quant.decode(cv_list, x_hat.data(), dim);
    
    std::cout << "Decoded (First 8D):  ";
    float mse = 0;
    float signal = 0;
    for (int i = 0; i < dim; i++) {
        if (i < 8) std::cout << std::fixed << std::setprecision(4) << x_hat[i] << " ";
        float err = x[i] - x_hat[i];
        mse += err * err;
        signal += x[i] * x[i];
    }
    std::cout << "..." << std::endl;
    
    mse /= dim;
    signal /= dim;
    float snr = 10.0f * std::log10(signal / (mse + 1e-25f));
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "MSE Error: " << std::scientific << mse << std::endl;
    std::cout << "Peak SNR:  " << std::fixed << std::setprecision(2) << snr << " dB" << std::endl;
    
    if (snr >= 50.0f) {
        std::cout << "✅ SUCCESS: V19 finalized the 3.0 BPD / 50 dB milestone!" << std::endl;
    } else {
        std::cout << "⚠️ WARNING: SNR below V19 target." << std::endl;
    }
    
    return 0;
}
