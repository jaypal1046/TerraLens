#include "lattice_quantizer.h"
#include <algorithm>
#include <numeric>

namespace TerraLens {

LatticeQuantizer::LatticeQuantizer(const Config& config) : config_(config) {
    // Initialize scales for recursive refinement (V16 logic)
    float s = config_.scale_gain;
    for (int i = 0; i < config_.max_stages; i++) {
        stage_scales_.push_back(s);
        s *= 0.5f; // Decaying residual scale
    }
}

void LatticeQuantizer::decode_dn(const float* x, float* y) {
    float sum = 0;
    for (int i = 0; i < 8; i++) {
        y[i] = std::round(x[i]);
        sum += y[i];
    }

    if (static_cast<int>(sum) % 2 != 0) {
        float max_diff = -1.0f;
        int max_idx = 0;
        for (int i = 0; i < 8; i++) {
            float diff = std::abs(x[i] - y[i]);
            if (diff > max_diff) {
                max_diff = diff;
                max_idx = i;
            }
        }
        y[max_idx] += (x[max_idx] > y[max_idx]) ? 1.0f : -1.0f;
    }
}

void LatticeQuantizer::quantize_e8(const float* x, float* y) {
    float y0[8], y1[8], x_minus_half[8];
    
    // y0 = decode_dn(x)
    decode_dn(x, y0);
    
    // y1 = decode_dn(x - 0.5) + 0.5
    for (int i = 0; i < 8; i++) x_minus_half[i] = x[i] - 0.5f;
    decode_dn(x_minus_half, y1);
    for (int i = 0; i < 8; i++) y1[i] += 0.5f;
    
    // Choose closer one
    float d0 = 0, d1 = 0;
    for (int i = 0; i < 8; i++) {
        d0 += (x[i] - y0[i]) * (x[i] - y0[i]);
        d1 += (x[i] - y1[i]) * (x[i] - y1[i]);
    }
    
    const float* best = (d0 < d1) ? y0 : y1;
    for (int i = 0; i < 8; i++) y[i] = best[i];
}

void LatticeQuantizer::encode_8d(const float* x, CompressedVector& out) {
    // Legacy single 8D encode - redirected to V19 logic below
}

std::vector<LatticeQuantizer::CompressedVector> LatticeQuantizer::encode(const float* x, int size) {
    int num_32d = (size + 31) / 32;
    std::vector<CompressedVector> output(num_32d);
    
    for (int b = 0; b < num_32d; b++) {
        const float* block_x = x + b * 32;
        CompressedVector& cv = output[b];
        
        // 1. Hierarchical Normalization (V19 style)
        double gm = 0;
        for (int i = 0; i < 32; i++) gm += (i < size - b*32) ? block_x[i] : 0;
        gm /= 32.0;
        
        double gs = 1e-12;
        for (int i = 0; i < 32; i++) {
            double diff = ((i < size - b*32) ? block_x[i] : 0) - gm;
            gs += diff * diff;
        }
        gs = std::sqrt(gs / 32.0) + 1e-12;
        
        cv.gm = static_cast<float>(gm);
        cv.gs = static_cast<float>(gs);
        
        // Local std (ls) calculation for 8D sub-chunks (V19 uses simplified single ls for now)
        cv.ls = 1.0f; // Placeholder for future 8D-local refinement
        
        double res[32];
        for (int i = 0; i < 32; i++) {
            double val = (b * 32 + i < size) ? static_cast<double>(block_x[i]) : 0.0;
            res[i] = (val - gm) / gs;
        }
        
        cv.qs.clear(); 
        cv.scales.clear();
        cv.pulse_masks.resize(config_.max_stages, false); 

        for (int s = 0; s < config_.max_stages; s++) {
            double rms = 0;
            for (int i = 0; i < 32; i++) rms += res[i] * res[i];
            rms = std::sqrt(rms / 32.0) + 1e-18;
            
            // V19 Pulsed Logic: Stages 2+ are sparse
            if (s >= 2 && rms < 0.001) { // Lowered threshold for 50dB stability
                break;
            }
            
            cv.pulse_masks[s] = true;
            double scale = 100.0 / rms;
            cv.scales.push_back(static_cast<float>(scale));
            
            for (int ch = 0; ch < 4; ch++) {
                float scaled_8d[8], q[8];
                for (int i = 0; i < 8; i++) scaled_8d[i] = static_cast<float>(res[ch*8 + i] * scale);
                
                quantize_e8(scaled_8d, q);
                
                std::vector<int16_t> q_int(8);
                for (int i = 0; i < 8; i++) {
                    q_int[i] = static_cast<int16_t>(q[i]);
                    res[ch*8 + i] -= static_cast<double>(q[i]) / scale;
                }
                cv.qs.push_back(q_int);
            }
        }
    }
    return output;
}

void LatticeQuantizer::decode(const std::vector<CompressedVector>& in, float* out, int size) {
    for (size_t b = 0; b < in.size(); b++) {
        const auto& cv = in[b];
        double res[32] = {0};
        
        int active_s = 0;
        for (int s = 0; s < config_.max_stages; s++) {
            if (!cv.pulse_masks[s]) continue;
            double scale = static_cast<double>(cv.scales[active_s]);
            
            for (int ch = 0; ch < 4; ch++) {
                const auto& q_chunk = cv.qs[active_s * 4 + ch];
                for (int i = 0; i < 8; i++) {
                    res[ch * 8 + i] += static_cast<double>(q_chunk[i]) / scale;
                }
            }
            active_s++;
        }
        
        for (int i = 0; i < 32; i++) {
            if (b * 32 + i < static_cast<size_t>(size)) {
                out[b * 32 + i] = static_cast<float>(res[i] * static_cast<double>(cv.gs) + static_cast<double>(cv.gm));
            }
        }
    }
}

} // namespace TerraLens
