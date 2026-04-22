#include "lattice_v19.h"

LatticeV19::LatticeV19(int dim, float target_bpd, int max_stages) 
    : dim(dim), target_bpd(target_bpd), max_stages(max_stages) {
    nch_32 = (dim + 31) / 32;
    pw_32 = nch_32 * 32;
    nch_8 = pw_32 / 8;
    density_sr = 0.1f; 
}

std::vector<float> LatticeV19::decode_dn(const std::vector<float>& x) {
    std::vector<float> y(8);
    float sum = 0;
    for (int i = 0; i < 8; i++) {
        y[i] = std::round(x[i]);
        sum += y[i];
    }

    if ((int)std::abs(sum) % 2 != 0) {
        int best_idx = 0;
        float max_diff = -1.0f;
        for (int i = 0; i < 8; i++) {
            float diff = std::abs(x[i] - y[i]);
            if (diff > max_diff) {
                max_diff = diff;
                best_idx = i;
            }
        }
        float diff_val = x[best_idx] - y[best_idx];
        y[best_idx] += (diff_val >= 0 ? 1.0f : -1.0f);
    }
    return y;
}

std::vector<float> LatticeV19::fast_e8_quantize(const std::vector<float>& x_block) {
    // x_block is 8D
    std::vector<float> y0 = decode_dn(x_block);
    
    std::vector<float> x_minus_05(8);
    for(int i=0; i<8; i++) x_minus_05[i] = x_block[i] - 0.5f;
    std::vector<float> y1 = decode_dn(x_minus_05);
    for(int i=0; i<8; i++) y1[i] += 0.5f;

    float d0 = 0, d1 = 0;
    for (int i = 0; i < 8; i++) {
        d0 += std::pow(x_block[i] - y0[i], 2);
        d1 += std::pow(x_block[i] - y1[i], 2);
    }

    return (d1 < d0) ? y1 : y0;
}

void LatticeV19::fit(const std::vector<float>& data) {
    // V19 Pulse Calibration - Simplified for C++ runtime
    scales.clear();
    std::vector<float> res = data;
    if (res.size() < (size_t)pw_32) res.resize(pw_32, 0.0f);

    for (int s = 0; s < max_stages; s++) {
        float rms = 0;
        for (float v : res) rms += v * v;
        rms = std::sqrt(rms / res.size()) + 1e-9f;
        float scale = 100.0f / rms;
        scales.push_back(scale);

        // Update residuals for next stage fit
        for (int i = 0; i < nch_8; i++) {
            std::vector<float> block(8);
            for(int j=0; j<8; j++) block[j] = res[i*8 + j] * scale;
            std::vector<float> q = fast_e8_quantize(block);
            for(int j=0; j<8; j++) res[i*8 + j] -= q[j] / scale;
        }
    }
    density_sr = 0.4f; // Fixed pulse for stability in C++ port
}

V19Compressed LatticeV19::encode(const std::vector<float>& data) {
    V19Compressed co;
    std::vector<float> X_p = data;
    X_p.resize(pw_32, 0.0f);

    // 1. Hierarchical Normalization
    float sum = std::accumulate(X_p.begin(), X_p.end(), 0.0f);
    co.gm = sum / pw_32;
    float var = 0;
    for(float v : X_p) var += std::pow(v - co.gm, 2);
    co.gs = std::sqrt(var / pw_32) + 1e-12f;

    std::vector<float> res(pw_32);
    for(int i=0; i<pw_32; i++) res[i] = (X_p[i] - co.gm) / co.gs;

    // Local Normalization (32D)
    co.ls.resize(nch_32);
    for(int i=0; i<nch_32; i++) {
        float l_var = 0;
        for(int j=0; j<32; j++) l_var += std::pow(res[i*32 + j], 2);
        co.ls[i] = std::sqrt(l_var / 32) + 1e-12f;
        for(int j=0; j<32; j++) res[i*32 + j] /= co.ls[i];
    }

    // 2. Multi-Stage Quantization
    for (int s = 0; s < (int)scales.size(); s++) {
        std::vector<float> q_stage(pw_32);
        for (int i = 0; i < nch_8; i++) {
            std::vector<float> block(8);
            for(int j=0; j<8; j++) block[j] = res[i*8 + j] * scales[s];
            std::vector<float> q = fast_e8_quantize(block);
            for(int j=0; j<8; j++) {
                q_stage[i*8 + j] = q[j];
                res[i*8 + j] -= q[j] / scales[s];
            }
        }
        co.qs.push_back(q_stage);
    }

    return co;
}

std::vector<float> LatticeV19::decode(const V19Compressed& co) {
    std::vector<float> res_8(pw_32, 0.0f);
    for (int s = 0; s < (int)co.qs.size(); s++) {
        for (int i = 0; i < pw_32; i++) {
            res_8[i] += co.qs[s][i] / scales[s];
        }
    }

    // Reverse Local Norm
    for (int i = 0; i < nch_32; i++) {
        for (int j = 0; j < 32; j++) {
            res_8[i * 32 + j] *= co.ls[i];
        }
    }

    // Reverse Global Norm
    std::vector<float> out(dim);
    for (int i = 0; i < dim; i++) {
        out[i] = res_8[i] * co.gs + co.gm;
    }

    return out;
}
