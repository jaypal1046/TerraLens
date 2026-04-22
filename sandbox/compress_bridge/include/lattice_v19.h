#ifndef LATTICE_V19_H
#define LATTICE_V19_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

struct V19Compressed {
    std::vector<std::vector<float>> qs; // Quantized stages
    float gm; // Global mean
    float gs; // Global std
    std::vector<float> ls; // Local stds (per 32D block)
    std::vector<bool> mask_r; // Pulsed refinement mask
};

class LatticeV19 {
public:
    LatticeV19(int dim, float target_bpd = 2.2f, int max_stages = 6);
    
    void fit(const std::vector<float>& data);
    V19Compressed encode(const std::vector<float>& data);
    std::vector<float> decode(const V19Compressed& co);

private:
    int dim;
    int pw_32;
    int nch_8;
    int nch_32;
    float target_bpd;
    int max_stages;
    std::vector<float> scales;
    float density_sr;

    std::vector<float> fast_e8_quantize(const std::vector<float>& x);
    std::vector<float> decode_dn(const std::vector<float>& x);
};

#endif
