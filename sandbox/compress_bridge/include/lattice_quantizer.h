#ifndef LATTICE_QUANTIZER_H
#define LATTICE_QUANTIZER_H

#include <vector>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace TerraLens {

// Higman-Sims V16 Lattice Quantizer
// Uses the E8 Gosset Lattice for near-lossless vector compression.
class LatticeQuantizer {
public:
    struct Config {
        int dim;
        int max_stages = 4;
        float scale_gain = 100.0f;
    };

    LatticeQuantizer(const Config& config);

    // Compressed representation of a vector
    struct CompressedVector {
        std::vector<std::vector<int16_t>> qs; // Lattice indices per stage
        std::vector<float> scales;           // Dynamic scales per stage
        std::vector<bool> pulse_masks;       // Sparse masks for refinement stages
        float gm, gs;                        // Global mean/std (32D block)
        float ls;                            // Local std (within 32D block)
    };

    // Encode a single 8D chunk
    void encode_8d(const float* x, CompressedVector& out);
    
    // Decode a single 8D chunk
    void decode_8d(const CompressedVector& in, float* out);

    // Full vector operations
    std::vector<CompressedVector> encode(const float* x, int size);
    void decode(const std::vector<CompressedVector>& in, float* out, int size);

private:
    Config config_;
    std::vector<float> stage_scales_;

    // Fast D_n Lattice Decoder (Core of E8)
    void decode_dn(const float* x, float* y);
    
    // Fast E8 Lattice Quantizer
    void quantize_e8(const float* x, float* y);
};

} // namespace TerraLens

#endif // LATTICE_QUANTIZER_H
