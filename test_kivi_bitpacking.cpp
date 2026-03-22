#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>

// Forward declarations - linked from ggml-quants.c
extern "C" {
    void quantize_row_kivi_2_ref(const float *x, void *y, int64_t k);
    void dequantize_row_kivi_2(const void *x, float *y, int64_t k);
}

// Block structure must match ggml-common.h
typedef struct {
    uint16_t d;     // ggml_half scale
    uint16_t m;     // ggml_half zero-point
    uint8_t qs[8];  // 32 × 2-bit values
} block_kivi_2;

// Simple half to float conversion (for display)
float half_to_float(uint16_t h) {
    // Simplified - just for inspection
    int s = (h >> 15) & 1;
    int e = (h >> 10) & 0x1f;
    int m = h & 0x3ff;
    
    if (e == 0 && m == 0) return s ? -0.0f : 0.0f;
    
    float f = 1.0f + m / 1024.0f;
    return (s ? -1.0f : 1.0f) * f * (1 << (e - 15));
}

int main() {
    printf("=== KIVI_2 Bit-Packing Sanity Check ===\n\n");
    
    // Test 1: Small block with known values
    printf("[Test 1] Small block (8 values) with known input\n");
    std::vector<float> test_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    
    // Quantize
    block_kivi_2 block;
    quantize_row_kivi_2_ref(test_data.data(), &block, 8);
    
    printf("Input:  ");
    for (float v : test_data) printf("%.1f ", v);
    printf("\n");
    
    printf("Block quantized:\n");
    printf("  scale (d): %.6f\n", half_to_float(block.d));
    printf("  zero-pt (m): %.6f\n", half_to_float(block.m));
    printf("  qs bytes: ");
    for (int i = 0; i < 1; i++) printf("%02x ", block.qs[i]);
    printf("\n\n");
    
    // Dequantize
    std::vector<float> reconstructed(8);
    dequantize_row_kivi_2(&block, reconstructed.data(), 8);
    
    printf("Output: ");
    for (float v : reconstructed) printf("%.1f ", v);
    printf("\n");
    
    // Calculate error
    double mse = 0.0;
    double max_error = 0.0;
    for (size_t i = 0; i < test_data.size(); i++) {
        float err = test_data[i] - reconstructed[i];
        mse += err * err;
        max_error = std::max(max_error, (double)std::abs(err));
    }
    mse /= test_data.size();
    
    printf("MSE: %.6f\n", mse);
    printf("Max Error: %.6f\n\n", max_error);
    
    // Test 2: Full 32-value block with random data
    printf("[Test 2] Full 32-value block with random data\n");
    std::vector<float> random_data(32);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    
    for (auto& v : random_data) {
        v = dis(gen);
    }
    
    printf("Input range: [%.2f, %.2f]\n", 
        *std::min_element(random_data.begin(), random_data.end()),
        *std::max_element(random_data.begin(), random_data.end()));
    
    // Quantize full block
    std::vector<uint8_t> quantized(12);
    quantize_row_kivi_2_ref(random_data.data(), quantized.data(), 32);
    
    block_kivi_2* block_ptr = (block_kivi_2*)quantized.data();
    printf("Scale: %.6f, Zero-point: %.6f\n", 
        half_to_float(block_ptr->d),
        half_to_float(block_ptr->m));
    
    // Dequantize full block
    std::vector<float> reconstructed_full(32);
    dequantize_row_kivi_2(quantized.data(), reconstructed_full.data(), 32);
    
    // Calculate MSE
    double mse_full = 0.0;
    double max_error_full = 0.0;
    for (int i = 0; i < 32; i++) {
        float err = random_data[i] - reconstructed_full[i];
        mse_full += err * err;
        max_error_full = std::max(max_error_full, (double)std::abs(err));
    }
    mse_full /= 32;
    
    printf("MSE: %.6f\n", mse_full);
    printf("Max Error: %.6f\n", max_error_full);
    
    // Diagnosis
    printf("\n=== DIAGNOSIS ===\n");
    if (mse_full < 0.1) {
        printf("✅ Bit-packing is CORRECT\n");
        printf("   Quantize/dequantize math looks good.\n");
        printf("   → Problem is likely SUSPECT 1: Missing Residual Window\n");
        printf("   → You need to keep last 128 tokens in FP16, not KIVI_2\n");
    } else if (mse_full < 1.0) {
        printf("⚠️  MSE is moderate - check if this is expected\n");
        printf("   Verify against the KIVI paper (they show ~1.0 MSE as acceptable)\n");
    } else {
        printf("❌ Bit-packing might be BROKEN\n");
        printf("   MSE is too high. Check:\n");
        printf("   1. Endianness of bit shifts (q0 << 0 vs q0 << 6)\n");
        printf("   2. Asymmetric formula: (q * scale) + zero_point\n");
        printf("   3. Min/max calculation during quantization\n");
    }
    
    return 0;
}
