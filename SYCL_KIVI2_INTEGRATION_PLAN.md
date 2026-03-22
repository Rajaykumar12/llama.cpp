# SYCL KIVI_2 GPU Support Integration Plan

## Current Status
- ✅ CPU KIVI_2 fully working (F16→KIVI_2 quantization in set_rows, flash attention dequantization)
- ✅ SYCL `quantize_row_kivi_2_sycl` kernel exists in `convert.cpp` (phase 2.3b)
- ❌ SYCL dispatcher doesn't route F16→KIVI_2 operations to the kernel

## Problem Statement
When using `-ctk kivi_2` with SYCL backend, inference fails with:
```
"pre-allocated tensor (cache_k_l0) in a buffer (SYCL0) that cannot run the operation (SET_ROWS)"
```

**Root Cause**: KV cache quantization happens via `SET_ROWS` operation, but SYCL dispatcher doesn't have a handler for `DST_TYPE=KIVI_2`.

## Architecture Overview

### Data Flow for KV Cache Update:
```
New Tokens (F16)
    ↓
[ggml_set_rows or ggml_cpy operation]
    ↓
[SYCL Dispatcher routes to appropriate kernel]
    ↓
[Quantization kernel: F16 → KIVI_2]
    ↓
[Quantized KV Cache (12-byte blocks)]
```

### SYCL Dispatcher Locations:
1. **CPY Operations**: `ggml/src/ggml-sycl/cpy.cpp::ggml_sycl_cpy()` (line 512)
   - Routes by `(src0->type, src1->type)` pairs
   - Calls `ggml_cpy_f32_q8_0_sycl()`, `ggml_cpy_f32_q4_0_sycl()`, etc.

2. **SET_ROWS Operations**: `ggml/src/ggml-sycl/set_rows.cpp::ggml_sycl_op_set_rows()` (line 250)
   - Routes by destination type
   - Uses template: `set_rows_sycl_q<TIdx, BlockType, QK, CpyBlck>()`

## Implementation Plan

### Phase 1: SET_ROWS Support (Primary Path)
This is the main KV cache update path when pre-allocating cache as KIVI_2.

#### 1.1 Register KIVI_2 in SET_ROWS Dispatcher
**File**: `ggml/src/ggml-sycl/set_rows.cpp`

Add case for KIVI_2 (after Q5_1 cases):
```cpp
else if (dst->type == GGML_TYPE_KIVI_2) {
    // Set rows with KIVI_2 quantization
    if (src1->type == GGML_TYPE_I64) {
        set_rows_sycl_q<int64_t, block_kivi_2, QK_KIVI_2, cpy_blck_f32_kivi_2>(
            ctx, src, src_indices, dst, ith, nth);
    } else if (src1->type == GGML_TYPE_I32) {
        set_rows_sycl_q<int32_t, block_kivi_2, QK_KIVI_2, cpy_blck_f32_kivi_2>(
            ctx, src, src_indices, dst, ith, nth);
    }
}
```

#### 1.2 Create Block Quantization Kernel
**File**: `ggml/src/ggml-sycl/cpy.hpp`

Add inline quantization kernel (after `cpy_blck_f32_q8_0`, etc.):
```cpp
inline void cpy_blck_f32_kivi_2(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    block_kivi_2 * dsti = (block_kivi_2 *) cdsti;
    
    // Find min/max for asymmetric quantization
    float min_val = xi[0], max_val = xi[0];
    for (int j = 0; j < QK_KIVI_2; j++) {
        min_val = sycl::fmin(min_val, xi[j]);
        max_val = sycl::fmax(max_val, xi[j]);
    }
    
    // Asymmetric quantization: scale = (max - min) / 3, zero_point = min
    const float scale = (max_val - min_val) / 3.0f;
    dsti->d = GGML_FP32_TO_FP16(scale);
    dsti->m = GGML_FP32_TO_FP16(min_val);
    
    // Pack 2-bit values
    const float inv_scale = scale > 0 ? 1.0f / scale : 0.0f;
    for (int j = 0; j < QK_KIVI_2; j += 16) {
        uint8_t byte = 0;
        for (int k = 0; k < 16 && j+k < QK_KIVI_2; k++) {
            const uint8_t v = (uint8_t)sycl::round((xi[j+k] - min_val) * inv_scale);
            byte |= ((v & 0x3) << (k % 8 ? 2 : 0));
        }
        dsti->qs[j/16] = byte;
    }
}
```

#### 1.3 Define QK_KIVI_2 Constant
**File**: `ggml/src/ggml-sycl/common.hpp` (or cpy.hpp)

```cpp
#ifndef QK_KIVI_2
#define QK_KIVI_2 32
#endif
```

### Phase 2: CPY Support (Alternative Path - F32→KIVI_2)
For compatibility if using `ggml_cpy` instead of `ggml_set_rows`.

#### 2.1 Add CPY Function
**File**: `ggml/src/ggml-sycl/cpy.cpp`

```cpp
static void ggml_cpy_f32_kivi_2_sycl(const char * cx, char * cdst, const int ne,
                                      const int ne00, const int ne01, const int ne02,
                                      const int nb00, const int nb01, const int nb02, const int nb03,
                                      const int ne10, const int ne11, const int ne12,
                                      const int nb10, const int nb11, const int nb12, const int nb13,
                                      queue_ptr stream) {
    GGML_ASSERT(ne % QK_KIVI_2 == 0);
    const int num_blocks = ne / QK_KIVI_2;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks), sycl::range<3>(1, 1, 1)),
                         [=](sycl::nd_item<3> item_ct1) {
                             cpy_f32_q<cpy_blck_f32_kivi_2, QK_KIVI_2>(
                                 cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}
```

#### 2.2 Register in CPY Dispatcher
**File**: `ggml/src/ggml-sycl/cpy.cpp::ggml_sycl_cpy()` (line 512)

Add routing case (after Q5_1 cases):
```cpp
} else if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_KIVI_2) {
    ggml_cpy_f32_kivi_2_sycl(src0_dd, dst_dd, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                             ne10, ne11, ne12, nb10, nb11, nb12, nb13, stream);
}
```

### Phase 3: Optional F16→KIVI_2 Direct Path
For maximum efficiency (skip F32 intermediate conversion).

#### 3.1 Add F16→KIVI_2 Handler
This would involve:
1. Creating `cpy_blck_f16_kivi_2()` kernel in `cpy.hpp`
2. Creating `ggml_cpy_f16_kivi_2_sycl()` in `cpy.cpp`
3. Registering in dispatcher

**Note**: Can be deferred if F32→KIVI_2 path is efficient enough.

---

## API Compatibility Check

### Existing Kernel Dependencies
```cpp
// Already defined in convert.cpp:
static void quantize_row_kivi_2_sycl(const src_t * x, void * y, int64_t k, sycl::queue & q)

// Dequantization already integrated:
- dequantize_row_kivi_2_sycl() exists and is registered in conversion dispatchers
```

### Required Type Definitions
```cpp
// In ggml-common.h:
#define QK_KIVI_2 32
typedef struct {
    ggml_half d;           // Scale factor
    ggml_half m;           // Zero-point
    uint8_t qs[QK_KIVI_2/16];  // 2-bit quantized values
} block_kivi_2;
```

---

## Testing Strategy

### 1. Verify KIVI_2 in SET_ROWS (Primary)
```bash
./build/bin/llama-cli -m model.gguf -ctk kivi_2 -ctv f16 -ngl 999 -p "Test" -n 10
```
Expected: Inference runs on GPU, same output as CPU version

### 2. Benchmark GPU vs CPU
```bash
# CPU KIVI_2 (baseline)
./build/bin/llama-cli -m model.gguf -ctk kivi_2 -ctv f16 -ngl 0

# SYCL GPU KIVI_2
./build/bin/llama-cli -m model.gguf -ctk kivi_2 -ctv f16 -ngl 999
```

### 3. Memory Validation
- Verify KV cache actual size: ~384MB for 7B model (5.33× smaller than F16)
- Compare with F16 baseline: ~2GB for same model

---

## Known Limitations

1. **SYCL SET_ROWS Kernel Limitation**: Current SYCL doesn't support `SET_ROWS` on quantized buffers
   - **Solution**: Implement `set_rows_sycl_q<>` template with inline quantization (Phase 1)

2. **Flash Attention**: SYCL FAT kernels only support F16/F32 KV
   - **Solution**: Use reference FA implementation with quantized K dequantization (like CPU path)
   - OR: Mark KIVI_2 + SYCL as GPU-BLAS-only (no flash attention optimization)

3. **F16→KIVI_2 Direct Path**: Not yet implemented
   - **Fallback**: Convert F16→F32→KIVI_2 via F32 path
   - **Future**: Implement direct F16→KIVI_2 conversion kernel

---

## Integration Checklist

- [ ] Add QK_KIVI_2 constant definition
- [ ] Add `cpy_blck_f32_kivi_2()` kernel to cpy.hpp
- [ ] Add `ggml_cpy_f32_kivi_2_sycl()` function to cpy.cpp
- [ ] Register CPY case in `ggml_sycl_cpy()` dispatcher
- [ ] Add SET_ROWS case in `ggml_sycl_op_set_rows()` dispatcher (primary path)
- [ ] Rebuild and test with SYCL
- [ ] Validate against CPU KIVI_2 output
- [ ] Benchmark performance improvement
- [ ] Document KIVI_2 GPU limitations if any

---

## Expected Outcomes

### Success Metrics:
1. ✅ SYCL build compiles without errors
2. ✅ Inference runs without "SET_ROWS not supported" errors
3. ✅ GPU memory usage: ~384MB KV cache (5.33× reduction)
4. ✅ Output matches CPU KIVI_2 baseline
5. ✅ Performance: Should be faster than CPU due to GPU parallelization

### Performance Expectations:
- **Inference Speed**: 2-4× faster than CPU (depending on GPU)
- **Memory Bandwidth**: VRAM limited ~400GB/s typical for discrete Intel Arc
- **KV Cache Impact**: ~5-10% speedup from reduced VRAM bandwidth (smaller cache)

