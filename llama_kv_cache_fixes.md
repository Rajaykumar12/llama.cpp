# llama.cpp KV Cache Tensor Bounds Fix

## Problem
GGML_ASSERT failure when running llama-cli with test_model.gguf:
```
/home/Fedora2/llama.cpp/ggml/src/ggml.c:1733: GGML_ASSERT(view_src == NULL || 
data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)) failed
```

Stack trace showed failure in `llama_kv_cache::get_k()` when creating tensor view.

## Root Causes Identified

### Bug 1: seq_to_stream Vector Resize (CRITICAL)
**Location:** [src/llama-kv-cache.cpp](src/llama-kv-cache.cpp) lines 88-90

**Issue:** When n_stream > 1, code was:
```cpp
seq_to_stream.resize(LLAMA_MAX_SEQ, 0);
if (n_stream > 1) {
    seq_to_stream.resize(n_stream, 0);  // ← SHRINKS vector from 4096 to n_stream!
```

This shrunk the vector from LLAMA_MAX_SEQ (4096) to n_stream (~2), causing out-of-bounds access when looking up sequence IDs during cache slot finding.

**Fix:** Removed the erroneous resize call, keeping full LLAMA_MAX_SEQ allocation:
```cpp
seq_to_stream.resize(LLAMA_MAX_SEQ, 0);
if (n_stream > 1) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        seq_to_stream[s] = s;
    }
}
```

### Bug 2: n_recent Tensor Bounds Overflow
**Location:** [src/llama-kv-cache.cpp](src/llama-kv-cache.cpp) lines 1078-1082

**Issue:** K tensor allocated with:
- k_size = std::min(kv_size, residual_window + 32) = min(context_size, 96)
- k->ne[1] = k_size (e.g., 96)

But get_k() computed:
- n_recent = n_kv - m_quant, where n_kv could be 256 (full context)
- Created view with ne2 = n_recent = 256 >> allocated k->ne[1] = 96

**Fix:** Clamp n_recent to actual allocated size:
```cpp
const uint32_t n_recent_unbounded = n_kv > m_quant ? n_kv - m_quant : 0;
const uint32_t n_recent = std::min(n_recent_unbounded, (uint32_t)k->ne[1]);
```

## Related Code Context
- KIVI_2: Dual-precision KV cache quantization (2-bit asymmetric)
- Residual window: 64 (hardcoded constant for recent unquantized cache)
- k_quant_count tracks how many tokens have been quantized
- k_recent_count tracks unquantized recent token count
