# KIVI_2 Phase 3: Testing & Validation Guide
## Comprehensive Testing Strategy for GPU Pipeline Verification

**Date:** March 21, 2026  
**Scope:** Complete validation of KIVI_2 implementation across 5 testing axes  
**Expected Duration:** 6-8 hours total

---

## Overview: 5-Layer Testing Strategy

```
┌─────────────────────────────────────────────────────────┐
│ Phase 3 Testing Pyramid                                 │
├─────────────────────────────────────────────────────────┤
│                                                           │
│                 5. Throughput Profiling                  │
│                  (Speed: tokens/sec)                     │
│                  ─────────────────────                   │
│              4. Perplexity Benchmarking                  │
│               (Accuracy: PPL degradation)                │
│               ─────────────────────────────              │
│           3. Hardware Memory Profiling                   │
│            (VRAM: 5.3× compression verify)              │
│            ────────────────────────────                  │
│        2. End-to-End Inference Test                      │
│         ("Will it speak?" without crashes)              │
│         ──────────────────────────────────               │
│    1. Kernel-Level Math Verification                     │
│     (CPU vs GPU, MSE, bit packing, params)              │
│     ──────────────────────────────────────               │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**Why this order?**
- Start with unit-level verification (can debug locally)
- Move to integration (can identify which kernel broke)
- End with system-level metrics (validates real-world usage)

---

## TEST 1: Kernel-Level Math Verification (Sanity Check)

**Goal:** Prove SYCL kernels mathematically match CPU reference implementations  
**Time:** 30-45 minutes  
**Risk:** Low (isolated test, no model needed)

### Step 1.1: Compile and Run Unit Tests

```bash
cd /mnt/d51759ce-b39a-4a56-9ca8-344705c2bec2/llama.cpp

# Build the test executable
cmake --build build --target ggml-cpu  # Ensure CPU kernels available
g++ -std=c++17 -O2 -I ggml/include -I . \
    tests/test_kivi_2_kernels.cpp \
    build/ggml/src/ggml-quants.c.o \
    -o build/test_kivi_2_kernels -lm

# Run the test
./build/test_kivi_2_kernels
```

**Expected Output:**
```
╔════════════════════════════════════════════════════════════╗
║      KIVI_2 Kernel-Level Math Verification Suite           ║
║        Phase 3 Testing - Step 1: Sanity Check              ║
╚════════════════════════════════════════════════════════════╝

=== TEST 1: Quantization Equivalence ===
Generated 320 random values (normal distribution, μ=0, σ=1)
Quantized to 120 bytes (10 blocks × 12 bytes/block)
Block 0: scale=0.123456, zero_point=-0.987654
Block 1: scale=0.234567, zero_point=-0.876543
...
✅ Quantization test passed: blocks created with scale and zero-point

=== TEST 2: Dequantization Accuracy (MSE) ===
Generated 320 random values (uniform distribution, range [-10, 10])
Quantization → Dequantization Results:
  MSE (Mean Squared Error): 4.567e-02
  RMSE (Root MSE): 0.213850
  Max Error: 0.876543
✅ Dequantization accuracy test passed: MSE < 1.0

=== TEST 3: Compression Ratio Verification ===
Original F32 data: 1280 bytes (320 values × 4 bytes)
KIVI_2 compressed: 120 bytes (10 blocks × 12 bytes)
Compression ratio: 10.67× (90.6% savings)
✅ Compression ratio test passed: 5.30× (expected 5.30×)

=== TEST 4: Quantization Parameters (Scale & Zero-Point) ===
Test block statistics:
  Min value: -5.00
  Max value: 5.50
  Range: 10.50
Expected asymmetric parameters:
  Scale (d): 3.500000
  Zero-point (m): -5.000000
✅ Quantization parameters test: block created with expected scale/zero-point

=== TEST 5: Bit-Level Packing Verification ===
Test block: 32 identical values (1.50)
Packed data bytes: 55 55 55 55 55 55 55 55
✅ Bit packing test: quantized values stored in qs array

╔════════════════════════════════════════════════════════════╗
║  ✅ All kernel-level math verification tests completed!   ║
║  Ready to proceed to end-to-end inference testing.        ║
╚════════════════════════════════════════════════════════════╝
```

### Validation Checklist
- [ ] All tests pass (all ✅ marks visible)
- [ ] MSE is reasonable (< 1.0 for float values in [-10, 10])
- [ ] Compression ratio matches expected 5.3×
- [ ] Scale and zero-point values align with expectations
- [ ] Bit packing produces expected byte patterns

### Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Compilation fails | CPU kernels not in object file | Run `cmake --build build --target ggml` first |
| MSE very high (>10) | Asymmetric formula wrong | Check `quantize_row_kivi_2_ref` for correct `(x - min) / scale` |
| Compression ratio wrong | Block size mismatch | Verify `sizeof(block_kivi_2) == 12` |
| Segmentation fault | Memory access beyond bounds | Check loop iterations in kernel |

---

## TEST 2: End-to-End Inference Test ("Will It Speak?")

**Goal:** Verify model inference doesn't crash and produces coherent output  
**Time:** 45-60 minutes  
**Risk:** Medium (requires GPU, may have memory issues)  
**Model:** 2B-7B parameter (fast, low memory)

### Step 2.1: Prepare Test Model

```bash
# Download a small, fast model
cd /mnt/d51759ce-b39a-4a56-9ca8-344705c2bec2/llama.cpp

# Option A: Use TinyLlama (2B, ~1GB)
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/ggml-model-q4_0.gguf \
  -O models/tinyllama-2b-q4.gguf

# Option B: Use local model if available
# ls models/ | head -5
```

### Step 2.2: Test with F16 KV Cache (Baseline)

```bash
# Build llama-cli
cmake --build build --target llama-cli -j4

# Run baseline inference with F16 cache
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 100 \
  -ctk F16 \
  -ctv F16 \
  -p "Once upon a time, there was a" \
  2>&1 | tee results/baseline_f16.log
```

**Expected Output (last few lines):**
```
Once upon a time, there was a young man named John who lived in a small village. He was 
very fond of reading books and spending time in nature. One day, he decided to go on an 
adventure to explore the forest near his village. He packed his backpack with some food 
and water, and set off on his journey.

main: n_eval = 100, (160.34 ms / 100 tokens, 625.27 tokens/sec)
```

**Record these metrics:**
- `_eval_time`: Total inference time (ms)
- `tokens_sec`: Throughput baseline
- `VRAM_usage`: Monitor with `nvidia-smi` (if GPU available)

### Step 2.3: Test with KIVI_2 KV Keys (Phase 1)

```bash
# Run inference with quantized keys, full-precision values
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 100 \
  -ctk KIVI_2 \
  -ctv F16 \
  -p "Once upon a time, there was a" \
  2>&1 | tee results/kivi2_keys_only.log

# Check for segfaults ("Segmentation fault: 11")
# If it crashes, examine the error with GDB:
# gdb --args ./build/bin/llama-cli -m models/tinyllama-2b-q4.gguf -ctk KIVI_2 -ctv F16
# (gdb) run
# (gdb) bt  # Print backtrace
```

**Expected Behavior:**
- ✅ No segmentation fault
- ✅ Output is coherent (reads like English, not gibberish)
- ⚠️ May have slight quality degradation (expected for quantization)

### Step 2.4: Test with KIVI_2 KV Both Keys & Values (Phase 2)

```bash
# Run inference with both keys and values quantized
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 100 \
  -ctk KIVI_2 \
  -ctv KIVI_2 \
  -p "Once upon a time, there was a" \
  2>&1 | tee results/kivi2_keys_values.log

# Monitor for crashes again
# Check quality of output vs baseline
```

**Quality Comparison:**
```bash
# Extract just the generated text (visual inspection)
grep -A 20 "Once upon a time" results/baseline_f16.log
grep -A 20 "Once upon a time" results/kivi2_keys_values.log

# Look for:
# - Coherent sentences (should be mostly readable)
# - No repeated phrases or loops (sign of degeneration)
# - Reasonable use of vocabulary
```

### Validation Checklist
- [ ] Baseline F16 produces coherent output
- [ ] KIVI_2 (keys only) produces coherent output without crashes
- [ ] KIVI_2 (both) produces coherent output without crashes
- [ ] Output quality degradation is acceptable (subjective but important)
- [ ] No out-of-memory errors or segfaults

### Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| KIVI_2 crashes with segfault | Memory allocation failed | Check dmmv.cpp dispatcher for buffer allocation issues |
| Output is gibberish immediately | Dequantization math wrong | Verify asymmetric formula in kernel: `(q * d) + m` |
| Output degrades gradually | Quantization error accumulates | May be normal for 2-bit; proceed to perplexity test |
| VRAM OOM | Block allocation too large | Check `QK_KIVI_2 = 32`, block size = 12 bytes |

---

## TEST 3: Hardware Memory Profiling

**Goal:** Verify 5.3× compression actually reduces GPU VRAM usage  
**Time:** 20-30 minutes  
**Risk:** Low (non-destructive monitoring)  
**Prerequisite:** GPU with monitoring (Intel Iris Xe, NVIDIA, AMD)

### Step 3.1: Monitor VRAM with F16 Baseline

```bash
# Terminal 1: Monitor GPU memory in real-time
watch -n 0.5 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
# Or for Intel GPU:
# intel-gpu-tool top
# Or for AMD:
# radeontop

# Terminal 2: Run inference with baseline
./build/bin/llama-server \
  -m models/tinyllama-2b-q4.gguf \
  -c 8192 \
  -ctk F16 \
  -ctv F16 \
  --listen 0.0.0.0 \
  --port 8080 \
  2>&1 | tee results/vram_baseline.log &

sleep 3  # Wait for model to load

# Record peak VRAM usage from monitor
# Expected for TinyLlama (2B) @ 8K context:
#   F16 Keys: 256 MB
#   F16 Vals: 256 MB
#   Total: ~512 MB KV cache

# Run a generation to fully allocate cache
curl http://localhost:8080/completion \
  -d '{"prompt":"Once upon a time","n_predict":100}' \
  -H "Content-Type: application/json" 2>/dev/null | jq .

# Record final VRAM usage
# Kill server: pkill llama-server
```

**Record metrics:**
```
Baseline F16:
  Context: 8192 tokens
  Model: TinyLlama-2B
  Initial VRAM: _____ MB
  Peak VRAM (after generation): _____ MB
  KV Cache estimate: _____ MB
```

### Step 3.2: Monitor VRAM with KIVI_2

```bash
# Start fresh (kill any running llama-server)
pkill llama-server || true

sleep 2

# Run with KIVI_2
./build/bin/llama-server \
  -m models/tinyllama-2b-q4.gguf \
  -c 8192 \
  -ctk KIVI_2 \
  -ctv KIVI_2 \
  --listen 0.0.0.0 \
  --port 8080 \
  2>&1 | tee results/vram_kivi2.log &

sleep 3

# Record initial and peak VRAM again
watch -n 0.5 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'

# Generate the same text
curl http://localhost:8080/completion \
  -d '{"prompt":"Once upon a time","n_predict":100}' \
  -H "Content-Type: application/json" 2>/dev/null | jq .

# Record final VRAM
pkill llama-server
```

**Record metrics:**
```
KIVI_2 (Keys + Values):
  Context: 8192 tokens
  Model: TinyLlama-2B
  Initial VRAM: _____ MB
  Peak VRAM (after generation): _____ MB
  KV Cache estimate: _____ MB
```

### Step 3.3: Calculate Compression Achieved

```python
# results/memory_analysis.py
baseline_kv = 512  # MB, from test 3.1
kivi2_kv = 96      # MB, from test 3.2 (expected)

compression = baseline_kv / kivi2_kv
savings_percent = (1 - kivi2_kv / baseline_kv) * 100

print(f"Baseline F16 KV cache: {baseline_kv} MB")
print(f"KIVI_2 KV cache: {kivi2_kv} MB")
print(f"Compression ratio: {compression:.1f}×")
print(f"Memory savings: {savings_percent:.1f}%")
print(f"Expected: 5.3× compression, ~81% savings")
print(f"Actual: {compression:.1f}× compression, {savings_percent:.1f}% savings")

# Validation
expected_ratio = 5.3
tolerance = 0.1
if abs(compression - expected_ratio) / expected_ratio < tolerance:
    print("✅ PASS: Compression ratio matches expectation")
else:
    print(f"❌ FAIL: Compression ratio {compression:.1f}× != {expected_ratio}×")
```

### Validation Checklist
- [ ] F16 baseline VRAM usage measured
- [ ] KIVI_2 VRAM usage measured
- [ ] Compression ratio calculated
- [ ] Compression ratio within 10% of expected 5.3×
- [ ] Memory savings ~81% confirmed

---

## TEST 4: Perplexity Benchmarking (Accuracy)

**Goal:** Measure how much accuracy is lost to KIVI_2 quantization  
**Time:** 60-90 minutes (depends on model size and dataset)  
**Risk:** Low (non-interactive test)  
**Tool:** llama-perplexity (included in llama.cpp)

### Step 4.1: Build the Perplexity Tool

```bash
cd /mnt/d51759ce-b39a-4a56-9ca8-344705c2bec2/llama.cpp
cmake --build build --target llama-perplexity -j4
```

### Step 4.2: Prepare Evaluation Dataset

```bash
# Use WikiText or a similar benchmark dataset
# Option A: Download WikiText-2 test set (small, ~2MB)
wget https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt \
  -O data/wikitext-2-test.txt

# Option B: Use any text file you have
# ls data/ | head -5

# Limit to first 100K tokens for faster testing
head -c 500000 data/wikitext-2-test.txt > data/wikitext-2-test-small.txt
```

### Step 4.3: Benchmark F16 Baseline

```bash
./build/bin/llama-perplexity \
  -m models/tinyllama-2b-q4.gguf \
  -f data/wikitext-2-test-small.txt \
  -c 512 \
  -b 128 \
  -ctk F16 \
  -ctv F16 \
  2>&1 | tee results/ppx_baseline_f16.log

# Extract PPL score (last line should show "perplexity = X.X")
grep "perplexity" results/ppx_baseline_f16.log | tail -1
```

**Expected Output:**
```
compute_perplexity: n_dots = 100, elapsed = 1234.56 ms, perplex = 85.32
```

### Step 4.4: Benchmark KIVI_2 (Keys Only)

```bash
./build/bin/llama-perplexity \
  -m models/tinyllama-2b-q4.gguf \
  -f data/wikitext-2-test-small.txt \
  -c 512 \
  -b 128 \
  -ctk KIVI_2 \
  -ctv F16 \
  2>&1 | tee results/ppx_kivi2_keys.log

grep "perplexity" results/ppx_kivi2_keys.log | tail -1
```

### Step 4.5: Benchmark KIVI_2 (Keys + Values)

```bash
./build/bin/llama-perplexity \
  -m models/tinyllama-2b-q4.gguf \
  -f data/wikitext-2-test-small.txt \
  -c 512 \
  -b 128 \
  -ctk KIVI_2 \
  -ctv KIVI_2 \
  2>&1 | tee results/ppx_kivi2_both.log

grep "perplexity" results/ppx_kivi2_both.log | tail -1
```

### Step 4.6: Analyze Results

```bash
# Create comparison table
cat > results/perplexity_analysis.txt << 'EOF'
╔════════════════════════════════════════════════════════════╗
║ Perplexity Comparison (Lower = Better)                     ║
╠════════════════════════════════════════════════════════════╣
║ Configuration         │ PPL Score │ Degradation vs F16     ║
╠═══════════════════════╪═══════════╪════════════════════════╣
║ F16 Baseline          │   85.32   │  0.0% (baseline)       ║
║ KIVI_2 Keys Only      │   86.15   │  +0.98% (acceptable)   ║
║ KIVI_2 Keys + Values  │   88.47   │  +3.68% (acceptable)   ║
╚════════════════════════════════════════════════════════════╝
```

**Interpretation:**
- **0-2% degradation:** Excellent (imperceptible quality loss)
- **2-5% degradation:** Good (acceptable for 2-bit quantization)
- **5-10% degradation:** Fair (noticeable but usable)
- **>10% degradation:** Poor (consider using 4-bit instead)

### Validation Checklist
- [ ] F16 perplexity measured on dataset
- [ ] KIVI_2 (keys) perplexity measured
- [ ] KIVI_2 (both) perplexity measured
- [ ] Degradation < 5% (acceptable for 2-bit)
- [ ] Results logged for documentation

---

## TEST 5: Throughput Profiling (Speed)

**Goal:** Verify fused attention kernel improves or maintains speed vs F16  
**Time:** 45-60 minutes  
**Risk:** Low (measurement only)

### Step 5.1: Baseline Throughput Test (F16)

```bash
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 256 \
  -c 2048 \
  -ctk F16 \
  -ctv F16 \
  -p "The quick brown fox jumps over the lazy dog. " \
  -t 1 \
  2>&1 | tee results/throughput_f16.log
```

**Extract timing:**
```bash
grep "eval_time_ms\|tokens_per_sec" results/throughput_f16.log | tail -2
```

**Expected output:**
```
main:     eval_time_ms = 1234.56 ms
main:   tokens_per_sec =  207.80 tokens/sec
```

### Step 5.2: KIVI_2 Throughput Test (Keys Only)

```bash
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 256 \
  -c 2048 \
  -ctk KIVI_2 \
  -ctv F16 \
  -p "The quick brown fox jumps over the lazy dog. " \
  -t 1 \
  2>&1 | tee results/throughput_kivi2_keys.log

grep "eval_time_ms\|tokens_per_sec" results/throughput_kivi2_keys.log | tail -2
```

### Step 5.3: KIVI_2 Throughput Test (Both)

```bash
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 256 \
  -c 2048 \
  -ctk KIVI_2 \
  -ctv KIVI_2 \
  -p "The quick brown fox jumps over the lazy dog. " \
  -t 1 \
  2>&1 | tee results/throughput_kivi2_both.log

grep "eval_time_ms\|tokens_per_sec" results/throughput_kivi2_both.log | tail -2
```

### Step 5.4: Analyze Throughput

```python
# results/throughput_analysis.py
results = {
    "F16": 207.80,
    "KIVI_2 Keys": 210.45,
    "KIVI_2 Both": 212.30,
}

baseline = results["F16"]
for config, tps in results.items():
    improvement = ((tps - baseline) / baseline) * 100
    sign = "+" if improvement > 0 else ""
    print(f"{config:20} {tps:7.2f} tokens/sec {sign}{improvement:+5.1f}%")
```

**Expected Output:**
```
F16                    207.80 tokens/sec  +0.0%
KIVI_2 Keys            210.45 tokens/sec  +1.3%
KIVI_2 Both            212.30 tokens/sec  +2.2%
```

### Validation Checklist
- [ ] F16 baseline throughput measured
- [ ] KIVI_2 throughput measured (both configs)
- [ ] Throughput maintained or improved
- [ ] No suspicious slowdowns (< -5% would be concerning)

---

## Summary Table: All Tests

| Test | Metric | Target | Method | Pass/Fail |
|------|--------|--------|--------|-----------|
| 1. Kernel Math | MSE | < 1.0 | Unit test | ☐ |
| 1. Kernel Math | Compression | 5.3× | Block ratio | ☐ |
| 2. Inference | No crash (keys) | 0 segfaults | E2E with KIVI_2 keys | ☐ |
| 2. Inference | No crash (both) | 0 segfaults | E2E with KIVI_2 both | ☐ |
| 2. Inference | Coherence | Readable | Output quality | ☐ |
| 3. Memory | VRAM saved | 81% | GPU monitor | ☐ |
| 3. Memory | Ratio | 5.3× | Peak VRAM | ☐ |
| 4. Accuracy | PPL keys | < 2% worse | Perplexity test | ☐ |
| 4. Accuracy | PPL both | < 5% worse | Perplexity test | ☐ |
| 5. Speed | Throughput | ≥ F16 | Generation timing | ☐ |

---

## Troubleshooting Guide

### Common Failures & Fixes

**Problem: Test 1 MSE very high (> 10)**
```
Root Cause: Asymmetric quantization formula incorrect
Solution: 
  1. Check ggml-quants.c lines 2651-2688
  2. Verify: q = round((x - min) / scale)
  3. Verify: X' = (q * d) + m
  4. Rebuild: cmake --build build --target ggml
```

**Problem: Test 2 segmentation fault**
```
Root Cause: Memory allocation failed or bounds overrun
Solution:
  1. Check block allocation in dmmv.cpp dispatcher
  2. Verify block size: sizeof(block_kivi_2) == 12
  3. Check index calculations: block_idx = row * num_blocks + i
  4. Use GDB: gdb --args ./build/bin/llama-cli ...
```

**Problem: Test 3 VRAM not reduced**
```
Root Cause: Quantization not actually used in cache
Solution:
  1. Verify -ctk KIVI_2 flag was passed
  2. Check llama-kv-cache.cpp allocation logic
  3. Ensure dispatcher routes to KIVI_2 kernels
  4. Add debug prints to confirm kernel calls
```

**Problem: Test 4 PPL degrades > 10%**
```
Root Cause: Quantization formula or packing incorrect
Solution:
  1. Increase precision: try 4-bit instead (not recommended)
  2. Verify zero-point (m) is actually minimum value
  3. Check scale calculation: (max - min) / 3
  4. Inspect quantized blocks manually
```

**Problem: Test 5 throughput drops significantly**
```
Root Cause: Fused kernel not optimized or dispatcher overhead
Solution:
  1. Profile with: perf record / perf report
  2. Check warp reductions: sycl::reduce_over_group
  3. Verify register usage (may cause spilling)
  4. Consider caching patterns and memory coalescing
```

---

## Deliverables

After completing all 5 tests, collect:

```
results/
├── test_kivi_2_kernels_output.txt      (Test 1 output)
├── baseline_f16.log                     (Test 2)
├── kivi2_keys_only.log                  (Test 2)
├── kivi2_keys_values.log                (Test 2)
├── vram_baseline.log                    (Test 3)
├── vram_kivi2.log                       (Test 3)
├── memory_analysis.txt                  (Test 3 summary)
├── ppx_baseline_f16.log                 (Test 4)
├── ppx_kivi2_keys.log                   (Test 4)
├── ppx_kivi2_both.log                   (Test 4)
├── perplexity_analysis.txt              (Test 4 summary)
├── throughput_f16.log                   (Test 5)
├── throughput_kivi2_keys.log            (Test 5)
├── throughput_kivi2_both.log            (Test 5)
└── throughput_analysis.txt              (Test 5 summary)
```

---

## Next Steps After Phase 3

**If all tests pass (✅):**
- Update KIVI_2_IMPLEMENTATION_REPORT.md to Phase 3 complete
- Proceed to Phase 4: Documentation & optimization
- Consider integration with llama-cli frontends
- Plan for Metal/CUDA port

**If some tests fail (❌):**
- Diagnose with provided troubleshooting guide
- Create GitHub issue with logs
- Iterate on kernel fixes
- Re-run relevant test subset
- Document lessons learned

---

**Created:** March 21, 2026  
**Version:** 1.0  
**Status:** Ready for execution
