# KIVI_2 Phase 3 Testing Checklist

**Date Started:** March 21, 2026  
**Expected Completion:** March 22-23, 2026  
**Total Estimated Time:** 6-8 hours

---

## Quick Navigation

📘 **Full Testing Guide:** [PHASE_3_TESTING_GUIDE.md](PHASE_3_TESTING_GUIDE.md)  
📊 **Implementation Status:** [KIVI_2_IMPLEMENTATION_REPORT.md](KIVI_2_IMPLEMENTATION_REPORT.md)

---

## TEST 1: Kernel-Level Math Verification ⏱️ 30-45 min

**Purpose:** Verify quantization math is correct before inference

```bash
# Build unit tests
cd /mnt/d51759ce-b39a-4a56-9ca8-344705c2bec2/llama.cpp
cmake --build build --target ggml  # Ensure CPU kernels
g++ -std=c++17 -O2 -I ggml/include -I . \
    tests/test_kivi_2_kernels.cpp \
    build/ggml/src/ggml-quants.c.o \
    -o build/test_kivi_2_kernels -lm

# Run test
./build/test_kivi_2_kernels 2>&1 | tee results/test1_output.txt
```

### Validation

| Item | Expected | Actual | Status |
|------|----------|--------|--------|
| MSE score | < 1.0 | ? | ☐ |
| Compression ratio | 5.3× | ? | ☐ |
| Scale parameters | Present | ? | ☐ |
| Bit packing | Correct | ? | ☐ |

**Pass/Fail:** ☐ PASS ☐ FAIL

---

## TEST 2: End-to-End Inference ("Will It Speak?") ⏱️ 45-60 min

**Purpose:** Verify model runs without crashes and produces coherent output

### 2.1 Prepare Test Model
```bash
cd /mnt/d51759ce-b39a-4a56-9ca8-344705c2bec2/llama.cpp
# Download TinyLlama 2B
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/ggml-model-q4_0.gguf \
  -O models/tinyllama-2b-q4.gguf
```

### 2.2 Build llama-cli
```bash
cmake --build build --target llama-cli -j4
```

### 2.3 Run Baseline (F16)
```bash
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 100 \
  -ctk F16 \
  -ctv F16 \
  -p "Once upon a time, there was a" \
  2>&1 | tee results/test2_baseline_f16.log
```

**Check:** ✅ No crashes, ✅ Readable output

### 2.4 Run KIVI_2 (Keys Only)
```bash
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 100 \
  -ctk KIVI_2 \
  -ctv F16 \
  -p "Once upon a time, there was a" \
  2>&1 | tee results/test2_kivi2_keys.log
```

**Check:** ✅ No segfault, ✅ Readable output

### 2.5 Run KIVI_2 (Both Keys & Values)
```bash
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 100 \
  -ctk KIVI_2 \
  -ctv KIVI_2 \
  -p "Once upon a time, there was a" \
  2>&1 | tee results/test2_kivi2_both.log
```

**Check:** ✅ No segfault, ✅ Readable output

### Validation

| Configuration | No Crash | Coherent | Quality | Status |
|---------------|----------|----------|---------|--------|
| F16 Baseline | ✓ | ? | ? | ☐ |
| KIVI_2 Keys | ? | ? | ? | ☐ |
| KIVI_2 Both | ? | ? | ? | ☐ |

**Visual Quality Check:**
```bash
# Compare outputs side-by-side
diff -y results/test2_baseline_f16.log results/test2_kivi2_both.log | head -40
```

**Pass/Fail:** ☐ PASS ☐ FAIL

---

## TEST 3: Hardware Memory Profiling ⏱️ 20-30 min

**Purpose:** Verify VRAM reduction (5.3× compression)

### 3.1 Monitor F16 Baseline
```bash
# Terminal 1: Monitor memory
watch -n 0.5 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'

# Terminal 2: Run server
./build/bin/llama-server \
  -m models/tinyllama-2b-q4.gguf \
  -c 8192 \
  -ctk F16 \
  -ctv F16 \
  --listen 0.0.0.0 \
  --port 8080 &

sleep 3
curl http://localhost:8080/completion \
  -d '{"prompt":"Once upon a time","n_predict":100}' \
  -H "Content-Type: application/json" 2>/dev/null | jq .

# Record peak: _________ MB
pkill llama-server
sleep 2
```

**F16 Result: _________ MB**

### 3.2 Monitor KIVI_2
```bash
# Terminal 1: Monitor memory
watch -n 0.5 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'

# Terminal 2: Run server
./build/bin/llama-server \
  -m models/tinyllama-2b-q4.gguf \
  -c 8192 \
  -ctk KIVI_2 \
  -ctv KIVI_2 \
  --listen 0.0.0.0 \
  --port 8080 &

sleep 3
curl http://localhost:8080/completion \
  -d '{"prompt":"Once upon a time","n_predict":100}' \
  -H "Content-Type: application/json" 2>/dev/null | jq .

# Record peak: _________ MB
pkill llama-server
```

**KIVI_2 Result: _________ MB**

### 3.3 Calculate Compression
```python
# results/memory_calc.py
f16_mb = float(input("F16 peak VRAM (MB): "))
kivi2_mb = float(input("KIVI_2 peak VRAM (MB): "))

ratio = f16_mb / kivi2_mb
savings = (1 - kivi2_mb / f16_mb) * 100

print(f"\nF16:        {f16_mb:.0f} MB")
print(f"KIVI_2:     {kivi2_mb:.0f} MB")
print(f"Ratio:      {ratio:.1f}× (expected 5.3×)")
print(f"Savings:    {savings:.1f}% (expected ~81%)")

if abs(ratio - 5.3) < 0.5:
    print("✅ PASS: Compression ratio acceptable")
else:
    print(f"❌ FAIL: Expected 5.3×, got {ratio:.1f}×")
```

### Validation

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| F16 VRAM | ~512 MB | ? MB | ☐ |
| KIVI_2 VRAM | ~96 MB | ? MB | ☐ |
| Compression | 5.3× | ?× | ☐ |
| Savings | ~81% | ?% | ☐ |

**Pass/Fail:** ☐ PASS ☐ FAIL

---

## TEST 4: Perplexity Benchmarking ⏱️ 60-90 min

**Purpose:** Measure accuracy loss (PPL degradation)

### 4.1 Prepare Dataset
```bash
# Download WikiText-2 test set
wget https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt \
  -O data/wikitext-2-test.txt

# Limit to 500K chars for faster testing
head -c 500000 data/wikitext-2-test.txt > data/wikitext-2-test-small.txt
```

### 4.2 Build Perplexity Tool
```bash
cmake --build build --target llama-perplexity -j4
```

### 4.3 Benchmark F16
```bash
./build/bin/llama-perplexity \
  -m models/tinyllama-2b-q4.gguf \
  -f data/wikitext-2-test-small.txt \
  -c 512 \
  -b 128 \
  -ctk F16 \
  -ctv F16 \
  2>&1 | tee results/test4_ppx_f16.log

grep "perplexity" results/test4_ppx_f16.log | tail -1
# Record PPL: _________
```

**F16 PPL: _________**

### 4.4 Benchmark KIVI_2 (Keys Only)
```bash
./build/bin/llama-perplexity \
  -m models/tinyllama-2b-q4.gguf \
  -f data/wikitext-2-test-small.txt \
  -c 512 \
  -b 128 \
  -ctk KIVI_2 \
  -ctv F16 \
  2>&1 | tee results/test4_ppx_kivi2_keys.log

grep "perplexity" results/test4_ppx_kivi2_keys.log | tail -1
# Record PPL: _________
```

**KIVI_2 Keys PPL: _________**

### 4.5 Benchmark KIVI_2 (Both)
```bash
./build/bin/llama-perplexity \
  -m models/tinyllama-2b-q4.gguf \
  -f data/wikitext-2-test-small.txt \
  -c 512 \
  -b 128 \
  -ctk KIVI_2 \
  -ctv KIVI_2 \
  2>&1 | tee results/test4_ppx_kivi2_both.log

grep "perplexity" results/test4_ppx_kivi2_both.log | tail -1
# Record PPL: _________
```

**KIVI_2 Both PPL: _________**

### Validation

| Configuration | PPL | Degradation | Status | Acceptable? |
|---------------|-----|-------------|--------|-------------|
| F16 Baseline | ? | 0.0% | ☐ | ✓ |
| KIVI_2 Keys | ? | ?% | ☐ | ☐ |
| KIVI_2 Both | ? | ?% | ☐ | ☐ |

**Degradation Interpretation:**
- 0-2%: Excellent
- 2-5%: Good (expected for 2-bit)
- 5-10%: Fair (acceptable)
- >10%: Poor (investigate)

**Pass/Fail:** ☐ PASS ☐ FAIL

---

## TEST 5: Throughput Profiling ⏱️ 45-60 min

**Purpose:** Verify no performance regression

### 5.1 Baseline Throughput
```bash
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 256 \
  -c 2048 \
  -ctk F16 \
  -ctv F16 \
  -p "The quick brown fox jumps over the lazy dog. " \
  -t 1 \
  2>&1 | tee results/test5_tps_f16.log

grep "tokens_per_sec\|eval_time_ms" results/test5_tps_f16.log | tail -2
# Record: _________ tokens/sec
```

**F16 Throughput: _________ tokens/sec**

### 5.2 KIVI_2 Keys Throughput
```bash
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 256 \
  -c 2048 \
  -ctk KIVI_2 \
  -ctv F16 \
  -p "The quick brown fox jumps over the lazy dog. " \
  -t 1 \
  2>&1 | tee results/test5_tps_kivi2_keys.log

grep "tokens_per_sec" results/test5_tps_kivi2_keys.log | tail -1
# Record: _________ tokens/sec
```

**KIVI_2 Keys Throughput: _________ tokens/sec**

### 5.3 KIVI_2 Both Throughput
```bash
./build/bin/llama-cli \
  -m models/tinyllama-2b-q4.gguf \
  -n 256 \
  -c 2048 \
  -ctk KIVI_2 \
  -ctv KIVI_2 \
  -p "The quick brown fox jumps over the lazy dog. " \
  -t 1 \
  2>&1 | tee results/test5_tps_kivi2_both.log

grep "tokens_per_sec" results/test5_tps_kivi2_both.log | tail -1
# Record: _________ tokens/sec
```

**KIVI_2 Both Throughput: _________ tokens/sec**

### Validation

| Configuration | TPS | Change | Status | Acceptable? |
|---------------|-----|--------|--------|-------------|
| F16 Baseline | ? | 0.0% | ☐ | ✓ |
| KIVI_2 Keys | ? | ?% | ☐ | ☐ |
| KIVI_2 Both | ? | ?% | ☐ | ☐ |

**Change Interpretation:**
- Positive: Fused kernel optimization worked!
- -5% to 0%: Good (negligible overhead)
- -5% to -10%: Acceptable (worth the 5.3× memory savings)
- < -10%: Investigate dispatcher or kernel implementation

**Pass/Fail:** ☐ PASS ☐ FAIL

---

## Summary & Next Steps

### Overall Progress

| Phase | Status | Date |
|-------|--------|------|
| Phase 1 | ✅ COMPLETE | Mar 15, 2026 |
| Phase 2.1 | ✅ COMPLETE | Mar 17, 2026 |
| Phase 2.3a | ✅ COMPLETE | Mar 18, 2026 |
| Phase 2.3b | ✅ COMPLETE | Mar 19, 2026 |
| Phase 2.3c | ✅ COMPLETE | Mar 21, 2026 |
| **Phase 3** | 🚀 IN PROGRESS | Started Mar 21 |
| Phase 4 | ⏹️ Pending | After Phase 3 |

### Phase 3 Completion Status

- [ ] TEST 1: Kernel Math ✅/❌
- [ ] TEST 2: Inference ✅/❌
- [ ] TEST 3: Memory ✅/❌
- [ ] TEST 4: Perplexity ✅/❌
- [ ] TEST 5: Throughput ✅/❌

**Overall:** ☐ ALL PASS ☐ 1+ FAIL

### If All Tests Pass

1. Update KIVI_2_IMPLEMENTATION_REPORT.md to "Phase 3 COMPLETE"
2. Proceed to Phase 4 (Documentation & optimization)
3. Consider integration with llama-cli frontends
4. Plan Metal/CUDA port

### If Tests Fail

1. Consult [PHASE_3_TESTING_GUIDE.md](PHASE_3_TESTING_GUIDE.md) Troubleshooting section
2. Create GitHub issue with logs
3. Iteratively fix and re-run failed tests
4. Document lessons learned

---

**Last Updated:** March 21, 2026  
**Next Review:** After TEST 1 completion
