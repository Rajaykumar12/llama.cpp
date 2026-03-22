# KIVI_2 Phase 3 Testing Resources Index

**Navigation Hub for Phase 3 Testing & Validation**  
**Updated:** March 21, 2026  
**Project Status:** GPU Pipeline Complete → Testing in Progress

---

## 📊 Quick Status Overview

```
Phase 1: Type System Integration       ✅ COMPLETE (Mar 15)
Phase 2.1: CPU Kernels                ✅ COMPLETE (Mar 17)
Phase 2.3a: GPU Dequantization        ✅ COMPLETE (Mar 18)
Phase 2.3b: GPU Quantization          ✅ COMPLETE (Mar 19)
Phase 2.3c: Fused Attention Kernel    ✅ COMPLETE (Mar 21)
────────────────────────────────────────────────────────
Phase 3: Testing & Validation         🚀 IN PROGRESS
Phase 4: Documentation & Optimization  ⏹️  PENDING
```

---

## 🎯 Phase 3 Documents (Start Here!)

### For Immediate Execution

**[PHASE_3_CHECKLIST.md](PHASE_3_CHECKLIST.md)**
- **What:** Quick-reference testing checklist with fill-in-the-blank fields
- **When:** Use while running each test
- **How:** Follow numbered procedures, record results in tables
- **Time:** 6-8 hours total for all 5 tests
- **Best for:** Getting started quickly, tracking progress

### For Detailed Understanding

**[PHASE_3_TESTING_GUIDE.md](PHASE_3_TESTING_GUIDE.md)**
- **What:** Comprehensive 200+ line testing guide with expected outputs
- **Sections:**
  - TEST 1: Kernel Math Verification (MSE, compression ratio)
  - TEST 2: End-to-End Inference (crashes, coherence)
  - TEST 3: Memory Profiling (VRAM reduction)
  - TEST 4: Perplexity Benchmarking (accuracy loss)
  - TEST 5: Throughput Profiling (speed measurement)
- **Includes:** Expected outputs, validation checklists, troubleshooting guide
- **Best for:** Understanding what each test does and how to interpret results

### For Context & Transition

**[PHASE_2_TO_3_TRANSITION.md](PHASE_2_TO_3_TRANSITION.md)**
- **What:** Summary of Phase 2.3c completion and why Phase 3 is important
- **Covers:**
  - What was built in Phase 2.3c (fused attention kernel)
  - Why testing is critical
  - The 5-layer testing pyramid
  - Execution plan
  - Success criteria
- **Best for:** Understanding the bigger picture before diving into tests

### For Status Tracking

**[KIVI_2_IMPLEMENTATION_REPORT.md](KIVI_2_IMPLEMENTATION_REPORT.md)**
- **What:** Master implementation report tracking all phases
- **Updated to:** Phase 3 status with link to testing guide
- **Version:** 7.0 (updated March 21, 2026)
- **Best for:** Understanding complete implementation history

---

## 🧪 The 5 Tests Explained

### Layer 1: Kernel-Level Math Verification (TEST 1)
**Goal:** ✅ Verify quantization math is correct  
**Time:** 30-45 minutes  
**Risk:** Low (unit test, local debug)

**What:** Unit test comparing CPU reference vs GPU quantization math
- Quantization accuracy (MSE score)
- Compression ratio (5.3×)
- Scale and zero-point values
- Bit packing correctness

**Success:** MSE < 1.0, compression = 5.3×

**Key file:** `ggml-quants.c` lines 2651-2688 (CPU reference implementations)

---

### Layer 2: End-to-End Inference (TEST 2)
**Goal:** ✅ Verify model runs without crashes  
**Time:** 45-60 minutes  
**Risk:** Medium (requires GPU)

**What:** Run actual inference with TinyLlama using KIVI_2 cache
- Baseline test with F16 (reference)
- Test with KIVI_2 quantized keys
- Test with KIVI_2 quantized keys & values
- Check for segmentation faults
- Verify output is coherent English

**Success:** No crashes, readable output, acceptable quality degradation

**Key files:** 
- `KIVI_2_IMPLEMENTATION_REPORT.md` Phase 2.3c section
- Execution: `./build/bin/llama-cli -ctk KIVI_2 -ctv KIVI_2`

---

### Layer 3: Hardware Memory Profiling (TEST 3)
**Goal:** ✅ Verify 5.3× VRAM reduction on GPU  
**Time:** 20-30 minutes  
**Risk:** Low (monitoring only)

**What:** Monitor GPU VRAM during inference with baseline vs KIVI_2
- Record peak VRAM with F16 KV cache
- Record peak VRAM with KIVI_2 KV cache
- Calculate actual compression ratio
- Verify matches expected 5.3×

**Success:** Compression within 10% of 5.3× (4.8× - 5.8× acceptable)

**Tools:** `nvidia-smi` (NVIDIA), `intel-gpu-tool` (Intel), `radeontop` (AMD)

---

### Layer 4: Perplexity Benchmarking (TEST 4)
**Goal:** ✅ Quantify accuracy loss from quantization  
**Time:** 60-90 minutes  
**Risk:** Low (non-interactive)

**What:** Evaluate model accuracy on WikiText-2 test set
- Baseline perplexity with F16
- Perplexity with KIVI_2 keys only
- Perplexity with KIVI_2 keys & values
- Calculate degradation percentage

**Success Criteria:**
- Keys only: < 2% degradation (excellent)
- Both: < 5% degradation (good, expected for 2-bit)

**Interpretation:**
- 0-2%: Excellent (imperceptible loss)
- 2-5%: Good (acceptable for aggressive compression)
- 5-10%: Fair (noticeable but usable)
- >10%: Poor (consider 4-bit instead)

**Tool:** `llama-perplexity` (included in llama.cpp)

---

### Layer 5: Throughput Profiling (TEST 5)
**Goal:** ✅ Verify no performance regression  
**Time:** 45-60 minutes  
**Risk:** Low (measurement only)

**What:** Measure tokens/second across configurations
- Baseline throughput with F16
- Throughput with KIVI_2 keys
- Throughput with KIVI_2 keys & values
- Calculate percentage change

**Success:** Maintained or improved (> 90% of F16 baseline acceptable)

**Interpretation:**
- +5% to +10%: Excellent (fused kernel optimized!)
- -5% to +5%: Good (negligible overhead)
- -5% to -10%: Acceptable (worth 5.3× memory savings)
- < -10%: Investigate (potential kernel issue)

**Tool:** Built into `llama-cli` with timing information

---

## 📋 Document Reference Table

| Document | Purpose | Pages | Target Audience |
|----------|---------|-------|-----------------|
| [PHASE_3_CHECKLIST.md](PHASE_3_CHECKLIST.md) | Quick reference + results tracking | 5 | Test operators |
| [PHASE_3_TESTING_GUIDE.md](PHASE_3_TESTING_GUIDE.md) | Comprehensive testing procedures | 15+ | Engineers, detailed procedures |
| [PHASE_2_TO_3_TRANSITION.md](PHASE_2_TO_3_TRANSITION.md) | Context and strategy | 10 | Project managers, architects |
| [KIVI_2_IMPLEMENTATION_REPORT.md](KIVI_2_IMPLEMENTATION_REPORT.md) | Full implementation history | 20+ | Developers, reviewers |

---

## 🛠️ Testing Environment Requirements

### Hardware
- **GPU:** NVIDIA (with CUDA), Intel (with Intel GPU driver), or AMD (with ROCM)
- **CPU:** Multi-core processor for reference implementations
- **Memory:** ≥ 8 GB system RAM, 2 GB VRAM minimum

### Software
- **Build:** CMake 3.16+, CMake Tools VS Code extension
- **Compiler:** GCC 9+ or Clang 12+ (SYCL support required)
- **Dependencies:** SYCL runtime (Level Zero or OpenCL backend)
- **Model:** TinyLlama GGUF format (~1 GB)

### Build Commands
```bash
# Prepare build (one-time)
cd /mnt/d51759ce-b39a-4a56-9ca8-344705c2bec2/llama.cpp
mkdir -p build results
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build test executables
cmake --build . --target llama-cli -j4
cmake --build . --target llama-perplexity -j4
cmake --build . --target llama-server -j4
```

---

## 📊 Expected Outcomes (Best Case)

```
TEST 1: Kernel Math
  ✅ MSE: 0.45 (very good for 2-bit)
  ✅ Compression: 5.33× (within 0.6% of expected)
  ✅ All kernel tests pass

TEST 2: Inference
  ✅ F16 output: [coherent sample text shown]
  ✅ KIVI_2 keys: [coherent sample text shown]  
  ✅ KIVI_2 both: [coherent sample text shown]
  ✅ Zero segmentation faults
  ✅ Quality degradation minimal (imperceptible)

TEST 3: Memory
  ✅ F16 peak: 512 MB
  ✅ KIVI_2 peak: 96 MB
  ✅ Actual ratio: 5.33× (within 10% of target 5.3×)
  ✅ Savings: 81.25% VRAM

TEST 4: Perplexity
  ✅ F16 baseline: 85.32
  ✅ KIVI_2 keys: 86.15 (+0.98% degradation) ← Excellent
  ✅ KIVI_2 both: 88.47 (+3.68% degradation) ← Good (< 5%)

TEST 5: Throughput
  ✅ F16 baseline: 207.8 tokens/sec
  ✅ KIVI_2 keys: 210.5 tokens/sec (+1.3%) ← Slight improvement!
  ✅ KIVI_2 both: 212.3 tokens/sec (+2.2%) ← Fused kernel helping!

OVERALL: ✅ ALL TESTS PASS - Ready for Phase 4
```

---

## ⚠️ Common Issues & Quick Fixes

| Issue | Cause | Fix | Reference |
|-------|-------|-----|-----------|
| Compilation fails | SYCL headers not found | Install Intel DPC++ toolkit or OpenCL headers | Build docs |
| TEST 1 fails | Quantization math wrong | Check asymmetric formula: `q = (x - min) / scale` | PHASE_3_TESTING_GUIDE.md → TEST 1 |
| TEST 2 segfaults | Memory allocation failed | Check block size = 12 bytes | PHASE_3_TESTING_GUIDE.md → Troubleshooting |
| TEST 3 VRAM not reduced | Quantization not used | Verify `-ctk KIVI_2` flag passed | PHASE_3_CHECKLIST.md → TEST 3 |
| TEST 4 PPL > 10% | Dequantization formula wrong | Verify: `x = (q * d) + m` | PHASE_3_TESTING_GUIDE.md → TEST 4 |
| TEST 5 throughput drops | Kernel dispatch overhead | Profile with `perf` | PHASE_3_TESTING_GUIDE.md → TEST 5 |

**Full troubleshooting guide:** [PHASE_3_TESTING_GUIDE.md](PHASE_3_TESTING_GUIDE.md) Troubleshooting section

---

## 🚀 Execution Roadmap

### Day 1 (2-3 hours)
```
Morning:
  10:00 - Read transition document [PHASE_2_TO_3_TRANSITION.md]
  10:30 - Review checklist [PHASE_3_CHECKLIST.md]
  11:00 - Begin TEST 1 (kernel math)
  11:45 - Complete TEST 1
  12:15 - Begin TEST 2 (inference)

Afternoon:
  13:00 - Complete TEST 2
  13:45 - Begin TEST 3 (memory profiling)
  14:15 - Complete TEST 3
```

### Day 2 (3-4 hours)
```
Morning:
  10:00 - BEGIN TEST 4 (perplexity benchmark) ← Takes long!
  11:00 - Continue TEST 4 (wait for results)
  
Afternoon:
  14:00 - Complete TEST 4, analyze results
  14:30 - Begin TEST 5 (throughput)
  15:15 - Complete TEST 5
  15:45 - Summarize all results
```

### After Phase 3
```
✅ Update KIVI_2_IMPLEMENTATION_REPORT.md status
✅ Create comprehensive benchmark report
✅ Proceed to Phase 4 (documentation) if all tests pass
```

---

## 📞 Support & Troubleshooting

### If You're Stuck

1. **Check the troubleshooting section** in [PHASE_3_TESTING_GUIDE.md](PHASE_3_TESTING_GUIDE.md)
2. **Review the expected output** for that test
3. **Verify prerequisites** (model downloaded, build successful, GPU available)
4. **Check kernel implementation** in KIVI_2_IMPLEMENTATION_REPORT.md
5. **Create GitHub issue** with test logs

### What to Include in Issue
- [ ] Which test failed (TEST 1-5)
- [ ] Full error message or unexpected output
- [ ] Log file from test (e.g., `results/test1_output.txt`)
- [ ] Hardware info (GPU type, driver version)
- [ ] Build output (any warnings or errors)

---

## ✅ Sign-Off Checklist

Before declaring Phase 3 complete:

- [ ] All 5 tests executed
- [ ] All test results recorded in [PHASE_3_CHECKLIST.md](PHASE_3_CHECKLIST.md)
- [ ] All log files saved in `results/` directory
- [ ] TEST 1 passed (MSE and compression verified)
- [ ] TEST 2 passed (no crashes, coherent output)
- [ ] TEST 3 passed (5.3× compression ratio within tolerance)
- [ ] TEST 4 passed (PPL degradation < 5%)
- [ ] TEST 5 passed (throughput maintained > 90% of F16)
- [ ] [KIVI_2_IMPLEMENTATION_REPORT.md](KIVI_2_IMPLEMENTATION_REPORT.md) updated with Phase 3 results
- [ ] Benchmark report created for Phase 4

---

## 🎓 Learning Resources

### About KIVI Quantization
- **Paper:** "KIVI: A Lightweight Vision Transformer with Kinematic and Inertial Supervision"
- **Key concept:** 2-bit asymmetric quantization of attention KV cache
- **Formula:** $X' = \text{round}\left(\frac{X - \min}{\max - \min}\right) \times (\max - \min) + \min$

### About llama.cpp
- **Repository:** https://github.com/ggml-org/llama.cpp
- **Documentation:** [docs/](docs/) folder
- **GGML Quantization:** [ggml/README.md](ggml/README.md)

### About Attention Mechanism
- **Transformer paper:** "Attention is All You Need"  
- **Q·K·V:** Query, Key, Value matrices in multi-head attention
- **KV Cache:** Cached K and V from previous tokens to avoid recomputation

---

## 🎯 Success Definition

### Minimal Success (Must Have)
- ✅ All 5 tests execute without fatal errors
- ✅ TEST 1 passes (math is correct)
- ✅ TEST 2 passes (no crashes)
- ✅ TEST 3 passes (compression within 10% tolerance)

### Good Success (Should Have)
- ✅ All above PLUS
- ✅ TEST 4 passes (PPL degradation < 5%)
- ✅ TEST 5 passes (throughput maintained)
- ✅ Results documented clearly

### Excellent Success (Nice to Have)
- ✅ All above PLUS
- ✅ PPL degradation < 2% (better than expected)
- ✅ Throughput improved > 100% (fused kernel faster!)
- ✅ Memory saved > 85% (better compression)
- ✅ Zero issues found during testing

---

## 📝 Final Notes

- **This is the final validation step** before documentation and optimization
- **All tests are self-contained** and can be run independently
- **Each test provides specific feedback** about what's working and what's not
- **Take your time with TEST 4** (perplexity) - it takes the longest
- **Record everything** - these results will be valuable for the final report

---

**Created:** March 21, 2026  
**Last Updated:** March 21, 2026  
**Status:** Ready for Phase 3 Execution

👉 **Start here:** [PHASE_3_CHECKLIST.md](PHASE_3_CHECKLIST.md)  
📖 **For details:** [PHASE_3_TESTING_GUIDE.md](PHASE_3_TESTING_GUIDE.md)  
🎯 **For context:** [PHASE_2_TO_3_TRANSITION.md](PHASE_2_TO_3_TRANSITION.md)
