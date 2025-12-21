# ARM Research: Distributed Phone Clustering

> **Research Summary for exo Distributed LLM Inference**

This document evaluates ARM optimizations, frameworks, and libraries for phone clustering. Each item is tagged with **USE** (recommended) or **CONSIDER** (worth evaluating but not immediately necessary).

**Related:** [ARM Optimization Guide](./ARM_OPTIMIZATION.md) for practical CPU specs and compiler flags.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [ARM-Optimized AI Kernels](#1-arm-optimized-ai-kernels)
3. [Mobile LLM Inference Frameworks](#2-mobile-llm-inference-frameworks)
4. [Distributed Inference Solutions](#3-distributed-inference-solutions)
5. [Linear Algebra & ML Libraries](#4-linear-algebra--ml-libraries)
6. [ARM Neural Network SDKs](#5-arm-neural-network-sdks)
7. [Model Optimization Tools](#6-model-optimization-tools)
8. [Clustering & Parallel Frameworks](#7-clustering--parallel-frameworks)
9. [Architecture-Specific Optimizations](#8-architecture-specific-optimizations)
10. [Recommendations for EXO](#recommendations-for-exo)

---

## Executive Summary

After extensive research, **EXO already has a strong foundation** as a distributed LLM inference backbone. Most existing solutions focus on single-device optimization rather than multi-device clustering. The key findings are:

### Key Takeaways

1. **No direct competitor** has achieved a better distributed phone clustering solution than EXO's approach.
2. **KleidiAI** from ARM is the most promising optimization layer for improving per-device inference speed.
3. **llama.cpp** already has excellent ARM NEON/SVE optimizations that EXO can leverage via its engine backends.
4. **GGML** tensor library forms the foundation for most efficient ARM LLM inference.
5. **Model quantization** (I8MM, dot product) extensions are critical for mobile performance.

### Priority Recommendations

| Priority | Technology | Action | Effort |
|----------|-----------|--------|--------|
| **HIGH** | KleidiAI integration | USE | Medium |
| **HIGH** | GGML ARM optimizations | Already using via llama.cpp | - |
| **MEDIUM** | ARM big.LITTLE scheduling | CONSIDER for power management | Low |
| **LOW** | PyArmadillo/mlpack | CONSIDER only if custom compute needed | High |

---

## 1. ARM-Optimized AI Kernels

### KleidiAI (ARM)

**🟢 USE**

ARM's official suite of highly optimized AI kernels, designed to accelerate AI workloads on ARM-based devices.

**Key Features:**
- Micro-kernels optimized for specific ARM architectures (Cortex-A, Cortex-X)
- Direct integration with popular frameworks: **llama.cpp**, MediaPipe, PyTorch, TensorFlow Lite
- NEON and SVE2 SIMD optimizations
- Int8 matrix multiplication (I8MM) acceleration
- FP16 and BF16 support

**Performance Claims:**
- Up to 2x speedup for LLM inference on compatible devices
- Optimized memory access patterns for cache efficiency

**Integration Path:**
- llama.cpp has merged KleidiAI support
- EXO can benefit by updating llama.cpp dependencies
- Direct GGML integration available

**Links:**
- [ARM AI Blog](https://interactive.arm.com/story/ai-on-mobile/page/5)
- [KleidiAI GitHub](https://github.com/ARM-software/kleidiai)

**Why USE:** Direct integration with llama.cpp means EXO gets these optimizations automatically when using the llama.cpp engine. Minimal effort for significant gains.

---

### GGML (ggerganov)

**🟢 USE** (Already integrated via llama.cpp)

Low-level tensor library designed for efficient ML inference on edge devices.

**Key Features:**
- Pure C implementation with no dependencies
- ARM NEON SIMD intrinsics for AArch64
- SVE/SVE2 support for newer ARMv9 devices
- Quantization support: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, etc.
- Optimized matrix multiplication kernels

**ARM-Specific Optimizations:**
```c
// GGML uses ARM-specific intrinsics
#if defined(__ARM_NEON)
    // NEON vectorized operations
    float32x4_t v = vld1q_f32(data);
#endif
#if defined(__ARM_FEATURE_SVE)
    // SVE scalable vector operations
#endif
```

**Critical Extensions Utilized:**
- `+dotprod` - 4-element dot product (2-4x speedup for quantized)
- `+fp16` - Half-precision floating point
- `+i8mm` - Int8 matrix multiply (ARMv8.6+)
- `+bf16` - Brain Float 16

**Why USE:** GGML is the foundation of llama.cpp and whisper.cpp. EXO's llama.cpp engine already uses this. Ensure we're using the latest version with all ARM optimizations enabled.

---

## 2. Mobile LLM Inference Frameworks

### llama.cpp

**🟢 USE** (Core engine)

The gold standard for efficient LLM inference on edge devices.

**ARM Optimizations:**
- Full NEON SIMD support
- SVE2 support for ARMv9 (Snapdragon 8 Gen 1+)
- Quantization (Q4_K_M recommended for mobile)
- Metal backend for Apple Silicon
- KleidiAI integration merged

**Compilation Flags for Termux:**
```bash
# Optimal for Snapdragon 8 Gen 3
make LLAMA_NO_CUDA=1 LLAMA_NO_METAL=1 \
    CFLAGS="-O3 -mcpu=cortex-x4 -march=armv9.2-a+sve2+i8mm+bf16 -flto" \
    CXXFLAGS="-O3 -mcpu=cortex-x4 -march=armv9.2-a+sve2+i8mm+bf16 -flto"
```

**Why USE:** EXO already uses llama.cpp as one of its inference backends. The key is ensuring optimal compilation flags are used per device architecture.

---

### MNN (Alibaba)

**🟠 CONSIDER**

Alibaba's Mobile Neural Network framework for efficient inference.

**Key Features:**
- Cross-platform (Android, iOS, Linux, Windows)
- Optimized for mobile ARM (ARM82 extensions)
- Supports TensorFlow, PyTorch, ONNX model formats
- Quantization-aware training support
- Low memory footprint

**ARM Optimizations:**
- ARM NEON intrinsics
- ARM FP16 support
- Vulkan GPU backend

**Links:**
- [MNN GitHub](https://github.com/alibaba/MNN)

**Why CONSIDER:** MNN is excellent for general ML inference but not specifically designed for LLM workloads. Could be useful if EXO expands to non-LLM AI tasks.

---

### NCNN (Tencent)

**🟠 CONSIDER**

High-performance neural network inference framework optimized for mobile.

**Key Features:**
- Pure C++ with no dependencies
- Vulkan GPU compute support
- ARM NEON optimization
- Very small binary size (~1MB)

**ARM Optimizations:**
- NEON assembly routines
- ARMv8.2 extensions support
- Cache-friendly memory access patterns

**Links:**
- [NCNN GitHub](https://github.com/Tencent/ncnn)

**Why CONSIDER:** NCNN excels at vision models but lacks specific LLM optimizations. Less relevant for EXO's primary use case.

---

### ExecuTorch (Meta)

**🟠 CONSIDER**

Meta's PyTorch-based runtime for on-device AI.

**Key Features:**
- Native PyTorch model support
- ARM delegate for optimized execution
- Quantization support
- Memory-efficient execution

**Links:**
- [ExecuTorch GitHub](https://github.com/pytorch/executorch)

**Why CONSIDER:** Still maturing. May be useful for PyTorch-native models but adds significant dependency overhead compared to GGML-based solutions.

---

## 3. Distributed Inference Solutions

### EXO (Current Project)

**🟢 USE** (This is us!)

EXO is already the leading solution for distributed LLM inference across heterogeneous devices.

**Current Strengths:**
- P2P networking with libp2p
- Dynamic topology discovery
- Multi-backend support (llama.cpp, MLX, etc.)
- Cross-platform (macOS, Linux, Android/Termux)

**What competitors lack:**
- Most solutions focus on single-device optimization
- No other project has achieved EXO's distributed clustering for phones
- Petals (closest alternative) is server-focused, not mobile-friendly

---

### Petals

**🔴 NOT RECOMMENDED** for mobile

Distributed inference for large models, but designed for server/desktop environments.

**Limitations for Mobile:**
- Python-heavy, high memory overhead
- Designed for GPUs and powerful CPUs
- No specific ARM mobile optimizations
- Network overhead not optimized for mobile

**Why NOT:** Petals is designed for BitTorrent-style distributed inference on servers. Not suitable for mobile phone clustering.

---

### llama.cpp RPC Backend

**🟠 CONSIDER** (for hybrid setups)

llama.cpp's built-in RPC backend for offloading compute to remote devices.

**Features:**
- Simple tensor transfer protocol
- Can offload layers to remote llama-server instances
- Low-latency communication

**Limitations:**
- Single-master architecture
- No automatic topology discovery
- No fault tolerance

**Why CONSIDER:** Could be useful for simple hybrid setups (phone + desktop), but EXO's P2P approach is superior for phone-to-phone clustering.

---

## 4. Linear Algebra & ML Libraries

### PyArmadillo

**🔴 NOT RECOMMENDED** for EXO

Python linear algebra library mirroring C++ Armadillo.

**Features:**
- MATLAB-like syntax
- LAPACK/BLAS integration
- Easy Python-to-C++ conversion

**Why NOT:** EXO doesn't need custom linear algebra. GGML/llama.cpp handles all tensor operations efficiently. Adding PyArmadillo would add complexity without benefit.

---

### mlpack

**🔴 NOT RECOMMENDED** for EXO

C++ ML library built on Armadillo.

**Features:**
- Scalable ML algorithms
- Header-only implementation
- ARM-friendly

**Why NOT:** mlpack is for training ML models, not LLM inference. Not relevant to EXO's use case.

---

### OpenBLAS (ARM build)

**🟠 CONSIDER** (for NumPy acceleration)

Optimized BLAS library with ARM support.

**Features:**
- ARM Cortex-A optimizations
- Multi-threaded matrix operations
- Used by NumPy/SciPy

**Why CONSIDER:** Only relevant if EXO's Python components use NumPy for heavy computation. The core inference doesn't need this.

---

## 5. ARM Neural Network SDKs

### Arm NN SDK

**🟠 CONSIDER**

ARM's official neural network SDK for optimized inference.

**Features:**
- TensorFlow Lite and ONNX support
- Optimized for Arm Cortex-A and Mali GPUs
- Includes Arm Compute Library

**Status:**
- PyArmNN (Python bindings) deprecated as of version 24.08
- C++ API still active

**Why CONSIDER:** Could be useful for adding GPU (Mali) acceleration on Samsung devices. However, llama.cpp + KleidiAI provides similar benefits with better LLM focus.

---

### Arm Compute Library (ACL)

**🟠 CONSIDER**

Low-level library with optimized compute primitives.

**Features:**
- NEON and SVE optimized kernels
- CPU and Mali GPU support
- GEMM, convolution, pooling operations

**Why CONSIDER:** Only if we need to write custom kernels. GGML already provides optimized operations.

---

## 6. Model Optimization Tools

### TensorFlow Model Optimization Toolkit - Weight Clustering

**🟠 CONSIDER** (for model preparation)

ARM-contributed weight clustering API for model compression.

**Features:**
- Reduces model size by clustering weights
- Improves inference speed
- Compatible with TensorFlow Lite

**Why CONSIDER:** Useful for preparing custom models, but EXO primarily uses pre-quantized GGUF models from Hugging Face.

---

### GGML Quantization

**🟢 USE** (Already using)

GGML's native quantization is the standard for mobile LLM inference.

**Recommended Quantizations for Mobile:**
| Quantization | Bits | Quality | Speed | Memory |
|--------------|------|---------|-------|--------|
| Q4_K_M | 4-bit | Good | Fast | Low |
| Q5_K_M | 5-bit | Better | Medium | Medium |
| Q8_0 | 8-bit | Best | Slow | High |
| Q4_0 | 4-bit | Acceptable | Fastest | Lowest |

**Why USE:** EXO should recommend Q4_K_M or Q5_K_M for phone clusters. Already integrated via llama.cpp.

---

## 7. Clustering & Parallel Frameworks

### clusterNOR

**🔴 NOT RECOMMENDED**

NUMA-optimized clustering framework for data centers.

**Why NOT:** Designed for server NUMA architectures, not mobile devices. Concepts not transferable to phone clustering.

---

### KScaNN

**🔴 NOT RECOMMENDED**

ARM-optimized approximate nearest neighbor search for Kunpeng 920.

**Why NOT:** Specific to Huawei Kunpeng server CPUs. Not applicable to mobile Cortex cores.

---

## 8. Architecture-Specific Optimizations

### ARM big.LITTLE Scheduling

**🟢 USE**

Heterogeneous core architecture present in all modern mobile SoCs.

**Strategy for EXO:**

| Core Type | Usage | Example |
|-----------|-------|---------|
| Prime (X-cores) | LLM inference, matrix multiply | Cortex-X4 |
| Performance (A7xx) | Token generation, preprocessing | Cortex-A720 |
| Efficiency (A5xx) | Network I/O, coordination | Cortex-A520 |

**Implementation:**
```bash
# Pin compute to performance cores (example for 8 Gen 3)
taskset -c 4-7 ./exo-worker  # Big cores
```

**Why USE:** Simple to implement via process affinity. Can improve sustained performance and thermals.

---

### ARMv9 Extensions

**🟢 USE** (via compiler flags)

Modern ARMv9 features for AI acceleration.

**Key Extensions:**
- **SVE2**: Scalable Vector Extension 2 (variable width SIMD)
- **I8MM**: Int8 Matrix Multiply (huge for quantized models)
- **BF16**: Brain Float 16 (ML-optimized format)

**Compiler Flags:**
```bash
# ARMv9 flagship (Snapdragon 8 Gen 1+)
-march=armv9-a+sve2+i8mm+bf16

# ARMv8.2 (most 2018-2022 devices)
-march=armv8.2-a+dotprod+fp16
```

**Why USE:** Enabling these extensions at compile time provides significant performance gains with no runtime overhead.

---

### NEON SIMD

**🟢 USE** (Always enabled on AArch64)

128-bit SIMD unit present in all 64-bit ARM CPUs.

**Critical Operations Accelerated:**
- Vector multiply-accumulate
- Quantized dot products
- Activation functions (ReLU, GELU)

**Why USE:** NEON is always available and llama.cpp/GGML already use it. Ensure compilation includes NEON optimizations.

---

## 9. Binary Neural Networks (BNNs)

### daBNN

**🔴 NOT RECOMMENDED** for LLM

Fast BNN inference framework for ARM.

**Why NOT:** Binary Neural Networks are for specialized vision models (1-bit weights). Not applicable to LLM inference which requires at minimum 4-bit quantization for quality.

---

## Recommendations for EXO

### Immediate Actions (USE)

1. **Update llama.cpp to latest version**
   - Gains KleidiAI integration automatically
   - Improved ARM NEON/SVE2 kernels
   - Better quantization support

2. **Optimize compilation per device architecture**
   - Implement device detection script
   - Compile with optimal `-mcpu` and `-march` flags
   - Already documented in `ARM_CORTEX_OPTIMIZATION_GUIDE.md`

3. **Implement big.LITTLE aware scheduling**
   - Pin inference threads to performance cores
   - Use efficiency cores for networking/coordination
   - Monitor thermals and throttle appropriately

4. **Recommend optimal quantizations**
   - Q4_K_M for memory-constrained phones
   - Q5_K_M for phones with 12GB+ RAM
   - Document performance expectations per device class

### Future Considerations (CONSIDER)

1. **Mali GPU acceleration**
   - Arm NN SDK supports Mali GPUs
   - Could offload some operations to GPU
   - Complex to implement, may not be worth it

2. **NPU integration**
   - Qualcomm AI Engine SDK for Snapdragon NPUs
   - Samsung Exynos NPU SDK
   - Very vendor-specific, high complexity

3. **Custom GGML kernels**
   - Write device-specific optimized kernels
   - Only if significant gains possible
   - High effort, low priority

### Not Recommended

1. **PyArmadillo, mlpack, clusterNOR** - Wrong abstraction level
2. **Petals** - Server-focused, not mobile-optimized
3. **daBNN** - Binary NNs not applicable to LLMs
4. **KScaNN** - Server CPU focused

---

## Conclusion

**EXO is already the best solution** for distributed phone clustering for LLM inference. The research confirms that:

1. No one else has achieved a comparable distributed phone cluster solution
2. The key optimizations (GGML, NEON, quantization) are already in place via llama.cpp
3. KleidiAI integration (via llama.cpp updates) is the biggest easy win
4. Architecture-specific compilation is important and already documented

The focus should remain on:
- **Reliability** of the distributed system
- **Ease of use** for Termux setup
- **Performance tuning** via optimal compilation flags
- **Documentation** of device-specific recommendations

---

## References

1. [ARM KleidiAI](https://github.com/ARM-software/kleidiai)
2. [llama.cpp](https://github.com/ggerganov/llama.cpp)
3. [GGML](https://github.com/ggerganov/ggml)
4. [ARM Developer Documentation](https://developer.arm.com/)
5. [MNN](https://github.com/alibaba/MNN)
6. [NCNN](https://github.com/Tencent/ncnn)
7. [ARM big.LITTLE](https://en.wikipedia.org/wiki/ARM_big.LITTLE)
8. [TensorFlow Model Optimization](https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html)
9. [Arm NN SDK](https://github.com/ARM-software/armnn)

---

*This research was compiled for the EXO project. Last updated: December 2024*

