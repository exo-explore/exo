# ARM Optimization Guide

> **CPU Specifications & Compiler Flags for Android/Termux**

Detailed specifications for ARM Cortex processors, compiler optimization flags, and performance tuning strategies for Termux.

**Related:** [ARM Research](./ARM_RESEARCH.md) for framework evaluation and technology recommendations.

---

## Table of Contents

1. [ARM Architecture Overview](#1-arm-architecture-overview)
2. [Cortex Core Specifications](#2-cortex-core-specifications)
3. [SoC Configurations by Manufacturer](#3-soc-configurations-by-manufacturer)
4. [Compiler Optimization Flags](#4-compiler-optimization-flags)
5. [Architecture Feature Extensions](#5-architecture-feature-extensions)
6. [Performance Tuning in Termux](#6-performance-tuning-in-termux)
7. [Device-Specific Optimizations](#7-device-specific-optimizations)
8. [Benchmarking & Profiling](#8-benchmarking--profiling)
9. [Quick Reference Tables](#9-quick-reference-tables)

---

## 1. ARM Architecture Overview

### Architecture Generations

| Architecture | Year | Key Features | Notable Cores |
|--------------|------|--------------|---------------|
| **ARMv7-A** | 2007 | 32-bit, NEON SIMD, VFPv3 | Cortex-A7, A9, A15, A17 |
| **ARMv8.0-A** | 2011 | 64-bit, AArch64, Advanced SIMD | Cortex-A53, A57, A72, A73 |
| **ARMv8.2-A** | 2016 | FP16, Dot Product, RAS | Cortex-A55, A75, A76, A77, A78 |
| **ARMv9.0-A** | 2021 | SVE2, MTE, BTI, PAC | Cortex-A710, A715, X1, X2, X3 |
| **ARMv9.2-A** | 2023 | Enhanced SVE2, SME | Cortex-A720, A725, X4, X925 |

### Core Classification (big.LITTLE / DynamIQ)

Modern ARM SoCs use heterogeneous computing with different core types:

| Core Type | Purpose | Examples | Characteristics |
|-----------|---------|----------|-----------------|
| **Prime/Ultra** | Peak performance | Cortex-X1, X2, X3, X4, X925 | Highest IPC, single-thread focus |
| **Performance (Big)** | Sustained performance | Cortex-A78, A715, A720 | Balance of power/performance |
| **Efficiency (Little)** | Power saving | Cortex-A55, A510, A520 | Low power, background tasks |

---

## 2. Cortex Core Specifications

### Cortex-X Series (Prime/Ultra Cores)

These cores deliver maximum single-threaded performance for demanding workloads.

| Core | Architecture | IPC vs Previous | Pipeline | Decode Width | L1 I$ | L1 D$ | L2 Cache | Notable Features |
|------|--------------|-----------------|----------|--------------|-------|-------|----------|------------------|
| **Cortex-X925** | ARMv9.2-A | +17% vs X4 | 10-wide | 10 | 64KB | 64KB | 1MB | Highest mobile IPC |
| **Cortex-X4** | ARMv9.2-A | +15% vs X3 | 10-wide | 10 | 64KB | 64KB | 1MB | 3nm optimized |
| **Cortex-X3** | ARMv9.0-A | +11% vs X2 | 6-wide | 6 | 64KB | 64KB | 512KB-1MB | Total Compute 2022 |
| **Cortex-X2** | ARMv9.0-A | +16% vs X1 | 5-wide | 5 | 64KB | 64KB | 512KB-1MB | First ARMv9 X-core |
| **Cortex-X1** | ARMv8.2-A | +22% vs A78 | 5-wide | 5 | 64KB | 64KB | 256KB-1MB | First X-series |

### Cortex-A Series (Performance Cores - Big)

| Core | Architecture | Process | Clock (Typ.) | IPC | L1 I$ | L1 D$ | L2 Cache | Key Optimizations |
|------|--------------|---------|--------------|-----|-------|-------|----------|-------------------|
| **Cortex-A720** | ARMv9.2-A | 4nm+ | 2.8-3.2 GHz | High | 64KB | 64KB | 256-512KB | Latest big core |
| **Cortex-A715** | ARMv9.0-A | 4nm | 2.5-3.0 GHz | +5% vs A710 | 64KB | 64KB | 256-512KB | 20% more efficient |
| **Cortex-A710** | ARMv9.0-A | 5nm | 2.5-2.85 GHz | A78-level | 64KB | 64KB | 256-512KB | First ARMv9 big |
| **Cortex-A78** | ARMv8.2-A | 5nm | 2.6-3.0 GHz | +7% vs A77 | 64KB | 64KB | 256-512KB | DynamIQ optimized |
| **Cortex-A78C** | ARMv8.2-A | 5nm | 2.4-2.8 GHz | A78-level | 64KB | 64KB | 512KB | Up to 8 cores |
| **Cortex-A77** | ARMv8.2-A | 7nm | 2.3-2.8 GHz | +20% vs A76 | 64KB | 64KB | 256-512KB | ML optimized |
| **Cortex-A76** | ARMv8.2-A | 7nm | 2.0-2.8 GHz | +25% vs A75 | 64KB | 64KB | 256-512KB | Dot Product support |
| **Cortex-A75** | ARMv8.2-A | 10nm | 2.0-2.5 GHz | High | 64KB | 64KB | 256-512KB | DynamIQ first gen |
| **Cortex-A73** | ARMv8.0-A | 10nm | 1.8-2.5 GHz | Moderate | 32-64KB | 32-64KB | 256KB-8MB | 30% more efficient |
| **Cortex-A72** | ARMv8.0-A | 16nm | 1.5-2.5 GHz | Moderate | 48KB | 32KB | 512KB-4MB | big.LITTLE era |
| **Cortex-A57** | ARMv8.0-A | 20nm | 1.5-2.0 GHz | Baseline | 48KB | 32KB | 512KB-2MB | First 64-bit big |

### Cortex-A Series (Efficiency Cores - Little)

| Core | Architecture | Efficiency | Clock (Typ.) | L1 I$ | L1 D$ | L2 Cache | Key Features |
|------|--------------|------------|--------------|-------|-------|----------|--------------|
| **Cortex-A520** | ARMv9.2-A | Best | 1.8-2.2 GHz | 32KB | 32KB | 128-256KB | Latest efficiency |
| **Cortex-A510** | ARMv9.0-A | Excellent | 1.8-2.0 GHz | 32KB | 32KB | 64-256KB | First ARMv9 little |
| **Cortex-A55** | ARMv8.2-A | Very Good | 1.4-2.0 GHz | 16-64KB | 16-64KB | 64-256KB | Most common |
| **Cortex-A53** | ARMv8.0-A | Good | 1.2-1.8 GHz | 8-64KB | 8-64KB | 128KB-2MB | Most deployed |
| **Cortex-A35** | ARMv8.0-A | High | 1.0-1.5 GHz | 8-64KB | 8-64KB | 128-512KB | Ultra-low power |

### Legacy 32-bit Cores (Reference)

| Core | Architecture | Clock (Typ.) | Features |
|------|--------------|--------------|----------|
| **Cortex-A17** | ARMv7-A | 1.8-2.2 GHz | NEON, VFPv4, 60% faster than A9 |
| **Cortex-A15** | ARMv7-A | 1.5-2.5 GHz | Out-of-order, 40% faster than A9 |
| **Cortex-A9** | ARMv7-A | 0.8-2.0 GHz | Out-of-order, VFPv3, NEON |
| **Cortex-A7** | ARMv7-A | 0.8-1.5 GHz | Efficient, paired with A15/A17 |
| **Cortex-A5** | ARMv7-A | 0.5-1.0 GHz | Ultra-low power |

---

## 3. SoC Configurations by Manufacturer

### Qualcomm Snapdragon (Most Common in Android)

#### Flagship Series (8 Gen)

| SoC | Year | CPU Configuration | Architecture | Recommended `-mcpu` |
|-----|------|-------------------|--------------|---------------------|
| **Snapdragon 8 Elite** | 2024 | 2x Oryon Prime @ 4.32 GHz + 6x Oryon Performance @ 3.53 GHz | Custom (ARMv9) | `oryon-1` or `cortex-x925` |
| **Snapdragon 8 Gen 3** | 2023 | 1x Cortex-X4 @ 3.3 GHz + 3x Cortex-A720 @ 3.15 GHz + 2x Cortex-A720 @ 2.96 GHz + 2x Cortex-A520 @ 2.27 GHz | ARMv9.2-A | `cortex-x4` |
| **Snapdragon 8 Gen 2** | 2022 | 1x Cortex-X3 @ 3.2 GHz + 2x Cortex-A715 @ 2.8 GHz + 2x Cortex-A710 @ 2.8 GHz + 3x Cortex-A510 @ 2.0 GHz | ARMv9.0-A | `cortex-x3` |
| **Snapdragon 8 Gen 1** | 2021 | 1x Cortex-X2 @ 3.0 GHz + 3x Cortex-A710 @ 2.5 GHz + 4x Cortex-A510 @ 1.8 GHz | ARMv9.0-A | `cortex-x2` |
| **Snapdragon 8 Gen 1+** | 2022 | 1x Cortex-X2 @ 3.2 GHz + 3x Cortex-A710 @ 2.75 GHz + 4x Cortex-A510 @ 2.0 GHz | ARMv9.0-A | `cortex-x2` |

#### Premium Series (888, 865, 855, etc.)

| SoC | Year | CPU Configuration | Architecture | Recommended `-mcpu` |
|-----|------|-------------------|--------------|---------------------|
| **Snapdragon 888/888+** | 2020/21 | 1x Cortex-X1 @ 2.84 GHz + 3x Cortex-A78 @ 2.42 GHz + 4x Cortex-A55 @ 1.8 GHz | ARMv8.2-A | `cortex-x1` |
| **Snapdragon 870** | 2021 | 1x Cortex-A77 @ 3.2 GHz + 3x Cortex-A77 @ 2.42 GHz + 4x Cortex-A55 @ 1.8 GHz | ARMv8.2-A | `cortex-a77` |
| **Snapdragon 865/865+** | 2020 | 1x Cortex-A77 @ 2.84 GHz + 3x Cortex-A77 @ 2.42 GHz + 4x Cortex-A55 @ 1.8 GHz | ARMv8.2-A | `cortex-a77` |
| **Snapdragon 855/855+** | 2019 | 1x Cortex-A76 @ 2.84 GHz + 3x Cortex-A76 @ 2.42 GHz + 4x Cortex-A55 @ 1.8 GHz | ARMv8.2-A | `cortex-a76` |
| **Snapdragon 845** | 2018 | 4x Cortex-A75 @ 2.8 GHz + 4x Cortex-A55 @ 1.8 GHz | ARMv8.2-A | `cortex-a75` |
| **Snapdragon 835** | 2017 | 4x Cortex-A73 @ 2.45 GHz + 4x Cortex-A53 @ 1.9 GHz | ARMv8.0-A | `cortex-a73` |

#### Mid-Range Series (7xx, 6xx)

| SoC | Year | CPU Configuration | Architecture | Recommended `-mcpu` |
|-----|------|-------------------|--------------|---------------------|
| **Snapdragon 7+ Gen 3** | 2024 | 1x Cortex-X4 @ 2.8 GHz + 4x Cortex-A720 @ 2.6 GHz + 3x Cortex-A520 @ 1.9 GHz | ARMv9.2-A | `cortex-x4` |
| **Snapdragon 778G/778G+** | 2021 | 1x Cortex-A78 @ 2.4 GHz + 3x Cortex-A78 @ 2.2 GHz + 4x Cortex-A55 @ 1.9 GHz | ARMv8.2-A | `cortex-a78` |
| **Snapdragon 765G** | 2020 | 1x Cortex-A76 @ 2.4 GHz + 1x Cortex-A76 @ 2.2 GHz + 6x Cortex-A55 @ 1.8 GHz | ARMv8.2-A | `cortex-a76` |
| **Snapdragon 720G** | 2020 | 2x Cortex-A76 @ 2.3 GHz + 6x Cortex-A55 @ 1.8 GHz | ARMv8.2-A | `cortex-a76` |
| **Snapdragon 695** | 2021 | 2x Cortex-A78 @ 2.2 GHz + 6x Cortex-A55 @ 1.8 GHz | ARMv8.2-A | `cortex-a78` |
| **Snapdragon 680/685** | 2021 | 4x Cortex-A73 @ 2.4 GHz + 4x Cortex-A53 @ 1.9 GHz | ARMv8.0-A | `cortex-a73` |

### MediaTek Dimensity

| SoC | Year | CPU Configuration | Architecture | Recommended `-mcpu` |
|-----|------|-------------------|--------------|---------------------|
| **Dimensity 9400** | 2024 | 1x Cortex-X925 @ 3.62 GHz + 3x Cortex-X4 @ 3.3 GHz + 4x Cortex-A720 @ 2.4 GHz | ARMv9.2-A | `cortex-x925` |
| **Dimensity 9300** | 2023 | 4x Cortex-X4 @ 3.25 GHz + 4x Cortex-A720 @ 2.0 GHz | ARMv9.2-A | `cortex-x4` |
| **Dimensity 9200/9200+** | 2022/23 | 1x Cortex-X3 @ 3.05 GHz + 3x Cortex-A715 @ 2.85 GHz + 4x Cortex-A510 @ 1.8 GHz | ARMv9.0-A | `cortex-x3` |
| **Dimensity 8300** | 2023 | 4x Cortex-A715 @ 3.35 GHz + 4x Cortex-A510 @ 2.2 GHz | ARMv9.0-A | `cortex-a715` |
| **Dimensity 8200** | 2022 | 1x Cortex-A78 @ 3.1 GHz + 3x Cortex-A78 @ 3.0 GHz + 4x Cortex-A55 @ 2.0 GHz | ARMv8.2-A | `cortex-a78` |
| **Dimensity 8100** | 2022 | 4x Cortex-A78 @ 2.85 GHz + 4x Cortex-A55 @ 2.0 GHz | ARMv8.2-A | `cortex-a78` |
| **Dimensity 7200** | 2023 | 2x Cortex-A715 @ 2.8 GHz + 6x Cortex-A510 @ 2.0 GHz | ARMv9.0-A | `cortex-a715` |
| **Dimensity 1200/1300** | 2021 | 1x Cortex-A78 @ 3.0 GHz + 3x Cortex-A78 @ 2.6 GHz + 4x Cortex-A55 @ 2.0 GHz | ARMv8.2-A | `cortex-a78` |

### Samsung Exynos

| SoC | Year | CPU Configuration | Architecture | Recommended `-mcpu` |
|-----|------|-------------------|--------------|---------------------|
| **Exynos 2400** | 2024 | 1x Cortex-X4 @ 3.2 GHz + 2x Cortex-A720 @ 2.9 GHz + 3x Cortex-A720 @ 2.6 GHz + 4x Cortex-A520 @ 1.95 GHz | ARMv9.2-A | `cortex-x4` |
| **Exynos 2200** | 2022 | 1x Cortex-X2 @ 2.8 GHz + 3x Cortex-A710 @ 2.5 GHz + 4x Cortex-A510 @ 1.8 GHz | ARMv9.0-A | `cortex-x2` |
| **Exynos 2100** | 2021 | 1x Cortex-X1 @ 2.9 GHz + 3x Cortex-A78 @ 2.8 GHz + 4x Cortex-A55 @ 2.2 GHz | ARMv8.2-A | `cortex-x1` |
| **Exynos 990** | 2020 | 2x Exynos M5 @ 2.73 GHz + 2x Cortex-A76 @ 2.5 GHz + 4x Cortex-A55 @ 2.0 GHz | ARMv8.2-A | `cortex-a76` |
| **Exynos 9820** | 2019 | 2x Exynos M4 @ 2.73 GHz + 2x Cortex-A75 @ 2.31 GHz + 4x Cortex-A55 @ 1.95 GHz | ARMv8.2-A | `cortex-a75` |

### Google Tensor

| SoC | Year | CPU Configuration | Architecture | Recommended `-mcpu` |
|-----|------|-------------------|--------------|---------------------|
| **Tensor G4** | 2024 | 1x Cortex-X4 @ 3.1 GHz + 3x Cortex-A720 @ 2.6 GHz + 4x Cortex-A520 @ 1.95 GHz | ARMv9.2-A | `cortex-x4` |
| **Tensor G3** | 2023 | 1x Cortex-X3 @ 2.91 GHz + 4x Cortex-A715 @ 2.37 GHz + 4x Cortex-A510 @ 1.7 GHz | ARMv9.0-A | `cortex-x3` |
| **Tensor G2** | 2022 | 2x Cortex-X1 @ 2.85 GHz + 2x Cortex-A78 @ 2.35 GHz + 4x Cortex-A55 @ 1.8 GHz | ARMv8.2-A | `cortex-x1` |
| **Tensor** | 2021 | 2x Cortex-X1 @ 2.8 GHz + 2x Cortex-A76 @ 2.25 GHz + 4x Cortex-A55 @ 1.8 GHz | ARMv8.2-A | `cortex-x1` |

---

## 4. Compiler Optimization Flags

### GCC/Clang Primary Flags

#### Target Architecture Flags

| Flag | Purpose | Example | Notes |
|------|---------|---------|-------|
| `-mcpu=<cpu>` | Target specific CPU | `-mcpu=cortex-a76` | Sets both arch and tune |
| `-march=<arch>` | Target architecture | `-march=armv8.2-a` | Enables instruction sets |
| `-mtune=<cpu>` | Optimize for CPU | `-mtune=cortex-a76` | Scheduling optimization |

**Important**: `-mcpu` automatically sets both `-march` and `-mtune`. Use it alone for simplicity.

#### Recommended `-mcpu` Values by Core

```bash
# Cortex-X Series (Prime cores)
-mcpu=cortex-x925    # Latest (2024)
-mcpu=cortex-x4      # 8 Gen 3, Exynos 2400
-mcpu=cortex-x3      # 8 Gen 2, Dimensity 9200
-mcpu=cortex-x2      # 8 Gen 1, Exynos 2200
-mcpu=cortex-x1      # 888, Exynos 2100, Tensor G2

# Cortex-A7xx Series (Big cores)
-mcpu=cortex-a720    # Latest mid-range
-mcpu=cortex-a715    # 8 Gen 2, Dimensity 9200
-mcpu=cortex-a710    # 8 Gen 1, Exynos 2200
-mcpu=cortex-a78     # 888, most 2021 flagships
-mcpu=cortex-a77     # 865, 870
-mcpu=cortex-a76     # 855, 765G, Dimensity 1000
-mcpu=cortex-a75     # 845, Exynos 9810
-mcpu=cortex-a73     # 835, mid-range 2020
-mcpu=cortex-a72     # 820, older flagships

# Cortex-A5xx Series (Little cores)
-mcpu=cortex-a520    # Latest efficiency
-mcpu=cortex-a510    # ARMv9 efficiency
-mcpu=cortex-a55     # Most common 2018-2023
-mcpu=cortex-a53     # Legacy efficiency
```

#### Architecture Feature Extensions

```bash
# ARMv8.2-A with common extensions
-march=armv8.2-a+dotprod+fp16+crypto

# ARMv9.0-A with SVE2
-march=armv9-a+sve2

# Explicit feature enable/disable
-march=armv8.2-a+dotprod         # Enable dot product
-march=armv8.2-a+fp16            # Enable FP16
-march=armv8.2-a+i8mm            # Enable Int8 Matrix Multiply
-march=armv8.2-a+bf16            # Enable BFloat16
-march=armv8.2-a+crypto          # Enable crypto extensions
```

### Optimization Level Flags

| Flag | Description | Use Case |
|------|-------------|----------|
| `-O0` | No optimization | Debugging |
| `-O1` | Basic optimization | Debug with some speed |
| `-O2` | Standard optimization | **Production (recommended)** |
| `-O3` | Aggressive optimization | Maximum speed, larger binary |
| `-Os` | Optimize for size | Memory-constrained |
| `-Ofast` | O3 + fast-math | Numeric-intensive code |

### Floating-Point and SIMD Flags

```bash
# For AArch64 (64-bit ARM) - NEON is always enabled
# FP16 (half precision)
-march=armv8.2-a+fp16

# For AArch32 (32-bit ARM) - explicit FPU selection
-mfpu=neon              # Basic NEON
-mfpu=neon-vfpv4        # NEON with VFPv4
-mfpu=neon-fp-armv8     # ARMv8 NEON
-mfloat-abi=hard        # Hardware floating point
-mfloat-abi=softfp      # FPU with soft calling convention
```

### Advanced Optimization Flags

```bash
# Link-Time Optimization
-flto                   # Enable LTO
-flto=auto             # Use available parallelism

# Profile-Guided Optimization
-fprofile-generate      # Instrumented build
-fprofile-use           # Optimized build

# Vectorization
-ftree-vectorize        # Auto-vectorization (in O3)
-fvect-cost-model=dynamic

# Loop Optimizations
-funroll-loops          # Unroll loops
-fprefetch-loop-arrays  # Prefetch arrays

# Function/Data Sections (for size)
-ffunction-sections
-fdata-sections
-Wl,--gc-sections       # Linker flag to remove unused

# Fast Math (use with caution)
-ffast-math             # Aggressive FP optimizations
-fno-math-errno         # Don't set errno for math funcs
-ffinite-math-only      # Assume no inf/nan
-funsafe-math-optimizations
```

### Complete Compilation Examples

```bash
# Optimal for Snapdragon 8 Gen 3 (Cortex-X4)
clang -O3 -mcpu=cortex-x4 -flto -ffast-math \
    -march=armv9.2-a+sve2+i8mm+bf16 \
    source.c -o output

# Optimal for Snapdragon 888 (Cortex-X1)
clang -O3 -mcpu=cortex-x1 -flto \
    -march=armv8.2-a+dotprod+fp16 \
    source.c -o output

# Optimal for mid-range (Cortex-A76 class)
gcc -O2 -mcpu=cortex-a76 -flto \
    -march=armv8.2-a+dotprod \
    source.c -o output

# Generic ARMv8.2-A (broad compatibility)
gcc -O2 -march=armv8.2-a+dotprod -mtune=cortex-a76 \
    source.c -o output

# Size-optimized for constrained devices
gcc -Os -mcpu=cortex-a55 -ffunction-sections -fdata-sections \
    -Wl,--gc-sections source.c -o output
```

---

## 5. Architecture Feature Extensions

### SIMD and Vector Extensions

| Extension | Architecture | Description | Compiler Flag | Use Case |
|-----------|--------------|-------------|---------------|----------|
| **NEON** | ARMv7+, ARMv8+ | 128-bit SIMD | Always on (AArch64) | Multimedia, signal processing |
| **SVE** | ARMv8.2+ | Scalable vectors (128-2048 bit) | `+sve` | Scientific computing |
| **SVE2** | ARMv9+ | Enhanced SVE | `+sve2` | ML, crypto, signal processing |
| **SME** | ARMv9.2+ | Scalable Matrix Extension | `+sme` | Matrix operations, ML |

### ML-Specific Extensions

| Extension | Architecture | Description | Compiler Flag | Performance Impact |
|-----------|--------------|-------------|---------------|-------------------|
| **Dot Product** | ARMv8.2+ | 4-element dot product | `+dotprod` | **2-4x** for quantized inference |
| **FP16** | ARMv8.2+ | Half-precision FP | `+fp16` | **2x** throughput, less precision |
| **BF16** | ARMv8.6+ | Brain Float 16 | `+bf16` | ML training/inference |
| **I8MM** | ARMv8.6+ | Int8 Matrix Multiply | `+i8mm` | **2-4x** for int8 models |

### Security Extensions

| Extension | Architecture | Description | Compiler Flag |
|-----------|--------------|-------------|---------------|
| **PAC** | ARMv8.3+ | Pointer Authentication | `+pauth` |
| **BTI** | ARMv8.5+ | Branch Target Identification | `+bti` |
| **MTE** | ARMv8.5+ | Memory Tagging Extension | `+mte` |

### Feature Detection in Code

```c
// Runtime feature detection using HWCAP
#include <sys/auxv.h>
#include <asm/hwcap.h>

int main() {
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    
    // Check for specific features
    if (hwcap & HWCAP_ASIMD) printf("NEON/ASIMD: supported\n");
    if (hwcap & HWCAP_FPHP) printf("FP16: supported\n");
    if (hwcap & HWCAP_ASIMDDP) printf("Dot Product: supported\n");
    if (hwcap & HWCAP_SVE) printf("SVE: supported\n");
    if (hwcap2 & HWCAP2_SVE2) printf("SVE2: supported\n");
    if (hwcap2 & HWCAP2_I8MM) printf("I8MM: supported\n");
    if (hwcap2 & HWCAP2_BF16) printf("BF16: supported\n");
    
    return 0;
}
```

```bash
# In Termux - check CPU features
cat /proc/cpuinfo | grep Features
```

### Feature Availability by Architecture

| Feature | ARMv8.0 | ARMv8.2 | ARMv8.4 | ARMv8.6 | ARMv9.0 | ARMv9.2 |
|---------|---------|---------|---------|---------|---------|---------|
| NEON | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Dot Product | - | Opt | ✓ | ✓ | ✓ | ✓ |
| FP16 | - | Opt | ✓ | ✓ | ✓ | ✓ |
| SVE | - | Opt | Opt | Opt | Opt | Opt |
| I8MM | - | - | - | Opt | Opt | ✓ |
| BF16 | - | - | - | Opt | Opt | ✓ |
| SVE2 | - | - | - | - | ✓ | ✓ |
| SME | - | - | - | - | - | Opt |

---

## 6. Performance Tuning in Termux

### Termux Constraints

Operating within Termux on non-rooted devices has specific limitations:

| Aspect | Limitation | Workaround |
|--------|------------|------------|
| **Kernel access** | No modification possible | User-space optimizations only |
| **CPU governors** | Cannot change | Rely on Android's scheduler |
| **CPU affinity** | Limited without root | Use `taskset` where possible |
| **Memory limits** | Per-app limits | Optimize memory usage |
| **GPU compute** | No direct CUDA/OpenCL | Use CPU with NEON/SVE |

### CPU Affinity and Task Pinning

While limited without root, some affinity control is possible:

```bash
# Check number of CPUs
nproc

# View CPU topology
cat /sys/devices/system/cpu/cpu*/topology/core_id
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# Attempt to pin process (may require Shizuku)
taskset -c 0-3 ./my_program  # Run on CPUs 0-3
taskset -c 4-7 ./my_program  # Run on CPUs 4-7 (usually big cores)
```

### Shizuku Integration for Enhanced Control

Shizuku allows ADB-level privileges without root:

```bash
# Install Shizuku app from Play Store/GitHub
# Start Shizuku service via wireless debugging

# In Termux, use rish for elevated commands
pkg install rish

# Run commands with ADB privileges
rish -c "settings get system accelerometer_rotation"
rish -c "am force-stop com.background.app"
```

### Compile-Time Optimizations for Termux

```bash
# Install build tools
pkg install clang cmake make

# Set optimal compiler flags
export CFLAGS="-O3 -mcpu=native -flto"
export CXXFLAGS="-O3 -mcpu=native -flto"
export LDFLAGS="-flto"

# Or detect and set for specific CPU
CPU_MODEL=$(cat /proc/cpuinfo | grep "CPU part" | head -1 | awk '{print $4}')
case $CPU_MODEL in
    0xd44) MCPU="cortex-x4" ;;      # Cortex-X4
    0xd43) MCPU="cortex-a715" ;;    # Cortex-A715
    0xd41) MCPU="cortex-a78" ;;     # Cortex-A78
    0xd0d) MCPU="cortex-a77" ;;     # Cortex-A77
    0xd0b) MCPU="cortex-a76" ;;     # Cortex-A76
    0xd05) MCPU="cortex-a55" ;;     # Cortex-A55
    *)     MCPU="native" ;;
esac
export CFLAGS="-O3 -mcpu=$MCPU -flto"
```

### Memory Optimization Strategies

```bash
# Monitor memory usage
watch -n 1 free -h

# Use htop for detailed view
pkg install htop
htop

# Set Python to optimize memory
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000

# For NumPy - limit threads to prevent memory bloat
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### Thermal Management Script

```bash
#!/bin/bash
# thermal_monitor.sh - Pause compute when too hot

TEMP_LIMIT=42  # Celsius

while true; do
    if command -v termux-battery-status &> /dev/null; then
        TEMP=$(termux-battery-status | jq -r '.temperature')
    else
        # Fallback to thermal zone
        TEMP=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
        TEMP=$((TEMP / 1000))
    fi
    
    if [ "$TEMP" -gt "$TEMP_LIMIT" ]; then
        echo "Temperature $TEMP°C exceeds limit. Pausing..."
        pkill -STOP -f "python|llama|compute"
        sleep 30
        pkill -CONT -f "python|llama|compute"
    fi
    
    sleep 10
done
```

### llama.cpp Compilation for Termux

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Detect optimal flags
NPROC=$(nproc)
MARCH="armv8.2-a+dotprod+fp16"  # Adjust based on device

# Compile with optimizations
make clean
make -j$NPROC \
    LLAMA_NO_METAL=1 \
    LLAMA_NO_CUDA=1 \
    CFLAGS="-O3 -march=$MARCH -flto" \
    CXXFLAGS="-O3 -march=$MARCH -flto"

# Verify NEON is enabled
./llama-cli --version 2>&1 | grep -i features
```

### Python Numeric Library Optimization

```bash
# Install NumPy with optimized BLAS
pip install numpy --no-binary :all: \
    --config-settings=setup-args="-Dallow-noblas=true"

# Or use pre-compiled optimized wheels
pip install numpy scipy

# For PyTorch - use CPU-optimized build
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Enable OpenMP threading
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

---

## 7. Device-Specific Optimizations

### Flagship Devices (2023-2024)

#### Samsung Galaxy S24 Ultra / S24+ (Snapdragon 8 Gen 3)

```bash
# CPU Configuration: 1x X4 + 3x A720 + 2x A720 + 2x A520
# Optimal flags
CFLAGS="-O3 -mcpu=cortex-x4 -march=armv9.2-a+sve2+i8mm+bf16 -flto"

# Pin to performance cores (4-7 are typically big cores)
taskset -c 4-7 ./compute_intensive_task
```

#### Google Pixel 8 Pro (Tensor G3)

```bash
# CPU Configuration: 1x X3 + 4x A715 + 4x A510
# Optimal flags
CFLAGS="-O3 -mcpu=cortex-x3 -march=armv9-a+sve2 -flto"

# Tensor G3 has enhanced ML accelerators - prefer CPU for inference
```

#### OnePlus 12 (Snapdragon 8 Gen 3)

```bash
# Same as S24 Ultra - Snapdragon 8 Gen 3
CFLAGS="-O3 -mcpu=cortex-x4 -march=armv9.2-a+sve2+i8mm+bf16 -flto"
```

### Flagship Devices (2021-2022)

#### Samsung Galaxy S22 / S23 (Snapdragon 8 Gen 1/2)

```bash
# S22: 1x X2 + 3x A710 + 4x A510
# S23: 1x X3 + 2x A715 + 2x A710 + 3x A510
CFLAGS="-O3 -mcpu=cortex-x2 -march=armv9-a+sve2 -flto"  # S22
CFLAGS="-O3 -mcpu=cortex-x3 -march=armv9-a+sve2 -flto"  # S23
```

#### Google Pixel 6/7 (Tensor G1/G2)

```bash
# Tensor G1/G2: 2x X1 + 2x A76/A78 + 4x A55
CFLAGS="-O3 -mcpu=cortex-x1 -march=armv8.2-a+dotprod+fp16 -flto"
```

### Mid-Range Devices

#### Samsung Galaxy A54 (Exynos 1380)

```bash
# 4x A78 + 4x A55
CFLAGS="-O2 -mcpu=cortex-a78 -march=armv8.2-a+dotprod -flto"
```

#### Xiaomi Redmi Note 12 Pro (Dimensity 1080)

```bash
# 2x A78 + 6x A55
CFLAGS="-O2 -mcpu=cortex-a78 -march=armv8.2-a+dotprod -flto"
```

#### Google Pixel 6a (Tensor)

```bash
# 2x X1 + 2x A76 + 4x A55
CFLAGS="-O2 -mcpu=cortex-a76 -march=armv8.2-a+dotprod -flto"
```

### Budget Devices

#### Samsung Galaxy A14 (Helio G80)

```bash
# 2x A75 + 6x A55
CFLAGS="-O2 -mcpu=cortex-a75 -march=armv8.2-a -flto"
```

#### Xiaomi Redmi 10 (Helio G88)

```bash
# 2x A75 + 6x A55
CFLAGS="-O2 -mcpu=cortex-a75 -march=armv8.2-a -flto"
```

---

## 8. Benchmarking & Profiling

### CPU Information Script

```bash
#!/bin/bash
# cpu_info.sh - Detailed CPU information

echo "=== CPU Architecture ==="
uname -m

echo -e "\n=== CPU Cores ==="
nproc

echo -e "\n=== CPU Model ==="
cat /proc/cpuinfo | grep "model name" | head -1

echo -e "\n=== CPU Features ==="
cat /proc/cpuinfo | grep Features | head -1

echo -e "\n=== CPU Frequencies ==="
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/; do
    if [ -f "$cpu/scaling_cur_freq" ]; then
        CORE=$(basename $(dirname $cpu))
        FREQ=$(cat $cpu/scaling_cur_freq)
        echo "$CORE: $((FREQ/1000)) MHz"
    fi
done

echo -e "\n=== Topology ==="
echo "Core mapping:"
for cpu in /sys/devices/system/cpu/cpu*/topology/; do
    if [ -d "$cpu" ]; then
        CORE=$(basename $(dirname $cpu))
        CLUSTER=$(cat $cpu/physical_package_id 2>/dev/null || echo "N/A")
        echo "$CORE -> Cluster $CLUSTER"
    fi
done
```

### Simple Benchmark Script

```bash
#!/bin/bash
# simple_bench.sh - Quick performance test

echo "=== Single-Core Performance ==="
time (for i in $(seq 1 1000000); do :; done)

echo -e "\n=== Multi-Core Performance ==="
time (
    for core in $(seq 0 $(($(nproc)-1))); do
        (for i in $(seq 1 1000000); do :; done) &
    done
    wait
)

echo -e "\n=== Memory Bandwidth (approximate) ==="
if command -v dd &> /dev/null; then
    dd if=/dev/zero of=/dev/null bs=1M count=1000 2>&1 | tail -1
fi
```

### Python Benchmark

```python
#!/usr/bin/env python3
# benchmark.py - Comprehensive benchmark

import time
import numpy as np
from multiprocessing import cpu_count

def cpu_intensive(n=10_000_000):
    """Simple CPU-bound task"""
    total = 0
    for i in range(n):
        total += i * i
    return total

def numpy_benchmark(size=1000):
    """NumPy SIMD benchmark"""
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    
    start = time.time()
    c = np.dot(a, b)
    elapsed = time.time() - start
    
    gflops = (2 * size**3) / elapsed / 1e9
    return gflops

def main():
    print(f"CPU Cores: {cpu_count()}")
    
    # Pure Python benchmark
    print("\n=== Pure Python Benchmark ===")
    start = time.time()
    cpu_intensive()
    print(f"Time: {time.time() - start:.2f}s")
    
    # NumPy benchmark (tests SIMD/NEON)
    print("\n=== NumPy SIMD Benchmark ===")
    gflops = numpy_benchmark()
    print(f"Performance: {gflops:.2f} GFLOPS")
    
    # Memory bandwidth test
    print("\n=== Memory Bandwidth ===")
    size = 100 * 1024 * 1024  # 100 MB
    arr = np.zeros(size // 8, dtype=np.float64)
    
    start = time.time()
    arr[:] = 1.0
    arr *= 2.0
    elapsed = time.time() - start
    
    bandwidth = (size * 2) / elapsed / 1e9  # GB/s
    print(f"Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    main()
```

### Profiling with perf (if available)

```bash
# Install perf (may require specific kernel support)
pkg install linux-tools-common

# Profile CPU cycles
perf stat -e cycles,instructions,cache-misses ./my_program

# Record execution profile
perf record -g ./my_program
perf report
```

---

## 9. Quick Reference Tables

### Architecture to `-march` Mapping

| Device Year | Common SoCs | Architecture | `-march` Value |
|-------------|-------------|--------------|----------------|
| 2024+ | 8 Gen 3, Dimensity 9300 | ARMv9.2-A | `armv9.2-a+sve2+i8mm+bf16` |
| 2022-2023 | 8 Gen 1/2, Dimensity 9200 | ARMv9.0-A | `armv9-a+sve2` |
| 2020-2022 | 888, 8cx, Tensor G1/G2 | ARMv8.2-A | `armv8.2-a+dotprod+fp16` |
| 2018-2020 | 855, 845, Dimensity 1000 | ARMv8.2-A | `armv8.2-a+dotprod` |
| 2017-2018 | 835, 660 | ARMv8.0-A | `armv8-a` |
| Pre-2017 | 820, 810 | ARMv8.0-A | `armv8-a` |

### CPU Part Number to Core Mapping

| CPU Part (hex) | Core Name | Architecture |
|----------------|-----------|--------------|
| 0xd44 | Cortex-X4 | ARMv9.2-A |
| 0xd43 | Cortex-A715 | ARMv9.0-A |
| 0xd42 | Cortex-A510 | ARMv9.0-A |
| 0xd41 | Cortex-A78 | ARMv8.2-A |
| 0xd40 | Cortex-X2 | ARMv9.0-A |
| 0xd0d | Cortex-A77 | ARMv8.2-A |
| 0xd0c | Neoverse N1 | ARMv8.2-A |
| 0xd0b | Cortex-A76 | ARMv8.2-A |
| 0xd0a | Cortex-A75 | ARMv8.2-A |
| 0xd09 | Cortex-A73 | ARMv8.0-A |
| 0xd08 | Cortex-A72 | ARMv8.0-A |
| 0xd07 | Cortex-A57 | ARMv8.0-A |
| 0xd05 | Cortex-A55 | ARMv8.2-A |
| 0xd04 | Cortex-A35 | ARMv8.0-A |
| 0xd03 | Cortex-A53 | ARMv8.0-A |

### Optimization Flag Quick Reference

```bash
# Safe, portable optimization (works on all ARMv8.2+ devices)
CFLAGS="-O2 -march=armv8.2-a -flto"

# Balanced performance (most 2020+ flagships)
CFLAGS="-O3 -march=armv8.2-a+dotprod+fp16 -flto"

# Maximum performance (2022+ flagships, ARMv9)
CFLAGS="-O3 -march=armv9-a+sve2 -flto -ffast-math"

# Native detection (auto-detect from running CPU)
CFLAGS="-O3 -mcpu=native -flto"

# Size-optimized (constrained devices)
CFLAGS="-Os -march=armv8-a -ffunction-sections -fdata-sections"
```

### Performance Expectations

| Device Class | Example SoC | 7B Model (Q4) | 13B Model (Q4) | Optimal Workers |
|--------------|-------------|---------------|----------------|-----------------|
| Ultra Flagship | 8 Gen 3 | 15-25 tok/s | 8-12 tok/s | 4-6 |
| Flagship | 8 Gen 1/2 | 10-18 tok/s | 5-9 tok/s | 4 |
| Upper Mid | 778G | 6-12 tok/s | 3-6 tok/s | 4 |
| Mid-Range | 695 | 4-8 tok/s | 2-4 tok/s | 2-4 |
| Budget | Helio G80 | 2-5 tok/s | N/A | 2 |

---

## Conclusion

Optimizing for ARM Cortex processors in Termux requires understanding:

1. **Your device's SoC** - Know which Cortex cores you have
2. **Architecture features** - Enable relevant extensions (dotprod, fp16, i8mm)
3. **Compiler flags** - Use `-mcpu` or `-march` appropriately
4. **Termux constraints** - Focus on user-space optimizations
5. **Thermal management** - Monitor temperature during sustained workloads

**Key Takeaways:**

- Use `-mcpu=native` for auto-detection when unsure
- `-march=armv8.2-a+dotprod` works on most 2018+ devices
- ARMv9 devices (2022+) benefit from `+sve2` and `+i8mm`
- Always test performance with your specific workload
- Monitor thermals - mobile chips throttle under sustained load

For distributed LLM inference with EXO, focus on:
- Maximizing per-device throughput with proper compiler flags
- Using efficiency cores for background tasks
- Implementing thermal throttling to maintain sustained performance
- Leveraging the Dot Product and FP16 extensions for quantized models

---

## References

1. [ARM Developer - Cortex-A Processors](https://developer.arm.com/Processors/Cortex-A)
2. [GCC ARM Options](https://gcc.gnu.org/onlinedocs/gcc/ARM-Options.html)
3. [LLVM ARM Target Features](https://llvm.org/docs/HowToBuildOnARM.html)
4. [ARM Architecture Reference Manual](https://developer.arm.com/documentation)
5. [Qualcomm Snapdragon Specs](https://www.qualcomm.com/snapdragon)
6. [MediaTek Dimensity Specs](https://www.mediatek.com/products/smartphones)
7. [llama.cpp ARM Optimizations](https://github.com/ggerganov/llama.cpp)

---

*This guide is maintained as part of the EXO project documentation. For updates and contributions, see the [main repository](https://github.com/exo-explore/exo).*

