# CMC Benchmark

<div align="center">

**A Regularization-Free Benchmark for Floating-Point Verification**

[![Tests](https://github.com/YOUR_USERNAME/cmc-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/cmc-benchmark/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://YOUR_USERNAME.github.io/cmc-benchmark/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)




</div>

---

## 🎯 What is CMC?

The **Curvature-Modularity Constant (CMC)** is a mathematically rigorous benchmark for verifying floating-point computations across heterogeneous precision environments (FP64, FP32, FP16, BF16, FP8). Unlike classical benchmarks (LINPACK, oscillatory integrals), CMC provides:

| Property | Why It Matters |
|----------|----------------|
| ✅ **Closed-form ground truth** | Exact value $I(a) = -\frac{3\pi\sqrt{2}}{8a}$ — no circular dependency on higher-precision runs |
| ✅ **Perfect conditioning** | $\kappa = 1$ for all $a > 0$ — errors scale linearly with machine epsilon |
| ✅ **No catastrophic cancellation** | Sign-definite integrand ($K(t) < 0$ everywhere) |
| ✅ **Absolute convergence** | $O(\|t\|^{-6})$ decay — no regularization or analytic continuation needed |
| ✅ **Singularity-free** | Smooth integrand, no adaptive mesh refinement required |

This makes CMC the ideal tool for **isolating pure floating-point representation errors** from algorithmic instability.

---

## 🚀 Quick Start

### Installation

```bash
pip install cmc-benchmark
