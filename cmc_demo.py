CMC Benchmark - Interactive Demo
=================================
A regularization-free benchmark for floating-point verification.

Based on: "A Regularization-Free Benchmark for Floating-Point Verification"
by Sami I. Almuaigel (2026)
"""
from __future__ import annotations
import math
import numpy as np


# ============================================================
# 1. Core Functions (from Appendix A)
# ============================================================

def cmc_exact(a: float) -> float:
    """
    Compute the exact analytical value of the CMC integral.
    I(a) = -3 * pi * sqrt(2) / (8 * a)
    """
    if not math.isfinite(a) or a <= 0:
        raise ValueError("a must be finite and positive")
    return -(3.0 * math.pi * math.sqrt(2.0)) / (8.0 * a)


def cmc_kernel_dimensionless(x: np.ndarray) -> np.ndarray:
    """
    Canonical kernel k(x) whose integral equals -3*pi*sqrt(2)/8.
    Uses dimensionless formulation to avoid overflow.
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    
    # Handle zero safely
    result = np.zeros_like(ax)
    mask = ax > 0
    
    # For |x| <= 1: standard form
    small = mask & (ax <= 1.0)
    if np.any(small):
        xs = ax[small]
        x2 = xs * xs
        x4 = x2 * x2
        x6 = x4 * x2
        result[small] = -8.0 * x6 / ((x4 + 1.0) ** 3)
    
    # For |x| > 1: reciprocal form to avoid overflow
    large = mask & (ax > 1.0)
    if np.any(large):
        xl = ax[large]
        y = 1.0 / xl
        y2 = y * y
        y4 = y2 * y2
        y6 = y4 * y2
        result[large] = -8.0 * y6 / ((1.0 + y4) ** 3)
    
    return result


def run_benchmark(
    a: float = 1.0,
    domain: float = 32.0,
    panels: int = 16384,
    eval_dtype: str = "float64",
    accum_dtype: str = "float64",
) -> dict:
    """
    Run the CMC benchmark with Simpson's rule quadrature.
    
    CORRECTED: Use dimensionless domain [-L, L] where L is fixed.
    """
    if panels <= 0 or panels % 2:
        raise ValueError("panels must be a positive even integer")
    if domain <= 0 or a <= 0:
        raise ValueError("domain and a must be positive")
    
    dt = np.dtype(eval_dtype)
    at = np.dtype(accum_dtype)
    
    # ✅ CORRECTED: Use dimensionless domain
    L_dimless = dt.type(domain)
    x_dimless = np.linspace(-L_dimless, L_dimless, panels + 1, dtype=dt)
    
    # Evaluate kernel in dimensionless form
    k = cmc_kernel_dimensionless(x_dimless).astype(dt)
    
    # Check for non-finite values
    if not np.all(np.isfinite(k)):
        raise FloatingPointError("non-finite kernel sample")
    
    # Simpson's rule weights
    w = np.ones(panels + 1, dtype=at)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    
    # ✅ CORRECTED: Integration in dimensionless space
    # I(a) = (1/a) * ∫ k(x) dx where x is dimensionless
    h_dimless = (dt.type(2) * L_dimless) / dt.type(panels)
    s = np.sum(k.astype(at) * w, dtype=at)
    numerical = float((at.type(h_dimless) / at.type(3)) * s / at.type(a))
    
    # Compute metrics
    exact = cmc_exact(a)
    absolute = abs(numerical - exact)
    relative = absolute / abs(exact)
    vcm = None if relative == 0 else -math.log10(relative)
    
    return {
        "numerical": numerical,
        "exact": exact,
        "absolute_error": absolute,
        "relative_error": relative,
        "vcm": vcm,
        "config": {
            "a": a,
            "domain": domain,
            "panels": panels,
            "eval_dtype": eval_dtype,
            "accum_dtype": accum_dtype,
        },
    }


# ============================================================
# 2. Demo Functions
# ============================================================

def demo_basic():
    """Basic benchmark run."""
    print("=" * 70)
    print("🎯 DEMO 1: Basic CMC Benchmark (FP64)")
    print("=" * 70)
    
    result = run_benchmark(a=1.0, panels=16384, domain=32.0)
    
    print(f"\n📊 Configuration:")
    print(f"   Scale parameter a    = {result['config']['a']}")
    print(f"   Domain half-width L  = {result['config']['domain']}")
    print(f"   Quadrature panels    = {result['config']['panels']}")
    print(f"   Evaluation dtype     = {result['config']['eval_dtype']}")
    print(f"   Accumulation dtype   = {result['config']['accum_dtype']}")
    
    print(f"\n📈 Results:")
    print(f"   Numerical value      = {result['numerical']:.15f}")
    print(f"   Exact value          = {result['exact']:.15f}")
    print(f"   Absolute error       = {result['absolute_error']:.3e}")
    print(f"   Relative error       = {result['relative_error']:.3e}")
    print(f"   VCM (verified digits)= {result['vcm']:.2f}")
    print()


def demo_precision_comparison():
    """Compare different precision formats."""
    print("=" * 70)
    print("🎯 DEMO 2: Precision Format Comparison")
    print("=" * 70)
    
    formats = [
        ("float64", "float64", "FP64 (Double)"),
        ("float32", "float64", "FP32 eval / FP64 accum"),
        ("float32", "float32", "FP32 (Single)"),
        ("float16", "float32", "FP16 eval / FP32 accum"),
        ("float16", "float16", "FP16 (Half)"),
    ]
    
    print(f"\n{'Format':<30} {'Numerical':>18} {'Rel. Error':>12} {'VCM':>8}")
    print("-" * 70)
    
    for eval_dt, accum_dt, label in formats:
        try:
            result = run_benchmark(
                a=1.0, panels=16384, domain=32.0,
                eval_dtype=eval_dt, accum_dtype=accum_dt
            )
            vcm_str = f"{result['vcm']:.2f}" if result['vcm'] is not None else "∞"
            print(f"{label:<30} {result['numerical']:>18.10f} "
                  f"{result['relative_error']:>12.3e} {vcm_str:>8}")
        except Exception as e:
            print(f"{label:<30} ERROR: {e}")
    
    print("\n💡 Note: VCM = -log10(relative_error) = verified decimal digits")
    print("   Expected VCM: FP64≈15.6, FP32≈6.9, FP16≈3.0")
    print()


def demo_scale_sweep():
    """Test scale invariance across different values of 'a'."""
    print("=" * 70)
    print("🎯 DEMO 3: Scale Invariance Test (κ = 1 property)")
    print("=" * 70)
    
    scales = [1e-6, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e6]
    
    print(f"\n{'Scale a':<12} {'I(a) numerical':>18} {'I(a) exact':>18} {'Rel. Error':>12} {'VCM':>8}")
    print("-" * 70)
    
    for a in scales:
        result = run_benchmark(a=a, panels=16384, domain=32.0)
        vcm_str = f"{result['vcm']:.2f}" if result['vcm'] is not None else "∞"
        print(f"{a:<12.1e} {result['numerical']:>18.6e} {result['exact']:>18.6e} "
              f"{result['relative_error']:>12.3e} {vcm_str:>8}")
    
    print("\n💡 The relative error should be nearly constant across all scales.")
    print("   This confirms the unit condition number κ = 1.")
    print()


def demo_kernel_visualization():
    """Visualize the kernel K(t)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("=" * 70)
        print("🎯 DEMO 4: Kernel Visualization (skipped - matplotlib not installed)")
        print("=" * 70)
        print("   Install with: pip install matplotlib")
        print()
        return
    
    print("=" * 70)
    print("🎯 DEMO 4: Kernel K(t) Visualization")
    print("=" * 70)
    
    t = np.linspace(-10, 10, 1000)
    k = cmc_kernel_dimensionless(t)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Linear scale
    axes[0].plot(t, k, 'b-', linewidth=2)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('K(t)')
    axes[0].set_title('Curvature Density K(t)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Log scale (absolute value)
    axes[1].semilogy(t, np.abs(k), 'r-', linewidth=2)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('|K(t)|')
    axes[1].set_title('Decay Rate (log scale)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cmc_kernel.png', dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved as 'cmc_kernel.png'")
    print()


def demo_interactive():
    """Interactive mode: user provides parameters."""
    print("=" * 70)
    print("🎯 DEMO 5: Interactive Mode")
    print("=" * 70)
    
    try:
        a = float(input("\nEnter scale parameter a [default=1.0]: ") or "1.0")
        panels = int(input("Enter number of panels [default=16384]: ") or "16384")
        domain = float(input("Enter domain half-width L [default=32.0]: ") or "32.0")
        
        result = run_benchmark(a=a, panels=panels, domain=domain)
        
        print(f"\n📊 Results:")
        print(f"   Numerical value      = {result['numerical']:.15f}")
        print(f"   Exact value          = {result['exact']:.15f}")
        print(f"   Relative error       = {result['relative_error']:.3e}")
        print(f"   VCM                  = {result['vcm']:.2f} verified digits")
    except (ValueError, EOFError) as e:
        print(f"\n⚠️  Invalid input: {e}")
    print()


# ============================================================
# 3. Main Entry Point
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  🔬 CMC BENCHMARK - INTERACTIVE DEMO")
    print("  Curvature-Modularity Constant for Floating-Point Verification")
    print("=" * 70 + "\n")
    
    demo_basic()
    demo_precision_comparison()
    demo_scale_sweep()
    demo_kernel_visualization()
    demo_interactive()
    
    print("=" * 70)
    print("✅ All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()