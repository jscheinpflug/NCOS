#!/usr/bin/env python3
"""Toy probe for one-loop (annulus-like) robustness of NCOS cancellation.

We model a second modulus t in addition to insertion variable tau.
Cancellation with critical phases is exact only if ordered sectors share the
same effective boundary kernel for each t.
"""

from __future__ import annotations

import cmath
import math
from typing import Callable, List

import numpy as np



def roots_of_unity_triplet() -> List[complex]:
    return [1.0 + 0.0j, cmath.exp(2j * math.pi / 3), cmath.exp(-2j * math.pi / 3)]



def trapz_complex(y: np.ndarray, x: np.ndarray) -> complex:
    return np.trapezoid(y, x)



def boundary_kernel_shared(t: float) -> complex:
    # Represents F(t,1)-F(t,0) from tau-derivative integral.
    return cmath.exp(0.7j * t) / (1.2 + t) - 1.0 / 1.2



def run_case(
    phase_fn: Callable[[int, float], complex],
    kernel_fn: Callable[[int, float], complex],
    t_grid: np.ndarray,
) -> complex:
    vals = []
    for t in t_grid:
        total = 0.0 + 0.0j
        for r in range(3):
            total += phase_fn(r, float(t)) * kernel_fn(r, float(t))
        vals.append(total)
    return trapz_complex(np.array(vals, dtype=np.complex128), t_grid)



def main() -> None:
    eta = roots_of_unity_triplet()
    t_grid = np.linspace(0.0, 1.0, 3001)

    # Case A: ideal annulus extension (shared kernel, t-independent phases)
    phase_A = lambda r, t: eta[r]
    kernel_A = lambda r, t: boundary_kernel_shared(t)
    A = run_case(phase_A, kernel_A, t_grid)

    # Case B: slight t-dependent phase drift (modular monodromy mismatch)
    phase_drift = [0.0, 0.7, -0.6]

    def phase_B(r: int, t: float) -> complex:
        return eta[r] * cmath.exp(1j * phase_drift[r] * t)

    kernel_B = kernel_A
    B = run_case(phase_B, kernel_B, t_grid)

    # Case C: slight ordering-dependent kernel deformation over modulus
    eps = [0.0, 0.03, -0.025]
    phase_C = phase_A

    def kernel_C(r: int, t: float) -> complex:
        return boundary_kernel_shared(t) + eps[r] * t * (1.0 - t)

    C = run_case(phase_C, kernel_C, t_grid)

    print("Annulus-modulus toy (integrate t in [0,1])")
    print("|A| shared-kernel/shared-phase case:", f"{abs(A):.6e}")
    print("|B| phase-drift case:", f"{abs(B):.6e}")
    print("|C| kernel-mismatch case:", f"{abs(C):.6e}")

    # Small-deformation scaling for phase drift strength lambda.
    lambdas = [1e-3, 3e-3, 1e-2, 3e-2]
    print("\nPhase-drift scaling (lambda rescales drift vector):")
    print("lambda      |A_lambda|    |A_lambda|/lambda")
    for lam in lambdas:
        def phase_l(r: int, t: float) -> complex:
            return eta[r] * cmath.exp(1j * lam * phase_drift[r] * t)

        Al = run_case(phase_l, kernel_A, t_grid)
        print(f"{lam:7.1e}   {abs(Al):10.4e}   {abs(Al)/lam:14.4e}")

    print(
        "\nInterpretation:\n"
        "A one-loop extension remains simple only if t-by-t ordered sectors\n"
        "retain the same effective boundary kernel and critical phase relation.\n"
        "Small modulus-dependent phase/kernel mismatches immediately lift\n"
        "the cancellation, typically linearly in the mismatch size."
    )


if __name__ == "__main__":
    main()
