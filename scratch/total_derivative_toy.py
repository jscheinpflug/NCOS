#!/usr/bin/env python3
"""Toy model for mechanisms (2) + (5): total-derivative and ordering cancellation."""

from __future__ import annotations

import cmath
import math
from typing import Callable, List


def phase_triplet() -> List[complex]:
    # Canonical 1 + exp(2pi i/3) + exp(-2pi i/3) = 0
    return [1.0 + 0.0j, cmath.exp(2j * math.pi / 3), cmath.exp(-2j * math.pi / 3)]


def boundary_integral_from_derivative(F: Callable[[float], complex]) -> complex:
    # int_0^1 dt dF/dt = F(1) - F(0)
    return F(1.0) - F(0.0)


def main() -> None:
    eta = phase_triplet()
    print("Phase sum:", sum(eta))

    # Case A: exact contour-pulling/equal-kernel situation.
    F_shared = lambda t: cmath.exp(0.3j * t) / (1.0 + t)
    I_shared = boundary_integral_from_derivative(F_shared)
    A_shared = sum(e * I_shared for e in eta)
    print("\nCase A: shared derivative kernel across orderings")
    print("  I_shared =", I_shared)
    print("  A_shared =", A_shared, "  |A_shared| =", abs(A_shared))

    # Case B: tiny ordering-dependent deformation (kernel mismatch).
    # Mimics failure of exact contour-pulling / non-identical ordered kernels.
    eps = [0.0, 0.04, -0.03]
    I_deformed = []
    for e in eps:
        # Linear piece ee*t gives boundary contribution ee, so kernels differ.
        F = lambda t, ee=e: cmath.exp(0.3j * t) / (1.0 + t) + ee * t
        I_deformed.append(boundary_integral_from_derivative(F))
    A_deformed = sum(eta[r] * I_deformed[r] for r in range(3))
    print("\nCase B: slightly different derivative kernels")
    print("  I_deformed =", I_deformed)
    print("  A_deformed =", A_deformed, "  |A_deformed| =", abs(A_deformed))

    print(
        "\nInterpretation:\n"
        "- Equal kernels + critical phases => exact cancellation.\n"
        "- Small kernel mismatch spoils cancellation.\n"
        "So phase structure alone is insufficient; contour/cohomological\n"
        "identities relating ordered kernels are essential."
    )


if __name__ == "__main__":
    main()
