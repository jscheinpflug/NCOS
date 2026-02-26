#!/usr/bin/env python3
"""Toy model: branch-cut monodromy obstruction to NCOS phase cancellation.

This models three ordering sectors with critical phase vector eta and an
extra branch-cut monodromy factor exp(2*pi*i*m_r*nu). At nu=0 (D1-like
holomorphic case), cancellation is exact. For nu!=0, cancellation is
lifted.
"""

from __future__ import annotations

import cmath
import math
from typing import List



def phase_triplet() -> List[complex]:
    return [1.0 + 0.0j, cmath.exp(2j * math.pi / 3), cmath.exp(-2j * math.pi / 3)]



def amplitude(nu: float, m: List[int]) -> complex:
    eta = phase_triplet()
    return sum(eta[r] * cmath.exp(2j * math.pi * m[r] * nu) for r in range(3))



def main() -> None:
    m = [0, 1, -1]
    print("Monodromy winding numbers m:", m)
    print("eta sum:", sum(phase_triplet()))

    nus = [0.0, 1e-4, 3e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.25]
    print("\n   nu         |A(nu)|")
    for nu in nus:
        A = amplitude(nu, m)
        print(f"{nu:8.4g}    {abs(A):10.6e}")

    # Numerical slope at the NCOS point nu=0.
    h = 1e-6
    slope = abs(amplitude(h, m) - amplitude(0.0, m)) / h
    print("\nEstimated linear slope near nu=0:", f"{slope:.6f}")

    print(
        "\nInterpretation:\n"
        "- nu=0 reproduces exact phase cancellation.\n"
        "- Any nonzero branch exponent nu induces monodromy mismatch and\n"
        "  lifts the cancellation linearly for small nu.\n"
        "This is a toy analogue of how along-brane branch cuts obstruct\n"
        "the contour-pulling decoupling argument beyond the D1 setting."
    )


if __name__ == "__main__":
    main()
