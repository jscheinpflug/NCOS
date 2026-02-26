#!/usr/bin/env python3
"""Minimal NSR-style derivative ansatz with transverse meromorphic factors.

Pure-NumPy version (no sympy dependency).

Goal:
Clarify why transverse dependence does not by itself obstruct the mechanism,
provided the integrated vertex contributes a full BRST-exact derivative of the
entire correlator factor, not just the longitudinal KN piece.
"""

from __future__ import annotations

import numpy as np



def L(t: np.ndarray) -> np.ndarray:
    return ((t + 2.0) ** 2 * (3.0 - t)) / (5.0 + t)



def dL(t: np.ndarray) -> np.ndarray:
    # L = N/D with N=(t+2)^2(3-t), D=(5+t)
    N = (t + 2.0) ** 2 * (3.0 - t)
    D = 5.0 + t
    dN = (t + 2.0) * (4.0 - 3.0 * t)
    dD = 1.0
    return (dN * D - N * dD) / (D**2)



def T(t: np.ndarray) -> np.ndarray:
    return 1.0 + 0.6 / (t + 2.0) - (2.0 / 7.0) / (3.0 - t)



def dT(t: np.ndarray) -> np.ndarray:
    return -0.6 / (t + 2.0) ** 2 - (2.0 / 7.0) / (3.0 - t) ** 2



def integrate(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))



def main() -> None:
    t = np.linspace(0.0, 1.0, 20001)

    l = L(t)
    dl = dL(t)
    tt = T(t)
    dtt = dT(t)

    I_naive = dl * tt
    Delta = l * dtt
    I_full = I_naive + Delta

    naive_int = integrate(I_naive, t)
    delta_int = integrate(Delta, t)
    full_int = integrate(I_full, t)

    boundary = (L(np.array([1.0])) * T(np.array([1.0])))[0] - (L(np.array([0.0])) * T(np.array([0.0])))[0]

    print("Numeric decomposition on t in [0,1]:")
    print("Integral(I_naive)         =", f"{naive_int:.10f}")
    print("Integral(Delta)           =", f"{delta_int:.10f}")
    print("Integral(I_full)          =", f"{full_int:.10f}")
    print("Boundary (L*T)|_1-(L*T)|_0=", f"{boundary:.10f}")

    mismatch = abs(full_int - boundary)
    print("|Integral(I_full)-Boundary| =", f"{mismatch:.3e}")

    print(
        "\nInterpretation:\n"
        "Meromorphic transverse dependence is compatible with exact-derivative\n"
        "localization, but only if the integrated vertex contributes the full\n"
        "BRST-completed derivative structure. Omitting the transverse-derivative\n"
        "piece leaves a finite residual."
    )


if __name__ == "__main__":
    main()
