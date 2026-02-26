#!/usr/bin/env python3
r"""Explicit nonplanar 2+2 cylinder analytic-structure check.

Model the singular piece of the nonplanar annulus after modular transform:
  A_n(s) ~ int_0^inf d\ell * \ell^{-1/2} exp[-\ell(\Lambda_n - s - i eps)]
       = sqrt(pi) (\Lambda_n - s - i eps)^(-1/2)
with
  \Lambda_n = 4n/alpha' + K\circ K/(4 pi^2 alpha'^2),
  K\circ K = theta_E^2 (s + K_T^2), theta_E = 2 pi alpha' Etilde.
Solving gives branch point
  s_n (1-Etilde^2) = Etilde^2 K_T^2 + 4n/alpha'.
"""

from __future__ import annotations

import cmath
import math



def s_threshold(alpha_p: float, etilde: float, kt2: float, n: int) -> float:
    return (etilde * etilde * kt2 + 4.0 * n / alpha_p) / (1.0 - etilde * etilde)



def amp_model(s: float, s_thr: float, eps: float = 1e-8) -> complex:
    return math.sqrt(math.pi) / cmath.sqrt((s_thr - s) - 1j * eps)



def threshold_discontinuity_demo() -> None:
    alpha_p = 0.03
    etilde = 0.97
    kt2 = 1.0
    n = 1
    s_thr = s_threshold(alpha_p, etilde, kt2, n)

    print("2+2 nonplanar threshold demo")
    print(f"alpha'={alpha_p}, Etilde={etilde}, K_T^2={kt2}, n={n}")
    print(f"s_threshold = {s_thr:.8f}")
    print("\n   s - s_thr      Re A_n(s)        Im A_n(s)        |A_n(s)|")
    for ds in [-1.0, -0.3, -0.1, -0.03, -0.01, 0.01, 0.03, 0.1, 0.3, 1.0]:
        s = s_thr + ds
        a = amp_model(s, s_thr)
        print(f"{ds:11.3e}   {a.real:13.6e}   {a.imag:13.6e}   {abs(a):13.6e}")

    # Disc across cut: A(s+i0)-A(s-i0) for s>s_thr.
    print("\nCut discontinuity scaling for s>s_thr")
    print("   ds>0         |Disc A_n|      |Disc A_n|*sqrt(ds)")
    for ds in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        s = s_thr + ds
        ap = math.sqrt(math.pi) / cmath.sqrt((s_thr - s) - 1j * 1e-12)
        am = math.sqrt(math.pi) / cmath.sqrt((s_thr - s) + 1j * 1e-12)
        disc = ap - am
        val = abs(disc)
        print(f"{ds:10.1e}   {val:12.6e}   {val*math.sqrt(ds):12.6e}")



def ncos_scaling_demo() -> None:
    print("\nNCOS scaling of nonplanar thresholds")
    alpha_eff = 1.0
    kt2 = 1.0
    n = 1
    print(" delta=1-E^2      alpha'         s_n")
    for delta in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        etilde = math.sqrt(1.0 - delta)
        alpha_p = alpha_eff * delta
        s_n = s_threshold(alpha_p, etilde, kt2, n)
        print(f"{delta:11.1e}   {alpha_p:11.2e}   {s_n:11.4e}")


if __name__ == "__main__":
    threshold_discontinuity_demo()
    ncos_scaling_demo()
