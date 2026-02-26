#!/usr/bin/env python3
"""Cylinder closed-channel threshold check from NCOS note formulas.

Implements the branch-point condition
  s_n (1-Etilde^2) = Etilde^2 K_T^2 + 4 n / alpha'
from the one-loop analysis in NCOS.tex.

Checks:
1) NCOS scaling alpha' = alpha_eff * (1-Etilde^2), Etilde -> 1^-.
2) Seiberg-Witten-like scaling alpha'->0 at fixed theta_E (thus Etilde~1/alpha').
"""

from __future__ import annotations

import math


def s_threshold(Etilde: float, alpha_p: float, Kt2: float, n: int) -> float:
    den = 1.0 - Etilde**2
    num = Etilde**2 * Kt2 + 4.0 * n / alpha_p
    return num / den


def ncos_scan() -> None:
    print("NCOS scaling: alpha' = alpha_eff * (1-Etilde^2), Etilde->1^-")
    alpha_eff = 1.0
    Kt2 = 1.0
    ns = [0, 1, 2]
    deltas = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]

    print("delta      Etilde      alpha'       s_n(n=0)      s_n(n=1)      s_n(n=2)")
    for d in deltas:
        E = math.sqrt(1.0 - d)
        ap = alpha_eff * d
        vals = [s_threshold(E, ap, Kt2, n) for n in ns]
        print(f"{d:7.1e}  {E:8.6f}   {ap:9.2e}   {vals[0]:11.4e}  {vals[1]:11.4e}  {vals[2]:11.4e}")


def sw_scan() -> None:
    print("\nSW-like scaling: alpha'->0 with fixed theta_E")
    # fixed theta_E=1 => Etilde = theta_E/(2*pi*alpha')
    theta_E = 1.0
    Kt2 = 1.0
    n = 1
    alphas = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]

    print("alpha'      Etilde         1-Etilde^2         s_n(n=1)")
    for ap in alphas:
        E = theta_E / (2.0 * math.pi * ap)
        den = 1.0 - E**2
        s = s_threshold(E, ap, Kt2, n)
        print(f"{ap:8.1e}   {E:11.4e}   {den:13.4e}   {s:12.4e}")


if __name__ == "__main__":
    ncos_scan()
    sw_scan()
