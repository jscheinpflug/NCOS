#!/usr/bin/env python3
"""Scratch probe for mechanisms (3) and (4):

- Is a massless NCOS leg generically "transparent" in the braided sense?
- Does the zero-mode Weyl structure by itself induce obvious discrete superselection?
"""

from __future__ import annotations

import cmath
import math
import random


def wedge_lightcone(kp: float, km: float, qp: float, qm: float, alpha_e: float = 1.0) -> float:
    r"""k ∧ q for theta^{01}=2*pi*alpha_e.

    With x^\pm = x^0 +/- x^1 and p_0=(p_+ + p_-)/2, p_1=(p_+ - p_-)/2:
      k0 q1 - k1 q0 = (k_- q_+ - k_+ q_-)/2.
    """
    theta = 2.0 * math.pi * alpha_e
    return 0.5 * theta * (km * qp - kp * qm)


def main() -> None:
    random.seed(11)
    alpha_e = 1.0

    # One massless right-moving leg: k_- = 0, k_+ != 0.
    kp, km = 1.0, 0.0

    n = 5000
    near_braid_identity = 0  # |exp(-i/2 k∧q)-1| small
    near_monodromy_identity = 0  # |exp(-i k∧q)-1| small

    wedges = []
    for _ in range(n):
        qp = random.uniform(-5.0, 5.0)
        qm = random.uniform(-5.0, 5.0)
        w = wedge_lightcone(kp, km, qp, qm, alpha_e=alpha_e)
        wedges.append(w)
        braid = cmath.exp(-0.5j * w)
        mono = cmath.exp(-1.0j * w)
        if abs(braid - 1.0) < 1e-3:
            near_braid_identity += 1
        if abs(mono - 1.0) < 1e-3:
            near_monodromy_identity += 1

    print("Massless leg: k_+=1, k_-=0")
    print("Samples:", n)
    print("Fraction with braid phase ~ 1:", near_braid_identity / n)
    print("Fraction with monodromy ~ 1:", near_monodromy_identity / n)
    print(
        "Interpretation: transparency is non-generic; it occurs only on\n"
        "measure-zero loci in continuous kinematics."
    )

    # Show codim-1 special locus: q_- = 0 => wedge = 0 for right-moving massless k.
    checks = []
    for qp in [-3.0, -1.0, 0.7, 2.4]:
        w = wedge_lightcone(kp, km, qp, 0.0, alpha_e=alpha_e)
        checks.append(w)
    print("\nSpecial locus q_-=0 gives wedges:", checks)
    print("Interpretation: exact transparency exists on restricted chiral kinematics.")

    # A simple continuity diagnostic for "superselection":
    w_min, w_max = min(wedges), max(wedges)
    print("\nRange of sampled k∧q values:", (w_min, w_max))
    print(
        "Interpretation: wedge values are continuous in kinematics;\n"
        "no obvious discrete sector label emerges from this alone."
    )


if __name__ == "__main__":
    main()
