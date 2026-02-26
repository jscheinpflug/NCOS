#!/usr/bin/env python3
"""Kinematic channel counting for a 1+1 massive Regge tower.

Purpose:
Show that 2->3 production is generically kinematically allowed once
center-of-mass energy is high enough, so integrability (if present)
must come from dynamics, not kinematic prohibition.
"""

from __future__ import annotations

import math
from typing import List, Tuple



def mass(level: int, alpha_e: float = 1.0) -> float:
    # Simple toy spectrum m_n^2 = n / alpha'_e, n>=1.
    return math.sqrt(level / alpha_e)



def allowed_channels_2to3(e_cm: float, nmax: int) -> List[Tuple[int, int, int]]:
    chans: List[Tuple[int, int, int]] = []
    for a in range(1, nmax + 1):
        for b in range(a, nmax + 1):
            for c in range(b, nmax + 1):
                if mass(a) + mass(b) + mass(c) <= e_cm:
                    chans.append((a, b, c))
    return chans



def allowed_channels_2to2(e_cm: float, nmax: int) -> List[Tuple[int, int]]:
    chans: List[Tuple[int, int]] = []
    for a in range(1, nmax + 1):
        for b in range(a, nmax + 1):
            if mass(a) + mass(b) <= e_cm:
                chans.append((a, b))
    return chans



def main() -> None:
    nmax = 20
    e_values = [3.0, 4.0, 5.0, 6.0, 8.0]

    print("Toy spectrum: m_n = sqrt(n), n>=1")
    print("Level cutoff nmax =", nmax)
    print("\n E_cm    #2->2 channels   #2->3 channels")
    for e_cm in e_values:
        c22 = allowed_channels_2to2(e_cm, nmax)
        c23 = allowed_channels_2to3(e_cm, nmax)
        print(f"{e_cm:5.1f}    {len(c22):14d}   {len(c23):14d}")

    e_cm = 6.0
    c23 = allowed_channels_2to3(e_cm, nmax)
    print("\nExample 2->3 channels at E_cm=6:", c23[:12], "...")

    print(
        "\nInterpretation:\n"
        "2->3 channels proliferate quickly with energy in an infinite\n"
        "massive tower. Therefore, if NCOS massive-sector integrability\n"
        "exists, it must be enforced by amplitude zeros/selection rules,\n"
        "not by kinematics alone."
    )


if __name__ == "__main__":
    main()
