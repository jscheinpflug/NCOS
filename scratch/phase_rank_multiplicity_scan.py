#!/usr/bin/env python3
"""Scan phase-only cancellation structure vs multiplicity.

For n-point tree amplitudes with orderings (fix leg 1), test whether there is
any momentum-independent linear relation among ordering phases.

If the sampled phase matrix has full column rank, phases alone are insufficient
for generic cancellation and dynamical kernel identities are required.
"""

from __future__ import annotations

import cmath
import itertools
import math
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

Order = Tuple[int, ...]
Momentum = Tuple[float, float]  # (k_+, k_-)



def random_momenta(n: int, rng: random.Random) -> Dict[int, Momentum]:
    ks: Dict[int, Momentum] = {}
    sp = 0.0
    sm = 0.0
    for i in range(1, n):
        kp = rng.uniform(-2.0, 2.0)
        km = rng.uniform(-2.0, 2.0)
        ks[i] = (kp, km)
        sp += kp
        sm += km
    ks[n] = (-sp, -sm)
    return ks



def wedge(k: Momentum, q: Momentum, alpha_e: float = 1.0) -> float:
    kp, km = k
    qp, qm = q
    theta = 2.0 * math.pi * alpha_e
    return 0.5 * theta * (km * qp - kp * qm)



def phase_for_order(order: Order, ks: Dict[int, Momentum]) -> complex:
    total = 0.0
    m = len(order)
    for p in range(m):
        for q in range(p + 1, m):
            total += wedge(ks[order[p]], ks[order[q]])
    return cmath.exp(-0.5j * total)



def phase_matrix(n: int, samples: int, seed: int = 101) -> Tuple[np.ndarray, Sequence[Order]]:
    orderings: Sequence[Order] = tuple((1,) + perm for perm in itertools.permutations(range(2, n + 1)))
    rng = random.Random(seed + n)
    rows: List[List[complex]] = []
    for _ in range(samples):
        ks = random_momenta(n, rng)
        rows.append([phase_for_order(o, ks) for o in orderings])
    return np.array(rows, dtype=np.complex128), orderings



def rank_and_nullity(M: np.ndarray, tol: float = 1e-10) -> Tuple[int, int]:
    s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    rank = int(np.sum(s > tol))
    nullity = M.shape[1] - rank
    return rank, nullity



def main() -> None:
    print("Phase-only multiplicity scan (fix leg 1 in ordering basis)")
    print("n   cols(orderings)   samples   rank   nullity   mean|sum_phases|   mean/sqrt(cols)")

    configs = {
        4: 32,   # cols=6
        5: 90,   # cols=24
        6: 260,  # cols=120
    }

    for n, samples in configs.items():
        M, orderings = phase_matrix(n=n, samples=samples)
        rank, nullity = rank_and_nullity(M)
        phase_sums = np.abs(np.sum(M, axis=1))
        mean_sum = float(np.mean(phase_sums))
        random_walk_ratio = mean_sum / math.sqrt(len(orderings))
        print(
            f"{n:1d}   {len(orderings):14d}   {samples:7d}   {rank:4d}   {nullity:7d}   {mean_sum:14.6f}   {random_walk_ratio:16.6f}"
        )

    print(
        "\nInterpretation:\n"
        "If rank reaches the number of ordering columns (nullity 0),\n"
        "there is no momentum-independent phase-only null vector at that n.\n"
        "Then simplification must come from phase + kernel identities together."
    )


if __name__ == "__main__":
    main()
