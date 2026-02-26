#!/usr/bin/env python3
"""5-point phase-only stress test for NCOS ordering cancellations.

Question:
Do Moyal phases alone produce a momentum-independent null vector across
all 5-point orderings (with physical 1+1 kinematics enforced)?
"""

from __future__ import annotations

import cmath
import itertools
import math
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

Order = Tuple[int, int, int, int, int]



def make_momenta(seed: int) -> Dict[int, Tuple[float, float]]:
    """Generate physical 1+1 momenta with momentum conservation.

    Returns k_i = (k_+, k_-) for i=1..5 with k5 = -sum_{1..4} k_i.
    """
    rng = random.Random(seed)
    ks: Dict[int, Tuple[float, float]] = {}
    sum_p = 0.0
    sum_m = 0.0
    for i in range(1, 5):
        kp = rng.uniform(-2.0, 2.0)
        km = rng.uniform(-2.0, 2.0)
        ks[i] = (kp, km)
        sum_p += kp
        sum_m += km
    ks[5] = (-sum_p, -sum_m)
    return ks



def wedge(k: Tuple[float, float], q: Tuple[float, float], alpha_e: float = 1.0) -> float:
    """k ∧ q for theta^{01}=2*pi*alpha_e in light-cone components."""
    kp, km = k
    qp, qm = q
    theta = 2.0 * math.pi * alpha_e
    return 0.5 * theta * (km * qp - kp * qm)



def ordered_phase(order: Order, ks: Dict[int, Tuple[float, float]]) -> complex:
    total = 0.0
    for p in range(len(order)):
        for q in range(p + 1, len(order)):
            i = order[p]
            j = order[q]
            total += wedge(ks[i], ks[j])
    return cmath.exp(-0.5j * total)



def svd_null_dim(matrix: np.ndarray, tol: float = 1e-10) -> int:
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(s > tol))
    return matrix.shape[1] - rank



def main() -> None:
    # Fix 1 first to remove cyclic redundancy in a simple way.
    orderings: Sequence[Order] = tuple((1,) + p for p in itertools.permutations((2, 3, 4, 5)))

    rows: List[List[complex]] = []
    n_samples = 180
    for seed in range(1000, 1000 + n_samples):
        ks = make_momenta(seed)
        rows.append([ordered_phase(o, ks) for o in orderings])

    M = np.array(rows, dtype=np.complex128)
    null_dim = svd_null_dim(M)

    phase_sums = np.sum(M, axis=1)
    mean_sum = float(np.mean(np.abs(phase_sums)))
    min_sum = float(np.min(np.abs(phase_sums)))

    print("5-pt ordering count:", len(orderings))
    print("Phase matrix shape:", M.shape)
    print("Nullspace dimension (phase-only, physical random kinematics):", null_dim)
    print("Mean |sum over all orderings of phases|:", f"{mean_sum:.6f}")
    print("Min  |sum over all orderings of phases| in sample:", f"{min_sum:.6f}")

    if null_dim == 0:
        print(
            "Interpretation: no universal phase-only null vector at 5 points;\n"
            "ordered-kernel identities are still required for generic cancellation."
        )
    else:
        print(
            "Interpretation: found a nontrivial nullspace; inspect ordering basis\n"
            "and kinematic sampling assumptions."
        )


if __name__ == "__main__":
    main()
