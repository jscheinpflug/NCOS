#!/usr/bin/env python3
"""Scratch check for 4-point NCOS ordering phases on a D1.

Goal:
1) Test whether phase factors by themselves admit a universal (momentum-independent)
   null vector across a small basis of orderings.
2) Exhibit special kinematic points where phase-only cancellation can occur.
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Order = Tuple[int, int, int, int]


@dataclass(frozen=True)
class WedgeData:
    a: float  # w12
    b: float  # w13
    c: float  # w23

    def w(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        if i > j:
            return -self.w(j, i)

        base: Dict[Tuple[int, int], float] = {
            (1, 2): self.a,
            (1, 3): self.b,
            (2, 3): self.c,
            # k4 = -(k1+k2+k3) implies:
            (1, 4): -(self.a + self.b),
            (2, 4): self.a - self.c,
            (3, 4): self.b + self.c,
        }
        return base[(i, j)]


def ordered_phase(order: Order, wd: WedgeData) -> complex:
    """NCOS phase for one ordering.

    For positions p<q on the real boundary, sgn(t_p - t_q) = -1.
    """
    total = 0.0
    for p in range(len(order)):
        for q in range(p + 1, len(order)):
            total += wd.w(order[p], order[q])
    return cmath.exp(-0.5j * total)


def svd_null_dim(matrix: np.ndarray, tol: float = 1e-10) -> int:
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(s > tol))
    return matrix.shape[1] - rank


def scan_special_points(
    orderings: Sequence[Order], grid: Iterable[float]
) -> List[Tuple[WedgeData, complex]]:
    """Find points where sum of phase factors is (nearly) zero."""
    hits: List[Tuple[WedgeData, complex]] = []
    for a in grid:
        for b in grid:
            for c in grid:
                wd = WedgeData(a=a, b=b, c=c)
                s = sum(ordered_phase(o, wd) for o in orderings)
                if abs(s) < 1e-8:
                    hits.append((wd, s))
    return hits


def main() -> None:
    # Three representative orderings (channel basis) for 4-pt discussions.
    orderings: Sequence[Order] = (
        (1, 2, 3, 4),
        (1, 3, 2, 4),
        (1, 2, 4, 3),
    )

    # 1) Generic-rank test: do phases alone admit a universal null vector?
    random.seed(7)
    samples = []
    for _ in range(40):
        wd = WedgeData(
            a=random.uniform(-3.0, 3.0),
            b=random.uniform(-3.0, 3.0),
            c=random.uniform(-3.0, 3.0),
        )
        row = [ordered_phase(o, wd) for o in orderings]
        samples.append(row)

    M = np.array(samples, dtype=np.complex128)
    null_dim = svd_null_dim(M)
    print("Phase-only matrix shape:", M.shape)
    print("Phase-only nullspace dimension (generic sample):", null_dim)
    if null_dim == 0:
        print(
            "Result: no momentum-independent phase null vector generically.\n"
            "Interpretation: phase factors alone do not force cancellation."
        )
    else:
        print(
            "Result: found nontrivial nullspace in generic sample.\n"
            "Interpretation: check implementation/ordering basis."
        )

    # 2) Special kinematic points where phase-only cancellation does occur.
    grid = [k * math.pi / 3.0 for k in range(-3, 4)]
    hits = scan_special_points(orderings, grid)
    print("\nSpecial-point scan over a,b,c in multiples of pi/3")
    print("Number of exact/near hits with |sum phases| < 1e-8:", len(hits))
    if hits:
        wd, _ = hits[0]
        phases = [ordered_phase(o, wd) for o in orderings]
        print("Example hit:")
        print(f"  a={wd.a:.6f}, b={wd.b:.6f}, c={wd.c:.6f}")
        print("  phases =", [complex(round(p.real, 6), round(p.imag, 6)) for p in phases])
        print(
            "Interpretation: phase-only cancellation exists on special loci,\n"
            "but is non-generic without further kernel identities."
        )


if __name__ == "__main__":
    main()
