#!/usr/bin/env python3
"""Quantify how NCOS phase cancellation degrades when ordered kernels differ."""

from __future__ import annotations

import cmath
import math
import random
from typing import List


def phase_triplet() -> List[complex]:
    return [1.0 + 0.0j, cmath.exp(2j * math.pi / 3), cmath.exp(-2j * math.pi / 3)]


def main() -> None:
    random.seed(23)
    eta = phase_triplet()
    base_kernel = 0.7 - 0.4j

    print("Phase sum:", sum(eta))
    print("Base cancellation amplitude:", sum(e * base_kernel for e in eta))

    eps_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    trials = 400

    print("\n eps        mean|A|      mean|A|/eps")
    for eps in eps_values:
        acc = 0.0
        for _ in range(trials):
            # Random mismatch among ordered kernels with size ~eps.
            d = []
            for _j in range(3):
                re = random.uniform(-1.0, 1.0)
                im = random.uniform(-1.0, 1.0)
                d.append(eps * (re + 1j * im))
            kernels = [base_kernel + d[j] for j in range(3)]
            A = sum(eta[j] * kernels[j] for j in range(3))
            acc += abs(A)
        mean_abs = acc / trials
        print(f"{eps:8.1e}  {mean_abs:10.4e}  {mean_abs/eps:10.4e}")

    print(
        "\nInterpretation:\n"
        "Residual amplitude scales linearly with ordering-kernel mismatch.\n"
        "So phase cancellation is stable only if a dynamical identity keeps\n"
        "ordered kernels equal/related (contour-pulling/cohomological mechanism)."
    )


if __name__ == "__main__":
    main()
