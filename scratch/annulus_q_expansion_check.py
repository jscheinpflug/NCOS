#!/usr/bin/env python3
"""
Check the q-expansion used in the nonplanar superstring annulus integrand.

We verify two things:
1) Symbolic low-order expansions of the Jacobi-product factors
     P_e(nu,q) = prod_{n>=1} (1 - 2 q^{2n} c + q^{4n}),
     P_o(nu,q) = prod_{n>=1} (1 - 2 q^{2n-1} c + q^{4n-2}),
   where c = cos(2*pi*nu).
2) Numerical consistency with theta-function identities used in hep-th/0311120:
     theta_1(pi nu, q) = 2 f(q^2) q^(1/4) sin(pi nu) P_e(nu,q),
     theta_4(pi nu, q) = f(q^2) P_o(nu,q),
   with f(q^2) = prod_{m>=1} (1 - q^{2m}).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple


# Polynomial in c representation: {power_of_c: integer_coeff}
Poly = Dict[int, int]
# Series in q with poly(c) coefficients: {power_of_q: Poly}
Series = Dict[int, Poly]


def poly_add(a: Poly, b: Poly) -> Poly:
    out = dict(a)
    for p, coeff in b.items():
        out[p] = out.get(p, 0) + coeff
        if out[p] == 0:
            del out[p]
    return out


def poly_mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for pa, ca in a.items():
        for pb, cb in b.items():
            p = pa + pb
            out[p] = out.get(p, 0) + ca * cb
            if out[p] == 0:
                del out[p]
    return out


def series_mul(a: Series, b: Series, qmax: int) -> Series:
    out: Series = {}
    for qa, pa in a.items():
        for qb, pb in b.items():
            qpow = qa + qb
            if qpow > qmax:
                continue
            cur = out.get(qpow, {})
            out[qpow] = poly_add(cur, poly_mul(pa, pb))
    return out


def poly_to_str(poly: Poly) -> str:
    if not poly:
        return "0"
    pieces = []
    for p in sorted(poly.keys(), reverse=True):
        coeff = poly[p]
        if coeff == 0:
            continue
        if p == 0:
            term = f"{coeff}"
        elif p == 1:
            term = "c" if coeff == 1 else ("-c" if coeff == -1 else f"{coeff}c")
        else:
            term = f"c^{p}" if coeff == 1 else ("-c^" + str(p) if coeff == -1 else f"{coeff}c^{p}")
        pieces.append(term)
    return " + ".join(pieces).replace("+ -", "- ")


def series_to_str(series: Series, qmax: int) -> str:
    pieces = []
    for qpow in range(qmax + 1):
        poly = series.get(qpow)
        if not poly:
            continue
        coeff_str = f"({poly_to_str(poly)})"
        if qpow == 0:
            pieces.append(coeff_str)
        elif qpow == 1:
            pieces.append(f"{coeff_str} q")
        else:
            pieces.append(f"{coeff_str} q^{qpow}")
    if not pieces:
        return "0"
    return " + ".join(pieces)


def factor_even(n: int) -> Series:
    # 1 - 2 c q^(2n) + q^(4n)
    return {0: {0: 1}, 2 * n: {1: -2}, 4 * n: {0: 1}}


def factor_odd(n: int) -> Series:
    # 1 - 2 c q^(2n-1) + q^(4n-2)
    return {0: {0: 1}, 2 * n - 1: {1: -2}, 4 * n - 2: {0: 1}}


def truncated_product_even(qmax: int) -> Series:
    out: Series = {0: {0: 1}}
    # factors with 2n <= qmax can contribute at or below qmax
    nmax = qmax // 2 + 1
    for n in range(1, nmax + 1):
        out = series_mul(out, factor_even(n), qmax)
    return out


def truncated_product_odd(qmax: int) -> Series:
    out: Series = {0: {0: 1}}
    # factors with (2n-1) <= qmax can contribute at or below qmax
    nmax = (qmax + 1) // 2 + 1
    for n in range(1, nmax + 1):
        out = series_mul(out, factor_odd(n), qmax)
    return out


def eval_poly(poly: Poly, c: float) -> float:
    return sum(coeff * (c ** p) for p, coeff in poly.items())


def eval_series(series: Series, q: float, c: float) -> float:
    return sum(eval_poly(poly, c) * (q ** qpow) for qpow, poly in series.items())


def theta1_series(nu: float, q: float, nsum: int = 200) -> float:
    # theta_1(pi*nu, q) in Jacobi nome convention
    acc = 0.0
    for n in range(nsum):
        acc += ((-1) ** n) * (q ** ((n + 0.5) ** 2)) * math.sin((2 * n + 1) * math.pi * nu)
    return 2.0 * acc


def theta4_series(nu: float, q: float, nsum: int = 200) -> float:
    # theta_4(pi*nu, q)
    acc = 1.0
    for n in range(1, nsum + 1):
        acc += 2.0 * ((-1) ** n) * (q ** (n * n)) * math.cos(2.0 * n * math.pi * nu)
    return acc


def f_q2(q: float, nprod: int = 400) -> float:
    # f(q^2) = prod_{m>=1}(1 - q^(2m))
    out = 1.0
    q2 = q * q
    for m in range(1, nprod + 1):
        out *= (1.0 - (q2 ** m))
    return out


def pe_product(nu: float, q: float, nprod: int = 200) -> float:
    c = math.cos(2.0 * math.pi * nu)
    out = 1.0
    for n in range(1, nprod + 1):
        out *= (1.0 - 2.0 * (q ** (2 * n)) * c + (q ** (4 * n)))
    return out


def po_product(nu: float, q: float, nprod: int = 200) -> float:
    c = math.cos(2.0 * math.pi * nu)
    out = 1.0
    for n in range(1, nprod + 1):
        out *= (1.0 - 2.0 * (q ** (2 * n - 1)) * c + (q ** (4 * n - 2)))
    return out


def check_theta_identities() -> None:
    print("Numerical check of theta-product identities")
    samples: Tuple[Tuple[float, float], ...] = (
        (0.17, 0.03),
        (0.31, 0.08),
        (0.41, 0.12),
    )
    for nu, q in samples:
        lhs1 = theta1_series(nu, q, nsum=300)
        rhs1 = 2.0 * f_q2(q, nprod=600) * (q ** 0.25) * math.sin(math.pi * nu) * pe_product(nu, q, nprod=250)
        rel1 = abs(lhs1 - rhs1) / max(1e-16, abs(lhs1))

        lhs4 = theta4_series(nu, q, nsum=300)
        rhs4 = f_q2(q, nprod=600) * po_product(nu, q, nprod=250)
        rel4 = abs(lhs4 - rhs4) / max(1e-16, abs(lhs4))

        print(f"nu={nu:.2f}, q={q:.2f}: relerr(theta1)={rel1:.3e}, relerr(theta4)={rel4:.3e}")


def check_truncated_series_accuracy() -> None:
    print("\nTruncated-series accuracy for P_e and P_o")
    qmax = 6
    pe_ser = truncated_product_even(qmax)
    po_ser = truncated_product_odd(qmax)
    samples: Tuple[Tuple[float, float], ...] = (
        (0.23, 0.04),
        (0.23, 0.08),
        (0.37, 0.05),
    )
    for nu, q in samples:
        c = math.cos(2.0 * math.pi * nu)
        pe_exact = pe_product(nu, q, nprod=250)
        po_exact = po_product(nu, q, nprod=250)
        pe_tr = eval_series(pe_ser, q, c)
        po_tr = eval_series(po_ser, q, c)
        err_pe = abs(pe_tr - pe_exact)
        err_po = abs(po_tr - po_exact)
        print(f"nu={nu:.2f}, q={q:.2f}: |P_e^tr-P_e|={err_pe:.3e}, |P_o^tr-P_o|={err_po:.3e}")


def main() -> None:
    qmax = 6
    pe_ser = truncated_product_even(qmax)
    po_ser = truncated_product_odd(qmax)

    print("Low-order q-expansion from Jacobi products (c = cos(2*pi*nu))")
    print(f"P_e(nu,q) up to q^{qmax}:")
    print(series_to_str(pe_ser, qmax))
    print(f"\nP_o(nu,q) up to q^{qmax}:")
    print(series_to_str(po_ser, qmax))

    check_theta_identities()
    check_truncated_series_accuracy()


if __name__ == "__main__":
    main()
