#!/usr/bin/env python3
"""Test the full n-point contour argument for massless decoupling.

The claim: for a massless right-mover at position tau_1, the FULL n-point
integrand (summed over all orderings with Moyal phases) is holomorphic
in tau_1 in the upper half-plane. Closing the contour in the UHP gives zero.

This script tests:
1. Whether the pairwise monodromy matching (pi*s_{1a} = k1^ka for all a)
   is sufficient to make the FULL integrand (product of KN factors times
   Moyal phase) holomorphic in tau_1.

2. An explicit numerical contour integral at 4-point and 5-point to verify
   vanishing.

3. The interplay between multiple insertion points: when tau_1 is continued
   to the UHP, the other insertion points remain on the real line. The
   product of |tau_1 - tau_a|^{s_{1a}} factors becomes a product of
   (tau_1 - tau_a)^{s_{1a}/2} * (tau_1* - tau_a)^{s_{1a}/2}.
   At the NCOS point, the Moyal phase converts this to:
   (tau_1 - tau_a)^{s_{1a}} for right-movers,
   which is holomorphic in tau_1.

4. For completeness, also test that for a massless LEFT-mover, closing
   in the LHP gives zero.
"""

from __future__ import annotations
import numpy as np
import cmath, math
from scipy.integrate import quad


def test_4pt_contour():
    """Compute the 4-point NCOS amplitude by explicit contour integration."""
    print("=" * 70)
    print("TEST 1: 4-POINT CONTOUR INTEGRAL")
    print("=" * 70)

    alpha_e = 1.0
    theta01 = 2 * math.pi * alpha_e

    # Fix z1=0 (massless right-mover), z3=1, z4=infinity
    # Integrate z2 along the real line
    # After summing orderings, this is equivalent to integrating z2 over R

    # Kinematics: all massless
    p, q = 0.5, 0.7
    k1 = (p/2, p/2)    # right-mover
    k2 = (-p/2, -p/2)  # right-mover (opposite)
    k3 = (q/2, -q/2)   # left-mover
    k4 = (-q/2, q/2)   # left-mover (opposite)

    s12 = 2 * alpha_e * (-k1[0]*k2[0] + k1[1]*k2[1])  # = 0
    s23 = 2 * alpha_e * (-k2[0]*k3[0] + k2[1]*k3[1])  # = pq
    s13 = -(s12 + s23)  # = -pq

    wedges = {}
    momenta = {1: k1, 2: k2, 3: k3, 4: k4}
    for i in range(1, 5):
        for j in range(i + 1, 5):
            wedges[(i,j)] = theta01 * (momenta[i][0]*momenta[j][1] - momenta[i][1]*momenta[j][0])

    print(f"\n  Kinematics: s12={s12:.4f}, s23={s23:.4f}, s13={s13:.4f}")

    # The NCOS integrand at fixed z1=0, z3=1, z4=inf:
    # I(z2) = sum_sigma e^{iPhi_sigma} * |z2|^{s12} * |z2-1|^{s23}
    # where sigma runs over the 3 orderings of z2 relative to {0, 1}.
    #
    # Actually, the Moyal phase depends on the ordering of z2 among the other
    # vertices. When z2 is in (0,1), (-inf,0), or (1,inf), the phase differs.
    #
    # The key insight: at the NCOS point, the Moyal phases convert the
    # |z|^s factors to holomorphic z^s factors (for a right-mover at z1=0).
    # So I(z2) = z2^{s12} * (z2-1)^{s23} (holomorphic in z2, no absolute values)
    # up to an overall z2-independent phase.

    # Let's verify this directly. The three regions and their phases:
    # Region B: z2 in (0,1)    -> ordering (1,2,3,4)
    # Region A: z2 in (-inf,0) -> ordering (2,1,3,4)  [z2 < z1=0]
    # Region C: z2 in (1,inf)  -> ordering (1,3,2,4)  [z2 > z3=1]

    # For each region, the KN factor is |z2|^{s12} * |z2-1|^{s23}.
    # The Moyal phase depends on the signs of z2 and z2-1.

    # In region B (0 < z2 < 1): z2 > 0, z2-1 < 0
    #   KN = z2^{s12} * (1-z2)^{s23}
    #   Going to region A (z2 < 0): z2 changes sign -> phase e^{i*pi*s12}
    #   Going to region C (z2 > 1): (z2-1) changes sign -> phase e^{i*pi*s23}
    #
    # If Moyal phases compensate these sign changes, the result is holomorphic.

    # Direct computation of the full-line integral with phases:
    def integrand_real(x, component='real'):
        """Full NCOS integrand on the real line."""
        if abs(x) < 1e-12 or abs(x - 1) < 1e-12:
            return 0.0

        # KN factor: |x|^{s12} * |x-1|^{s23}
        kn = abs(x)**s12 * abs(x-1)**s23

        # Moyal phase: depends on signs of x and x-1
        # The full phase for ordering with z1=0, z3=1, z4=inf, z2=x:
        # phi = w12*sgn(0-x)/2 + w13*sgn(0-1)/2 + w14*0
        #     + w23*sgn(x-1)/2 + w24*0 + w34*0
        # (w14 and w24 and w34 involve k4, which is at infinity)
        # Actually for the standard 4-point, z4=inf drops out.
        # The ordering determines the Moyal phase.

        # Simpler: the monodromy-relation approach.
        # Phase from crossing z2 past z1=0:
        #   when x < 0: extra phase from z2 being left of z1
        #   when x > 0: no extra phase
        # Phase from crossing z2 past z3=1:
        #   when x > 1: extra phase from z2 being right of z3
        #   when x < 1: no extra phase

        # The Moyal phase for the NCOS integrand:
        phase = 0.0
        # z2 vs z1=0: contributes w12 * sgn(0 - x) / 2 = -w12 * sgn(x) / 2
        phase += -wedges[(1,2)] * np.sign(x) / 2
        # z2 vs z3=1: contributes w23 * sgn(x - 1) / 2
        phase += wedges[(2,3)] * np.sign(x - 1) / 2 if (2,3) in wedges else -wedges[(2,3)] * np.sign(x - 1) / 2

        # Hmm, this needs more care. Let me use the full formula.
        # Actually, the phase structure for the NCOS amplitude with
        # z1=0 < z3=1 < z4=inf, z2=x is:
        # For the pair (1,2): sgn(tau_1 - tau_2) = sgn(0 - x) = -sgn(x)
        # For the pair (2,3): sgn(tau_2 - tau_3) = sgn(x - 1)
        # For the pair (2,4): effectively 0 (z4 -> inf, but the wedge w24 is finite)
        # For the pair (1,3): sgn(0-1) = -1 (independent of z2)
        # For the pair (1,4): sgn(0-inf) = -1 (independent of z2)
        # For the pair (3,4): sgn(1-inf) = -1 (independent of z2)

        phi = 0.0
        phi += wedges[(1,2)] * (-np.sign(x)) / 2   # pair (1,2)
        phi += wedges[(1,3)] * (-1) / 2             # pair (1,3)
        phi += wedges[(1,4)] * (-1) / 2 if (1,4) in wedges else 0  # pair (1,4)
        phi += wedges[(2,3)] * np.sign(x-1) / 2     # pair (2,3)
        phi += wedges[(2,4)] * np.sign(x) / 2 if (2,4) in wedges else 0  # pair (2,4) [z2 vs z4 at inf: same sign as z2]
        phi += wedges[(3,4)] * (-1) / 2 if (3,4) in wedges else 0  # pair (3,4)

        val = kn * cmath.exp(1j * phi)
        return val.real if component == 'real' else val.imag

    # Integrate over the real line (truncated)
    L = 50.0
    n_pts = 10000
    x_vals = np.linspace(-L, L, n_pts + 1)
    # Remove singular points
    x_vals = x_vals[(np.abs(x_vals) > 0.001) & (np.abs(x_vals - 1) > 0.001)]
    dx = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 0.01

    A_real = sum(integrand_real(x, 'real') for x in x_vals) * dx
    A_imag = sum(integrand_real(x, 'imag') for x in x_vals) * dx
    A_total = complex(A_real, A_imag)

    print(f"\n  Full-line integral (L={L}):")
    print(f"    A = {A_total:.6e}")
    print(f"    |A| = {abs(A_total):.6e}")

    # Also test the holomorphic version: z2^{s12} * (z2-1)^{s23}
    # This should be the NCOS integrand if monodromy matching works.
    # Its contour integral over R should vanish (close in UHP).
    def integrand_holo(x, component='real'):
        if abs(x) < 1e-12 or abs(x-1) < 1e-12:
            return 0.0
        # z2^{s12}: for x<0, this has phase e^{i*pi*s12}
        # (z2-1)^{s23}: for x<1, this has phase e^{i*pi*s23} * |1-x|^{s23}
        z2_s12 = abs(x)**s12 * cmath.exp(1j * math.pi * s12 * (1 if x < 0 else 0))
        z2m1_s23 = abs(x-1)**s23 * cmath.exp(1j * math.pi * s23 * (1 if x < 1 else 0))
        val = z2_s12 * z2m1_s23
        return val.real if component == 'real' else val.imag

    A_holo_real = sum(integrand_holo(x, 'real') for x in x_vals) * dx
    A_holo_imag = sum(integrand_holo(x, 'imag') for x in x_vals) * dx
    A_holo = complex(A_holo_real, A_holo_imag)

    print(f"\n  Holomorphic integrand (z2^s12 * (z2-1)^s23):")
    print(f"    A_holo = {A_holo:.6e}")
    print(f"    |A_holo| = {abs(A_holo):.6e}")


def test_5pt_contour():
    """5-point contour test with one massless leg."""
    print("\n" + "=" * 70)
    print("TEST 2: 5-POINT CONTOUR INTEGRAL")
    print("=" * 70)

    alpha_e = 1.0
    theta01 = 2 * math.pi * alpha_e

    # Kinematics: k1 massless right-mover, k2-k4 generic, k5 from conservation
    k1 = (1.0, 1.0)   # massless right-mover
    k2 = (0.5, -0.5)  # massless left-mover
    k3 = (0.8, 0.3)   # massive
    k4 = (-0.7, 0.4)  # massive
    k5 = (-(k1[0]+k2[0]+k3[0]+k4[0]), -(k1[1]+k2[1]+k3[1]+k4[1]))

    momenta = {1: k1, 2: k2, 3: k3, 4: k4, 5: k5}

    # Verify monodromy matching for leg 1
    print(f"\n  Monodromy matching for massless right-mover k1:")
    all_match = True
    for a in range(2, 6):
        ka = momenta[a]
        s_1a = 2 * alpha_e * (-k1[0]*ka[0] + k1[1]*ka[1])
        w_1a = theta01 * (k1[0]*ka[1] - k1[1]*ka[0])
        delta = abs(cmath.exp(1j * (math.pi * s_1a - w_1a)) - 1.0)
        print(f"    vs leg {a}: s_1a={s_1a:.4f}, w_1a/(pi)={w_1a/math.pi:.4f}, delta={delta:.2e}")
        if delta > 1e-10:
            all_match = False

    if all_match:
        print(f"\n  All pairwise matchings exact -> integrand holomorphic in tau_1")
        print(f"  -> contour argument gives A_NCOS = 0")

    # Mandelstam invariants
    wedges = {}
    for i in range(1, 6):
        for j in range(i+1, 6):
            wedges[(i,j)] = theta01 * (momenta[i][0]*momenta[j][1] - momenta[i][1]*momenta[j][0])

    def s_ij(i, j):
        return 2 * alpha_e * (-momenta[i][0]*momenta[j][0] + momenta[i][1]*momenta[j][1])

    # For 5-point, fix z1=0, z4=1, z5=inf, integrate z2, z3
    # The number of orderings is 3! = 6 (orderings of z2, z3 among {0, 1})
    # Actually with z1=0, z4=1, z5=inf fixed, we integrate z2 and z3 over R,
    # and the different orderings correspond to different regions.

    # Numerical contour integral in 2D
    # At the NCOS point, the integrand should be holomorphic in z_2 (and z_3 is
    # real, but the integral over z_2 at fixed z_3 gives zero by the
    # same contour argument).

    # Actually, the argument is that the integrand is holomorphic in tau_1
    # (the massless leg), not in the other variables. In the gauge where
    # tau_1 is integrated and the others are fixed... hmm.

    # Let's use a different gauge: fix z2=0, z4=1, z5=inf, integrate z1 and z3.
    # Then the contour argument says the integral over z1 vanishes for each z3.

    s_12 = s_ij(1, 2)
    s_13 = s_ij(1, 3)
    s_14 = s_ij(1, 4)
    s_23 = s_ij(2, 3)
    s_24 = s_ij(2, 4)
    s_34 = s_ij(3, 4)

    print(f"\n  Mandelstam invariants:")
    print(f"    s12={s_12:.4f}, s13={s_13:.4f}, s14={s_14:.4f}")
    print(f"    s23={s_23:.4f}, s24={s_24:.4f}, s34={s_34:.4f}")

    # Numerical check: fix z3 and integrate z1 over real line
    # (z2=0, z4=1, z5=inf)
    z3_fixed = 0.5
    n_pts = 5000
    L = 30.0
    z1_vals = np.linspace(-L, L, n_pts)
    # Remove singularities
    z1_vals = z1_vals[(np.abs(z1_vals) > 0.005) &
                      (np.abs(z1_vals - z3_fixed) > 0.005) &
                      (np.abs(z1_vals - 1) > 0.005)]
    dz1 = z1_vals[1] - z1_vals[0]

    A_total = 0.0
    for z1 in z1_vals:
        # KN factors involving z1: |z1|^{s12} * |z1-z3|^{s13} * |z1-1|^{s14}
        kn_z1 = (abs(z1)**s_12 * abs(z1 - z3_fixed)**s_13 * abs(z1 - 1)**s_14)

        # Moyal phase depending on ordering of z1 among {z2=0, z3=z3_fixed, z4=1}
        phi = 0.0
        phi += wedges[(1,2)] * np.sign(z1 - 0) / 2      # z1 vs z2=0
        phi += wedges[(1,3)] * np.sign(z1 - z3_fixed) / 2  # z1 vs z3
        phi += wedges[(1,4)] * np.sign(z1 - 1) / 2       # z1 vs z4=1
        # z1 vs z5=inf: sgn(z1 - inf) = -1
        phi += wedges[(1,5)] * (-1) / 2 if (1,5) in wedges else 0
        # z2 vs z3,z4,z5 phases are z1-independent (constant)
        # They contribute an overall phase that doesn't affect vanishing.

        val = kn_z1 * cmath.exp(1j * phi)
        A_total += val * dz1

    print(f"\n  z1 integral at fixed z3={z3_fixed} (z2=0, z4=1, z5=inf):")
    print(f"    A = {A_total:.6e}")
    print(f"    |A| = {abs(A_total):.2e}")

    # Test at another z3 value
    for z3_fixed in [0.3, 0.7, -0.5, 2.0]:
        z1_vals_v2 = np.linspace(-L, L, n_pts)
        z1_vals_v2 = z1_vals_v2[(np.abs(z1_vals_v2) > 0.005) &
                                (np.abs(z1_vals_v2 - z3_fixed) > 0.005) &
                                (np.abs(z1_vals_v2 - 1) > 0.005)]
        dz1 = z1_vals_v2[1] - z1_vals_v2[0]

        A_v2 = 0.0
        for z1 in z1_vals_v2:
            kn_z1 = (abs(z1)**s_12 * abs(z1 - z3_fixed)**s_13 * abs(z1 - 1)**s_14)
            phi = 0.0
            phi += wedges[(1,2)] * np.sign(z1) / 2
            phi += wedges[(1,3)] * np.sign(z1 - z3_fixed) / 2
            phi += wedges[(1,4)] * np.sign(z1 - 1) / 2
            phi += wedges[(1,5)] * (-1) / 2 if (1,5) in wedges else 0
            val = kn_z1 * cmath.exp(1j * phi)
            A_v2 += val * dz1

        print(f"    z3={z3_fixed:5.1f}: |A| = {abs(A_v2):.2e}")


def test_holomorphic_factorization():
    """Verify that at the NCOS point, the integrand factorizes as
    (holomorphic in z1) * (z1-independent),
    so the z1 integral gives zero by Cauchy's theorem."""
    print("\n" + "=" * 70)
    print("TEST 3: HOLOMORPHIC FACTORIZATION CHECK")
    print("=" * 70)

    alpha_e = 1.0
    theta01 = 2 * math.pi * alpha_e

    # For massless right-mover k1, at NCOS point:
    # |z1 - z_a|^{s_{1a}} * e^{i*w_{1a}*sgn(z1-z_a)/2}
    # = |z1-z_a|^{s_{1a}} * e^{i*pi*s_{1a}*sgn(z1-z_a)/2}  [using w_{1a} = pi*s_{1a}]
    # = |z1-z_a|^{s_{1a}} * {e^{i*pi*s_{1a}/2} if z1>z_a, e^{-i*pi*s_{1a}/2} if z1<z_a}
    #
    # Now |z1-z_a|^{s} * e^{i*pi*s*sgn(z1-z_a)/2}:
    # For z1 > z_a: (z1-z_a)^s * e^{i*pi*s/2} [WRONG, let me redo]
    #
    # Actually: |z1-z_a|^s = (z1-z_a)^s for z1>z_a, and |z_a-z1|^s = (z_a-z1)^s for z1<z_a.
    # For z1 > z_a: |z1-z_a|^s * e^{i*pi*s/2} = (z1-z_a)^s * e^{i*pi*s/2}
    # For z1 < z_a: |z1-z_a|^s * e^{-i*pi*s/2} = (z_a-z1)^s * e^{-i*pi*s/2}
    #             = (-(z1-z_a))^s * e^{-i*pi*s/2} = (z1-z_a)^s * e^{i*pi*s} * e^{-i*pi*s/2}
    #             = (z1-z_a)^s * e^{i*pi*s/2}
    #
    # So BOTH regions give (z1-z_a)^s * e^{i*pi*s/2} -- holomorphic in z1!

    print("\n  Verification of holomorphic factorization:")
    print("  For massless right-mover at NCOS point,")
    print("  |z-a|^s * exp(-i*pi*s*sgn(z-a)/2) = UHP bdy of (z-a)^s * exp(-i*pi*s/2)")
    print()

    s = 0.35  # example exponent
    a = 0.7   # example position

    print(f"  s = {s}, a = {a}")
    print(f"  {'z':>8s}  {'LHS (real)':>14s}  {'LHS (imag)':>14s}  {'RHS (real)':>14s}  {'RHS (imag)':>14s}  {'|diff|':>10s}")

    for z in [-2.0, -1.0, -0.1, 0.3, 0.5, 0.69, 0.71, 0.9, 1.5, 3.0]:
        # LHS: |z-a|^s * exp(-i*pi*s*sgn(z-a)/2) [correct Moyal sign]
        lhs = abs(z - a)**s * cmath.exp(-1j * math.pi * s * np.sign(z - a) / 2)

        # RHS: UHP boundary of (z-a)^s * exp(-i*pi*s/2)
        # For z < a: UHP limit gives (z-a)^s -> |z-a|^s * exp(i*pi*s)
        if z - a > 0:
            rhs_power = (z - a)**s
        else:
            rhs_power = abs(z - a)**s * cmath.exp(1j * math.pi * s)
        rhs = rhs_power * cmath.exp(-1j * math.pi * s / 2)

        diff = abs(lhs - rhs)
        print(f"  {z:8.2f}  {lhs.real:14.6f}  {lhs.imag:14.6f}  {rhs.real:14.6f}  {rhs.imag:14.6f}  {diff:10.2e}")

    print(f"\n  => The NCOS KN+Moyal factor for each pair (1,a) is holomorphic")
    print(f"     in z_1 (for right-mover) at the NCOS point.")
    print(f"     Product over all a gives a holomorphic function.")
    print("     Integral over R gives 0 by closing in UHP (decay ~ z_1^{sum s_{1a}}).")


def test_uhp_decay():
    """Verify that the holomorphic integrand decays in the UHP,
    so we can close the contour."""
    print("\n" + "=" * 70)
    print("TEST 4: UHP DECAY OF HOLOMORPHIC INTEGRAND")
    print("=" * 70)

    alpha_e = 1.0

    # For k1 right-mover: sum_a s_{1a} = sum_a 2*alpha'_e * k1.ka
    # By momentum conservation: sum_a ka = -k1 (if summing over a=2..n)
    # Actually sum_{all} k_i = 0, so sum_{a!=1} ka = -k1.
    # Then sum_{a!=1} k1.ka = k1.(-k1) = -k1^2 = 0 for massless k1.
    # So sum_a s_{1a} = 0.

    # This means the holomorphic function goes like z_1^0 = const at infinity.
    # That's NOT decaying! The contour argument needs more care.

    # Actually, for the individual factors:
    # prod_a (z1 - z_a)^{s_{1a}} ~ z_1^{sum s_{1a}} = z_1^0 = 1
    # So the integrand approaches a constant at infinity.
    # The contour integral doesn't automatically vanish just from UHP closure.

    # However, in the ordered integral there's a measure factor and the
    # integrand has the OTHER KN factors (between non-1 legs) which also
    # depend on the other integration variables.

    # For the 4-point case with (z1=integrated, z3=1, z4=inf), z2 is also
    # fixed to some value. Then:
    # I(z1) = (z1-z2)^{s12} * (z1-z3)^{s13} * (z1-z4)^{s14}
    # As z1 -> inf: ~ z1^{s12+s13+s14} = z1^0 = const.
    # So the integral over R doesn't converge absolutely.

    # BUT: the actual prescription is the SL(2,R) gauge-fixed amplitude,
    # which includes the |z1-z3|^2 * |z1-z4|^2 * |z3-z4|^2 Jacobian...
    # In the standard gauge z_a=0, z_b=1, z_c=inf, we integrate the other
    # variables. If z1 is one of the fixed points, there's no integral over z1.

    # So the correct setup is: z1 is NOT fixed (it's the massless leg we want
    # to show decouples). We fix three of the OTHER legs.
    # E.g., for 4-point: fix z2=0, z3=1, z4=inf, integrate z1 over R.
    # The SL(2,R) Jacobian gives a factor, and the gauge-fixed integrand is:
    # |z1|^{s12} * |z1-1|^{s13}  [z4 at inf drops out]
    # The NCOS version is: z1^{s12} * (z1-1)^{s13} (holomorphic).
    # As z1 -> inf: z1^{s12+s13} = z1^{-s23} (using s12+s13+s23=0).
    # For s23 > 0 (physical kinematics): decays as z1^{-s23} -> 0. GOOD.
    # For s23 < 0: diverges. But this is outside the convergence region anyway.

    p, q = 0.5, 0.7
    s12 = 0  # right-mover vs right-mover
    s23 = alpha_e * p * q  # = 0.35
    s13 = -s23  # = -0.35

    print(f"\n  4-point: s12={s12}, s13={s13:.4f}, s23={s23:.4f}")
    print(f"  Decay exponent at infinity: s12+s13 = {s12+s13:.4f} = -s23 = {-s23:.4f}")
    print(f"  For s23 > 0: integrand ~ z1^{{{s12+s13:.4f}}} -> 0 in UHP. GOOD.")

    # Verify numerically: the holomorphic integral z1^{s12}*(z1-1)^{s13} dz1 over R
    # should be 0 for the NCOS case.
    n_pts = 50000
    L = 200.0
    z1_vals = np.linspace(-L, L, n_pts)
    z1_vals = z1_vals[(np.abs(z1_vals) > 0.001) & (np.abs(z1_vals - 1) > 0.001)]
    dz1 = z1_vals[1] - z1_vals[0]

    A = 0.0
    for z1 in z1_vals:
        # (z1)^{s12} * (z1-1)^{s13} with appropriate branch cuts
        # s12 = 0, so the first factor is 1
        # (z1-1)^{s13}: branch cut on (-inf, 1)
        if z1 > 1:
            f = (z1 - 1)**s13
        else:
            f = abs(z1 - 1)**s13 * cmath.exp(1j * math.pi * s13)
        A += f * dz1

    print(f"\n  Numerical contour integral of z1^{s12}*(z1-1)^{s13} over [{-L},{L}]:")
    print(f"    A = {A:.6e}")
    print(f"    |A| = {abs(A):.2e}")

    # Compare with the non-holomorphic version (without NCOS phases):
    A_abs = 0.0
    for z1 in z1_vals:
        A_abs += abs(z1)**s12 * abs(z1-1)**s13 * dz1

    print(f"\n  Compare with non-holomorphic |z1|^{s12}*|z1-1|^{s13}:")
    print(f"    integral = {A_abs:.6f} (non-zero, as expected)")


if __name__ == "__main__":
    test_4pt_contour()
    test_5pt_contour()
    test_holomorphic_factorization()
    test_uhp_decay()
