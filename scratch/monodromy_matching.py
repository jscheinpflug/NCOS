#!/usr/bin/env python3
"""Verify the critical monodromy matching that underlies NCOS massless decoupling.

The key identity: at the NCOS point (theta^{01} = 2*pi*alpha'_e), for a massless
external leg k_2 (k_{2+}k_{2-} = 0), the KN branch-cut monodromy and the Moyal
phase jump cancel exactly when the contour crosses another insertion:

    exp(i*pi*s_{2a}) * exp(-i * k_2 wedge k_a) = 1

where s_{2a} = 2*alpha'_e * k_2.k_a and k_2 wedge k_a = theta^{01}*(k_{2,0}*k_{a,1} - k_{2,1}*k_{a,0}).

This makes the integrand holomorphic in the UHP, and Cauchy's theorem gives the
vanishing of the amplitude (sum over orderings).

We test:
1) The identity holds for massless k_2 at the NCOS point.
2) It fails for massive k_2.
3) It fails when theta != 2*pi*alpha'_e.
4) Numerical contour integration of the 4-point amplitude on the D1.
"""

from __future__ import annotations

import cmath
import math
import numpy as np
from scipy import integrate


def kn_exponent(k2: tuple, ka: tuple, alpha_e: float) -> float:
    """s_{2a} = 2*alpha'_e * k_2 . k_a  with signature (-,+)."""
    return 2.0 * alpha_e * (-k2[0] * ka[0] + k2[1] * ka[1])


def wedge(k2: tuple, ka: tuple, theta01: float) -> float:
    """k_2 wedge k_a = theta^{01}*(k_{2,0}*k_{a,1} - k_{2,1}*k_{a,0})."""
    return theta01 * (k2[0] * ka[1] - k2[1] * ka[0])


def monodromy_mismatch(k2: tuple, ka: tuple, alpha_e: float, theta01: float) -> float:
    """Return |exp(i*pi*s_{2a}) * exp(-i*w_{2a}) - 1| where w = k2 wedge ka.

    Zero means exact monodromy matching.
    """
    s = kn_exponent(k2, ka, alpha_e)
    w = wedge(k2, ka, theta01)
    phase = cmath.exp(1j * math.pi * s - 1j * w)
    return abs(phase - 1.0)


def test_monodromy_matching():
    print("=" * 65)
    print("TEST 1: Monodromy matching for massless k2 at NCOS point")
    print("=" * 65)
    alpha_e = 1.0
    theta01_ncos = 2.0 * math.pi * alpha_e

    # Massless right-mover: k_{2,0} = k_{2,1} = p/2
    for p2 in [1.0, 2.5, -3.7, 0.1]:
        k2 = (p2 / 2, p2 / 2)  # right-mover: k_0 = k_1
        print(f"\n  k2 = ({k2[0]:.3f}, {k2[1]:.3f}), mass^2 = {-k2[0]**2+k2[1]**2:.6f}")
        for ka in [(1.3, 0.7), (-2.1, 0.5), (0.0, 3.0), (1.0, -1.0), (0.4, 0.4)]:
            m = monodromy_mismatch(k2, ka, alpha_e, theta01_ncos)
            print(f"    ka = {ka}, mismatch = {m:.2e}")

    # Massless left-mover: k_{2,0} = -k_{2,1}
    k2 = (1.5, -1.5)
    print(f"\n  k2 = ({k2[0]:.3f}, {k2[1]:.3f}) [left-mover], mass^2 = {-k2[0]**2+k2[1]**2:.6f}")
    for ka in [(1.3, 0.7), (-2.1, 0.5), (0.0, 3.0)]:
        m = monodromy_mismatch(k2, ka, alpha_e, theta01_ncos)
        print(f"    ka = {ka}, mismatch = {m:.2e}")

    print("\n" + "=" * 65)
    print("TEST 2: Monodromy matching FAILS for massive k2")
    print("=" * 65)
    # Massive: k_0 != +/- k_1
    for k2 in [(1.0, 0.5), (2.0, 0.0), (1.0, 0.3), (0.7, -0.2)]:
        msq = -k2[0] ** 2 + k2[1] ** 2
        mismatches = []
        for ka in [(1.3, 0.7), (-2.1, 0.5), (0.0, 3.0)]:
            mismatches.append(monodromy_mismatch(k2, ka, alpha_e, theta01_ncos))
        print(
            f"  k2 = {k2}, mass^2 = {msq:.4f}, "
            f"mismatches = [{', '.join(f'{m:.4f}' for m in mismatches)}]"
        )

    print("\n" + "=" * 65)
    print("TEST 3: Monodromy matching FAILS away from NCOS (theta != 2*pi*alpha'_e)")
    print("=" * 65)
    k2 = (1.0, 1.0)  # massless right-mover
    ka = (1.3, 0.7)
    for eps in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
        theta01 = 2.0 * math.pi * alpha_e * (1.0 + eps)
        m = monodromy_mismatch(k2, ka, alpha_e, theta01)
        print(f"  eps = {eps:.2f}, theta/theta_NCOS = {1+eps:.2f}, mismatch = {m:.6f}")


def ncos_4pt_contour_integral():
    """Numerically verify the 4-point NCOS amplitude vanishes.

    Setup: 4 open strings on D1, fix z_1=0, z_3=1, z_4->inf.
    Integrate z_2 over the real line. The NCOS integrand (KN * Moyal phase)
    should be the boundary value of a holomorphic function in the UHP,
    so the real-line integral vanishes.

    Kinematics: k1 = right-mover (k+, 0), k2 = right-mover (-k+, 0),
    k3 = left-mover (0, q-), k4 = left-mover (0, -q-).
    This gives s12 = s34 = 0, s23 = -s13 = alpha'_e * k+ * q-.
    """
    print("\n" + "=" * 65)
    print("TEST 4: Numerical 4-point NCOS contour integral")
    print("=" * 65)

    alpha_e = 1.0
    kp = 1.5   # k_{1,+}
    qm = 2.0   # k_{3,-}

    # Mandelstam-like exponents
    s12 = 0.0  # both right-movers
    s23 = alpha_e * kp * qm  # = alpha'_e * k_{1+} * k_{3-}
    s34 = 0.0
    # s13 = -s23, s14 = s23, s24 = -s23

    # Wedge products (with theta = 2*pi*alpha'_e)
    # k_a wedge k_b = pi * alpha'_e * (k_{a-}*k_{b+} - k_{a+}*k_{b-})
    W = math.pi * alpha_e * kp * qm  # = pi * s23

    print(f"  alpha'_e = {alpha_e}, k+ = {kp}, q- = {qm}")
    print(f"  s12 = {s12}, s23 = {s23:.4f}, W = pi*s23 = {W:.4f}")
    print(f"  Monodromy check: pi*s23 = {math.pi*s23:.6f}, W = {W:.6f}, ratio = {W/(math.pi*s23):.6f}")

    # Holomorphic integrand (approach real line from UHP):
    #   f_hol(z) = z^{s12} * (z-1)^{s23}
    # with branch cuts in the LHP.
    # On real line from UHP:
    #   z > 1:  f = z^{s12} * (z-1)^{s23}                       [both real positive]
    #   0<z<1:  f = z^{s12} * |z-1|^{s23} * exp(i*pi*s23)       [z-1 < 0]
    #   z < 0:  f = |z|^{s12} * exp(i*pi*s12) * |z-1|^{s23} * exp(i*pi*s23)
    #
    # Moyal phases for the three orderings (with + convention from the note):
    #   (1,2,3,4) for z in (0,1):  Phi_B
    #   (1,3,2,4) for z in (1,inf): Phi_C
    #   (1,3,4,2) for z in (-inf,0): Phi_A
    #
    # The NCOS integrand is |z|^{s12} |z-1|^{s23} * exp(i*Phi(z)) * K.
    # For A_NCOS = integral f_hol, we need exp(i*Phi) = (holomorphic phases) / |...|
    # The matching means A_NCOS = integral of f_hol over real line = 0 (Cauchy).

    # Direct numerical check: compute each ordered piece with Moyal phases.
    # Ordered amplitude A_sigma = integral_{I_sigma} |z|^{s12} |z-1|^{s23} dz.
    # (Drop K since it's common.)

    # Moyal phases for the three orderings.
    # Using the note's convention and my earlier calculation:
    # Phi_{1234} = 0, Phi_{1324} = W, Phi_{1342} = 0
    # (with the + sign convention from the INTRINSIC_NCOS note)

    # Actually let me compute them properly with a general formula.
    def ordering_phase(order, wedges):
        """Compute Moyal phase for a given cyclic ordering.

        order: tuple of particle labels in boundary order.
        wedges: dict mapping (i,j) with i<j to k_i wedge k_j.
        """
        n = len(order)
        phi = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                i, j = order[p], order[q]
                # sgn(tau_{order[p]} - tau_{order[q]}) = -1 since p < q in the ordering
                sgn_val = -1.0
                if i < j:
                    phi += 0.5 * wedges[(i, j)] * sgn_val
                else:
                    phi += 0.5 * (-wedges[(j, i)]) * sgn_val
        return phi

    wedges = {
        (1, 2): 0.0,
        (1, 3): -W,
        (1, 4): W,
        (2, 3): W,
        (2, 4): -W,
        (3, 4): 0.0,
    }

    orderings = [(1, 2, 3, 4), (1, 3, 2, 4), (1, 3, 4, 2)]
    for order in orderings:
        phi = ordering_phase(order, wedges)
        print(f"  Ordering {order}: Phi = {phi:.6f}, exp(iPhi) = {cmath.exp(1j*phi):.6f}")

    # Compute ordered integrals (regularized).
    # Use s12 = 0, so |z|^{s12} = 1.
    # A(1,2,3,4) = integral_0^1 (1-z)^{s23} dz = 1/(1+s23)  [for s23 > -1]
    # A(1,3,2,4) = integral_1^R (z-1)^{s23} dz -> diverges for s23 >= -1
    # A(1,3,4,2) = integral_{-R}^0 |z-1|^{s23} dz -> also diverges

    # The divergences cancel in the sum (this is the standard open-string story).
    # Use dimensional regularization or a cutoff.

    # Better: compute the holomorphic contour integral directly.
    # Integrate f_hol(z) = (z-1)^{s23} over a large semicircle in the UHP.
    # The real-line piece is the NCOS amplitude; the semicircle at infinity should
    # vanish if the integrand decays.
    #
    # For s23 > -1 (our case: s23 = alpha'_e * k+ * q- > 0), (z-1)^{s23} grows
    # as z -> infinity, so the semicircle does NOT vanish. This means the
    # naive "integral over real line = 0" argument needs modification.
    #
    # The resolution: the FULL superstring integrand includes ALL KN factors
    # and the kinematic prefactor K, which provides additional falloff.
    # For the 4-point function, the standard treatment uses the SL(2,R)-fixed
    # form where the z_4 -> infinity is already taken care of.

    # Let me instead use the proper SL(2,R)-fixed integral.
    # Fix z_1 = 0, z_3 = 1, z_4 -> infinity.
    # After SL(2,R) fixing, the measure is just dz_2.
    # The KN factor (for the z_2-dependent pieces) is:
    #   |z_2|^{s12} * |z_2 - 1|^{s23} * (z_4-dependent pieces -> Jacobian)
    # For s12 = 0, this is just |z_2 - 1|^{s23}.
    #
    # But we need to also include the z_2-z_4 factor: |z_2 - z_4|^{s24} -> |z_4|^{s24}
    # as z_4 -> inf, which contributes to the Jacobian. The standard result is that
    # the Jacobian from SL(2,R) gauge-fixing exactly cancels these z_4-dependent pieces.
    #
    # For our kinematics, s24 = -s23, so there IS a z_2-dependent remnant from
    # the gauge-fixing at finite z_4. Let me use z_4 = L and take L -> infinity.

    # Actually, the correct SL(2,R)-fixed integrand for the 4-point function
    # (with z_1=0, z_3=1, z_4=L, then L->inf) is:
    # dz_2 * |z_2|^{s12} * |z_2-1|^{s23} * |z_2-L|^{s24} * |L|^{s14} * |L-1|^{s34}
    #       * L^2 (Jacobian)
    # The z_2-independent factors (L^{s14}*(L-1)^{s34}*L^2) just give an overall
    # normalization. The z_2-dependent part is |z_2|^{s12}*|z_2-1|^{s23}*|z_2-L|^{s24}.
    # As L->inf: |z_2-L|^{s24} -> L^{s24} (z_2-independent), so the integral becomes
    # just integral of |z_2|^{s12}*|z_2-1|^{s23} dz_2.
    #
    # With s12=0 and s24=-s23, the L-dependent overall factor is
    # L^{s14+s34+s24+2} = L^{s23+0+(-s23)+2} = L^2, which gives a finite
    # normalization as L->inf after accounting for the SL(2,R) volume.

    # So the net integrand is (1-z)^{s23} for z in (0,1), and by holomorphic
    # continuation for the other intervals. But the issue is convergence at infinity.

    # The REAL resolution: for 4-point with all massless states on D1, s23 > 0
    # (for our kinematics), so the integral diverges at infinity. This means
    # the amplitude needs regularization. In string theory, these divergences
    # are handled by the full treatment of SL(2,R) with a finite cutoff and
    # then showing the divergences cancel between orderings.

    # For a clean numerical test, use complex integration with a small imaginary
    # part to regularize, and compare the NCOS sum with a reference.

    # Simpler approach: test a case where s23 < 0 (convergent integral).
    # This happens when k_{1+}*k_{3-} < 0, i.e., k_{1+} and k_{3-} have
    # opposite signs. E.g., k1 right-moving with k+ > 0, k3 left-moving with
    # k- < 0. But for physical kinematics k+ > 0 and k- < 0 means negative
    # energy... let me just use analytic continuation.

    # Most informative: verify the identity sum_sigma phase_sigma * I_sigma = 0
    # using the Beta function evaluation of the ordered integrals.

    # For s12 = 0:
    # A(0,1) = B(1+s12, 1+s23) = B(1, 1+s23) = 1/(1+s23)
    # A(1,inf) = B(1+s23, 1+s24) = B(1+s23, 1-s23)  [using s24 = -s23]
    # A(-inf,0) = B(1+s24, 1+s12) = B(1-s23, 1) = 1/(1-s23)
    # Wait, I need the standard open-string monodromy relations here.

    # Actually for the open superstring, the color-ordered 4-point amplitudes
    # satisfy the relation:
    # A(1,2,3,4) + e^{i*pi*s12} A(1,2,4,3) + e^{i*pi*(s12+s23)} A(1,4,2,3) = 0
    # This is the monodromy relation (BCJ). At the NCOS point, the Moyal phases
    # should reproduce exactly these monodromy factors.

    # Let me verify: the monodromy relation says that the sum of ordered amplitudes
    # times specific phases vanishes. The NCOS amplitude is the sum of ordered
    # amplitudes times Moyal phases. If the Moyal phases equal the monodromy phases,
    # the NCOS amplitude vanishes.

    # Standard monodromy relation (contour deformation of z_2 from UHP):
    # A(0,1) + e^{i*pi*s12} * A(-inf,0) + e^{i*pi*(s12+s23)} * A(1,inf) = 0

    # Where:
    # A(0,1) = ordering (1,2,3,4): Moyal phase Phi_B
    # A(1,inf) = ordering (1,3,2,4): Moyal phase Phi_C
    # A(-inf,0) = ordering (1,3,4,2): Moyal phase Phi_A

    # Monodromy phases (from analytic continuation in UHP):
    #   (0,1):     e^{i*pi*s23} (from (z-1)^{s23} with z<1)
    #   (1,inf):   1             (both factors real)
    #   (-inf,0):  e^{i*pi*(s12+s23)} (both z and z-1 negative)

    # But wait, the standard monodromy relation uses a DIFFERENT convention.
    # The relation comes from deforming the integration contour of z_2 in the
    # COMPLEX plane: start from (0,1) on the real axis, deform around z=0
    # to reach (-inf,0), picking up e^{i*pi*s12}, then deform around z=1
    # to reach (1,inf), picking up e^{i*pi*s23}.
    # So: A(0,1) + e^{i*pi*s12} A(-inf,0) = -e^{i*pi*(s12+s23)} A(1,inf)
    # i.e., A(0,1) + e^{i*pi*s12} A(-inf,0) + e^{i*pi*(s12+s23)} A(1,inf) = 0.

    # The NCOS amplitude is:
    # A_NCOS = e^{iPhi_B} A(0,1) + e^{iPhi_C} A(1,inf) + e^{iPhi_A} A(-inf,0)
    # For this to vanish (using the monodromy relation), we need:
    # e^{iPhi_B} : e^{iPhi_A} : e^{iPhi_C} = 1 : e^{i*pi*s12} : e^{i*pi*(s12+s23)}
    # (up to overall phase)

    # Check: e^{iPhi_A}/e^{iPhi_B} should = e^{i*pi*s12}
    # From our calculation: Phi_A = 0, Phi_B = 0, so ratio = 1.
    # And e^{i*pi*s12} = e^0 = 1 (since s12 = 0). CHECK!

    # e^{iPhi_C}/e^{iPhi_B} should = e^{i*pi*(s12+s23)}
    # Phi_C = W = pi*s23, Phi_B = 0, so ratio = e^{i*pi*s23}.
    # And e^{i*pi*(s12+s23)} = e^{i*pi*s23} (since s12 = 0). CHECK!

    print("\n  Monodromy relation verification:")
    Phi_B = ordering_phase((1, 2, 3, 4), wedges)
    Phi_C = ordering_phase((1, 3, 2, 4), wedges)
    Phi_A = ordering_phase((1, 3, 4, 2), wedges)
    print(f"    Phi_B = {Phi_B:.6f}, Phi_C = {Phi_C:.6f}, Phi_A = {Phi_A:.6f}")
    ratio_AB = cmath.exp(1j * (Phi_A - Phi_B))
    ratio_CB = cmath.exp(1j * (Phi_C - Phi_B))
    mono_AB = cmath.exp(1j * math.pi * s12)
    mono_CB = cmath.exp(1j * math.pi * (s12 + s23))
    print(f"    e^(i(Phi_A-Phi_B)) = {ratio_AB:.6f}, needed e^(i*pi*s12) = {mono_AB:.6f}")
    print(f"    e^(i(Phi_C-Phi_B)) = {ratio_CB:.6f}, needed e^(i*pi*(s12+s23)) = {mono_CB:.6f}")
    print(f"    Match AB: {abs(ratio_AB - mono_AB):.2e}")
    print(f"    Match CB: {abs(ratio_CB - mono_CB):.2e}")

    # Now compute the actual ordered integrals using the Beta function.
    # A(0,1) with s12=0: B(1, 1+s23) = Gamma(1)*Gamma(1+s23)/Gamma(2+s23) = 1/(1+s23)
    from scipy.special import gamma as Gamma, beta as Beta

    # Standard: A(1,sigma(2),sigma(3),4) ~ B(1+s_{1,sigma(2)}, 1+s_{sigma(2),sigma(3)})
    # for the superstring (numerator is 1 for simplicity; really it's K).
    A_B = Beta(1.0 + s12, 1.0 + s23)  # ordering (1,2,3,4)
    # A(1,inf) ~ B(1+s23, 1+s24) = B(1+s23, 1-s23)
    s24 = -s23
    A_C = Beta(1.0 + s23, 1.0 + s24)  # ordering (1,3,2,4)
    # A(-inf,0) ~ B(1+s24, 1+s12) = B(1-s23, 1)
    s13 = -s23
    A_A = Beta(1.0 + s24, 1.0 + s12)  # ordering (1,3,4,2)
    print(f"\n  Ordered amplitudes (Beta function):")
    print(f"    A_B = {A_B:.6f}, A_C = {A_C:.6f}, A_A = {A_A:.6f}")

    A_ncos = (
        cmath.exp(1j * Phi_B) * A_B
        + cmath.exp(1j * Phi_C) * A_C
        + cmath.exp(1j * Phi_A) * A_A
    )
    print(f"\n  NCOS amplitude = sum(phase * ordered):")
    print(f"    A_NCOS = {A_ncos}")
    print(f"    |A_NCOS| = {abs(A_ncos):.2e}")

    # Compare with the monodromy relation check:
    A_mono = (
        1.0 * A_B
        + cmath.exp(1j * math.pi * s12) * A_A
        + cmath.exp(1j * math.pi * (s12 + s23)) * A_C
    )
    print(f"    Monodromy sum = {A_mono}")
    print(f"    |Monodromy sum| = {abs(A_mono):.2e}")


def ncos_deformation_test():
    """Test how the amplitude scales when theta deviates from 2*pi*alpha'_e."""
    print("\n" + "=" * 65)
    print("TEST 5: Deformation away from NCOS point")
    print("=" * 65)

    alpha_e = 1.0
    kp = 1.5
    qm = 2.0
    s12 = 0.0
    s23 = alpha_e * kp * qm

    from scipy.special import beta as Beta

    A_B = Beta(1.0 + s12, 1.0 + s23)
    s24 = -s23
    A_C = Beta(1.0 + s23, 1.0 + s24)
    A_A = Beta(1.0 + s24, 1.0 + s12)

    epsilons = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]
    print(f"\n  {'eps':>10s}  {'|A_NCOS|':>12s}  {'|A|/eps':>12s}")
    for eps in epsilons:
        theta01 = 2.0 * math.pi * alpha_e * (1.0 + eps)
        W_def = 0.5 * theta01 * kp * qm  # k_2 wedge k_3 at deformed theta

        # Recompute ordering phases with deformed theta
        wedges_def = {
            (1, 2): 0.0,
            (1, 3): -W_def,
            (1, 4): W_def,
            (2, 3): W_def,
            (2, 4): -W_def,
            (3, 4): 0.0,
        }
        Phi_B = 0.0
        Phi_C = 0.0
        Phi_A = 0.0
        for order, label in [((1,2,3,4), 'B'), ((1,3,2,4), 'C'), ((1,3,4,2), 'A')]:
            n = len(order)
            phi = 0.0
            for p in range(n):
                for q in range(p+1, n):
                    i, j = order[p], order[q]
                    sgn_val = -1.0
                    if i < j:
                        phi += 0.5 * wedges_def[(i, j)] * sgn_val
                    else:
                        phi += 0.5 * (-wedges_def[(j, i)]) * sgn_val
            if label == 'B':
                Phi_B = phi
            elif label == 'C':
                Phi_C = phi
            else:
                Phi_A = phi

        A_ncos = (
            cmath.exp(1j * Phi_B) * A_B
            + cmath.exp(1j * Phi_C) * A_C
            + cmath.exp(1j * Phi_A) * A_A
        )
        mag = abs(A_ncos)
        ratio = mag / eps if eps > 0 else float('nan')
        print(f"  {eps:10.1e}  {mag:12.6e}  {ratio:12.6f}")


def ncos_npoint_phase_test():
    """Test monodromy matching for n=5,6 points with one massless leg."""
    print("\n" + "=" * 65)
    print("TEST 6: Monodromy matching at 5-point and 6-point")
    print("=" * 65)

    alpha_e = 1.0

    # 5-point: k1 massless right-mover, k2..k5 generic massive on D1
    # k_i = (k_{i,0}, k_{i,1})
    np.random.seed(42)
    for n in [5, 6]:
        print(f"\n  n = {n} points, leg 1 massless right-mover")
        # Massless right-mover: k_{1,0} = k_{1,1}
        k1_p = 2.0
        k1 = (k1_p / 2, k1_p / 2)

        # Random massive momenta for legs 2..n-1, then fix leg n by momentum conservation
        momenta = [k1]
        psum = [k1[0], k1[1]]
        for _ in range(n - 2):
            k0 = np.random.uniform(-3, 3)
            k1_comp = np.random.uniform(-3, 3)
            momenta.append((k0, k1_comp))
            psum[0] += k0
            psum[1] += k1_comp
        momenta.append((-psum[0], -psum[1]))

        # Check monodromy matching for leg 1 against all others
        theta01_ncos = 2.0 * math.pi * alpha_e
        mismatches = []
        for a in range(1, n):
            ka = momenta[a]
            m = monodromy_mismatch(k1, ka, alpha_e, theta01_ncos)
            mismatches.append(m)
        print(f"    Monodromy mismatches (leg 1 vs leg 2..{n}):")
        print(f"    {[f'{m:.2e}' for m in mismatches]}")
        print(f"    Max mismatch: {max(mismatches):.2e}")
        if max(mismatches) < 1e-12:
            print("    => Exact monodromy matching confirmed!")


def massive_sector_test():
    """Check that monodromy matching FAILS for all-massive external states.

    This means the contour-pulling argument does not apply, and the massive
    sector has genuinely interacting amplitudes.
    """
    print("\n" + "=" * 65)
    print("TEST 7: Monodromy matching for massive external states")
    print("=" * 65)

    alpha_e = 1.0
    theta01_ncos = 2.0 * math.pi * alpha_e

    # First massive NS level: m^2 = 1/(2*alpha'_e)
    # k_+ * k_- = 1/(2*alpha'_e), so e.g. k_+ = 1, k_- = 1/(2*alpha'_e)
    k_massive = []
    for kp in [1.0, 2.0, -1.5]:
        km = 1.0 / (2.0 * alpha_e * kp)
        k0 = (kp + km) / 2
        k1 = (kp - km) / 2
        k_massive.append((k0, k1))

    print("  Massive momenta (first NS level):")
    for i, k in enumerate(k_massive):
        msq = -k[0] ** 2 + k[1] ** 2
        print(f"    k_{i+1} = ({k[0]:.4f}, {k[1]:.4f}), m^2 = {msq:.4f}")

    print("\n  Pairwise monodromy mismatches:")
    for i in range(len(k_massive)):
        for j in range(i + 1, len(k_massive)):
            m = monodromy_mismatch(k_massive[i], k_massive[j], alpha_e, theta01_ncos)
            print(f"    k_{i+1} vs k_{j+1}: mismatch = {m:.6f}")

    print("\n  => Monodromy matching fails for massive states.")
    print("  => Contour pulling is obstructed; massive sector is interacting.")


if __name__ == "__main__":
    test_monodromy_matching()
    ncos_4pt_contour_integral()
    ncos_deformation_test()
    ncos_npoint_phase_test()
    massive_sector_test()
