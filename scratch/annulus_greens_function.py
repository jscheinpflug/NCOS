#!/usr/bin/env python3
"""Test whether the monodromy matching identity pi*s_{ka}(t) = k wedge k_a
holds pointwise in the annulus modulus t, using the actual open-string
boundary Green's function on the annulus.

The key question: at one loop, the KN exponents depend on t through the
annulus Green's function G(tau_a, tau_b; t). Does the critical identity
hold for each t, or only after integrating over t?

Background:
  On the annulus with modulus q = exp(-pi*t), the boundary Green's function
  for an open string between boundaries at Im(z)=0 and Im(z)=t/2 is:
    G(tau1, tau2) = -alpha'_e * ln|theta_1(tau12 | it)|^2
                    + alpha'_e * (2*pi/t) * tau12^2  + const.
  where tau12 = tau1 - tau2 and theta_1 is the Jacobi theta function.

  The Moyal phase contribution from the antisymmetric part is:
    (i/2) * theta^{01} * sgn(tau12)
  which is the same as at tree level (the antisymmetric part of the
  boundary Green's function is topological/independent of modulus).

  The KN exponent is s_{ka}(t) = 2*alpha'_e * k . k_a evaluated
  using the SYMMETRIC part of the annulus Green's function, which
  IS t-dependent.

This script tests:
  1. Whether the antisymmetric (Moyal) part is truly t-independent.
  2. Whether the symmetric (KN) part's t-dependence spoils the
     monodromy matching.
  3. Whether the contour argument still works at each t.
"""

from __future__ import annotations
import numpy as np
from scipy.special import ellipj, ellipk
import cmath, math


def jacobi_theta1(z, tau_mod, nterms=200):
    """Jacobi theta_1(z|tau) = 2 * sum_{n=0}^{inf} (-1)^n q^{(n+1/2)^2} sin((2n+1)*pi*z)
    where q = exp(i*pi*tau). For pure imaginary tau = i*t, q = exp(-pi*t)."""
    q = np.exp(-np.pi * tau_mod)  # tau_mod = t > 0, so q = e^{-pi*t}
    result = 0.0
    for n in range(nterms):
        result += (-1)**n * q**((n + 0.5)**2) * np.sin((2*n + 1) * np.pi * z)
    return 2.0 * result


def jacobi_theta1_derivative(z, tau_mod, nterms=200):
    """d/dz theta_1(z|tau)."""
    q = np.exp(-np.pi * tau_mod)
    result = 0.0
    for n in range(nterms):
        result += (-1)**n * q**((n + 0.5)**2) * (2*n + 1) * np.pi * np.cos((2*n + 1) * np.pi * z)
    return 2.0 * result


def annulus_green_symmetric(tau12, t, alpha_e=1.0, nterms=200):
    """Symmetric part of the boundary Green's function on the annulus.

    For two points on the same boundary of the annulus with modulus t:
      G_sym(tau12, t) = -alpha'_e * ln|theta_1(tau12/L | it/L)|^2
                        + alpha'_e * (2*pi/(t/L)) * (tau12/L)^2 + const

    We work with the worldsheet period L=1 (tau in [0,1]) so:
      G_sym(tau12, t) = -alpha'_e * ln|theta_1(tau12 | it)|^2
                        + alpha'_e * (2*pi/t) * tau12^2

    The t-dependent piece is the linear-in-tau12^2 term, which comes from
    the zero mode on the annulus.
    """
    # Avoid tau12 = 0 (self-contraction divergence)
    if abs(tau12) < 1e-15:
        return 0.0

    th1 = jacobi_theta1(tau12, t, nterms)
    if abs(th1) < 1e-300:
        return float('inf')

    return -alpha_e * np.log(th1**2) + alpha_e * (2 * np.pi / t) * tau12**2


def annulus_green_antisymmetric(tau12, theta01):
    """Antisymmetric part of the boundary Green's function.
    This is TOPOLOGICAL (t-independent):
      G_anti(tau12) = (i/2) * theta^{01} * sgn(tau12)
    """
    return 0.5 * theta01 * np.sign(tau12)


def kn_exponent_annulus(k1, k2, tau12, t, alpha_e=1.0):
    """The Koba-Nielsen exponent from the symmetric Green's function,
    evaluated at modulus t:
      exp(2 * k1.k2 * G_sym(tau12, t))
    The exponent in the integrand is:
      s_{12}(t, tau12) = 2 * alpha'_e * k1.k2 * f(tau12, t)
    where f is the appropriate function from G_sym.

    For the monodromy matching test, what matters is the DISCONTINUITY
    of the symmetric Green's function as tau12 crosses zero, which
    corresponds to the log|theta_1| term (the theta_1 has a simple zero
    at tau12=0).

    Key insight: theta_1(z|tau) has zeros at z = m + n*tau for integers m,n.
    Near z=0: theta_1(z|tau) ~ theta_1'(0|tau) * z.
    So ln|theta_1(z)|^2 ~ ln|theta_1'(0)|^2 + 2*ln|z|.

    The monodronmy (going around z=0) picks up:
      Delta[ln|z|^2] = Delta[ln(z) + ln(z*)] = 2*pi*i for z -> z*e^{2pi*i}
    Wait, ln|z|^2 = ln(z*z) which has no monodromy.

    Actually what we need is: the ANALYTIC continuation.
    The KN factor is |z|^{2*alpha'_e*k.k'} which for the purpose of
    branch cuts means z^{alpha'_e*k.k'} * z_bar^{alpha'_e*k.k'}.
    On the real line, going from tau12 > 0 to tau12 < 0, z = tau12
    crosses through zero. The analytic continuation picks up a phase
    exp(i*pi*alpha'_e*k.k') = exp(i*pi*s_{12}/(2*alpha'_e) ...
    no wait. Let me reconsider.

    For the open string on the boundary, tau12 is real. The OPE is:
      : e^{ik1.X}(tau1) :: e^{ik2.X}(tau2) : ~ |tau12|^{2*alpha'_e*k1.k2} * ...
    When tau12 changes sign (crossing), the log|tau12| is continuous
    but for complex-valued continuation, tau12^{s} has monodromy e^{i*pi*s}.

    On the annulus, theta_1(tau12|it) replaces tau12 for the
    short-distance behavior. Since theta_1 vanishes linearly at tau12=0,
    the monodromy structure is the same: the branch cut discontinuity
    when tau12 passes through 0 is e^{i*pi * s_{12}} where
    s_{12} = 2*alpha'_e*k1.k2.

    The KEY POINT: the monodromy is determined by the ORDER of the zero
    of theta_1, which is 1 (simple zero), independent of t.
    So the branch-cut discontinuity is e^{i*pi*s_{12}} for ALL t.
    The Moyal phase jump is e^{i*k1 wedge k2} for ALL t (it's topological).
    Therefore, if pi*s_{12} = k1 wedge k2 at the NCOS point, the
    monodromy matching holds for ALL t.
    """
    dot12 = -k1[0] * k2[0] + k1[1] * k2[1]  # Minkowski dot
    s12 = 2 * alpha_e * dot12
    return s12


def test_monodromy_t_independence():
    """Test that the monodromy matching holds at each value of t."""
    print("=" * 70)
    print("TEST 1: MONODROMY MATCHING vs ANNULUS MODULUS t")
    print("=" * 70)

    alpha_e = 1.0
    theta01 = 2 * np.pi * alpha_e

    # Massless right-mover
    k1 = (1.5, 1.5)  # right-mover

    # Various other legs (massless and massive)
    test_legs = [
        ((0.8, -0.8), "massless left-mover"),
        ((2.0, 2.0), "massless right-mover"),
        ((1.0, 0.3), "massive (m^2 = 0.91)"),
        ((-0.5, 1.2), "massive (m^2 = 1.19)"),
    ]

    # The monodromy matching identity is:
    # pi * s_{1a} = k1 wedge ka
    # This should hold regardless of t because:
    # - s_{1a} = 2*alpha'_e * k1.ka (the Mandelstam invariant, a property of momenta)
    # - k1 wedge ka = theta01 * (k1_0*ka_1 - k1_1*ka_0) (also a property of momenta)
    # Neither depends on t!

    # The t-dependence in the annulus enters through the Green's function
    # G(tau12, t), but the MONODROMY (branch-cut discontinuity) is determined
    # by the zero structure of theta_1, which is t-independent.

    print("\n  The monodromy matching identity pi*s_{1a} = k1 ^ ka")
    print("  depends only on momenta, NOT on t. Verification:\n")

    for ka, label in test_legs:
        s_1a = 2 * alpha_e * (-k1[0]*ka[0] + k1[1]*ka[1])
        w_1a = theta01 * (k1[0]*ka[1] - k1[1]*ka[0])
        mismatch = abs(cmath.exp(1j * (np.pi * s_1a - w_1a)) - 1.0)
        print(f"  ka = {ka} ({label})")
        print(f"    pi*s_1a = {np.pi*s_1a:.6f}, k1^ka = {w_1a:.6f}, "
              f"|mismatch| = {mismatch:.2e}")

    print("\n  => Identity is t-independent (algebraic, not dynamical).")
    print("     The annulus modulus t does NOT spoil monodromy matching.")


def test_annulus_greens_function_structure():
    """Examine the t-dependence of the annulus Green's function."""
    print("\n" + "=" * 70)
    print("TEST 2: ANNULUS GREEN'S FUNCTION STRUCTURE")
    print("=" * 70)

    alpha_e = 1.0

    # Evaluate the symmetric Green's function at fixed tau12 for various t
    tau12_values = [0.1, 0.2, 0.3]
    t_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n  G_sym(tau12, t) for alpha'_e = {alpha_e}:")
    print(f"  {'tau12':>8s}", end="")
    for t in t_values:
        print(f"  {'t='+str(t):>12s}", end="")
    print()

    for tau12 in tau12_values:
        print(f"  {tau12:8.3f}", end="")
        for t in t_values:
            G = annulus_green_symmetric(tau12, t, alpha_e)
            print(f"  {G:12.4f}", end="")
        print()

    # Show that the MONODROMY is t-independent
    # Near tau12 = 0: G_sym ~ -alpha'_e * ln(theta_1'(0|it)^2) - 2*alpha'_e * ln|tau12|
    # The coefficient of ln|tau12| is -2*alpha'_e, independent of t.
    print(f"\n  Coefficient of ln|tau12| near tau12=0 (should be -2*alpha'_e = {-2*alpha_e}):")

    for t in t_values:
        # Numerical derivative: d G_sym / d ln|tau12| at small tau12
        eps = 1e-6
        tau_small = 0.001
        G1 = annulus_green_symmetric(tau_small, t, alpha_e)
        G2 = annulus_green_symmetric(tau_small * (1 + eps), t, alpha_e)
        dG_dlntau = (G2 - G1) / (np.log(tau_small * (1 + eps)) - np.log(tau_small))
        print(f"    t = {t:5.1f}: dG/d(ln|tau12|) = {dG_dlntau:.6f}")


def test_theta1_zero_structure():
    """Verify that theta_1 has a simple zero at tau12=0 for all t."""
    print("\n" + "=" * 70)
    print("TEST 3: theta_1 ZERO STRUCTURE (simple zero at tau12=0)")
    print("=" * 70)

    t_values = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n  theta_1'(0|it) for various t:")
    for t in t_values:
        th1_prime = jacobi_theta1_derivative(0, t)
        # For comparison: theta_1'(0|it) = 2*pi*eta(it)^3 where eta is Dedekind
        # For large t: theta_1'(0) -> 2*pi*q^{1/4}*(1 + ...) where q = e^{-pi*t}
        print(f"    t = {t:5.1f}: theta_1'(0|it) = {th1_prime:.8f}")

    # Verify theta_1(z) ~ theta_1'(0) * z near z=0
    print(f"\n  Check theta_1(z)/z -> theta_1'(0) as z->0 for t=1.0:")
    t_test = 1.0
    th1_prime = jacobi_theta1_derivative(0, t_test)
    for z in [0.1, 0.01, 0.001, 0.0001]:
        th1 = jacobi_theta1(z, t_test)
        ratio = th1 / z
        print(f"    z = {z:.4f}: theta_1(z)/z = {ratio:.8f}, "
              f"theta_1'(0) = {th1_prime:.8f}, "
              f"|diff| = {abs(ratio - th1_prime):.2e}")


def test_full_annulus_amplitude_toy():
    """Toy model: 3-ordering amplitude with annulus Green's function,
    integrating over both vertex positions AND modulus t.

    A_NCOS = integral_0^infty dt * sum_sigma e^{iPhi_sigma}
             * integral_{chamber_sigma} prod |theta_1(tau_ij|it)|^{s_ij} d tau

    If the monodromy matching holds at each t, the inner integral
    (sum over orderings at fixed t) vanishes, so the full thing vanishes.
    """
    print("\n" + "=" * 70)
    print("TEST 4: TOY ANNULUS AMPLITUDE (t-integrated)")
    print("=" * 70)

    alpha_e = 1.0
    theta01 = 2 * np.pi * alpha_e

    # 3-ordering toy with KN factor replaced by theta_1-based factor
    # Massless right-mover k1 and two other legs
    k1 = (1.0, 1.0)
    k2 = (0.7, -0.7)
    k3 = (-1.7, -0.3)

    s12 = 2 * alpha_e * (-k1[0]*k2[0] + k1[1]*k2[1])
    s13 = 2 * alpha_e * (-k1[0]*k3[0] + k1[1]*k3[1])
    s23 = 2 * alpha_e * (-k2[0]*k3[0] + k2[1]*k3[1])

    w12 = theta01 * (k1[0]*k2[1] - k1[1]*k2[0])
    w13 = theta01 * (k1[0]*k3[1] - k1[1]*k3[0])
    w23 = theta01 * (k2[0]*k3[1] - k2[1]*k3[0])

    print(f"\n  Kinematics: s12={s12:.4f}, s13={s13:.4f}, s23={s23:.4f}")
    print(f"  Wedges:     w12={w12:.4f}, w13={w13:.4f}, w23={w23:.4f}")
    print(f"  Check: pi*s12 = {np.pi*s12:.6f}, w12 = {w12:.6f}, diff = {abs(np.pi*s12 - w12):.2e}")

    # Three orderings of tau2 relative to (tau1=0, tau3=fixed)
    # Phases: ordering (1,2,3): Phi = (w12*sgn(12) + w13*sgn(13) + w23*sgn(23))/2
    # Here sgn indicates the ordering convention.

    # For the contour argument: at fixed t, the integrand for tau2 is
    # |theta_1(tau2|it)|^{s12} * |theta_1(tau2 - tau3|it)|^{s23} * Moyal_phase
    # At the NCOS point, the monodromy matching makes this holomorphic in tau2,
    # so the integral over the full real period vanishes.

    # Numerical test: integrate a simplified version
    # Use t=1 and tau3 = 0.5 (fixed), integrate tau2 over [0,1)
    t_test = 1.0
    tau3 = 0.5
    N_sample = 1000

    # Full amplitude = integral of holomorphic function over a closed cycle
    # At NCOS point, this should vanish. Away from NCOS, it shouldn't.

    def integrand(tau2, t, eps_deform=0.0):
        """Annulus KN-like integrand with Moyal phase."""
        theta_eff = theta01 * (1 + eps_deform)

        tau12 = tau2  # tau1 = 0
        tau23 = tau2 - tau3

        if abs(tau12) < 1e-12 or abs(tau23) < 1e-12:
            return 0.0

        # theta_1 factors (using absolute value for the KN part)
        th12 = jacobi_theta1(tau12, t, nterms=50)
        th23 = jacobi_theta1(tau23, t, nterms=50)

        if abs(th12) < 1e-300 or abs(th23) < 1e-300:
            return 0.0

        # KN factor: |theta_1|^{s_ij}
        # For complex continuation: sign matters
        kn = abs(th12)**s12 * abs(th23)**s23

        # Moyal phase from ordering
        w_eff_12 = theta_eff * (k1[0]*k2[1] - k1[1]*k2[0])
        w_eff_23 = theta_eff * (k2[0]*k3[1] - k2[1]*k3[0])
        phase = 0.5 * w_eff_12 * np.sign(tau12) + 0.5 * w_eff_23 * np.sign(tau23)

        return kn * cmath.exp(1j * phase)

    # Midpoint integration over [eps, 1-eps] (avoiding zeros of theta_1)
    eps_cut = 0.005
    tau2_vals = np.linspace(eps_cut, 1 - eps_cut, N_sample)
    dt = tau2_vals[1] - tau2_vals[0]

    # At NCOS point
    A_ncos = sum(integrand(tau2, t_test, 0.0) for tau2 in tau2_vals) * dt

    # Away from NCOS
    A_deformed = {}
    for eps in [0.01, 0.03, 0.1]:
        A_def = sum(integrand(tau2, t_test, eps) for tau2 in tau2_vals) * dt
        A_deformed[eps] = A_def

    print(f"\n  Annulus toy amplitude at t={t_test}:")
    print(f"    |A_NCOS| = {abs(A_ncos):.2e}  (should be ~0)")
    for eps, A in A_deformed.items():
        print(f"    |A(eps={eps})| = {abs(A):.2e}, |A|/eps = {abs(A)/eps:.4f}")


def test_t_integrated_amplitude():
    """Integrate the toy annulus amplitude over t as well."""
    print("\n" + "=" * 70)
    print("TEST 5: t-INTEGRATED ANNULUS AMPLITUDE")
    print("=" * 70)

    alpha_e = 1.0
    theta01 = 2 * np.pi * alpha_e

    k1 = (1.0, 1.0)
    k2 = (0.7, -0.7)
    k3 = (-1.7, -0.3)

    s12 = 2 * alpha_e * (-k1[0]*k2[0] + k1[1]*k2[1])
    s23 = 2 * alpha_e * (-k2[0]*k3[0] + k2[1]*k3[1])

    tau3 = 0.5
    N_tau = 200
    N_t = 30
    eps_cut = 0.005

    t_values = np.linspace(0.3, 5.0, N_t)  # modulus range
    tau2_vals = np.linspace(eps_cut, 1 - eps_cut, N_tau)
    dtau = tau2_vals[1] - tau2_vals[0]
    dt_mod = t_values[1] - t_values[0]

    def integrand_at_t(tau2, t, eps_deform=0.0):
        theta_eff = theta01 * (1 + eps_deform)
        tau12 = tau2
        tau23 = tau2 - tau3

        if abs(tau12) < 1e-12 or abs(tau23) < 1e-12:
            return 0.0

        th12 = jacobi_theta1(tau12, t, nterms=30)
        th23 = jacobi_theta1(tau23, t, nterms=30)

        if abs(th12) < 1e-300 or abs(th23) < 1e-300:
            return 0.0

        kn = abs(th12)**s12 * abs(th23)**s23

        w_eff_12 = theta_eff * (k1[0]*k2[1] - k1[1]*k2[0])
        w_eff_23 = theta_eff * (k2[0]*k3[1] - k2[1]*k3[0])
        phase = 0.5 * w_eff_12 * np.sign(tau12) + 0.5 * w_eff_23 * np.sign(tau23)

        return kn * cmath.exp(1j * phase)

    # Integrate over both tau2 and t
    A_ncos_total = 0.0
    A_def_total = {0.01: 0.0, 0.05: 0.0}

    contributions_by_t = []

    for t in t_values:
        A_t = sum(integrand_at_t(tau2, t, 0.0) for tau2 in tau2_vals) * dtau
        contributions_by_t.append(abs(A_t))
        A_ncos_total += A_t * dt_mod
        for eps in A_def_total:
            A_t_def = sum(integrand_at_t(tau2, t, eps) for tau2 in tau2_vals) * dtau
            A_def_total[eps] += A_t_def * dt_mod

    print(f"\n  t-integrated annulus amplitude (t in [{t_values[0]:.1f}, {t_values[-1]:.1f}]):")
    print(f"    |A_NCOS| = {abs(A_ncos_total):.2e}")
    for eps, A in A_def_total.items():
        print(f"    |A(eps={eps})| = {abs(A):.2e}, |A|/eps = {abs(A)/eps:.4f}")

    # Show contribution by t
    print(f"\n  |A(t)| at NCOS point for selected t values:")
    for i in range(0, len(t_values), max(1, len(t_values)//8)):
        print(f"    t = {t_values[i]:5.2f}: |A(t)| = {contributions_by_t[i]:.2e}")


if __name__ == "__main__":
    test_monodromy_t_independence()
    test_annulus_greens_function_structure()
    test_theta1_zero_structure()
    test_full_annulus_amplitude_toy()
    test_t_integrated_amplitude()
