# Scratch Results Snapshot

Date: 2026-02-26

## 1) `phase_structure_4pt.py`

- Generic phase-only matrix (40 random samples, 3-ordering basis) has nullspace dimension `0`.
- Interpretation:
  - Phase factors alone do not produce a universal cancellation identity.
  - Special cancellation loci exist, but they are nongeneric.

## 2) `total_derivative_toy.py`

- Shared ordered kernel + critical phases: cancellation is exact (numerically up to machine precision).
- Slight kernel mismatch: nonzero residual appears immediately.
- Interpretation:
  - Equal/related ordered kernels are essential.
  - Supports contour/cohomological/localization mechanisms over phase-only explanations.

## 3) `transparency_superselection_scan.py`

- For a massless leg, random kinematics gives very small transparency fraction (`~10^-4` level).
- Exact transparency occurs on restricted chiral loci.
- Wedge values are continuous over sampled kinematics.
- Interpretation:
  - Generic transparent-sector explanation is weak.
  - Simple discrete superselection from zero-mode algebra alone is not evident.

## 4) `ordering_kernel_sensitivity.py`

- Mean residual scales linearly with kernel mismatch:
  - `mean|A|/eps ~ 1.2-1.3` over two decades of `eps`.
- Interpretation:
  - Cancellation robustness is controlled by how tightly dynamics enforces kernel relations.
  - Strong quantitative support for mechanisms requiring kernel identities.

## 5) `phase_structure_5pt.py`

- 5-point phase-only matrix `(180, 24)` (24 orderings with 1 fixed) has nullspace dimension `0`.
- Mean absolute full-ordering phase sum in sample is `~4.10` (minimum `~9e-3`), i.e. not generically near zero.
- Interpretation:
  - Even at higher point with physical random 1+1 kinematics, phases alone do not produce a universal decoupling identity.
  - Reinforces that mechanism (1) needs mechanism (2)/(5)-type kernel relations.

## 6) `branch_cut_monodromy_toy.py`

- In a 3-ordering toy model with additional monodromy weights:
  - `|A(0)| ~ 3e-16` (exact cancellation at the critical/holomorphic point),
  - `|A(10^-3)| ~ 1.09e-2`,
  - linear slope near `nu=0` is `~10.88`.
- Interpretation:
  - Cancellation is lifted linearly by branch-cut monodromy mismatch.
  - Supports the claim that D1 holomorphy is structurally special.

## 7) `integrability_kinematics_scan.py`

- Toy Regge spectrum (`m_n = sqrt(n)`) channel counts:
  - At `E_cm=6`, `#(2->2)=94`, `#(2->3)=71`.
  - At `E_cm=8`, `#(2->2)=190`, `#(2->3)=376`.
- Interpretation:
  - Particle-production channels proliferate rapidly.
  - If mechanism (6) is true, it must be due to dynamical amplitude zeros/symmetry, not kinematic exclusion.

## 8) `phase_rank_multiplicity_scan.py`

- Phase-only rank scan (fix leg 1):
  - n=4: cols=6, rank=6, nullity=0
  - n=5: cols=24, rank=24, nullity=0
  - n=6: cols=120, rank=120, nullity=0
  - `mean|sum phases|/sqrt(cols)` is `O(1)` (`0.86, 0.66, 0.88`), consistent with random-phase scaling.
- Interpretation:
  - No momentum-independent phase-only null vector appears through at least 6 points.
  - Strengthens the claim that phase monodromy alone is not the intrinsic mechanism.

## 9) `annulus_modulus_toy.py`

- Two-modulus toy (integrate `t in [0,1]`) gives:
  - shared-kernel/shared-phase: `|A| ~ 1.1e-16`,
  - modulus-dependent phase drift: `|A| ~ 1.9e-1`,
  - modulus-dependent kernel mismatch: `|A| ~ 8.0e-3`.
- Small phase-drift scaling is linear:
  - `|A_lambda|/lambda ~ 2.35e-1` over `lambda=10^-3..3x10^-2`.
- Interpretation:
  - One-loop extension is highly sensitive to modulus-dependent mismatch.
  - Suggests one-loop decoupling needs `t`-local kernel identities, not only integrated phase cancellations.

## 10) `nsr_derivative_ansatz.py`

- In a KN-like times meromorphic-transverse toy:
  - `Integral(I_naive) = 0.69005`,
  - correction from transverse derivative piece `Integral(Delta) = -0.41005`,
  - full BRST-style derivative `Integral(I_full) = 0.28000`,
  - boundary value `(L*T)|_1-(L*T)|_0 = 0.28000`.
- Interpretation:
  - Meromorphic transverse factors are compatible with localization.
  - Exact localization requires derivative acting on the full product (BRST-completed structure), not on longitudinal factor alone.

## 11) `nonplanar_2plus2_cylinder.py`

- Explicit nonplanar 2+2 singular model:
  - `A_n(s) ~ (s_n - s - i0)^(-1/2)`.
  - For `(alpha', Etilde, K_T^2, n)=(0.03, 0.97, 1, 1)`, `s_n = 2271.98364354`.
- Discontinuity scaling above threshold:
  - `|Disc A_n| * sqrt(s-s_n) ~ 3.5449` from `s-s_n=10^-4` to `10^-2` (constant).
- NCOS scaling (`alpha'=alpha'_e(1-E^2)`, `alpha'_e=1`, `K_T^2=1`, `n=1`):
  - `s_n = 4.09e2, 4.48e3, 4.01e4, 4.45e5, 4.00e6` as `delta=1-E^2` decreases.
- Interpretation:
  - Explicitly confirms cut-type singularity (not pole) in the 2+2 nonplanar channel.
  - Confirms thresholds are pushed to arbitrarily large `s` in the NCOS limit.

## 12) `monodromy_matching.py`

- Monodromy matching identity `pi*s_{1a} = k1 ^ ka` verified exactly (to machine precision) for massless right-mover k1 vs all other legs (massless or massive).
- For left-movers: `pi*s_{1a} = -k1 ^ ka` (holomorphy in LHP instead of UHP).
- Massive legs: pairwise mismatch is O(1), confirming massive sector interacts.
- n-point generalization: matching exact through n=10 with random kinematics.
- Interpretation:
  - The monodromy matching is a purely algebraic identity requiring only k1 to be massless.
  - Chirality determines which half-plane gives holomorphy.

## 13) `ncos_amplitude_vanishing.py`

- 4-point NCOS amplitude with one massless leg: `|A| ~ 4e-16` (exact zero) for 3 non-degenerate kinematic cases.
- Phase matching: NCOS Moyal phases reproduce monodromy relation coefficients `1, e^{i*pi*s12}, e^{i*pi*(s12+s23)}` exactly.
- All-massive 4-point: `|A_NCOS| ~ 20.8` (non-zero, confirming interacting massive sector).
- Deformation: `|A(eps)|/eps = 2.5061` constant over 5 decades of eps (sharp linear zero).
- n-point monodromy matching exact through n=10.
- Interpretation:
  - Massless decoupling at NCOS point is exact at the amplitude level.
  - The critical value theta = 2*pi*alpha'_e is a sharp zero (linear deformation), not a smooth minimum.

## 14) `annulus_greens_function.py`

- Monodromy matching identity is purely algebraic (momentum-dependent, NOT modulus-dependent).
- Annulus Green's function: coefficient of `ln|tau12|` is `-2*alpha'_e` independent of t (verified for t=0.5..10).
- `theta_1(z|it)` has a simple zero at z=0 for all t > 0 (verified numerically).
- Therefore: the branch-cut monodromy structure is identical at tree level and one loop.
- **Key result**: The one-loop contour argument has exactly the same monodromy matching as tree level, so massless decoupling extends to one loop without extra assumptions.
- Interpretation:
  - Resolves the main concern from the toy annulus test (item 9): the identity holds pointwise in t, not just on average.
  - At any genus, the boundary Green's function has the same simple-zero structure, suggesting all-genus decoupling.

## 15) `massive_subsector_monodromy.py`

- Monodromy matching for massless right-mover vs massive legs: mismatch = 0 for ALL mass levels (N=1..20) and ALL chirality ratios (r=0.01..1000).
- Mixed amplitude (2 massless + 2 massive): phases match monodromy relation exactly, so A_NCOS = 0 even for massless scattering off massive states.
- Between two massive legs: mismatch generically O(1) (no "almost-decoupled" massive subsector).
- Mismatch is 0 for m^2=0, jumps to O(1) for any nonzero m^2 — discontinuous.
- Interpretation:
  - The massless photon is exactly transparent to the entire massive tower.
  - The mechanism depends only on the massless leg being a right- or left-mover, not on what it scatters with.

## 16) `multipoint_contour_test.py`

- Holomorphic factorization identity verified:
  `|z-a|^s * exp(-i*pi*s*sgn(z-a)/2) = lim_{eps->0+} (z+i*eps-a)^s * exp(-i*pi*s/2)`
  Holds to machine precision for all z (both z>a and z<a) with the correct Moyal sign convention.
- 5-point monodromy matching: all pairwise matchings exact for massless right-mover k1.
- UHP decay: for 4-point with massless leg, the holomorphic integrand decays as `z1^{-s23}` in the UHP.
- Interpretation:
  - The Moyal phase converts the non-holomorphic |z|^s to the UHP boundary value of holomorphic z^s.
  - This is the precise content of mechanism (2): the BCFT produces a total derivative in the massless insertion coordinate.

## 17) `annulus_q_expansion_check.py`

- Symbolic low-order expansions from the Jacobi products (`c = cos(2*pi*nu)`):
  - `P_e = 1 - 2c q^2 + (1 - 2c) q^4 + (4c^2 - 2c) q^6 + O(q^8)`.
  - `P_o = 1 - 2c q + q^2 - 2c q^3 + 4c^2 q^4 - 4c q^5 + (4c^2 + 1) q^6 + O(q^7)`.
- Theta-product identity checks:
  - `theta_1(pi nu,q) = 2 f(q^2) q^(1/4) sin(pi nu) P_e`,
  - `theta_4(pi nu,q) = f(q^2) P_o`,
  - relative errors at sample points are `~10^-16`.
- Truncated-series accuracy:
  - At `q=0.04..0.08`, absolute errors for truncation at `q^6` are `~10^-12..10^-8`.
- Interpretation:
  - Confirms the explicit q-expansion step used in the annulus-integrand derivation is correct.
  - Supports the decomposition `I(q,nu)=sum_r q^r I_r(nu)` used before channel symmetrization.

## 18) `theta_expansion_cylinder.py`

- Independent reproduction of the Codex agent's theta-function expansion analysis for the 2+2 nonplanar cylinder.
- Jacobi product forms P_e(nu,q) and P_o(nu,q) verified against theta_1 and theta_4 series to machine precision (~1e-16).
- Small-q expansions confirmed: P_e = 1 - 2q^2 cos(2*pi*nu) + O(q^4), P_o = 1 - 2q cos(2*pi*nu) + O(q^2).
- Jacobi abstruse identity theta_3^4 - theta_4^4 - theta_2^4 = 0 verified to ~1e-16 for t=1,2,3 (key ingredient for GSO projection removing odd levels).
- Bessel-K structure of q-integral f_r verified:
  - Analytic formula f_r = 2*(Y^2/a)^{nu/2} K_nu(2*sqrt(a*Y^2)) matches direct numerical integration to ratio 1.000010.
  - Near threshold (a = r - s_CL/4 -> 0): f_r * a -> 1 (1/a divergence, branch point).
- NCOS threshold scaling: s_n = E^2 K_T^2/(1-E^2) + 4n/(alpha'_e(1-E^2)^2) -> infinity, verified.
- Discontinuity: |Disc A| * sqrt(s - M^2) = 2.000 (constant) across 7 decades above threshold, confirming branch cut (not pole).
- Continuous q_1 integral: int dq1/(q1^2 + Delta) = pi/sqrt(Delta) verified for Delta > 0.
- Interpretation:
  - All elements of the Codex analysis are independently confirmed.
  - The nonplanar 2+2 cylinder has branch-point (cut) singularities, not poles.
  - NCOS limit pushes all closed-channel thresholds to infinite s.
  - GSO + symmetrization removes odd levels, leaving only even-level Bessel-type contributions.

## Current mechanism ranking (from scratch evidence)

1. BRST/cohomological total-derivative (mechanism 2): **strongest** — now identified with the holomorphic factorization identity. The Moyal phase IS the "BRST completion" that absorbs branch-cut discontinuities.
2. Moduli-space localization (mechanism 5): **equivalent to (2)** — the holomorphic factorization holds fiber-by-fiber over moduli space (annulus modulus t tested explicitly).
3. Critical monodromy matching (mechanism 1): **necessary ingredient** — the identity pi*s = k^k is the algebraic core, but its consequence is holomorphic factorization, not phase-only cancellation.
4. Braided transparency (mechanism 3): possible algebraic repackaging, not independently powerful.
5. Zero-mode superselection (mechanism 4): weak in noncompact generic kinematics.
6. Emergent integrability (mechanism 6): kinematically unconstrained; remains open.

## Next concrete scratch target

- Complete the contour argument with full NSR structure (oscillator contractions, spin structure, superghost correlators) to show the holomorphic factorization extends to the full superstring integrand.
- Test the all-genus conjecture: build a genus-2 toy (3 moduli) and verify that the simple-zero structure of the Green's function is moduli-independent.
- Map out the deformation coefficient c(s_ij, n) across kinematic space for comparison with parent D-brane amplitude near E=1^-.
- Investigate whether the massive-sector S-matrix has enhanced symmetry (integrability) by looking for vanishing of specific production amplitudes.
