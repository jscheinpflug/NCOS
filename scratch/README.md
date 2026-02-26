# Scratch Calculations: Intrinsic NCOS Mechanisms

This folder contains quick calculations to stress-test mechanism ideas in `INTRINSIC_NCOS.tex`.

## Files

- `phase_structure_4pt.py`
  - Tests whether phase factors alone admit a momentum-independent null vector across orderings.
  - Also finds special kinematic points where phase-only cancellation happens.
  - Relevant to mechanisms 1 and 3.

- `total_derivative_toy.py`
  - Demonstrates why cancellation is robust only when ordered kernels are equal/related by total-derivative/contour arguments.
  - Relevant to mechanisms 2 and 5.

- `transparency_superselection_scan.py`
  - Monte-Carlo probe of whether a massless leg is generically transparent in braided phases.
  - Checks whether wedge values look discretized (simple superselection proxy) or continuous.
  - Relevant to mechanisms 3 and 4.

- `ordering_kernel_sensitivity.py`
  - Quantifies how cancellation residuals scale when ordered kernels are slightly unequal.
  - Relevant to mechanisms 2 and 5 (need for contour/cohomological kernel identities).

- `phase_structure_5pt.py`
  - Extends the phase-only null-vector test to 5-point orderings with physical random 1+1 kinematics.
  - Relevant to mechanisms 1 and 3.

- `branch_cut_monodromy_toy.py`
  - Toy model for how branch-cut monodromies spoil critical-phase cancellation away from the holomorphic D1 setup.
  - Relevant to mechanisms 1, 2, and the higher-dimensional caveat.

- `integrability_kinematics_scan.py`
  - Counts kinematically allowed 2->2 and 2->3 channels in a toy Regge-tower spectrum.
  - Relevant to mechanism 6 (integrability cannot follow from kinematics alone).

- `phase_rank_multiplicity_scan.py`
  - Scans phase-only matrix rank for n=4,5,6 orderings.
  - Tests whether any momentum-independent phase-only null vector survives at higher multiplicity.
  - Relevant to mechanisms 1 and 2.

- `annulus_modulus_toy.py`
  - Adds a second modulus (annulus-like) and tests cancellation robustness under modulus-dependent phase/kernel mismatch.
  - Relevant to one-loop extension of mechanisms 1/2/5.

- `nsr_derivative_ansatz.py`
  - Minimal derivative-decomposition toy with longitudinal KN-like factor times meromorphic transverse factor.
  - Shows that BRST-completed derivative of the full product is needed for exact localization.
  - Relevant to mechanisms 2 and 5.

- `nonplanar_2plus2_cylinder.py`
  - Explicit singular-piece check for the nonplanar 2+2 annulus channel.
  - Verifies square-root branch behavior and NCOS scaling of branch-point thresholds.
  - Relevant to the cylinder closed-channel test.

- `monodromy_matching.py`
  - Verifies the critical monodromy matching identity pi*s_{1a} = k1 ^ ka for massless right-movers.
  - Tests massless vs massive, left-mover sign flip, n-point generalization through n=10.
  - Relevant to mechanism 1 and the holomorphic factorization identity.

- `ncos_amplitude_vanishing.py`
  - Full numerical verification that the NCOS 4-point amplitude vanishes with one massless leg.
  - Tests deformation away from NCOS point (linear scaling |A|/eps = const).
  - Tests all-massive amplitude (non-zero, confirming interacting massive sector).
  - Relevant to all mechanisms.

- `annulus_greens_function.py`
  - Tests whether the monodromy matching identity holds at each value of the annulus modulus t.
  - Verifies t-independence of ln|tau12| coefficient in the annulus Green's function.
  - Checks theta_1 simple-zero structure at all t.
  - Key result: one-loop contour argument has same monodromy as tree level.
  - Relevant to one-loop extension of mechanisms 1/2/5.

- `massive_subsector_monodromy.py`
  - Exhaustive scan of monodromy mismatch vs chirality ratio and mass level.
  - Tests mixed (2 massless + 2 massive) amplitude vanishing.
  - Shows no "almost-decoupled" massive subsector exists.
  - Relevant to massive sector structure and mechanism 1.

- `multipoint_contour_test.py`
  - Verifies the holomorphic factorization identity with correct Moyal sign convention.
  - Tests 4-point and 5-point contour integrals.
  - Checks UHP decay for contour closure.
  - Relevant to mechanism 2 (BRST/cohomological total-derivative).

- `annulus_q_expansion_check.py`
  - Derives low-order q-series coefficients of the annulus Jacobi-product factors `P_e` and `P_o`.
  - Verifies numerically the theta-product identities used in the nonplanar 2+2 derivation.
  - Relevant to the explicit annulus-integrand expansion step.

- `theta_expansion_cylinder.py`
  - Independent reproduction of the full theta-function expansion analysis for the 2+2 nonplanar cylinder.
  - Verifies Jacobi product forms, small-q expansions, Jacobi abstruse identity (GSO), Bessel-K structure of q-integral, NCOS threshold scaling, and branch-cut discontinuity structure.
  - Relevant to the full cylinder diagram analysis and closed-string decoupling.

## Run

```bash
cd /Users/xiyin/ResearchIdeas/NCOS/scratch
python3 phase_structure_4pt.py
python3 total_derivative_toy.py
python3 transparency_superselection_scan.py
python3 ordering_kernel_sensitivity.py
python3 phase_structure_5pt.py
python3 branch_cut_monodromy_toy.py
python3 integrability_kinematics_scan.py
python3 phase_rank_multiplicity_scan.py
python3 annulus_modulus_toy.py
python3 nsr_derivative_ansatz.py
python3 nonplanar_2plus2_cylinder.py
python3 monodromy_matching.py
python3 ncos_amplitude_vanishing.py
python3 annulus_greens_function.py
python3 massive_subsector_monodromy.py
python3 multipoint_contour_test.py
python3 annulus_q_expansion_check.py
python3 theta_expansion_cylinder.py
```

## Interpretation

- If `phase_structure_4pt.py` reports full rank, phases alone are not enough generically.
- If `total_derivative_toy.py` shows exact cancellation only in the shared-kernel case, that supports the cohomological/localization mechanism as the stronger explanation.
- If `transparency_superselection_scan.py` reports near-zero transparency fraction at random kinematics, transparency/superselection mechanisms are likely subleading unless extra structure is added.
- If `ordering_kernel_sensitivity.py` shows linear growth of residuals with kernel mismatch, then equal-kernel identities are quantitatively essential.
- If `phase_structure_5pt.py` is full-rank phase-only, higher-point simplification also needs dynamical kernel identities.
- If `branch_cut_monodromy_toy.py` shows linear lift of cancellation at nonzero branch exponent, contour obstruction away from D1 is expected.
- If `integrability_kinematics_scan.py` shows many open 2->3 channels, integrability must come from dynamical zeros/selection rules.
- If `phase_rank_multiplicity_scan.py` stays full rank as n increases, phase-only null vectors are not the all-multiplicity mechanism.
- If `annulus_modulus_toy.py` shows linear mismatch sensitivity in the modulus integral, one-loop decoupling needs t-by-t kernel identities.
- If `nsr_derivative_ansatz.py` shows a finite correction term from the transverse derivative piece, BRST completion is structurally essential even with meromorphic transverse factors.
- If `nonplanar_2plus2_cylinder.py` shows constant `|Disc A| sqrt(s-s_th)` and rapidly growing `s_th` in NCOS scaling, the channel has a cut (not pole) and decouples at finite energy.
- If `monodromy_matching.py` gives zero mismatch for all massless right-mover vs arbitrary-leg pairs, the matching is a universal algebraic identity.
- If `ncos_amplitude_vanishing.py` shows |A| ~ machine epsilon for massless and O(1) for massive, the decoupling is exact and the massive sector interacts.
- If `annulus_greens_function.py` shows the ln|tau12| coefficient is -2*alpha'_e independent of t, the one-loop monodromy structure is identical to tree level.
- If `massive_subsector_monodromy.py` shows zero mismatch at all mass levels/chiralities for massless reference but O(1) between massive pairs, the massless photon is exactly transparent to the entire massive tower.
- If `multipoint_contour_test.py` confirms the holomorphic factorization identity with the correct sign, the Moyal phase literally converts |z|^s to the boundary of holomorphic z^s.
- If `annulus_q_expansion_check.py` reproduces the first q-coefficients and matches theta identities to machine precision, the annulus q-expansion step used in the derivation is reliable.
- If `theta_expansion_cylinder.py` confirms Bessel-K structure, constant |Disc|*sqrt(s-M^2), and diverging NCOS thresholds, the full cylinder analysis is independently verified.
