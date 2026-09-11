# Lean formalization

Lean 4 (v4.33.1) with Mathlib (tag `v4.33.1`).  Build with

    cd Lean && lake exe cache get && lake build

`lake build` completes with no `sorry` and no additional axiom: every theorem below
depends on `propext`, `Classical.choice`, `Quot.sound` only, which is what
`#print axioms` reports for ordinary Mathlib results.

## What this checks, and what it does not

The three results of `Paper/main.tex` that this folder addresses are

* **Theorem 1**, the elliptic estimates for the Kolmogorov solution `u` and for the
  discrete Poisson solution `φ_h` (Section 2.3, proved in Appendix A);
* **Lemma 1**, the local error decomposition of SRK-I and SRK-II (Section 3.2, proved
  in Appendix B.3);
* **Lemma 2**, the decomposition of the product `ΔQ` (Section 3.2, proved in
  Appendix B.4).

Their statements involve stochastic objects — a Brownian path, the law of the diffusion,
conditional expectations, the reflection coupling of Eberle and the Bismut–Elworthy–Li
derivative formula — for which Mathlib has no library.  A complete formalization of the
three statements is therefore out of reach today, and this folder does not claim one.

What is formalized is the part of each proof that is not stochastic: the analytic
skeleton, the algebra of the two schemes, and every numerical constant.  The stochastic
inputs enter as hypotheses of the Lean statements, in the exact form in which the paper
states them (for instance `‖B_s‖_p ≤ C √s`, or `E[B_s B_h^⊤] = s I`).  Where a
constant is claimed in the paper, the Lean statement carries that constant, so a wrong
coefficient would not compile.

## Poisson/Basic.lean

| Lean name | Paper |
|---|---|
| `exp_neg_le_inv_one_add` | `1 + a ≤ e^a`, used twice in Appendix A |
| `div_one_add_le_one_sub_exp_neg` | `1 - e^{-a} ≥ a/(1+a)`, Appendix A.3 |
| `exp_sub_one_le_mul_exp` | `e^a - 1 ≤ a e^a`, Appendix A.2 |

## Poisson/Theorem1.lean

| Lean name | Paper |
|---|---|
| `duhamel_K` | the Duhamel bound behind `‖K_t[w]‖ ≤ M t e^{2Mt}` (A.2) |
| `theta_source` | the source count `3 M² s e^{3Ms} + M e^{3Ms} = M(1+3Ms)e^{3Ms}` for `Θ` (A.2) |
| `variance_bound` | `(2d/M)(e^{2Mτ}-1) ≤ 4dτe^{2Mτ}`, the variance bound `(eq: variance bound)` (A.2) |
| `decay_of_split` | the split at `t = 1` of Appendix A.3: a bound constant on `[0,1]` and decaying beyond it decays for all `t ≥ 0` |
| `grid_sum_le` | `h/(1 - e^{-λh}) ≤ 2/λ` for `λh ≤ 1` (A.3) |
| `grid_tsum` | `∑_k e^{-λkh} = (1-e^{-λh})⁻¹` |
| `poisson_bound_of_decay` | the final step of A.3: from `‖D u(·,kh)‖ ≤ K e^{-λkh}` to `h ∑_k ‖D u(·,kh)‖ ≤ 2K/λ`, uniformly in `h`, which is `(eq: poisson bound)` |

Not formalized: the reflection coupling (Lemma `lem: contraction`, quoted from Eberle),
the derivative formulas of Lemmas `lem: BEL`, `lem: hessian from lipschitz`,
`lem: third from lipschitz`, and the pointwise bound `(eq: u bound)`.

## Poisson/Lemma1.lean

| Lean name | Paper |
|---|---|
| `DI`, `DIIpm`, `DII` | the stage displacements of `(eq: srk1)`, `(eq: srk2)` and their mean `(eq: mean stage)` |
| `DIIpm_eq` | `D_II^± = D_II ± (1/h) ∫₀ʰ B_s ds` |
| `DII_is_mean` | `½(D_II^+ + D_II^-) = D_II` |
| `two_thirds_DI` | `(2/3) D_I = D_II`, the first identity of Step 1 |
| `square_average`, `cube_average`, `square_average_vec` | the averaging identities behind the second and third identities of Step 1 |
| `srkI_quadratic`, `srkII_quadratic` | `(2/3)(3/2)² = 3/2` and `½((1+1/√2)²+(1-1/√2)²) = 3/2`, Step 3 |
| `srkI_cubic`, `srkII_cubic` | `(2/3)(3/2)³ = 9/4` and `½((1+1/√2)³+(1-1/√2)³) = 5/2`, Step 4, that is `c₃ = 9/4` and `c₃ = 5/2` |
| `c_one_values` | `c₁ = w/2`, that is `3/4` and `1/2`, Step 3 |
| `step4_coefficient` | `(1/6) c₃ h (√2/h)³ = (√2/3)(c₃/h²)`, the fourth coefficient of `(eq: martingale Z)` |
| `step5_centering` | `h²/2 - (3/(2h))(h³/3) = 0`, the centering of Step 5 |
| `step5_coefficient_unique` | `3/(2h)` is the only weight that centres that term (not in the paper) |
| `taylor_remainder_constant` | `(1/6)∫₀¹(1-θ)³dθ = 1/24`, giving `|r(v)| ≤ (M/24)|v|⁴` |
| `moment_integral_one`, `moment_integral_two` | the two `h^{5/2}` integrals of Step 6 |
| `step6_sum` | the four bounds of Step 6 sum to `C h²(1+|Z₀|)` for `h ≤ 1` |

Not formalized: the Taylor expansion of `∇U` as an identity between random vectors, and
the moment bounds of the Brownian functionals, which are hypotheses.

## Poisson/Lemma2.lean

| Lean name | Paper |
|---|---|
| `pairing_integral_one` | `∫₀ʰ (h-s) s ds = h³/6` |
| `pairing_integral_three` | `∫₀ʰ (s - c₁h) s ds = h³(1/3 - c₁/2)` |
| `pairing_integral_four` | `∫₀ʰ s² ds = h³/3` |
| `pairing_term_one` | first term of `(eq: pairing identity)`: `-√2 · 2√2 · h³/6 = -2h³/3` |
| `pairing_term_three` | third term: `-4h³(1/3 - c₁/2)` |
| `pairing_term_three_values` | its two specializations, `+h³/6` for `c₁ = 3/4` and `-h³/3` for `c₁ = 1/2` |
| `pairing_term_four_bracket` | `h³/3 - (c₃/h²)(h³/3)(h²/2) = h³(1/3 - c₃/6)` |
| `pairing_term_four` | fourth term: `2√2 · (√2/3) · 3 · h³(1/3 - c₃/6) = (4h³/3)(1 - c₃/2)` |
| `pairing_srkI`, `pairing_srkII` | the two conditional means quoted in Appendix B.4, Step 1 |
| `product_mean_zero_order`, `product_remainder_order` | the orders `h^{5/2}` and `h³` of `(eq: cross moment)` |

Not formalized: the Gaussian moment formulas `E[B_{s,j}B_{s,k}B_{s,l}B_{h,m}] = σ_{jklm}s²`
and its counterpart for `∫₀ʰ B_s ds`, and the parity argument that kills the second term;
these are the hypotheses under which the terms above are computed.

## Result

No discrepancy was found between the Lean development and `Paper/main.tex`.  Every
constant checked here — `3/4`, `1/2`, `9/4`, `5/2`, `3/(2h)`, `√2/3`, `3h³/8`, `h³/4`,
`h³/6`, `1/3 - c₁/2`, `1/3 - c₃/6`, `4h³/3`, `1/24`, `2/λ` — is the constant the paper
prints.
