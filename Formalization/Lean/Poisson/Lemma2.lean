/-
Lemma 2 of `Paper/main.tex` (decomposition of the product `Delta Q`),
Appendix B.1 and B.3.

The content of Lemma 2 that is not algebra common with Lemma 1 is the pairing identity
`(eq: pairing identity)`: the conditional mean of `2√2 M(ΔZ₀) B_h^⊤`.  Its three
non-vanishing terms are computed here from the moments of the Brownian path, which are
the hypotheses of each statement, and every constant of the paper is checked.
-/
import Poisson.Basic

namespace Poisson

open Real intervalIntegral

/-! ### The two time integrals of the pairing identity -/

/-- First pairing, from `E[B_s B_h^⊤] = s I`: `∫₀ʰ (h - s) s ds = h³/6`. -/
theorem pairing_integral_one (h : ℝ) :
    (∫ s in (0:ℝ)..h, (h - s) * s) = h ^ 3 / 6 := by
  have hderiv : ∀ s ∈ Set.uIcc (0:ℝ) h,
      HasDerivAt (fun x : ℝ => (h/2) * x ^ 2 - (1/3 : ℝ) * x ^ 3) ((h - s) * s) s := by
    intro s _
    have d1 : HasDerivAt (fun x : ℝ => (h/2) * x ^ 2) ((h/2) * (2 * s ^ 1)) s :=
      (hasDerivAt_pow 2 s).const_mul (h/2)
    have d2 : HasDerivAt (fun x : ℝ => (1/3 : ℝ) * x ^ 3) ((1/3 : ℝ) * (3 * s ^ 2)) s :=
      (hasDerivAt_pow 3 s).const_mul (1/3 : ℝ)
    have h3 := d1.sub d2
    have heq : (h/2) * (2 * s ^ 1) - (1/3 : ℝ) * (3 * s ^ 2) = (h - s) * s := by ring
    rw [heq] at h3
    exact h3
  have hcont : Continuous fun s : ℝ => (h - s) * s :=
    (continuous_const.sub continuous_id).mul continuous_id
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv
    (hcont.intervalIntegrable (μ := MeasureTheory.volume) 0 h)]
  ring

/-- Third pairing, from `E[B_s B_h^⊤] = s I`: `∫₀ʰ (s - c₁ h) s ds = h³ (1/3 - c₁/2)`. -/
theorem pairing_integral_three (h c₁ : ℝ) :
    (∫ s in (0:ℝ)..h, (s - c₁ * h) * s) = h ^ 3 * (1/3 - c₁ / 2) := by
  have hderiv : ∀ s ∈ Set.uIcc (0:ℝ) h,
      HasDerivAt (fun x : ℝ => (1/3 : ℝ) * x ^ 3 - (c₁ * h / 2) * x ^ 2) ((s - c₁ * h) * s) s := by
    intro s _
    have d1 : HasDerivAt (fun x : ℝ => (1/3 : ℝ) * x ^ 3) ((1/3 : ℝ) * (3 * s ^ 2)) s :=
      (hasDerivAt_pow 3 s).const_mul (1/3 : ℝ)
    have d2 : HasDerivAt (fun x : ℝ => (c₁ * h / 2) * x ^ 2) ((c₁ * h / 2) * (2 * s ^ 1)) s :=
      (hasDerivAt_pow 2 s).const_mul (c₁ * h / 2)
    have h3 := d1.sub d2
    have heq : (1/3 : ℝ) * (3 * s ^ 2) - (c₁ * h / 2) * (2 * s ^ 1) = (s - c₁ * h) * s := by ring
    rw [heq] at h3
    exact h3
  have hcont : Continuous fun s : ℝ => (s - c₁ * h) * s :=
    (continuous_id.sub continuous_const).mul continuous_id
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv
    (hcont.intervalIntegrable (μ := MeasureTheory.volume) 0 h)]
  ring

/-- Fourth pairing, the time integral of the fourth moment: `∫₀ʰ s² ds = h³/3`. -/
theorem pairing_integral_four (h : ℝ) : (∫ s in (0:ℝ)..h, s ^ 2) = h ^ 3 / 3 := by
  have hderiv : ∀ s ∈ Set.uIcc (0:ℝ) h,
      HasDerivAt (fun x : ℝ => (1/3 : ℝ) * x ^ 3) (s ^ 2) s := by
    intro s _
    have d1 : HasDerivAt (fun x : ℝ => (1/3 : ℝ) * x ^ 3) ((1/3 : ℝ) * (3 * s ^ 2)) s :=
      (hasDerivAt_pow 3 s).const_mul (1/3 : ℝ)
    have heq : (1/3 : ℝ) * (3 * s ^ 2) = s ^ 2 := by ring
    rw [heq] at d1
    exact d1
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv
    ((continuous_pow 2).intervalIntegrable (μ := MeasureTheory.volume) 0 h)]
  ring

/-! ### The three terms of the pairing identity -/

/-- First term of `(eq: pairing identity)`: the `∇²U` term of the mean-zero term pairs
with `2√2 B_h^⊤` to `-(2h³/3)(∇²U)²`.  The scalar computation is
`-√2 · 2√2 · h³/6 = -2h³/3`, the same for both integrators. -/
theorem pairing_term_one (h : ℝ) :
    -(Real.sqrt 2) * (2 * Real.sqrt 2) * (h ^ 3 / 6) = -(2 * h ^ 3 / 3) := by
  have h2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  calc -(Real.sqrt 2) * (2 * Real.sqrt 2) * (h ^ 3 / 6)
      = -(2 * (Real.sqrt 2 * Real.sqrt 2)) * (h ^ 3 / 6) := by ring
    _ = -(2 * 2) * (h ^ 3 / 6) := by rw [h2]
    _ = -(2 * h ^ 3 / 3) := by ring

/-- Third term of `(eq: pairing identity)`: the `∇³U[∇U]` term pairs to
`-4h³(1/3 - c₁/2)`. -/
theorem pairing_term_three (h c₁ : ℝ) :
    -(Real.sqrt 2) * (2 * Real.sqrt 2) * (h ^ 3 * (1/3 - c₁ / 2))
      = -(4 * h ^ 3 * (1/3 - c₁ / 2)) := by
  have h2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  calc -(Real.sqrt 2) * (2 * Real.sqrt 2) * (h ^ 3 * (1/3 - c₁ / 2))
      = -(2 * (Real.sqrt 2 * Real.sqrt 2)) * (h ^ 3 * (1/3 - c₁ / 2)) := by ring
    _ = -(2 * 2) * (h ^ 3 * (1/3 - c₁ / 2)) := by rw [h2]
    _ = -(4 * h ^ 3 * (1/3 - c₁ / 2)) := by ring

/-- Its two specializations: `+h³/6` for SRK-I (`c₁ = 3/4`) and `-h³/3` for SRK-II
(`c₁ = 1/2`). -/
theorem pairing_term_three_values (h : ℝ) :
    -(4 * h ^ 3 * (1/3 - (3/4 : ℝ) / 2)) = h ^ 3 / 6 ∧
    -(4 * h ^ 3 * (1/3 - (1/2 : ℝ) / 2)) = -(h ^ 3 / 3) := by
  constructor <;> ring

/-- Fourth term of `(eq: pairing identity)`.  The bracket pairs with `B_{h,m}` to
`σ_{jklm} h³ (1/3 - c₃/6)`: the exact part contributes `h³/3` and the stage part, carried
by the weight `c₃/h²`, contributes `(c₃/h²)(h³/3)(h²/2) = c₃h³/6`. -/
theorem pairing_term_four_bracket {h c₃ : ℝ} (hh : h ≠ 0) :
    h ^ 3 / 3 - (c₃ / h ^ 2) * ((h ^ 3 / 3) * (h ^ 2 / 2)) = h ^ 3 * (1/3 - c₃ / 6) := by
  field_simp; ring

/-- Fourth term of `(eq: pairing identity)`, assembled.  The three Kronecker products of
`σ` each contract to `(∇²ΔU)_{im}`, so the pairing carries the factor `3`, and
`2√2 · (√2/3) · 3 · h³(1/3 - c₃/6) = (4h³/3)(1 - c₃/2)`. -/
theorem pairing_term_four (h c₃ : ℝ) :
    (2 * Real.sqrt 2) * (Real.sqrt 2 / 3) * 3 * (h ^ 3 * (1/3 - c₃ / 6))
      = (4 * h ^ 3 / 3) * (1 - c₃ / 2) := by
  have h2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  calc (2 * Real.sqrt 2) * (Real.sqrt 2 / 3) * 3 * (h ^ 3 * (1/3 - c₃ / 6))
      = 2 * (Real.sqrt 2 * Real.sqrt 2) * (h ^ 3 * (1/3 - c₃ / 6)) := by ring
    _ = 2 * 2 * (h ^ 3 * (1/3 - c₃ / 6)) := by rw [h2]
    _ = (4 * h ^ 3 / 3) * (1 - c₃ / 2) := by ring

/-- The conditional mean of `2√2 M(ΔZ₀) B_h^⊤` for SRK-I, `c₁ = 3/4`, `c₃ = 9/4`:
`-(h³/6)(4(∇²U)² - ∇³U[∇U] + ∇²ΔU)`, the value stated in Appendix B.3, Step 1. -/
theorem pairing_srkI (h : ℝ) :
    -(2 * h ^ 3 / 3) = -(h ^ 3 / 6) * 4 ∧
    -(4 * h ^ 3 * (1/3 - (3/4 : ℝ) / 2)) = -(h ^ 3 / 6) * (-1) ∧
    (4 * h ^ 3 / 3) * (1 - (9/4 : ℝ) / 2) = -(h ^ 3 / 6) * 1 := by
  refine ⟨by ring, by ring, by ring⟩

/-- The same for SRK-II, `c₁ = 1/2`, `c₃ = 5/2`:
`-(h³/3)(2(∇²U)² + ∇³U[∇U] + ∇²ΔU)`. -/
theorem pairing_srkII (h : ℝ) :
    -(2 * h ^ 3 / 3) = -(h ^ 3 / 3) * 2 ∧
    -(4 * h ^ 3 * (1/3 - (1/2 : ℝ) / 2)) = -(h ^ 3 / 3) * 1 ∧
    (4 * h ^ 3 / 3) * (1 - (5/2 : ℝ) / 2) = -(h ^ 3 / 3) * 1 := by
  refine ⟨by ring, by ring, by ring⟩

/-! ### The orders of Lemma 2 -/

/-- Step 2 of Appendix B.3: the mean-zero term of the product is of order `h^{5/2}`,
from `‖M(ΔZ₀)‖₄ ≤ C h² (1+|Z₀|)` and `‖B_h‖₄ ≤ C √h`. -/
theorem product_mean_zero_order {C h Z : ℝ} (hC : 0 ≤ C) (hh : 0 < h) (hZ : 0 ≤ Z) :
    (C * h ^ 2 * (1 + Z)) * (C * Real.sqrt h) = C ^ 2 * (h ^ 2 * Real.sqrt h) * (1 + Z) := by
  ring

/-- And the remainder is of order `h³`: `h^{5/2} · √h = h³`. -/
theorem product_remainder_order {h : ℝ} (hh : 0 ≤ h) :
    (h ^ 2 * Real.sqrt h) * Real.sqrt h = h ^ 3 := by
  have hs : Real.sqrt h * Real.sqrt h = h := Real.mul_self_sqrt hh
  calc (h ^ 2 * Real.sqrt h) * Real.sqrt h = h ^ 2 * (Real.sqrt h * Real.sqrt h) := by ring
    _ = h ^ 2 * h := by rw [hs]
    _ = h ^ 3 := by ring

end Poisson
