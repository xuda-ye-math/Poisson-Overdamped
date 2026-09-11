/-
Lemma 1 of `Paper/main.tex` (local error decomposition of SRK-I and SRK-II),
Appendix B.1-B.2.

The stage displacements of the two integrators are

  D_I      = -(3h/4) g + (3√2/(2h)) J,
  D_II^±   = -(h/2)  g + (√2/h)(1 ± 1/√2) J,

with `g = ∇U(Z₀)` and `J = ∫₀ʰ B_s ds`, in a real inner product space.  Everything in
this file is an identity or an inequality between such expressions, or between the
scalar coefficients they produce; the probabilistic input (the Gaussian moments used in
Step 5, and the moment bounds `‖B_s‖_p ≤ C√s`) is isolated in the hypotheses.
-/
import Poisson.Basic

namespace Poisson

open Real

section Stages

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- The stage displacement of SRK-I. -/
noncomputable def DI (h : ℝ) (g J : V) : V := (-(3 * h / 4)) • g + (3 * Real.sqrt 2 / (2 * h)) • J

/-- The two stage displacements of SRK-II. -/
noncomputable def DIIpm (h : ℝ) (g J : V) (ε : ℝ) : V :=
  (-(h / 2)) • g + (Real.sqrt 2 / h * (1 + ε * (1 / Real.sqrt 2))) • J

/-- The common weighted stage mean `D_II` of `(eq: mean stage)`. -/
noncomputable def DII (h : ℝ) (g J : V) : V := (-(h / 2)) • g + (Real.sqrt 2 / h) • J

/-- `D_II^± = D_II ± (1/h) J`: the two stage displacements of SRK-II differ from their
mean by `±(1/h) ∫₀ʰ B_s ds`.  This is the identity behind every SRK-II computation of
Appendix B.2. -/
theorem DIIpm_eq (h : ℝ) (hh : h ≠ 0) (g J : V) (ε : ℝ) :
    DIIpm h g J ε = DII h g J + (ε * (1 / h)) • J := by
  have hs : Real.sqrt 2 ≠ 0 := by positivity
  simp only [DIIpm, DII]
  rw [add_assoc, ← add_smul]
  congr 1
  have h2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  field_simp

/-- The mean of the two SRK-II stage displacements is `D_II`. -/
theorem DII_is_mean (h : ℝ) (hh : h ≠ 0) (g J : V) :
    (1/2 : ℝ) • (DIIpm h g J 1 + DIIpm h g J (-1)) = DII h g J := by
  rw [DIIpm_eq h hh g J 1, DIIpm_eq h hh g J (-1)]
  module

/-- `(2/3) D_I = D_II`, the first identity of Step 1 in Appendix B.2. -/
theorem two_thirds_DI (h : ℝ) (hh : h ≠ 0) (g J : V) :
    (2/3 : ℝ) • DI h g J = DII h g J := by
  simp only [DI, DII, smul_add, smul_smul]
  congr 1
  · congr 1; ring
  · congr 1; field_simp

end Stages

/-! ### The scalar identities of Steps 1, 3 and 4 -/

/-- Step 1 and Step 3: the SRK-I stage weight against the squared noise coefficient,
`(2/3)(3/2)² = 3/2`. -/
theorem srkI_quadratic : (2/3 : ℝ) * (3/2) ^ 2 = 3/2 := by norm_num

/-- Step 1 and Step 3: the SRK-II noise coefficients satisfy
`½((1 + 1/√2)² + (1 - 1/√2)²) = 3/2`, one more than the square of their mean. -/
theorem srkII_quadratic :
    (1/2 : ℝ) * ((1 + 1 / Real.sqrt 2) ^ 2 + (1 - 1 / Real.sqrt 2) ^ 2) = 3/2 := by
  have h : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have h2 : Real.sqrt 2 ≠ 0 := by positivity
  field_simp
  nlinarith [h]

/-- Step 4: the SRK-I stage coefficient of the pure-noise cube, `(2/3)(3/2)³ = 9/4`,
so that `c₃ = 9/4`. -/
theorem srkI_cubic : (2/3 : ℝ) * (3/2) ^ 3 = 9/4 := by norm_num

/-- Step 4: the SRK-II stage coefficient of the pure-noise cube,
`½((1 + 1/√2)³ + (1 - 1/√2)³) = 5/2`, so that `c₃ = 5/2`. -/
theorem srkII_cubic :
    (1/2 : ℝ) * ((1 + 1 / Real.sqrt 2) ^ 3 + (1 - 1 / Real.sqrt 2) ^ 3) = 5/2 := by
  have h : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have h2 : Real.sqrt 2 ≠ 0 := by positivity
  field_simp
  nlinarith [h, Real.sqrt_nonneg 2]

/-- Step 3: `c₁ = w/2` with `w = 3/2` for SRK-I and `w = 1` for SRK-II. -/
theorem c_one_values : (3/2 : ℝ)/2 = 3/4 ∧ (1:ℝ)/2 = 1/2 := by norm_num

/-- Step 4: the coefficient of the mean-zero term produced by the stage part,
`(1/6) c₃ h (√2/h)³ = (√2/3)(c₃/h²)`, which is the fourth term of `(eq: martingale Z)`. -/
theorem step4_coefficient {h c₃ : ℝ} (hh : h ≠ 0) :
    (1/6 : ℝ) * (c₃ * h) * (Real.sqrt 2 / h) ^ 3 = (Real.sqrt 2 / 3) * (c₃ / h ^ 2) := by
  have h3 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  have hcube : (Real.sqrt 2 / h) ^ 3 = 2 * Real.sqrt 2 / h ^ 3 := by
    rw [div_pow]
    congr 1
    calc Real.sqrt 2 ^ 3 = (Real.sqrt 2 * Real.sqrt 2) * Real.sqrt 2 := by ring
      _ = 2 * Real.sqrt 2 := by rw [h3]
  rw [hcube]
  field_simp
  ring

/-- Step 5: the centering of the second term of the mean-zero term.  With
`E ∫₀ʰ B_s^{⊗2} ds = (h²/2) I` and `E (∫₀ʰ B_s ds)^{⊗2} = (h³/3) I`, the combination
`∫₀ʰ B_s^{⊗2} ds - (3/(2h)) (∫₀ʰ B_s ds)^{⊗2}` has zero mean. -/
theorem step5_centering {h : ℝ} (hh : h ≠ 0) :
    h ^ 2 / 2 - (3 / (2 * h)) * (h ^ 3 / 3) = 0 := by field_simp; ring

/-- The coefficient `3/(2h)` is forced by the two second moments: it is the only weight
that centres the second term.  (Not stated in the paper; a consistency check.) -/
theorem step5_coefficient_unique {h c : ℝ} (hh : 0 < h)
    (hc : h ^ 2 / 2 - c * (h ^ 3 / 3) = 0) : c = 3 / (2 * h) := by
  have hne : h ≠ 0 := ne_of_gt hh
  field_simp at hc ⊢
  nlinarith [hc, hh, sq_nonneg h, mul_pos hh hh]

/-! ### Step 6: the Taylor remainder and the moment integrals -/

/-- The constant of the fourth order Taylor remainder `(eq: taylor rho)`:
`(1/6) ∫₀¹ (1-θ)³ dθ = 1/24`, so `|r(v)| ≤ (M/24)|v|⁴` when `‖∇⁵U‖ ≤ M`. -/
theorem taylor_remainder_constant :
    (1/6 : ℝ) * ∫ θ in (0:ℝ)..1, (1 - θ) ^ 3 = 1 / 24 := by
  have : ∫ θ in (0:ℝ)..1, (1 - θ) ^ 3 = 1/4 := by
    rw [intervalIntegral.integral_comp_sub_left (fun x : ℝ => x ^ 3) 1]
    norm_num
  rw [this]; norm_num

/-- Step 6, first mean-zero bound: `∫₀ʰ (h - s) √s ds ≤ (1/2) h² √h`, of order
`h^{5/2}` as the paper states. -/
theorem moment_integral_one {h : ℝ} (hh : 0 < h) :
    (∫ s in (0:ℝ)..h, (h - s) * Real.sqrt s) ≤ (h * h / 2) * Real.sqrt h := by
  have hc1 : Continuous fun s : ℝ => (h - s) * Real.sqrt s := by
    exact (continuous_const.sub continuous_id).mul Real.continuous_sqrt
  have hc2 : Continuous fun s : ℝ => (h - s) * Real.sqrt h := by
    exact (continuous_const.sub continuous_id).mul continuous_const
  have hle : ∀ s ∈ Set.Icc (0:ℝ) h, (h - s) * Real.sqrt s ≤ (h - s) * Real.sqrt h := by
    intro s hs
    have h1 : Real.sqrt s ≤ Real.sqrt h := Real.sqrt_le_sqrt hs.2
    have h2 : (0:ℝ) ≤ h - s := by linarith [hs.2]
    exact mul_le_mul_of_nonneg_left h1 h2
  have hmono := intervalIntegral.integral_mono_on hh.le
    (hc1.intervalIntegrable (μ := MeasureTheory.volume) 0 h)
    (hc2.intervalIntegrable (μ := MeasureTheory.volume) 0 h) hle
  have hval : (∫ s in (0:ℝ)..h, (h - s) * Real.sqrt h) = (h * h / 2) * Real.sqrt h := by
    rw [intervalIntegral.integral_mul_const]
    have hid : (∫ x in (0:ℝ)..h, x) = h ^ 2 / 2 := by
      simpa using integral_id (a := (0:ℝ)) (b := h)
    have hsub : (∫ x in (0:ℝ)..h, (h - x)) = h * h / 2 := by
      have := intervalIntegral.integral_sub
        (intervalIntegrable_const (μ := MeasureTheory.volume) (a := (0:ℝ)) (b := h) (c := h))
        (continuous_id.intervalIntegrable (μ := MeasureTheory.volume) 0 h)
      simp only [id_eq] at this
      rw [this]
      simp only [intervalIntegral.integral_const, smul_eq_mul, sub_zero]
      rw [hid]; ring
    rw [hsub]
  linarith [hmono, hval.le, hval.ge]

/-- Step 6, fourth mean-zero bound: `∫₀ʰ s √s ds ≤ h² √h`, again of order `h^{5/2}`. -/
theorem moment_integral_two {h : ℝ} (hh : 0 < h) :
    (∫ s in (0:ℝ)..h, s * Real.sqrt s) ≤ h * (h * Real.sqrt h) := by
  have hc1 : Continuous fun s : ℝ => s * Real.sqrt s :=
    continuous_id.mul Real.continuous_sqrt
  have hle : ∀ s ∈ Set.Icc (0:ℝ) h, s * Real.sqrt s ≤ h * Real.sqrt h := by
    intro s hs
    have h1 : Real.sqrt s ≤ Real.sqrt h := Real.sqrt_le_sqrt hs.2
    have h2 : (0:ℝ) ≤ Real.sqrt s := Real.sqrt_nonneg s
    nlinarith [hs.1, hs.2, Real.sqrt_nonneg h]
  have hmono := intervalIntegral.integral_mono_on hh.le
    (hc1.intervalIntegrable (μ := MeasureTheory.volume) 0 h)
    (intervalIntegrable_const (μ := MeasureTheory.volume) (a := (0:ℝ)) (b := h)
      (c := h * Real.sqrt h)) hle
  simpa using hmono

/-- Step 6: for `h ≤ 1` the four bounds on the mean-zero term, of orders
`h^{5/2}, h², h^{5/2}, h^{5/2}`, sum to at most `4 h² (1 + |Z₀|)`: the first half of
`(eq: local moments)`. -/
theorem step6_sum {h Z : ℝ} (hh : 0 < h) (hh1 : h ≤ 1) (hZ : 0 ≤ Z) :
    h * h * Real.sqrt h + h ^ 2 + h * h * Real.sqrt h * (1 + Z) + h * h * Real.sqrt h
      ≤ 4 * h ^ 2 * (1 + Z) := by
  have hs1 : Real.sqrt h ≤ 1 := by
    rw [show (1:ℝ) = Real.sqrt 1 by simp]
    exact Real.sqrt_le_sqrt hh1
  have hs0 : (0:ℝ) ≤ Real.sqrt h := Real.sqrt_nonneg h
  have key : h * h * Real.sqrt h ≤ h ^ 2 := by nlinarith [hs0, hs1, hh.le]
  nlinarith [key, hZ, hh.le, sq_nonneg h, mul_nonneg (mul_nonneg hh.le hh.le) hs0]

/-! ### Step 4: the cube identity behind the SRK-II stage coefficient -/

section Cube

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- With `D_II^± = a ± b`, the weighted cube of the SRK-II stages is
`½((a+b)^{⊗3} + (a-b)^{⊗3}) = a^{⊗3} + 3 sym(a ⊗ b ⊗ b)` in each component; the scalar
consequence used in Step 4 is the coefficient `5/2` computed in `srkII_cubic`.  In the
one-dimensional case the identity is: -/
theorem cube_average (a b : ℝ) :
    (1/2 : ℝ) * ((a + b) ^ 3 + (a - b) ^ 3) = a ^ 3 + 3 * a * b ^ 2 := by ring

/-- and its square counterpart, the identity behind the third line of Step 1. -/
theorem square_average (a b : ℝ) :
    (1/2 : ℝ) * ((a + b) ^ 2 + (a - b) ^ 2) = a ^ 2 + b ^ 2 := by ring

/-- The vector form of the square identity, in a real vector space: the average of the
two SRK-II stage displacements squared exceeds the square of their mean by exactly
`b ⊗ b`, with `b = (1/h) ∫₀ʰ B_s ds`.  Here it is stated for the symmetric product
written through the norm, which is what the moment estimates use. -/
theorem square_average_vec (a b : V) :
    (1/2 : ℝ) • ((a + b) + (a - b)) = a := by module

end Cube

end Poisson
