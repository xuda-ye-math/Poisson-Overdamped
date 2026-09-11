/-
Shared setup for the formalization of

  Theorem 1 (elliptic estimates for the Kolmogorov solution),
  Lemma 1   (local error decomposition of SRK-I and SRK-II),
  Lemma 2   (decomposition of the product Delta Q),

of `Paper/main.tex`.  What is formalized, and what is assumed, is stated in
`Lean/README.md`.
-/
import Mathlib

namespace Poisson

open Real

/-- `exp (-a) ≤ (1 + a)⁻¹` for `0 ≤ a`, from `1 + a ≤ exp a`. -/
theorem exp_neg_le_inv_one_add {a : ℝ} (ha : 0 ≤ a) : exp (-a) ≤ (1 + a)⁻¹ := by
  have h1 : (0:ℝ) < 1 + a := by linarith
  have h2 : 1 + a ≤ exp a := by
    have := Real.add_one_le_exp a; linarith
  have h3 : (0:ℝ) < exp a := exp_pos a
  rw [Real.exp_neg, inv_le_inv₀ h3 h1]
  exact h2

/-- `a / (1 + a) ≤ 1 - exp (-a)` for `0 ≤ a`: the inequality used to sum the decay over
the time grid in Appendix A.3. -/
theorem div_one_add_le_one_sub_exp_neg {a : ℝ} (ha : 0 ≤ a) :
    a / (1 + a) ≤ 1 - exp (-a) := by
  have h1 : (0:ℝ) < 1 + a := by linarith
  have h2 : exp (-a) ≤ (1 + a)⁻¹ := exp_neg_le_inv_one_add ha
  have h3 : a / (1 + a) = 1 - (1 + a)⁻¹ := by
    have : (1 + a) ≠ 0 := ne_of_gt h1
    field_simp
    ring
  rw [h3]; linarith

/-- `exp a - 1 ≤ a * exp a` for `0 ≤ a`: the step `e^a - 1 ≤ a e^a` of Appendix A.2.
It is `1 - a ≤ exp (-a)` multiplied by `exp a`. -/
theorem exp_sub_one_le_mul_exp {a : ℝ} (ha : 0 ≤ a) : exp a - 1 ≤ a * exp a := by
  have h1 : 1 - a ≤ exp (-a) := by
    have := Real.add_one_le_exp (-a); linarith
  have h2 : (0:ℝ) < exp a := exp_pos a
  have h3 : (1 - a) * exp a ≤ exp (-a) * exp a := mul_le_mul_of_nonneg_right h1 h2.le
  have h4 : exp (-a) * exp a = 1 := by rw [← Real.exp_add]; simp
  rw [h4] at h3
  nlinarith [h3]

end Poisson
