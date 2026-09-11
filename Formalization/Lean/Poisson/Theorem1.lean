/-
Theorem 1 of `Paper/main.tex` (elliptic estimates for the Kolmogorov solution).

The stochastic ingredients of Appendix A -- the reflection coupling of Eberle and the
Bismut-Elworthy-Li derivative formula -- are taken as hypotheses, in the exact form in
which the paper states them.  What is proved here is the analytic skeleton of the
appendix: the Duhamel bounds on the variation processes (A.2), the variance bound of
the Hessian lemma (A.2), the splitting at `t = 1` (A.3), and the summation over the
time grid that turns the decay of `u` into the bound on the discrete Poisson solution
`phi_h` (A.3).
-/
import Poisson.Basic

namespace Poisson

open Real MeasureTheory intervalIntegral

/-! ### A.2, Lemma "variation bounds": the Duhamel estimates -/

/-- The Duhamel integral behind `‖K_t[w]‖ ≤ M t e^{2Mt}` (Appendix A.2):
`∫₀ᵗ e^{M(t-s)} · M · e^{2Ms} ds = e^{Mt}(e^{Mt} - 1) ≤ M t e^{2Mt}` after dividing by `M`. -/
theorem duhamel_K {M t : ℝ} (hM : 0 < M) (ht : 0 ≤ t) :
    exp (M * t) * (exp (M * t) - 1) ≤ M * t * exp (2 * (M * t)) := by
  have h1 : exp (M * t) - 1 ≤ M * t * exp (M * t) :=
    exp_sub_one_le_mul_exp (by positivity)
  have h2 : (0:ℝ) < exp (M * t) := exp_pos _
  have h3 : exp (M * t) * exp (M * t) = exp (2 * (M * t)) := by
    rw [← Real.exp_add]; ring_nf
  calc exp (M * t) * (exp (M * t) - 1)
      ≤ exp (M * t) * (M * t * exp (M * t)) := by nlinarith
    _ = M * t * exp (2 * (M * t)) := by rw [← h3]; ring

/-- The source bound behind `‖Θ_t‖ ≤ M(1 + 3Mt) t e^{3Mt}` (Appendix A.2): the three
terms carrying `∇³U` contribute `M² s e^{3Ms}` each and the `∇⁴U` term `M e^{3Ms}`. -/
theorem theta_source {M s : ℝ} (hM : 0 < M) (hs : 0 ≤ s) :
    3 * (M ^ 2 * s * exp (3 * (M * s))) + M * exp (3 * (M * s))
      = M * (1 + 3 * M * s) * exp (3 * (M * s)) := by ring

/-! ### A.2, Lemma "hessian from lipschitz": the variance bound -/

/-- The integrated form of `d/dt E|Y|² ≤ 2M E|Y|² + 4d` with `E|Y_0|² = 0`, and the
step `e^a - 1 ≤ a e^a` that turns it into the bound used in the paper:
`(2d/M)(e^{2Mτ} - 1) ≤ 4 d τ e^{2Mτ}`. -/
theorem variance_bound {M d τ : ℝ} (hM : 0 < M) (hd : 0 ≤ d) (hτ : 0 ≤ τ) :
    (2 * d / M) * (exp (2 * (M * τ)) - 1) ≤ 4 * d * τ * exp (2 * (M * τ)) := by
  have ha : (0:ℝ) ≤ 2 * (M * τ) := by positivity
  have h1 : exp (2 * (M * τ)) - 1 ≤ 2 * (M * τ) * exp (2 * (M * τ)) :=
    exp_sub_one_le_mul_exp ha
  have h2 : (0:ℝ) < exp (2 * (M * τ)) := exp_pos _
  have h3 : (0:ℝ) ≤ 2 * d / M := by positivity
  calc (2 * d / M) * (exp (2 * (M * τ)) - 1)
      ≤ (2 * d / M) * (2 * (M * τ) * exp (2 * (M * τ))) := by nlinarith
    _ = 4 * d * τ * exp (2 * (M * τ)) := by field_simp; ring

/-! ### A.3: the split at `t = 1` -/

/-- Appendix A.3.  A quantity bounded by a constant on `[0,1]` and by the decayed
constant beyond `t = 1` decays exponentially for all `t ≥ 0`.  This is the shape of the
argument for `‖∇²u(·,t)‖` and `‖∇³u(·,t)‖`: the short-time bounds of A.2 give the first
hypothesis, the gradient decay of A.1 composed with the smoothing lemma gives the second. -/
theorem decay_of_split {C₀ C₁ lam L : ℝ} {F : ℝ → ℝ}
    (hlam : 0 < lam) (hL : 0 ≤ L) (hC₀ : 0 ≤ C₀) (hC₁ : 0 ≤ C₁)
    (hshort : ∀ t, 0 ≤ t → t ≤ 1 → F t ≤ C₀ * L)
    (hlong : ∀ t, 1 ≤ t → F t ≤ C₁ * L * exp (-(lam * (t - 1)))) :
    ∀ t, 0 ≤ t → F t ≤ (C₀ + C₁) * exp lam * L * exp (-(lam * t)) := by
  intro t ht
  have hexp : (0:ℝ) < exp (-(lam * t)) := exp_pos _
  have hE : (0:ℝ) < exp lam := exp_pos _
  have hCL : (0:ℝ) ≤ C₀ * L := mul_nonneg hC₀ hL
  have hCL1 : (0:ℝ) ≤ C₁ * L := mul_nonneg hC₁ hL
  rcases le_or_gt t 1 with h | h
  · have h1 : F t ≤ C₀ * L := hshort t ht h
    have h2 : (1:ℝ) ≤ exp lam * exp (-(lam * t)) := by
      rw [← Real.exp_add]
      apply Real.one_le_exp
      nlinarith
    nlinarith [h1, hCL, hCL1, hexp.le, hE.le]
  · have h1 : F t ≤ C₁ * L * exp (-(lam * (t - 1))) := hlong t h.le
    have h2 : exp (-(lam * (t - 1))) = exp lam * exp (-(lam * t)) := by
      rw [← Real.exp_add]; ring_nf
    rw [h2] at h1
    have h3 : C₁ * L * (exp lam * exp (-(lam * t)))
        ≤ (C₀ + C₁) * exp lam * L * exp (-(lam * t)) := by
      have : C₁ * L ≤ (C₀ + C₁) * L := by nlinarith [hL, hC₀]
      nlinarith [hexp.le, hE.le, this, mul_nonneg hE.le hexp.le]
    linarith [h1, h3]

/-! ### A.3: summation over the time grid -/

/-- The geometric sum of Appendix A.3: for `0 < λ h ≤ 1`,
`h ∑_{k ≥ 0} e^{-λ k h} = h / (1 - e^{-λ h}) ≤ 2 / λ`.
This is the step that converts the decay of `u` into the bound on `φ_h`, uniform in `h`. -/
theorem grid_sum_le {lam h : ℝ} (hlam : 0 < lam) (hh : 0 < h) (hlh : lam * h ≤ 1) :
    h / (1 - exp (-(lam * h))) ≤ 2 / lam := by
  have ha : 0 < lam * h := by positivity
  have hden : (0:ℝ) < 1 + lam * h := by linarith
  have h1 : (lam * h) / (1 + lam * h) ≤ 1 - exp (-(lam * h)) :=
    div_one_add_le_one_sub_exp_neg ha.le
  have h2 : (0:ℝ) < (lam * h) / (1 + lam * h) := by positivity
  have h3 : (0:ℝ) < 1 - exp (-(lam * h)) := lt_of_lt_of_le h2 h1
  have h4 : (lam * h) / 2 ≤ (lam * h) / (1 + lam * h) := by
    rw [div_le_div_iff₀ (by norm_num : (0:ℝ) < 2) hden]
    nlinarith [ha, hlh]
  rw [div_le_div_iff₀ h3 hlam]
  nlinarith [h1, h4, h3]

/-- The same statement with the series written out: `∑' k, e^{-λ k h} = (1 - e^{-λ h})⁻¹`. -/
theorem grid_tsum {lam h : ℝ} (hlam : 0 < lam) (hh : 0 < h) :
    ∑' k : ℕ, exp (-(lam * h)) ^ k = (1 - exp (-(lam * h)))⁻¹ := by
  have h1 : |exp (-(lam * h))| < 1 := by
    rw [abs_of_pos (exp_pos _)]
    exact Real.exp_lt_one_iff.mpr (by nlinarith)
  exact tsum_geometric_of_lt_one (exp_pos _).le (by rwa [abs_of_pos (exp_pos _)] at h1)

/-! ### A.3: from the decay of `u` to the bound on `φ_h`

The discrete Poisson solution is `φ_h = h ∑_{k ≥ 0} u(·, kh)`.  Given the decay of a
derivative of `u`, the sum of the series is bounded uniformly in `h`, which is
`(eq: poisson bound)`. -/

/-- If `‖D u(x, t)‖ ≤ K e^{-λ t}` for all `t = kh`, then `h ∑_k ‖D u(x, kh)‖ ≤ 2K/λ`,
uniformly in `h ∈ (0, 1/λ]`.  This is the last step of Appendix A.3. -/
theorem poisson_bound_of_decay {lam h K : ℝ} (hlam : 0 < lam) (hh : 0 < h)
    (hlh : lam * h ≤ 1) (hK : 0 ≤ K) (D : ℕ → ℝ)
    (hD : ∀ k : ℕ, |D k| ≤ K * exp (-(lam * (k * h)))) :
    h * ∑' k : ℕ, |D k| ≤ 2 * K / lam := by
  have hr : exp (-(lam * h)) < 1 := by
    apply Real.exp_lt_one_iff.mpr
    nlinarith
  have hr0 : (0:ℝ) < exp (-(lam * h)) := exp_pos _
  have hpow : ∀ k : ℕ, |D k| ≤ K * exp (-(lam * h)) ^ k := by
    intro k
    have : exp (-(lam * (k * h))) = exp (-(lam * h)) ^ k := by
      rw [← Real.exp_nat_mul]
      congr 1
      ring
    rw [← this]
    exact hD k
  have hsummable : Summable fun k : ℕ => K * exp (-(lam * h)) ^ k :=
    (summable_geometric_of_lt_one hr0.le hr).mul_left K
  have hsummable' : Summable fun k : ℕ => |D k| :=
    hsummable.of_nonneg_of_le (fun k => abs_nonneg _) hpow
  have hsum : ∑' k : ℕ, |D k| ≤ ∑' k : ℕ, K * exp (-(lam * h)) ^ k :=
    hsummable'.tsum_le_tsum hpow hsummable
  have hgeom : ∑' k : ℕ, K * exp (-(lam * h)) ^ k = K * (1 - exp (-(lam * h)))⁻¹ := by
    rw [tsum_mul_left, tsum_geometric_of_lt_one hr0.le hr]
  rw [hgeom] at hsum
  have hden : (0:ℝ) < 1 - exp (-(lam * h)) := by linarith
  have hkey : h / (1 - exp (-(lam * h))) ≤ 2 / lam := grid_sum_le hlam hh hlh
  have h1 : h * ∑' k : ℕ, |D k| ≤ h * (K * (1 - exp (-(lam * h)))⁻¹) := by
    exact mul_le_mul_of_nonneg_left hsum hh.le
  have h2 : h * (K * (1 - exp (-(lam * h)))⁻¹) = K * (h / (1 - exp (-(lam * h)))) := by
    field_simp
  rw [h2] at h1
  have h3 : K * (h / (1 - exp (-(lam * h)))) ≤ K * (2 / lam) :=
    mul_le_mul_of_nonneg_left hkey hK
  calc h * ∑' k : ℕ, |D k| ≤ K * (h / (1 - exp (-(lam * h)))) := h1
    _ ≤ K * (2 / lam) := h3
    _ = 2 * K / lam := by ring

end Poisson
