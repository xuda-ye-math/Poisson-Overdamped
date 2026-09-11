# Formalization

Machine checks of the two decompositions at the heart of the paper, Lemma 1
(the local error `ΔZ_0`, Appendix B.3 of `Paper/main.tex`) and Lemma 2 (the
product `ΔQ_0`, Appendix B.4), and of the constants of Theorem 1 (Appendix A).
The question they settle is the one a reader is least able to check by hand:
that the coefficients printed in the mean-zero terms, the remainders and the
pairing identity are correct, and that each is the only value that works.

Two independent routes are used. They assume different things, and agree.

```
Formalization/
  Wick/     exact symbolic computation (Python, sympy)
  Lean/     proof assistant (Lean 4, Mathlib)
```

## Wick: exact symbolic computation

`Wick/` treats the Brownian functionals of the two lemmas as what they are,
polynomials in the values of a Gaussian path, and computes their expectations
exactly by Isserlis' theorem from the covariance
`E[B_{s,a} B_{t,b}] = δ_{ab} min(s,t)` alone. No moment formula from the paper
is assumed; the moments are derived.

- `verify.py` computes `E[M(ΔZ_0)]` and `E[M(ΔQ_0)]` and checks that both
  vanish identically, as polynomials in the derivatives of `U`, in `h`, and in
  the scheme constants `c₁`, `c₃` kept symbolic so that one run covers SRK-I
  (`c₁ = 3/4`, `c₃ = 9/4`) and SRK-II (`c₁ = 1/2`, `c₃ = 5/2`). It then
  solves the vanishing conditions for each coefficient and confirms that the
  printed value is the unique solution.
- `remainder.py` checks the identities `ΔZ_0 = M(ΔZ_0) + R(ΔZ_0)` and
  `ΔQ_0 = M(ΔQ_0) + R(ΔQ_0)` themselves, pathwise, for a free polynomial path
  and a free polynomial gradient increment, with the Taylor identity and the
  stage weights as the only inputs; no expectation is taken.
- `wick.py` is the engine (Isserlis pairing over ordered regions of the cube).

`Wick/README.md` describes the checks and records which expectation pins which
coefficient. `Wick/verify_output.txt` and `Wick/remainder_output.txt` are the
saved outputs of the two scripts; every check reports `OK`, every perturbed
coefficient is reported `detected`, and both runs end in `all checks passed`.
To rerun,

```sh
cd Formalization/Wick
python verify.py      # sympy required
python remainder.py
```

Both scripts carry the potential as symbolic symmetric tensors in dimension 3
(`D = 3`); the identities are multilinear contractions, so this suffices to
detect a wrong coefficient. A run at `D = 5` removes any dependence on the
dimension.

## Lean: proof assistant

`Lean/` formalizes the non-stochastic part of the same proofs: the algebra of
the two schemes, every numerical constant, and the analytic skeleton of
Theorem 1. The stochastic inputs — the moment bounds of the Brownian
functionals and the Gaussian moment formulas — enter as hypotheses of the Lean
statements, in the form the paper states them. Where the paper prints a
constant, the Lean statement carries that constant, so a wrong coefficient
would not compile.

`lake build` completes with no `sorry` and no additional axiom.
`Lean/build_output.txt` is the saved build: the four modules compile with only
linter warnings, and `#print axioms` reports for each of the seven theorems of
`Poisson/Theorem1.lean` — the elliptic estimates — the three standard axioms
`propext`, `Classical.choice`, `Quot.sound` alone. `Lean/README.md` lists every
theorem with the step of the paper it checks. To build,

```sh
cd Formalization/Lean
lake exe cache get && lake build
```

with the toolchain pinned in `lean-toolchain` (Lean 4 v4.33.1, Mathlib
`v4.33.1`).

## What is and is not covered

Covered, by both routes: the coefficients `3/(2h)`, `c₁`, `c₃` of the
mean-zero term of Lemma 1, the constants of its remainder, and the three
coefficients of the pairing identity of Lemma 2, each shown to be unique.
Covered by Wick alone: the decompositions as pathwise identities, and the
Gaussian moment formulas. Covered by Lean alone: the constants of Theorem 1.

Not covered by either: the moment bounds `‖B_s‖_p ≤ C√s`, the Grönwall bound
on the exact flow, and the Hölder estimates that give the orders `h²` and `h³`
of Lemma 1 and `h^{5/2}` and `h³` of Lemma 2. These are hypotheses in Lean and
outside the scope of Wick; they are proved in Appendix B of the paper.
