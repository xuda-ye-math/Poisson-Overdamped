# Wick check of the mean-zero terms

Symbolic verification of the coefficients in the two mean-zero terms of
`Paper/main.tex`: `eq: martingale Z` for the local error and
`eq: product martingale` for the product.

Run with an interpreter that has sympy:

```
~/.envs/jflows/bin/python verify.py
```

## What is checked

Both mean-zero terms are asserted to have zero conditional mean, and that
assertion is what fixes their coefficients. The script computes

    E[ M(dZ_0) ]      and      E[ M(dQ_0) ]

exactly and checks that both vanish identically — as polynomials in the entries
of grad U, grad^2 U, grad^3 U and grad^4 U, in the step size h, and in the
scheme constants c1 and c3, which are left symbolic so that one run covers
SRK-I (c1 = 3/4, c3 = 9/4) and SRK-II (c1 = 1/2, c3 = 5/2) together.

It then treats each coefficient as an unknown and solves the vanishing
condition for it, which shows the value in the paper is the only one that works,
not merely a value that works.

`wick.py` is the engine. Each functional is written as

    int_{[0,h]^n} w(s_1,...,s_n) B_{t_1,a_1} ... B_{t_k,a_k} ds ,

the expectation is Isserlis' theorem over the k factors with covariance
E[B_{s,a} B_{t,b}] = delta_{ab} min(s,t), and the min is resolved by splitting
the cube into the n! regions where the integration variables are ordered. Every
result is an exact rational function of h; nothing is numerical, and no formula
from the paper is assumed — the moments are computed from the covariance alone.

The potential is carried as symbolic fully symmetric tensors in dimension 3.
The identities are contractions, so a small dimension settles them: a wrong
rational coefficient cannot cancel against symbolic tensor entries.

## Result

All checks pass. The run also reports which of the two expectations pins which
coefficient, and the two are complementary:

| perturbed coefficient | `E[M(dZ)]` | `E[M(dQ)]` |
| --- | --- | --- |
| stage square `3/(2h)` | caught | invisible |
| shift `c1` | invisible | caught |
| `grad^4 U` prefactor `sqrt2/3` | invisible | caught |
| stage cube `c3/h^2` | invisible | caught |

The second term of `M(dZ_0)` is even in the Brownian path, so its coefficient is
the only one the zero mean of `M(dZ_0)` can see; the first, third and fourth
terms are odd and have zero mean whatever their coefficients. Those three are
pinned instead by the pairing with `B_h`, that is by Lemma 2. Neither check
alone covers `M(dZ_0)`; together they cover it.

# The remainders

```
~/.envs/jflows/bin/python remainder.py
```

A remainder is defined as `R = dZ_0 - M`, so `eq: remainder I`,
`eq: remainder II` and `eq: product remainder` are identities in the Brownian
path, not statements about moments, and no expectation can test them: moving any
mean-zero functional from `R` into `M` and back leaves `E[M] = 0` untouched.
`remainder.py` tests them directly instead, with no randomness at all.

The step is driven by a free polynomial path: `B_s = sum_n b_n s^n` and
`grad U(Z_0(s)) - grad U(Z_0) = sum_n f_n s^n` with symbolic vector
coefficients, from which `W_s`, `q_s` of `eq: g s` and `e_s` of `eq: r s`
follow. One step is then expanded from the Taylor identity
`eq: taylor identity` and the stage weights — not copied from the paper's
display — giving `dZ_0` as the three brackets carrying `grad^2 U`, `grad^3 U`
and `grad^4 U`. Every integral is a polynomial integral over `[0,h]`, so

    dZ_0 - M(dZ_0) - R(dZ_0) = 0

reduces to a polynomial identity in the free symbols, checked exactly for both
integrators. The terms in `r` appear identically on both sides and are omitted.
The last check confirms that the matrix added in `eq: product martingale`
cancels against the one subtracted in `eq: product remainder`, so that
`M(dQ_0) + R(dQ_0) = dQ_0`.

All checks pass, and perturbing any of the four printed remainder constants —
`3h^3/8`, `9 sqrt2/(2h^2)` for SRK-I, `h^3/4`, `5 sqrt2/h^2` for SRK-II — is
detected.

## The two scripts together

`remainder.py` constrains only the sum `M + R`: an error in `M` cancelled by an
equal error in `R` would pass it. `verify.py` pins `M` on its own, through the
two zero-mean conditions. Neither alone settles both printed expressions; the
two together do.

The size claims stay out of scope. That `R` is of order `h^3` is what pins `c1`
and `c3` inside the proof of Lemma 1; here they are pinned independently,
through the pairing identity, with no moment bound anywhere.
