"""Coefficient check for the two mean-zero terms of Paper/main.tex.

Lemma 1 writes the local error as M(dZ) + R(dZ), Lemma 2 writes the product as
M(dQ) + R(dQ), and both M's are asserted to have zero conditional mean.  That
assertion is what fixes their coefficients, so this script computes

    E[ M(dZ) ]   and   E[ M(dQ) ]

exactly, by Isserlis' theorem on the Brownian functionals, and checks that both
vanish identically -- as polynomials in the entries of grad U, grad^2 U,
grad^3 U, grad^4 U and in the step size h, with the scheme constants c1 and c3
left symbolic, so that the check covers SRK-I and SRK-II at once.

It then treats each coefficient as an unknown and solves for it, which shows the
value in the paper is the only one that makes the expectation vanish.

Conventions follow Section 2.1 of the manuscript:
    (grad^3 U [Q])_i    = sum_{jk} d_i d_j d_k U  Q_{jk}
    (grad^3 U [v])_{ij} = sum_k    d_i d_j d_k U  v_k
    (grad^4 U [S])_i    = sum_{jkl} d_i d_j d_k d_l U  S_{jkl}
    (grad^2 Laplace U)_{ij} = sum_k d_i d_j d_k d_k U
"""

from functools import lru_cache
from itertools import product

import sympy as sp

from wick import expect

D = 3  # dimension; the identities are contractions, so a small D settles them

h = sp.symbols("h", positive=True)
c1, c3 = sp.symbols("c1 c3")
s, s1, s2, s3 = sp.symbols("s s1 s2 s3", positive=True)
sqrt2 = sp.sqrt(2)


# ---------------------------------------------------------------- the potential

def _sym(name, idx):
    return sp.Symbol("%s_%s" % (name, "".join(map(str, sorted(idx)))))


g = [sp.Symbol("g_%d" % i) for i in range(D)]                       # grad U
A = [[_sym("A", (i, j)) for j in range(D)] for i in range(D)]        # grad^2 U
T = [[[_sym("T", (i, j, k)) for k in range(D)]
      for j in range(D)] for i in range(D)]                          # grad^3 U
F = [[[[_sym("F", (i, j, k, l)) for l in range(D)] for k in range(D)]
      for j in range(D)] for i in range(D)]                          # grad^4 U

A2 = [[sum(A[i][r] * A[r][j] for r in range(D)) for j in range(D)] for i in range(D)]
lapU = [[sum(F[i][j][k][k] for k in range(D)) for j in range(D)] for i in range(D)]


# ------------------------------------------------- the Brownian functionals of M
# Each helper returns the exact expectation of one bracket of M(dZ) against an
# optional extra factor B_{h,m}; m = None means no extra factor.

def _tail(m):
    return [(m, h)] if m is not None else []


@lru_cache(maxsize=None)
def e_weighted(j, m):
    """E[ int_0^h (h-s) B_{s,j} ds * B_{h,m} ]"""
    return expect([(j, s)] + _tail(m), variables=[s], weight=(h - s), h=h)


cc = sp.Symbol("cc")  # placeholder for the shift, substituted per call


@lru_cache(maxsize=None)
def e_shifted(k, m):
    """E[ int_0^h (s - cc h) B_{s,k} ds * B_{h,m} ], the shift left free"""
    return expect([(k, s)] + _tail(m), variables=[s], weight=(s - cc * h), h=h)


@lru_cache(maxsize=None)
def e_square_exact(j, k, m):
    """E[ int_0^h B_{s,j} B_{s,k} ds * B_{h,m} ]"""
    return expect([(j, s), (k, s)] + _tail(m), variables=[s], h=h)


@lru_cache(maxsize=None)
def e_square_stage(j, k, m):
    """E[ (int_0^h B ds)_j (int_0^h B ds)_k * B_{h,m} ]"""
    return expect([(j, s1), (k, s2)] + _tail(m), variables=[s1, s2], h=h)


@lru_cache(maxsize=None)
def e_cube_exact(j, k, l, m):
    """E[ int_0^h B_{s,j} B_{s,k} B_{s,l} ds * B_{h,m} ]"""
    return expect([(j, s), (k, s), (l, s)] + _tail(m), variables=[s], h=h)


@lru_cache(maxsize=None)
def e_cube_stage(j, k, l, m):
    """E[ (int B)_j (int B)_k (int B)_l * B_{h,m} ]"""
    return expect([(j, s1), (k, s2), (l, s3)] + _tail(m), variables=[s1, s2, s3], h=h)


# ------------------------------------------------------------------- E[ M(dZ) ]

def expected_M_dZ(i, m=None, alpha=None, shift=None, beta=None, gamma=None):
    """E[ M(dZ)_i * B_{h,m} ], with the four coefficients of eq: martingale Z
    left free: alpha weights the stage square, shift is the c1 of the third
    term, beta the whole grad^4 U bracket, gamma the stage cube inside it."""
    alpha = sp.Rational(3, 2) / h if alpha is None else alpha
    shift = c1 if shift is None else shift
    beta = sqrt2 / 3 if beta is None else beta
    gamma = c3 / h ** 2 if gamma is None else gamma

    total = -sqrt2 * sum(A2[i][j] * e_weighted(j, m) for j in range(D))

    total += sum(
        T[i][j][k] * (e_square_exact(j, k, m) - alpha * e_square_stage(j, k, m))
        for j, k in product(range(D), repeat=2)
    )

    total += -sqrt2 * sum(
        T[i][j][k] * g[j] * e_shifted(k, m).subs(cc, shift)
        for j, k in product(range(D), repeat=2)
    )

    total += beta * sum(
        F[i][j][k][l] * (e_cube_exact(j, k, l, m) - gamma * e_cube_stage(j, k, l, m))
        for j, k, l in product(range(D), repeat=3)
    )
    return sp.expand(total)


# ------------------------------------------------------------------- E[ M(dQ) ]

def expected_M_dQ(i, m, p=None, q=None, r=None, **inner):
    """E[ M(dQ)_{im} ] for eq: product martingale, with the three added
    coefficients left free; `inner` perturbs M(dZ) inside the pairing."""
    p = sp.Rational(2, 3) * h ** 3 if p is None else p
    q = 4 * h ** 3 * (sp.Rational(1, 3) - c1 / 2) if q is None else q
    r = -sp.Rational(4, 3) * h ** 3 * (1 - c3 / 2) if r is None else r

    pairing = 2 * sqrt2 * expected_M_dZ(i, m, **inner)
    added = p * A2[i][m] + q * sum(T[i][m][k] * g[k] for k in range(D)) + r * lapU[i][m]
    return sp.expand(pairing + added)


# ------------------------------------------------------------------------ checks

def check_zero(name, entries):
    bad = [(idx, e) for idx, e in entries if sp.simplify(e) != 0]
    print("%-46s %s" % (name, "OK" if not bad else "FAILED"))
    for idx, e in bad:
        print("      %s  ->  %s" % (idx, sp.simplify(e)))
    return not bad


def solve_for(name, expr_builder, unknown, expected):
    """Solve E[...] = 0 for one coefficient and compare with the paper's value."""
    solutions = set()
    for idx in expr_builder():
        sol = sp.solve(sp.Eq(idx, 0), unknown)
        solutions.update(sp.simplify(v) for v in sol)
    solutions.discard(None)
    unique = {sp.nsimplify(v) for v in solutions}
    ok = len(unique) == 1 and sp.simplify(unique.pop() - expected) == 0
    print("%-46s %s   (paper: %s)" % (name, "OK" if ok else "FAILED", expected))
    return ok


def main():
    ok = True

    print("Zero conditional mean, coefficients as printed")
    ok &= check_zero(
        "  E[ M(dZ_0) ] = 0",
        [(i, expected_M_dZ(i)) for i in range(D)],
    )
    ok &= check_zero(
        "  E[ M(dQ_0) ] = 0",
        [((i, m), expected_M_dQ(i, m)) for i, m in product(range(D), repeat=2)],
    )

    print("\nEach coefficient is the only one that makes the mean vanish")
    a = sp.Symbol("a")
    ok &= solve_for(
        "  stage square in M(dZ_0)",
        lambda: [expected_M_dZ(i, alpha=a) for i in range(D)],
        a, sp.Rational(3, 2) / h,
    )
    p, q, r = sp.symbols("p q r")
    ok &= solve_for(
        "  (grad^2 U)^2 in M(dQ_0)",
        lambda: [expected_M_dQ(i, m, p=p) for i, m in product(range(D), repeat=2)],
        p, sp.Rational(2, 3) * h ** 3,
    )
    ok &= solve_for(
        "  grad^3 U[grad U] in M(dQ_0)",
        lambda: [expected_M_dQ(i, m, q=q) for i, m in product(range(D), repeat=2)],
        q, 4 * h ** 3 * (sp.Rational(1, 3) - c1 / 2),
    )
    ok &= solve_for(
        "  grad^2 Laplace U in M(dQ_0)",
        lambda: [expected_M_dQ(i, m, r=r) for i, m in product(range(D), repeat=2)],
        r, -sp.Rational(4, 3) * h ** 3 * (1 - c3 / 2),
    )

    print("\nWhich expectation pins which coefficient")
    print("  %-34s %-12s %s" % ("perturbed coefficient", "E[M(dZ)]", "E[M(dQ)]"))
    perturbations = [
        ("stage square  3/(2h) + 1", dict(alpha=sp.Rational(3, 2) / h + 1)),
        ("shift         c1 + 1", dict(shift=c1 + 1)),
        ("grad^4 U      2 sqrt2 / 3", dict(beta=2 * sqrt2 / 3)),
        ("stage cube    (c3 + 1)/h^2", dict(gamma=(c3 + 1) / h ** 2)),
    ]
    for label, kwargs in perturbations:
        in_dZ = any(sp.simplify(expected_M_dZ(i, **kwargs)) != 0 for i in range(D))
        in_dQ = any(sp.simplify(expected_M_dQ(i, m, **kwargs)) != 0
                    for i, m in product(range(D), repeat=2))
        print("  %-34s %-12s %s" % (label, "caught" if in_dZ else "invisible",
                                    "caught" if in_dQ else "invisible"))
        # every coefficient of M(dZ) must be pinned by at least one of the two
        ok &= (in_dZ or in_dQ)

    print("\n%s" % ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
