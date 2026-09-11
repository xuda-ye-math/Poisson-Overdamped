"""Algebraic check of the remainders of Paper/main.tex.

A remainder is defined by R = dZ_0 - M, so `eq: remainder I`, `eq: remainder II`
and `eq: product remainder` are identities in the Brownian path, not statements
about moments; no expectation can test them (see README).  This script tests
them directly, by expanding one step exactly and comparing term by term.

Nothing here is random.  The step is driven by a free polynomial path

    B_s = sum_n b_n s^n ,        grad U(Z_0(s)) - grad U(Z_0) = sum_n f_n s^n ,

with b_n, f_n and grad U(Z_0) symbolic vectors, from which

    W_s = - int_0^s ( grad U(Z_0) + phi_{s'} ) ds' + sqrt2 B_s
    q_s = - int_0^s phi_{s'} ds'
    e_s = phi_s - sqrt2 grad^2 U(Z_0) B_s

follow, matching \\eqref{eq: g s} and \\eqref{eq: r s}.  Every integral is then a
polynomial integral over [0,h], so each claimed identity reduces to a polynomial
identity in the free symbols, checked exactly.

The Taylor remainders r(W_s) and r(D) appear on both sides unchanged and are
omitted; the brackets carrying grad^2 U, grad^3 U and grad^4 U are what the
coefficients live in.
"""

from itertools import product

import sympy as sp

D = 3   # dimension
N = 2   # path modes: B_s = b_1 s + ... + b_N s^N

s, h = sp.symbols("s h", positive=True)
sqrt2 = sp.sqrt(2)


def _sym(name, idx):
    return sp.Symbol("%s_%s" % (name, "".join(map(str, sorted(idx)))))


g = [sp.Symbol("g_%d" % i) for i in range(D)]
A = [[_sym("A", (i, j)) for j in range(D)] for i in range(D)]
T = [[[_sym("T", (i, j, k)) for k in range(D)]
      for j in range(D)] for i in range(D)]
F = [[[[_sym("F", (i, j, k, l)) for l in range(D)] for k in range(D)]
      for j in range(D)] for i in range(D)]

b = [[sp.Symbol("b%d_%d" % (n, i)) for i in range(D)] for n in range(1, N + 1)]
f = [[sp.Symbol("f%d_%d" % (n, i)) for i in range(D)] for n in range(0, N + 1)]


# ------------------------------------------------------------- vector utilities

def vadd(*vs):
    return [sum(v[i] for v in vs) for i in range(D)]


def smul(c, v):
    return [sp.expand(c * x) for x in v]


def matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(D)) for i in range(D)]


def integrate_vec(v, upper=None):
    upper = h if upper is None else upper
    return [sp.integrate(x, (s, 0, upper)) for x in v]


def outer2(u, v):
    return [[sp.expand(u[i] * v[j]) for j in range(D)] for i in range(D)]


def outer3(u, v, w):
    return [[[sp.expand(u[i] * v[j] * w[k]) for k in range(D)]
             for j in range(D)] for i in range(D)]


def integrate_t2(Q):
    return [[sp.integrate(Q[i][j], (s, 0, h)) for j in range(D)] for i in range(D)]


def integrate_t3(S):
    return [[[sp.integrate(S[i][j][k], (s, 0, h)) for k in range(D)]
             for j in range(D)] for i in range(D)]


def t2_combine(pairs):
    return [[sum(c * Q[i][j] for c, Q in pairs) for j in range(D)] for i in range(D)]


def t3_combine(pairs):
    return [[[sum(c * S[i][j][k] for c, S in pairs) for k in range(D)]
             for j in range(D)] for i in range(D)]


def apply3(Q):
    """grad^3 U [Q] for a matrix Q."""
    return [sum(T[i][j][k] * Q[j][k] for j, k in product(range(D), repeat=2))
            for i in range(D)]


def apply4(S):
    """grad^4 U [S] for a third order tensor S."""
    return [sum(F[i][j][k][l] * S[j][k][l]
                for j, k, l in product(range(D), repeat=3)) for i in range(D)]


def is_zero_vec(v):
    return all(sp.expand(x) == 0 for x in v)


# --------------------------------------------------------------- the path pieces

phi = [sum(f[n][i] * s ** n for n in range(N + 1)) for i in range(D)]
B = [sum(b[n - 1][i] * s ** n for n in range(1, N + 1)) for i in range(D)]

int_phi = integrate_vec(phi, s)                       # int_0^s phi
q = smul(-1, int_phi)                                 # eq: g s
W = vadd(smul(-s, g), smul(-1, int_phi), smul(sqrt2, B))
e = vadd(phi, smul(-sqrt2, matvec(A, B)))             # eq: r s

IB = integrate_vec(B)                                 # int_0^h B ds
IW = integrate_vec(W)
Bh = [x.subs(s, h) for x in B]

# stage displacements
D_I = vadd(smul(-3 * h / 4, g), smul(3 * sqrt2 / (2 * h), IB))
D_IIp = vadd(smul(-h / 2, g), smul(sqrt2 / h * (1 + 1 / sqrt2), IB))
D_IIm = vadd(smul(-h / 2, g), smul(sqrt2 / h * (1 - 1 / sqrt2), IB))
D_II = smul(sp.Rational(2, 3), D_I)

# integrals that both sides use
I_hs_B = integrate_vec(smul(h - s, B))
I_B2 = integrate_t2(outer2(B, B))
I_B3 = integrate_t3(outer3(B, B, B))
I_W2 = integrate_t2(outer2(W, W))
I_W3 = integrate_t3(outer3(W, W, W))
I_hs_e = integrate_vec(smul(h - s, e))
I_qB = integrate_t2(outer2(q, B))
WmB = vadd(W, smul(-sqrt2, B))
I_WmB2 = integrate_t2(outer2(WmB, WmB))


def mean_zero(c1, c3):
    """M(dZ_0) of eq: martingale Z."""
    term1 = smul(-sqrt2, matvec(A, matvec(A, I_hs_B)))
    term2 = apply3(t2_combine([(1, I_B2), (-sp.Rational(3, 2) / h, outer2(IB, IB))]))
    shifted = integrate_vec(smul(s - c1 * h, B))
    term3 = smul(-sqrt2, apply3(outer2(g, shifted)))
    term4 = smul(sqrt2 / 3, apply4(t3_combine(
        [(1, I_B3), (-c3 / h ** 2, outer3(IB, IB, IB))])))
    return vadd(term1, term2, term3, term4)


def remainder_I(det_coeff=None, cube_coeff=None):
    """R(dZ_0) of eq: remainder I, without the terms in r."""
    det_coeff = 3 * h ** 3 / 8 if det_coeff is None else det_coeff
    cube_coeff = 9 * sqrt2 / (2 * h ** 2) if cube_coeff is None else cube_coeff
    t1 = smul(-1, matvec(A, I_hs_e))
    t2 = smul(sqrt2, apply3(I_qB))
    t3 = smul(sp.Rational(1, 2), apply3(
        t2_combine([(1, I_WmB2), (-det_coeff, outer2(g, g))])))
    t4 = smul(sp.Rational(1, 6), apply4(t3_combine([
        (1, I_W3), (-2 * sqrt2, I_B3),
        (-2 * h / 3, outer3(D_I, D_I, D_I)),
        (cube_coeff, outer3(IB, IB, IB))])))
    return vadd(t1, t2, t3, t4)


def remainder_II(det_coeff=None, cube_coeff=None):
    """R(dZ_0) of eq: remainder II, without the terms in r."""
    det_coeff = h ** 3 / 4 if det_coeff is None else det_coeff
    cube_coeff = 5 * sqrt2 / h ** 2 if cube_coeff is None else cube_coeff
    t1 = smul(-1, matvec(A, I_hs_e))
    t2 = smul(sqrt2, apply3(I_qB))
    t3 = smul(sp.Rational(1, 2), apply3(
        t2_combine([(1, I_WmB2), (-det_coeff, outer2(g, g))])))
    stage_cube = t3_combine([(1, outer3(D_IIp, D_IIp, D_IIp)),
                             (1, outer3(D_IIm, D_IIm, D_IIm))])
    t4 = smul(sp.Rational(1, 6), apply4(t3_combine([
        (1, I_W3), (-2 * sqrt2, I_B3),
        (-h / 2, stage_cube),
        (cube_coeff, outer3(IB, IB, IB))])))
    return vadd(t1, t2, t3, t4)


def local_error(scheme):
    """dZ_0 in bracket form, without the terms in r; from eq: local error integral
    and the Taylor identity eq: taylor identity."""
    if scheme == "I":
        stage2 = t2_combine([(sp.Rational(2, 3) * h, outer2(D_I, D_I))])
        stage3 = t3_combine([(sp.Rational(2, 3) * h, outer3(D_I, D_I, D_I))])
    else:
        stage2 = t2_combine([(h / 2, outer2(D_IIp, D_IIp)),
                             (h / 2, outer2(D_IIm, D_IIm))])
        stage3 = t3_combine([(h / 2, outer3(D_IIp, D_IIp, D_IIp)),
                             (h / 2, outer3(D_IIm, D_IIm, D_IIm))])
    br2 = matvec(A, vadd(IW, smul(-h, D_II)))
    br3 = smul(sp.Rational(1, 2), apply3(t2_combine([(1, I_W2), (-1, stage2)])))
    br4 = smul(sp.Rational(1, 6), apply4(t3_combine([(1, I_W3), (-1, stage3)])))
    return vadd(br2, br3, br4)


def smul_t2(c, Q):
    return [[sp.expand(c * Q[i][j]) for j in range(D)] for i in range(D)]


# ------------------------------------------------------------------------ checks

def report(name, ok):
    print("%-52s %s" % (name, "OK" if ok else "FAILED"))
    return ok


def main():
    ok = True

    print("dZ_0 = M(dZ_0) + R(dZ_0), as an identity in the path")
    for scheme, c1, c3, rem in [("I", sp.Rational(3, 4), sp.Rational(9, 4), remainder_I),
                                ("II", sp.Rational(1, 2), sp.Rational(5, 2), remainder_II)]:
        diff = vadd(local_error(scheme), smul(-1, mean_zero(c1, c3)), smul(-1, rem()))
        ok &= report("  SRK-%s: eq: remainder %s" % (scheme, scheme), is_zero_vec(diff))

    print("\nA wrong remainder coefficient is detected")
    perturbed = [
        ("  SRK-I  deterministic 3h^3/8 -> h^3/8",
         "I", sp.Rational(3, 4), sp.Rational(9, 4),
         lambda: remainder_I(det_coeff=h ** 3 / 8)),
        ("  SRK-I  stage cube 9 sqrt2/(2h^2) -> 5 sqrt2/h^2",
         "I", sp.Rational(3, 4), sp.Rational(9, 4),
         lambda: remainder_I(cube_coeff=5 * sqrt2 / h ** 2)),
        ("  SRK-II deterministic h^3/4 -> 3h^3/8",
         "II", sp.Rational(1, 2), sp.Rational(5, 2),
         lambda: remainder_II(det_coeff=3 * h ** 3 / 8)),
        ("  SRK-II stage cube 5 sqrt2/h^2 -> 9 sqrt2/(2h^2)",
         "II", sp.Rational(1, 2), sp.Rational(5, 2),
         lambda: remainder_II(cube_coeff=9 * sqrt2 / (2 * h ** 2))),
    ]
    for label, scheme, c1, c3, rem in perturbed:
        diff = vadd(local_error(scheme), smul(-1, mean_zero(c1, c3)), smul(-1, rem()))
        caught = not is_zero_vec(diff)
        print("%-52s %s" % (label, "detected" if caught else "MISSED"))
        ok &= caught

    print("\nThe product remainder, eq: product remainder")
    # dQ_0 = dZ_0 (2 sqrt2 B_h + Gamma_0)^T, and the added matrix of
    # eq: product martingale must cancel against the one in eq: product remainder.
    c1s, c3s = sp.symbols("c1 c3")
    Msym = [sp.Symbol("M_%d" % i) for i in range(D)]
    Rsym = [sp.Symbol("R_%d" % i) for i in range(D)]
    Tsum = [sp.Symbol("S_%d" % i) for i in range(D)]        # 2 sqrt2 B_h + Gamma_0
    A2 = [[sum(A[i][r] * A[r][j] for r in range(D)) for j in range(D)] for i in range(D)]
    lapU = [[sum(F[i][j][k][k] for k in range(D)) for j in range(D)] for i in range(D)]
    added = [[sp.Rational(2, 3) * h ** 3 * A2[i][j]
              + 4 * h ** 3 * (sp.Rational(1, 3) - c1s / 2)
              * sum(T[i][j][k] * g[k] for k in range(D))
              - sp.Rational(4, 3) * h ** 3 * (1 - c3s / 2) * lapU[i][j]
              for j in range(D)] for i in range(D)]
    Bh_sym = [sp.Symbol("Bh_%d" % i) for i in range(D)]
    M_dQ = [[2 * sqrt2 * Msym[i] * Bh_sym[j] + added[i][j] for j in range(D)]
            for i in range(D)]
    Gamma = [Tsum[i] - 2 * sqrt2 * Bh_sym[i] for i in range(D)]
    R_dQ = [[Msym[i] * Gamma[j] + Rsym[i] * Tsum[j] - added[i][j] for j in range(D)]
            for i in range(D)]
    dQ = [[(Msym[i] + Rsym[i]) * Tsum[j] for j in range(D)] for i in range(D)]
    resid = [[sp.expand(M_dQ[i][j] + R_dQ[i][j] - dQ[i][j]) for j in range(D)]
             for i in range(D)]
    ok &= report("  M(dQ_0) + R(dQ_0) = dQ_0",
                 all(resid[i][j] == 0 for i, j in product(range(D), repeat=2)))

    print("\n%s" % ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
