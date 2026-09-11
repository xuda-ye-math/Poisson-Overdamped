"""Exact Gaussian expectations of polynomial functionals of Brownian motion.

A functional is written as

    coeff * \\int_{[0,h]^n} w(s_1,...,s_n) B_{t_1,a_1} ... B_{t_k,a_k} ds

with each time t_r either one of the integration variables or the fixed time h,
and each a_r a coordinate of the d-dimensional Brownian motion.  The expectation
is Isserlis' theorem over the k factors, with covariance

    E[ B_{s,a} B_{t,b} ] = delta_{ab} min(s,t) ,

followed by exact integration.  The min is resolved by splitting the cube into
the n! regions on which the integration variables are ordered, so every result
is an exact rational function of h.
"""

from itertools import permutations

import sympy as sp


def _matchings(indices):
    """Every perfect matching of a list of even length."""
    if not indices:
        yield []
        return
    first, rest = indices[0], indices[1:]
    for i, partner in enumerate(rest):
        remaining = rest[:i] + rest[i + 1:]
        for tail in _matchings(remaining):
            yield [(first, partner)] + tail


def _isserlis(atoms):
    """Sum over pairings of the atoms, each atom a pair (coordinate, time)."""
    if len(atoms) % 2:
        return sp.Integer(0)
    total = sp.Integer(0)
    for matching in _matchings(list(range(len(atoms)))):
        term = sp.Integer(1)
        for i, j in matching:
            (coord_i, time_i), (coord_j, time_j) = atoms[i], atoms[j]
            if coord_i != coord_j:
                term = sp.Integer(0)
                break
            term *= sp.Min(time_i, time_j)
        total += term
    return total


def _integrate_cube(expr, variables, h):
    """Integrate expr over [0,h]^n, resolving Min by ordering the variables."""
    if not variables:
        return sp.simplify(expr)
    expr = expr.replace(sp.Min, lambda *args: sp.Min(*args))  # canonicalise
    total = sp.Integer(0)
    for order in permutations(variables):
        rank = {v: i for i, v in enumerate(order)}

        def pick(*args, _rank=rank):
            # On this region the variables increase along `order`; h is the largest.
            return min(args, key=lambda a: _rank.get(a, len(_rank)))

        region = expr.replace(sp.Min, pick)
        # 0 <= order[0] <= order[1] <= ... <= order[-1] <= h
        bounds = list(order[1:]) + [h]
        for var, upper in zip(order, bounds):
            region = sp.integrate(region, (var, 0, upper))
        total += region
    return sp.simplify(sp.expand(total))


def expect(atoms, variables=(), weight=1, h=None):
    """Exact expectation of weight * (product of the atoms), integrated over the cube.

    atoms:     sequence of (coordinate, time); time is a variable or h
    variables: the integration variables, each ranging over [0, h]
    weight:    the integrand's deterministic weight
    """
    if h is None:
        raise ValueError("h is required")
    return _integrate_cube(sp.expand(weight * _isserlis(list(atoms))), list(variables), h)
