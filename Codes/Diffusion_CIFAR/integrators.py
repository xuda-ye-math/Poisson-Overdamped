"""The four integrators of the long-time study and the reference, in torch.

Every scheme is written for dX = -grad U(X, tau) dtau + sqrt 2 dB and mirrors
../Convergence/integrators.py term for term.  The drift here depends on the
time tau, which the Convergence schemes never see.  The rule that carries it
through the stages is the augmented state (X, tau) with tau' = 1 and no noise
on tau: pushing that state through each scheme's stage formulas gives the
time at which every stage evaluates the drift, and those are the times
written below.  The coefficients are untouched.

Each step consumes the pair

    DB = B_{tau+h} - B_tau,     DA = int_tau^{tau+h} ( B_s - B_tau ) ds,

drawn once per path on a fine grid (fine_noise, the exact joint law of the
increment and its time integral, as ../Convergence/noise.py) and aggregated
exactly onto the coarse steps (coarsen), so every scheme, and the reference
on the fine grid, integrate the same Brownian path.
"""

import math

import torch

SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)


def fine_noise(gen, n_fine, shape, dt, device):
    """(dB, dA) of shape (n_fine, *shape) on a grid of step dt."""
    xi = torch.randn((n_fine, *shape), generator=gen, device=device)
    eta = torch.randn((n_fine, *shape), generator=gen, device=device)
    return math.sqrt(dt) * xi, dt**1.5 * (0.5 * xi + eta / (2.0 * SQRT3))


def coarsen(dB, dA, n_steps, dt):
    """Aggregate the fine pair onto n_steps coarse steps, exactly:
    DB = sum dB_i,  DA = sum [ (B_{t_i} - B_{kh}) dt + dA_i ]."""
    n_fine = dB.shape[0]
    m = n_fine // n_steps
    assert m * n_steps == n_fine, (n_fine, n_steps)
    b = dB.reshape(n_steps, m, *dB.shape[1:])
    a = dA.reshape(n_steps, m, *dA.shape[1:])
    incl = b.cumsum(dim=1)
    return incl[:, -1], ((incl - b) * dt + a).sum(dim=1)


# --------------------------------------------------------------------------
# one step of each scheme, from state z at time tau
# --------------------------------------------------------------------------

def stochastic_heun(gU, z, tau, h, dB, dA):
    g0 = gU(z, tau)
    zp = z - h * g0 + SQRT2 * dB
    return z - 0.5 * h * (g0 + gU(zp, tau + h)) + SQRT2 * dB


def srk1(gU, z, tau, h, dB, dA):
    """Integrator I; the stage sits at time tau + 3h/4."""
    g0 = gU(z, tau)
    stage = z - 0.75 * h * g0 + (3.0 * SQRT2 / (2.0 * h)) * dA
    return z - (h / 3.0) * g0 - (2.0 * h / 3.0) * gU(stage, tau + 0.75 * h) + SQRT2 * dB


def srk2(gU, z, tau, h, dB, dA):
    """Integrator II, delta = 1/sqrt 2; both stages sit at time tau + h/2."""
    delta = 1.0 / SQRT2
    base = z - 0.5 * h * gU(z, tau)
    hp = base + (SQRT2 / h) * (1.0 + delta) * dA
    hm = base + (SQRT2 / h) * (1.0 - delta) * dA
    return z - 0.5 * h * (gU(hp, tau + 0.5 * h) + gU(hm, tau + 0.5 * h)) + SQRT2 * dB


def _rk3_flow(gU, x, tau, h):
    """One Kutta RK3 step of xdot = -grad U(x, tau); stages at tau, tau + h/2, tau + h."""
    k1 = -gU(x, tau)
    k2 = -gU(x + 0.5 * h * k1, tau + 0.5 * h)
    k3 = -gU(x + h * (2.0 * k2 - k1), tau + h)
    return x + (h / 6.0) * (k1 + 4.0 * k2 + k3)


def random_splitting_rk3(gU, z, tau, h, dB, dA, flag):
    """Random splitting LMC with the RK3 drift flow; flag picks the order."""
    drift_first = _rk3_flow(gU, z, tau, h) + SQRT2 * dB
    noise_first = _rk3_flow(gU, z + SQRT2 * dB, tau, h)
    return torch.where(flag, drift_first, noise_first)


def srk_ld(gU, z, tau, h, dB, dA):
    """The reference scheme of Li, Wu, Mackey and Erdogdu (strong order 3/2);
    stages at tau and tau + h."""
    c1 = 0.5 + 1.0 / math.sqrt(6.0)
    c2 = 0.5 - 1.0 / math.sqrt(6.0)
    c3 = 1.0 / math.sqrt(12.0)
    xi = dB / math.sqrt(h)
    eta = 2.0 * SQRT3 * dA / h**1.5 - SQRT3 * xi
    s = math.sqrt(2.0 * h)
    h1 = z + s * (c1 * xi + c3 * eta)
    h2 = z - h * gU(z, tau) + s * (c2 * xi + c3 * eta)
    return z - 0.5 * h * (gU(h1, tau) + gU(h2, tau + h)) + SQRT2 * dB


def integrate(step, gU, z, h, DB, DA, flags=None):
    """Run `step` over every coarse step from time 0; flags only for the splitting."""
    for k in range(DB.shape[0]):
        tau = k * h
        if flags is None:
            z = step(gU, z, tau, h, DB[k], DA[k])
        else:
            z = step(gU, z, tau, h, DB[k], DA[k], flags[k])
    return z


# name -> (one step, gradient evaluations per step, label)
METHODS = {
    "srk1":                 (srk1,                 2, "stochastic Runge–Kutta I"),
    "srk2":                 (srk2,                 3, "stochastic Runge–Kutta II"),
    "stochastic_heun":      (stochastic_heun,      2, "stochastic Heun"),
    "random_splitting_rk3": (random_splitting_rk3, 3, "random splitting LMC (RK3)"),
}
