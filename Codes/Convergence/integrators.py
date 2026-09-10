"""The integrators of Table 1 of the manuscript, plus a reference integrator.

Every scheme is written for

    dX = -grad U(X) dt + sqrt(2) dB,

with step size h, and is driven by the SAME Brownian path as the reference
solution, which is what makes the strong error meaningful.  Each step consumes

    DB_k = B_{(k+1)h} - B_{kh},
    DA_k = int_{kh}^{(k+1)h} ( B_s - B_{kh} ) ds,

both of which are assembled exactly from the fine grid (see noise.py).

COST is the number of grad U evaluations per step, which is what the
experiment holds fixed across methods.
"""

from functools import partial

import jax
import jax.numpy as jnp
from potential import grad_potential as gU

SQRT2 = jnp.sqrt(2.0)


def _scan(step, z0, xs, traj=False):
    if not traj:
        z, _ = jax.lax.scan(lambda z, x: (step(z, x), None), z0, xs)
        return z
    z, ys = jax.lax.scan(lambda z, x: (lambda zn: (zn, zn))(step(z, x)), z0, xs)
    return jnp.concatenate([z0[None], ys], axis=0)


# --------------------------------------------------------------------------
# 1 gradient evaluation per step
# --------------------------------------------------------------------------

def euler_maruyama(z0, h, DB, DA=None, extra=None, traj=False):
    def step(z, dB):
        return z - h * gU(z) + SQRT2 * dB
    return _scan(step, z0, DB, traj)


def leimkuhler_matthews(z0, h, DB, DA=None, extra=None, traj=False):
    """Z_{k+1} = Z_k - h grad U(Z_k) + sqrt(h/2) (chi_k + chi_{k+1}).

    Coupled to the driving path by chi_k = DB_k / sqrt(h), so that the noise
    is (DB_k + DB_{k+1}) / sqrt(2).  The scheme looks one step into the
    future, so the last step needs chi_N; `extra` supplies it as an
    independent N(0, h I) increment beyond the horizon.
    """
    DBnext = jnp.concatenate([DB[1:], extra[None]], axis=0)
    def step(z, d):
        dB, dBn = d
        return z - h * gU(z) + (dB + dBn) / SQRT2
    return _scan(step, z0, (DB, DBnext), traj)


# --------------------------------------------------------------------------
# 2 gradient evaluations per step
# --------------------------------------------------------------------------

def stochastic_heun(z0, h, DB, DA=None, extra=None, traj=False):
    def step(z, dB):
        g0 = gU(z)
        zp = z - h * g0 + SQRT2 * dB
        return z - 0.5 * h * (g0 + gU(zp)) + SQRT2 * dB
    return _scan(step, z0, DB, traj)


def randomized_midpoint(z0, h, DB, DA=None, extra=None, traj=False):
    """extra = (DBmid, zeta): the Brownian increment up to the random time
    kh + zeta_k h, and zeta_k itself."""
    DBmid, zeta = extra
    def step(z, d):
        dB, dBm, zt = d
        zm = z - zt * h * gU(z) + SQRT2 * dBm
        return z - h * gU(zm) + SQRT2 * dB
    return _scan(step, z0, (DB, DBmid, zeta), traj)


def srk1(z0, h, DB, DA, extra=None, traj=False):
    """Integrator I, the two-stage scheme of Yang and Wang."""
    def step(z, d):
        dB, dA = d
        g0 = gU(z)
        stage = z - 0.75 * h * g0 + (3.0 * SQRT2 / (2.0 * h)) * dA
        return z - (h / 3.0) * g0 - (2.0 * h / 3.0) * gU(stage) + SQRT2 * dB
    return _scan(step, z0, (DB, DA), traj)


# --------------------------------------------------------------------------
# 3 gradient evaluations per step
# --------------------------------------------------------------------------

def srk2(z0, h, DB, DA, extra=None, traj=False):
    """Integrator II, the three-stage scheme, delta = 1/sqrt(2)."""
    delta = 1.0 / SQRT2
    def step(z, d):
        dB, dA = d
        base = z - 0.5 * h * gU(z)
        hp = base + (SQRT2 / h) * (1.0 + delta) * dA
        hm = base + (SQRT2 / h) * (1.0 - delta) * dA
        return z - 0.5 * h * (gU(hp) + gU(hm)) + SQRT2 * dB
    return _scan(step, z0, (DB, DA), traj)


def _rk3_flow(x, h):
    """One Kutta RK3 step of xdot = -grad U(x); 3 gradient evaluations."""
    k1 = -gU(x)
    k2 = -gU(x + 0.5 * h * k1)
    k3 = -gU(x + h * (2.0 * k2 - k1))
    return x + (h / 6.0) * (k1 + 4.0 * k2 + k3)


def random_splitting_rk3(z0, h, DB, DA=None, extra=None, traj=False):
    """Random splitting LMC with the RK3 flow of xdot = -grad U(x); `extra`
    holds the Bernoulli(1/2) order flags.  A third order ODE solver is enough
    for a second order integrator.
    """
    flags = extra
    def step(z, d):
        dB, fl = d
        drift_first = _rk3_flow(z, h) + SQRT2 * dB
        noise_first = _rk3_flow(z + SQRT2 * dB, h)
        return jnp.where(fl, drift_first, noise_first)
    return _scan(step, z0, (DB, flags), traj)


# --------------------------------------------------------------------------
# 4 gradient evaluations per step
# --------------------------------------------------------------------------

def _rk4_flow(x, h):
    """One RK4 step of xdot = -grad U(x); 4 gradient evaluations."""
    k1 = -gU(x)
    k2 = -gU(x + 0.5 * h * k1)
    k3 = -gU(x + 0.5 * h * k2)
    k4 = -gU(x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def random_splitting(z0, h, DB, DA=None, extra=None, traj=False):
    """Random splitting LMC; `extra` holds the Bernoulli(1/2) order flags.

    The drift half step is the RK4 flow of xdot = -grad U(x), which is the
    second order or better ODE solver the analysis calls for.
    """
    flags = extra
    def step(z, d):
        dB, fl = d
        drift_first = _rk4_flow(z, h) + SQRT2 * dB
        noise_first = _rk4_flow(z + SQRT2 * dB, h)
        return jnp.where(fl, drift_first, noise_first)
    return _scan(step, z0, (DB, flags), traj)


# --------------------------------------------------------------------------
# reference integrator: SRK-LD of Li, Wu, Mackey and Erdogdu (strong order 3/2)
# --------------------------------------------------------------------------

def srk_ld(z0, h, DB, DA, extra=None, traj=False):
    """Not one of the compared methods; used only to build the reference."""
    c1 = 0.5 + 1.0 / jnp.sqrt(6.0)
    c2 = 0.5 - 1.0 / jnp.sqrt(6.0)
    c3 = 1.0 / jnp.sqrt(12.0)
    def step(z, d):
        dB, dA = d
        xi = dB / jnp.sqrt(h)
        eta = 2.0 * jnp.sqrt(3.0) * dA / h**1.5 - jnp.sqrt(3.0) * xi
        s = jnp.sqrt(2.0 * h)
        h1 = z + s * (c1 * xi + c3 * eta)
        h2 = z - h * gU(z) + s * (c2 * xi + c3 * eta)
        return z - 0.5 * h * (gU(h1) + gU(h2)) + SQRT2 * dB
    return _scan(step, z0, (DB, DA), traj)


# name -> (function, gradient evaluations per step, label for plots)
METHODS = {
    "euler_maruyama":     (euler_maruyama,        1, "Euler–Maruyama"),
    "leimkuhler_matthews":(leimkuhler_matthews,   1, "Leimkuhler–Matthews"),
    "stochastic_heun":    (stochastic_heun,       2, "stochastic Heun"),
    "randomized_midpoint":(randomized_midpoint,   2, "randomized midpoint"),
    "srk1":               (srk1,                   2, "stochastic Runge–Kutta I"),
    "srk2":               (srk2,                   3, "stochastic Runge–Kutta II"),
    "random_splitting_rk3":(random_splitting_rk3, 3, "random splitting LMC (RK3)"),
    "random_splitting":   (random_splitting,      4, "random splitting LMC (RK4)"),
}


@partial(jax.jit, static_argnums=(4,))
def srk_ld_blocks(z0, dt, dB, dA, n_out):
    """Reference trajectory recorded every `n_fine / n_out` fine steps.

    The inner scan keeps the fine resolution while the outer scan emits only
    the block endpoints, so the stored trajectory is the reference evaluated
    on a grid that every coarse grid of the experiment divides.  Jitted, so
    the reshaped fine arrays it consumes are freed when the call returns.
    """
    n_fine = dB.shape[0]
    m = n_fine // n_out
    assert m * n_out == n_fine, (n_fine, n_out)
    c1 = 0.5 + 1.0 / jnp.sqrt(6.0)
    c2 = 0.5 - 1.0 / jnp.sqrt(6.0)
    c3 = 1.0 / jnp.sqrt(12.0)

    def fine_step(z, d):
        dBi, dAi = d
        xi = dBi / jnp.sqrt(dt)
        eta = 2.0 * jnp.sqrt(3.0) * dAi / dt**1.5 - jnp.sqrt(3.0) * xi
        s = jnp.sqrt(2.0 * dt)
        h1 = z + s * (c1 * xi + c3 * eta)
        h2 = z - dt * gU(z) + s * (c2 * xi + c3 * eta)
        return z - 0.5 * dt * (gU(h1) + gU(h2)) + SQRT2 * dBi, None

    def block(z, d):
        zb, _ = jax.lax.scan(fine_step, z, d)
        return zb, zb

    shape = (n_out, m) + dB.shape[1:]
    _, ys = jax.lax.scan(block, z0, (dB.reshape(shape), dA.reshape(shape)))
    return jnp.concatenate([z0[None], ys], axis=0)
