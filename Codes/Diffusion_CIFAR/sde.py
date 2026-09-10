"""The variance preserving diffusion, in the Langevin form the integrators take.

Forward process on [0, 1]:

    dX = -1/2 beta(t) X dt + sqrt(beta(t)) dW,     beta(t) = BETA_MIN + t (BETA_MAX - BETA_MIN),

so X_t = alpha(t) X_0 + sigma(t) xi with alpha(t) = exp(-tau(t)), tau(t) = int_0^t beta/2,
and sigma^2 = 1 - alpha^2.

Reverse process.  In the reversed time s = 1 - t the reverse SDE reads
dX = ( 1/2 beta X + beta grad log p_t(X) ) ds + sqrt(beta) dB, and the time change
d(tau') = 1/2 beta ds turns it into

    dX = ( X + 2 s_theta(X, t) ) d(tau') + sqrt 2 dB ,

overdamped Langevin dynamics dX = -grad U d(tau') + sqrt 2 dB with the time dependent
potential gradient grad U(x, tau') = -( x + 2 s_theta(x, t(tau')) ).  This is the
exact form the integrators of ../Convergence are written for, with tau' in place of
time; tau' runs from 0 (t = 1) to TAU_END = tau(1) - tau(T_END).

The chain is stopped at t = T_END rather than 0: the score grows like 1/sigma(t), so
an explicit step needs h times 2/sigma^2 of order one, and at T_END = 0.12 the
variance sigma^2 = 0.14 keeps that product near 1 for N = 60 steps.  The endpoint is
denoised by Tweedie's formula x_0 = ( x + sigma^2 s_theta(x, t) ) / alpha(t), the same
map for every integrator.
"""

import math

import torch

BETA_MIN = 0.1
BETA_MAX = 20.0
T_END = 0.12


def beta(t):
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


def tau(t):
    """int_0^t beta(u)/2 du = -log alpha(t)."""
    return (BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t**2) / 2.0


def tau_inv(u):
    """The t with tau(t) = u, the positive root of the quadratic."""
    d = BETA_MAX - BETA_MIN
    return (-BETA_MIN + torch.sqrt(BETA_MIN**2 + 4.0 * d * u)) / d


def alpha(t):
    return torch.exp(-tau(t))


def sigma(t):
    return torch.sqrt(1.0 - torch.exp(-2.0 * tau(t)))


TAU_END = float(tau(torch.tensor(1.0)) - tau(torch.tensor(T_END)))


def t_of(tau_prime):
    """Forward time at reverse Langevin time tau', a tensor of any shape."""
    return tau_inv(tau(torch.tensor(1.0, dtype=tau_prime.dtype, device=tau_prime.device)) - tau_prime)


def score(model, x, t):
    """s_theta(x, t) = -eps_theta(x, t) / sigma(t) for a batch x and a batch of times t."""
    return -model(x, t) / sigma(t)[:, None, None, None]


def grad_U(model):
    """grad U(x, tau') = -( x + 2 s_theta(x, t(tau')) ), tau' a Python float."""
    def gU(x, tau_prime):
        t = t_of(torch.full((x.shape[0],), tau_prime, dtype=x.dtype, device=x.device))
        return -(x + 2.0 * score(model, x, t))
    return gU


def tweedie(model, x, t_scalar):
    """x_0 = ( x + sigma^2 s_theta(x, t) ) / alpha(t) at a common time t."""
    t = torch.full((x.shape[0],), t_scalar, dtype=x.dtype, device=x.device)
    return (x + sigma(t)[:, None, None, None] ** 2 * score(model, x, t)) / alpha(t)[:, None, None, None]
