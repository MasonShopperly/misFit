# SPDX-License-Identifier: Apache-2.0
"""Lifting-line aerodynamics coupled to a cantilever torsion beam. Differentiable, float64.

This module is the physics only. No constant here belongs to an experimental plan; the plan
constants live in `model.py`.

MODEL.

Aerodynamics -- classical Prandtl lifting line in Glauert's Fourier form. With
`y = -(b/2) cos(t)` and `Gamma(t) = 2 b V sum_n A_n sin(n t)`, the induced angle is
`alpha_i = sum_n n A_n sin(n t) / sin(t)` and Kutta-Joukowski plus the section lift law give the
monoplane equation

    sum_n A_n sin(n t) [ sin(t) + n mu(t) ]  =  mu(t) sin(t) alpha_tot(t),   mu = a0 c / (4 b)

collocated at N stations. Symmetric wing and symmetric twist, so only odd harmonics are carried.
Outputs follow from the same coefficients: `CL = pi AR A_1` and `CDi = pi AR sum_n n A_n^2`.

Structure -- linear FE torsion of a cantilever, root clamped, tip free, on its OWN uniform mesh:

    K theta = f,     K_e = (GJ_e / h) [[1, -1], [-1, 1]]

The two meshes deliberately do not match, because in real aeroelasticity they never do. Transfer is
conservative: displacements interpolate `theta_aero = P theta_struct`, and loads transfer as
`f_struct = P^T (w * t_aero)` with `w` the spanwise quadrature weights, so virtual work is preserved
in both directions rather than only one.

Coupling -- lift twists the wing, twist changes the lift. Eliminating the aerodynamic unknowns
leaves `(K - q G) theta = q g0`, which is linear in the state for a fixed design and NONLINEAR in
the design (GJ enters K, twist enters g0). `q G` is the aerodynamic stiffness; the smallest
generalised eigenvalue of `K theta = q G theta` is the divergence dynamic pressure.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

A0 = 2.0 * jnp.pi          # section lift-curve slope, 1/rad (thin-airfoil)
RHO = 1.225                # kg/m^3, sea level


@dataclass(frozen=True)
class Wing:
    """Geometry and discretisation. All lengths in metres, angles in radians."""

    b: float = 16.0        # full span
    c: float = 1.0         # chord (rectangular planform)
    e: float = 0.15        # (x_ea - x_ac)/c > 0: aerodynamic centre AHEAD of the elastic axis,
    #                        so positive lift twists the section nose-up (washin -> divergence)
    gj_ref: float = 2.0e5  # N m^2, reference torsional rigidity
    n_aero: int = 40       # lifting-line stations per semispan (odd harmonics 1,3,...,2N-1)
    n_elem: int = 24       # torsion elements per semispan

    @property
    def semispan(self) -> float:
        return 0.5 * self.b

    @property
    def area(self) -> float:
        return self.b * self.c

    @property
    def ar(self) -> float:
        return self.b ** 2 / self.area


def aero_grid(w: Wing):
    """Cosine collocation on the right semispan, midpoint rule -- neither tip nor root is a node."""
    k = jnp.arange(1, w.n_aero + 1)
    t = (k - 0.5) * jnp.pi / (2.0 * w.n_aero)      # t in (0, pi/2); t->0 is the tip
    eta = jnp.cos(t)                                # |y| / semispan, 1 at tip, 0 at root
    dt = jnp.pi / (2.0 * w.n_aero)
    wq = w.semispan * jnp.sin(t) * dt               # dy = semispan * sin(t) dt
    return t, eta, wq


def struct_grid(w: Wing):
    """Uniform torsion mesh on the semispan; node 0 is the clamped root."""
    y = jnp.linspace(0.0, w.semispan, w.n_elem + 1)
    return y


def transfer_matrix(w: Wing):
    """P: structural nodes -> aero stations, piecewise-linear. Its transpose moves the loads back."""
    _, eta, _ = aero_grid(w)
    ys = struct_grid(w)
    ya = eta * w.semispan
    h = ys[1] - ys[0]
    idx = jnp.clip(jnp.floor(ya / h).astype(int), 0, w.n_elem - 1)
    frac = (ya - ys[idx]) / h
    cols = jnp.arange(w.n_elem + 1)
    return ((cols == idx[:, None]) * (1.0 - frac)[:, None]
            + (cols == (idx + 1)[:, None]) * frac[:, None])


def harmonics(w: Wing):
    """Odd harmonics only: symmetric planform, symmetric twist, symmetric loading."""
    return 2 * jnp.arange(1, w.n_aero + 1) - 1


def monoplane_matrix(w: Wing):
    """M[k, j] = sin(n_j t_k) [sin(t_k) + n_j mu], and the RHS scaling mu sin(t)."""
    t, _, _ = aero_grid(w)
    n = harmonics(w)
    mu = A0 * w.c / (4.0 * w.b)
    m = jnp.sin(jnp.outer(t, n)) * (jnp.sin(t)[:, None] + n[None, :] * mu)
    return m, mu * jnp.sin(t)


def solve_aero(w: Wing, alpha_tot):
    """Fourier coefficients for a prescribed total incidence distribution at the aero stations."""
    m, rhs_scale = monoplane_matrix(w)
    return jnp.linalg.solve(m, rhs_scale * alpha_tot)


def coefficients(w: Wing, a_n):
    n = harmonics(w)
    cl = jnp.pi * w.ar * a_n[0]
    cdi = jnp.pi * w.ar * jnp.sum(n * a_n ** 2)
    return cl, cdi


def sectional_lift(w: Wing, a_n, v):
    """L'(y) = rho V Gamma(y) at the aero stations, N/m."""
    t, _, _ = aero_grid(w)
    n = harmonics(w)
    gamma = 2.0 * w.b * v * (jnp.sin(jnp.outer(t, n)) @ a_n)
    return RHO * v * gamma


def stiffness(w: Wing, gj_nodes):
    """Assemble K for spanwise-varying GJ, then clamp the root by deleting row/column 0."""
    h = w.semispan / w.n_elem
    gj_e = 0.5 * (gj_nodes[:-1] + gj_nodes[1:])
    kloc = gj_e / h
    nn = w.n_elem + 1
    i = jnp.arange(w.n_elem)
    k = jnp.zeros((nn, nn))
    k = k.at[i, i].add(kloc).at[i + 1, i + 1].add(kloc)
    k = k.at[i, i + 1].add(-kloc).at[i + 1, i].add(-kloc)
    return k[1:, 1:]


def aero_operators(w: Wing, v):
    """The two blocks the coupled system needs, both linear.

    Returns (G, g0_unit) with the free part of the torque already folded in, such that the
    reduced coupled system is  (K - G) theta = g0_unit @ alpha_free, theta on the free nodes.
    G scales exactly with dynamic pressure, which is what makes divergence an eigenvalue problem.
    """
    m, rhs_scale = monoplane_matrix(w)
    _, _, wq = aero_grid(w)
    p = transfer_matrix(w)[:, 1:]                   # drop the clamped root column
    t_grid, _, _ = aero_grid(w)
    n = harmonics(w)
    sin_mat = jnp.sin(jnp.outer(t_grid, n))
    # A = M^-1 diag(rhs_scale) alpha_tot  ->  L' = rho V 2 b V sin_mat A  ->  torque = L' e c
    lift_op = RHO * v * 2.0 * w.b * v * sin_mat @ jnp.linalg.solve(m, jnp.diag(rhs_scale))
    torque_op = w.e * w.c * lift_op                 # N m per metre of span, per unit incidence
    g_full = p.T @ (wq[:, None] * torque_op)        # conservative load transfer
    return g_full @ p, g_full


def gj_profile(w: Wing, s_ctrl):
    """log-stiffness control points -> GJ at the structural nodes (3 points: root, mid, tip)."""
    ys = struct_grid(w) / w.semispan
    knots = jnp.array([0.0, 0.5, 1.0])
    return w.gj_ref * jnp.exp(jnp.interp(ys, knots, s_ctrl))


def twist_profile(w: Wing, t_ctrl):
    """geometric-twist control points -> twist at the aero stations (3 points: root, mid, tip)."""
    _, eta, _ = aero_grid(w)
    knots = jnp.array([0.0, 0.5, 1.0])
    return jnp.interp(eta, knots, t_ctrl)


def solve_coupled(w: Wing, t_ctrl, s_ctrl, alpha, v):
    """The state that neither component can produce alone.

    Returns (theta_nodes, a_n, cl, cdi). theta_nodes includes the clamped root value 0.
    """
    gj = gj_profile(w, s_ctrl)
    k = stiffness(w, gj)
    g, g_full = aero_operators(w, v)
    alpha_free = alpha + twist_profile(w, t_ctrl)
    theta_free = jnp.linalg.solve(k - g, g_full @ alpha_free)
    theta = jnp.concatenate([jnp.zeros(1), theta_free])
    p = transfer_matrix(w)
    a_n = solve_aero(w, alpha_free + p @ theta)
    cl, cdi = coefficients(w, a_n)
    return theta, a_n, cl, cdi


def solve_rigid(w: Wing, t_ctrl, s_ctrl, alpha, v):
    """The same wing with the structure switched off: theta == 0. Stiffness cannot act here."""
    del s_ctrl
    alpha_free = alpha + twist_profile(w, t_ctrl)
    a_n = solve_aero(w, alpha_free)
    cl, cdi = coefficients(w, a_n)
    return jnp.zeros(w.n_elem + 1), a_n, cl, cdi


def divergence_q(w: Wing, s_ctrl):
    """Smallest q with (K - q G_unit) singular, from the generalised eigenproblem K x = q G x.

    Solved as `K^-1 G x = (1/q) x` rather than `G^-1 K x = q x`. The two are equivalent only when G
    is invertible, and it frequently is not: G = P^T diag(w) T P inherits the rank of the transfer
    matrix P, so any structural element containing no aerodynamic station leaves G singular. K is
    symmetric positive definite for a clamped beam and always invertible, so inverting that one
    instead is unconditionally safe. Getting this backwards silently returned a divergence pressure
    100x too high at one mesh pair and inf at another.
    """
    k = stiffness(w, gj_profile(w, s_ctrl))
    g_at_v, _ = aero_operators(w, 1.0)
    g_unit = g_at_v / (0.5 * RHO * 1.0 ** 2)       # G is exactly linear in q
    ev = jnp.linalg.eigvals(jnp.linalg.solve(k, g_unit))
    real = jnp.where(jnp.abs(ev.imag) <= 1e-8 * jnp.abs(ev.real) + 1e-30, ev.real, -jnp.inf)
    return 1.0 / jnp.max(real)


def trim_alpha(w: Wing, t_ctrl, s_ctrl, v, cl_target, solver):
    """Root incidence that puts the wing on its target lift. Exact, not iterated.

    CL is affine in alpha: the coupled system is linear in the total incidence and CL is linear in
    the resulting loading, so two solves at alpha = 0 and 1 determine the line exactly. Trimming
    rather than penalising lift removes a hand-chosen weight from the objective, and it is what an
    aerostructural design loop actually does -- the aircraft is flown at its weight, not at a
    convenient angle. It also sharpens the comparison: the rigid model trims to a DIFFERENT alpha
    than the flexible aircraft needs, which is a real consequence of the approximation rather than
    an artifact of scoring.
    """
    _, _, cl0, _ = solver(w, t_ctrl, s_ctrl, 0.0, v)
    _, _, cl1, _ = solver(w, t_ctrl, s_ctrl, 1.0, v)
    return (cl_target - cl0) / (cl1 - cl0)


def solve_trimmed(w: Wing, t_ctrl, s_ctrl, v, cl_target, solver):
    """The trimmed state, and the incidence it took to get there."""
    a = trim_alpha(w, t_ctrl, s_ctrl, v, cl_target, solver)
    theta, a_n, cl, cdi = solver(w, t_ctrl, s_ctrl, a, v)
    return theta, a_n, cl, cdi, a


def sectional_cl(w: Wing, a_n, v):
    """Section lift coefficient at the aero stations -- the quantity a stall margin is set on."""
    return sectional_lift(w, a_n, v) / (0.5 * RHO * v ** 2 * w.c)


def mass_proxy(w: Wing, s_ctrl):
    """Torsion-box mass at fixed section geometry is linear in GJ: nondimensional integral of GJ."""
    gj = gj_profile(w, s_ctrl)
    return jnp.trapezoid(gj, struct_grid(w)) / (w.gj_ref * w.semispan)
