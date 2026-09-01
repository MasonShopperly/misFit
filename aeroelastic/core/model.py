#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""The reference wing, the flight condition and the trimmed coupled objective.

Every experiment in this repository is scored on the same objective at the same condition, so the
constants and the objective terms live here and are imported rather than restated. Changing a value
here changes every result that follows, which is why nothing else defines them locally.

    W_MASS      the drag/mass exchange rate in the objective
    H_FD        the finite-difference step the directional derivatives use

The objective every study minimises is

    J(x) = CDi(x) / CDi_ref + W_MASS * mass(x) / mass_ref

evaluated at the trimmed state, so a design cannot buy drag by shedding lift.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax.numpy as jnp                                                   # noqa: E402

import aero_struct as A                                                   # noqa: E402
from aero_struct import RHO, Wing                                         # noqa: E402

W = Wing()
V = 80.0                    # q/q_D = 0.371 at the reference stiffness
CL_TARGET = 0.5             # the aircraft is flown at its weight; the wing is trimmed to it
H_FD = 1e-6
W_MASS = 0.35               # the drag/mass exchange rate


def _terms(x, mode):
    """The three quantities the studies score: induced drag, mass proxy and q/q_D.

    `mode` selects the solver the wing is trimmed with. q/q_D is a property of the structure alone,
    so it is the same number in both modes, and it is what reports the divergence margin that the
    rigid model has no way to see.
    """
    t_ctrl, s_ctrl = x[:3], x[3:]
    solver = A.solve_coupled if mode == "coupled" else A.solve_rigid
    _, _, _, cdi, _ = A.solve_trimmed(W, t_ctrl, s_ctrl, V, CL_TARGET, solver)
    q_ratio = (0.5 * RHO * V ** 2) / A.divergence_q(W, s_ctrl)
    return cdi, A.mass_proxy(W, s_ctrl), q_ratio


REFS = None


def refs():
    global REFS
    if REFS is None:
        cdi, mass, _ = _terms(jnp.zeros(6), "coupled")
        REFS = (float(cdi), float(mass))
    return REFS
