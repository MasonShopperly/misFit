# SPDX-License-Identifier: Apache-2.0
"""struct.S1 — cantilever torsion. NumPy and SciPy, differentiated by a HAND-DERIVED adjoint.

Deliberately not JAX. A composed system whose components all differentiate the same way tests
nothing about composition; the interesting boundary is the one where an autodiff component must
exchange derivatives with a solver that has none. Structural codes are the realistic instance of
that — they are old, they are Fortran or C, and their sensitivities are hand-derived. So this
component assembles K in NumPy, solves with SciPy, and implements its own JVP and VJP from the
derivation below. `verify_adjoint.py` checks every one of them against JAX.

STATE.  K(s) theta_f = P^T (w * tau),   root clamped, so theta_f omits node 0.
        Returned at the AERO stations, theta_a = P theta_f, because that is the only form the
        aerodynamic component can consume. The transfer lives here, with the mesh that owns it.

        theta_a = G(s) tau,     G = P K(s)^-1 P^T diag(w)

DERIVATIVES.  K is symmetric, which is what makes the adjoint solve reuse the primal factorisation.

  d(theta_a)/d(tau)  VJP:   tau_bar = diag(w) P K^-1 P^T theta_a_bar
  d(theta_a)/d(s)    VJP:   lambda = K^-1 P^T theta_a_bar
                            s_bar_i = - lambda^T (dK/ds_i) theta_f
                            and  lambda^T (dK/ds_i) theta_f
                              = sum_e (d gj_e/ds_i / h) (lam[e]-lam[e+1]) (th[e]-th[e+1])
  with gj_j = GJ_REF exp(sum_i B[j,i] s_i)  =>  d gj_j / d s_i = gj_j B[j,i],
       gj_e = (gj_e_left + gj_e_right)/2,
  where B is the piecewise-linear interpolation from the three control knots onto the nodes.
"""
import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64

B_SPAN, N_AERO, N_ELEM, GJ_REF = 16.0, 40, 24, 2.0e5
SEMISPAN = 0.5 * B_SPAN
N_CTRL = 3

_K_IDX = np.arange(1, N_AERO + 1)
_T = (_K_IDX - 0.5) * np.pi / (2.0 * N_AERO)
_ETA = np.cos(_T)                                     # |y|/semispan at the aero stations
_WQ = SEMISPAN * np.sin(_T) * (np.pi / (2.0 * N_AERO))
_YS = np.linspace(0.0, SEMISPAN, N_ELEM + 1)
_H = _YS[1] - _YS[0]
_KNOTS = np.array([0.0, 0.5, 1.0])


def _interp_basis(x, knots):
    """Rows sum to 1: the piecewise-linear weight of each knot at each sample point."""
    b = np.zeros((len(x), len(knots)))
    for j, xv in enumerate(x):
        i = int(np.clip(np.searchsorted(knots, xv) - 1, 0, len(knots) - 2))
        f = (xv - knots[i]) / (knots[i + 1] - knots[i])
        f = min(max(f, 0.0), 1.0)
        b[j, i] += 1.0 - f
        b[j, i + 1] += f
    return b


_B_NODES = _interp_basis(_YS / SEMISPAN, _KNOTS)      # (N_ELEM+1, 3)
_P = _interp_basis(_ETA * SEMISPAN / SEMISPAN, np.linspace(0.0, 1.0, N_ELEM + 1))
_P_FREE = _P[:, 1:]                                   # the clamped root column is not a DOF
_TRAPW = np.full(N_ELEM + 1, _H)
_TRAPW[0] = _TRAPW[-1] = 0.5 * _H


class InputSchema(BaseModel):
    torque: Differentiable[Array[(N_AERO,), Float64]]
    s_ctrl: Differentiable[Array[(N_CTRL,), Float64]]


class OutputSchema(BaseModel):
    theta_aero: Differentiable[Array[(N_AERO,), Float64]]
    mass: Differentiable[Array[(), Float64]]


def _gj(s_ctrl):
    return GJ_REF * np.exp(_B_NODES @ np.asarray(s_ctrl, dtype=np.float64))


def _stiffness(gj):
    gj_e = 0.5 * (gj[:-1] + gj[1:])
    k = np.zeros((N_ELEM + 1, N_ELEM + 1))
    for e in range(N_ELEM):
        kl = gj_e[e] / _H
        k[e, e] += kl
        k[e + 1, e + 1] += kl
        k[e, e + 1] -= kl
        k[e + 1, e] -= kl
    return k[1:, 1:]


def _solve(torque, s_ctrl):
    gj = _gj(s_ctrl)
    k = _stiffness(gj)
    rhs = _P_FREE.T @ (_WQ * np.asarray(torque, dtype=np.float64))
    theta_f = np.linalg.solve(k, rhs)
    return gj, k, theta_f


def _mass(gj):
    return float(_TRAPW @ gj / (GJ_REF * SEMISPAN))


def apply(inputs: InputSchema) -> OutputSchema:
    _, _, theta_f = _solve(inputs.torque, inputs.s_ctrl)
    return OutputSchema(theta_aero=_P_FREE @ theta_f, mass=np.float64(_mass(_gj(inputs.s_ctrl))))


def _dk_quadform(gj, s_index, lam_f, th_f):
    """lambda^T (dK/ds_i) theta, assembled element by element. Node 0 is clamped, hence the zero."""
    lam = np.concatenate([[0.0], lam_f])
    th = np.concatenate([[0.0], th_f])
    dgj = gj * _B_NODES[:, s_index]                    # d gj_j / d s_i = gj_j B[j,i]
    dgj_e = 0.5 * (dgj[:-1] + dgj[1:])
    e = np.arange(N_ELEM)
    return float(np.sum((dgj_e / _H) * (lam[e] - lam[e + 1]) * (th[e] - th[e + 1])))


def vector_jacobian_product(inputs: InputSchema, vjp_inputs: set[str],
                            vjp_outputs: set[str], cotangent_vector):
    gj, k, theta_f = _solve(inputs.torque, inputs.s_ctrl)
    ta_bar = np.asarray(cotangent_vector.get("theta_aero", np.zeros(N_AERO)), dtype=np.float64)
    mass_bar = float(np.asarray(cotangent_vector.get("mass", 0.0)))
    lam = np.linalg.solve(k, _P_FREE.T @ ta_bar)       # K is symmetric: one solve serves both
    out = {}
    if "torque" in vjp_inputs:
        out["torque"] = _WQ * (_P_FREE @ lam)
    if "s_ctrl" in vjp_inputs:
        s_bar = np.array([-_dk_quadform(gj, i, lam, theta_f) for i in range(N_CTRL)])
        s_bar += mass_bar * (_B_NODES.T @ (_TRAPW * gj)) / (GJ_REF * SEMISPAN)
        out["s_ctrl"] = s_bar
    return out


def jacobian_vector_product(inputs: InputSchema, jvp_inputs: set[str],
                            jvp_outputs: set[str], tangent_vector):
    gj, k, theta_f = _solve(inputs.torque, inputs.s_ctrl)
    d_tau = np.asarray(tangent_vector.get("torque", np.zeros(N_AERO)), dtype=np.float64)
    d_s = np.asarray(tangent_vector.get("s_ctrl", np.zeros(N_CTRL)), dtype=np.float64)
    rhs = _P_FREE.T @ (_WQ * d_tau)
    if np.any(d_s):
        dgj = gj * (_B_NODES @ d_s)
        dgj_e = 0.5 * (dgj[:-1] + dgj[1:])
        th = np.concatenate([[0.0], theta_f])
        dk_th = np.zeros(N_ELEM + 1)
        for e in range(N_ELEM):
            v = (dgj_e[e] / _H) * (th[e] - th[e + 1])
            dk_th[e] += v
            dk_th[e + 1] -= v
        rhs = rhs - dk_th[1:]
    d_theta_f = np.linalg.solve(k, rhs)
    out = {}
    if "theta_aero" in jvp_outputs:
        out["theta_aero"] = _P_FREE @ d_theta_f
    if "mass" in jvp_outputs:
        out["mass"] = np.float64((_TRAPW * gj) @ (_B_NODES @ d_s) / (GJ_REF * SEMISPAN))
    return out


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    """Assembled column by column from the JVP, so it cannot disagree with it."""
    cols = {"torque": N_AERO, "s_ctrl": N_CTRL}
    out = {o: {} for o in jac_outputs}
    for name in jac_inputs:
        blocks = {o: [] for o in jac_outputs}
        for j in range(cols[name]):
            seed = np.zeros(cols[name])
            seed[j] = 1.0
            d = jacobian_vector_product(inputs, {name}, jac_outputs, {name: seed})
            for o in jac_outputs:
                blocks[o].append(np.atleast_1d(d[o]))
        for o in jac_outputs:
            m = np.stack(blocks[o], axis=-1)
            out[o][name] = m if m.shape[0] > 1 else m.reshape(-1)
    return out


def abstract_eval(abstract_inputs):
    return {"theta_aero": {"shape": (N_AERO,), "dtype": "float64"},
            "mass": {"shape": (), "dtype": "float64"}}
