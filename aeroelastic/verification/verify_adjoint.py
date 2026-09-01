#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Is the hand-derived structural adjoint right, and do the two components compose?

A hand-derived adjoint that is subtly wrong would poison every result downstream while
looking healthy, so it is checked three independent ways before it is used for anything:

  DOT-PRODUCT IDENTITY   <VJP(v), u> == <v, JVP(u)> for random u, v. Self-contained: it needs no
                         reference implementation and catches any transpose error.
  AGAINST JAX            the same derivatives taken by autodiff through an equivalent JAX model.
                         This is the heterogeneity the architecture claims, tested rather than
                         asserted.
  AGAINST DIFFERENCES    central differences of the component's own apply.

Then the composition itself: the partitioned fixed point A1 <-> S1 must reproduce the monolithic
coupled solve, and its convergence factor must be the one theory predicts, q/q_D.

    .venv/bin/python aeroelastic/verification/verify_adjoint.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AE = HERE.parent
sys.path.insert(0, str(AE / "core"))

import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
import numpy as np                                                        # noqa: E402

import aero_struct as A                                                   # noqa: E402
from aero_struct import RHO, Wing                                         # noqa: E402

FAILURES: list[str] = []
SEED = 20260806
W = Wing()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check(name, got, tol):
    ok = got <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<56s} {got:>10.3e}  tol {tol:.0e}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    S1 = load("s1_api", AE / "components" / "structure" / "tesseract_api.py")
    A1 = load("a1_api", AE / "components" / "aero" / "tesseract_api.py")
    rng = np.random.default_rng(SEED)

    tau = rng.normal(0.0, 400.0, S1.N_AERO)
    s = rng.normal(0.0, 0.3, 3)
    inp = S1.InputSchema(torque=tau, s_ctrl=s)

    print("DOT-PRODUCT IDENTITY -- <VJP(v), u> == <v, JVP(u)>")
    worst = 0.0
    for _ in range(6):
        u_t = rng.normal(0, 1, S1.N_AERO)
        u_s = rng.normal(0, 1, 3)
        v_th = rng.normal(0, 1, S1.N_AERO)
        v_m = float(rng.normal())
        fwd = S1.jacobian_vector_product(inp, {"torque", "s_ctrl"}, {"theta_aero", "mass"},
                                         {"torque": u_t, "s_ctrl": u_s})
        rev = S1.vector_jacobian_product(inp, {"torque", "s_ctrl"}, {"theta_aero", "mass"},
                                         {"theta_aero": v_th, "mass": v_m})
        lhs = float(v_th @ fwd["theta_aero"] + v_m * float(fwd["mass"]))
        rhs = float(rev["torque"] @ u_t + rev["s_ctrl"] @ u_s)
        worst = max(worst, abs(lhs - rhs) / max(abs(lhs), 1e-300))
    check("adjoint identity, worst relative", worst, 1e-12)

    print("\nAGAINST JAX -- the NumPy hand adjoint vs autodiff of an equivalent JAX model")

    def jax_struct(tau_j, s_j):
        gj = A.gj_profile(W, s_j)
        k = A.stiffness(W, gj)
        p = A.transfer_matrix(W)[:, 1:]
        _, _, wq = A.aero_grid(W)
        theta_f = jnp.linalg.solve(k, p.T @ (wq * tau_j))
        return p @ theta_f, A.mass_proxy(W, s_j)

    out = S1.apply(inp)
    th_j, m_j = jax_struct(jnp.asarray(tau), jnp.asarray(s))
    check("apply: theta_aero vs JAX", float(np.max(np.abs(np.asarray(out.theta_aero) - th_j))
                                            / np.max(np.abs(th_j))), 1e-12)
    check("apply: mass vs JAX", abs(float(out.mass) - float(m_j)) / abs(float(m_j)), 1e-12)

    v_th = rng.normal(0, 1, S1.N_AERO)
    rev = S1.vector_jacobian_product(inp, {"torque", "s_ctrl"}, {"theta_aero"},
                                     {"theta_aero": v_th, "mass": 0.0})
    gj_t, gj_s = jax.grad(lambda t, ss: jnp.sum(v_th * jax_struct(t, ss)[0]), argnums=(0, 1))(
        jnp.asarray(tau), jnp.asarray(s))
    check("VJP wrt torque vs JAX", float(np.max(np.abs(rev["torque"] - gj_t))
                                         / np.max(np.abs(gj_t))), 1e-11)
    check("VJP wrt s_ctrl vs JAX", float(np.max(np.abs(rev["s_ctrl"] - gj_s))
                                         / np.max(np.abs(gj_s))), 1e-11)

    print("\nAGAINST CENTRAL DIFFERENCES of the component's own apply")
    h = 1e-6
    d_s = rng.normal(0, 1, 3)
    fwd = S1.jacobian_vector_product(inp, {"s_ctrl"}, {"theta_aero", "mass"}, {"s_ctrl": d_s})
    tp = S1.apply(S1.InputSchema(torque=tau, s_ctrl=s + h * d_s))
    tm = S1.apply(S1.InputSchema(torque=tau, s_ctrl=s - h * d_s))
    fd = (np.asarray(tp.theta_aero) - np.asarray(tm.theta_aero)) / (2 * h)
    check("JVP wrt s_ctrl vs central differences",
          float(np.max(np.abs(fwd["theta_aero"] - fd)) / np.max(np.abs(fd))), 1e-7)

    print("\nJACOBIAN consistency -- assembled columns must equal the JVP they came from")
    jac = S1.jacobian(inp, {"s_ctrl"}, {"theta_aero"})
    col = S1.jacobian_vector_product(inp, {"s_ctrl"}, {"theta_aero"},
                                     {"s_ctrl": np.array([0.0, 1.0, 0.0])})["theta_aero"]
    check("jacobian column 1 vs JVP", float(np.max(np.abs(jac["theta_aero"]["s_ctrl"][:, 1] - col))
                                            / np.max(np.abs(col))), 1e-15)

    print("\nCOMPOSITION -- the partitioned fixed point A1 <-> S1 vs the monolithic coupled solve")
    alpha = 0.04
    t_ctrl = jnp.array([0.05, 0.01, -0.06])
    for v in (40.0, 60.0, 80.0, 100.0):
        s_c = jnp.zeros(3)
        theta_a = np.zeros(A1.N_AERO)
        twist = np.asarray(A.twist_profile(W, t_ctrl))
        errs = []
        for _ in range(200):
            ao = A1.apply(A1.InputSchema(alpha_tot=alpha + twist + theta_a, v=v))
            so = S1.apply(S1.InputSchema(torque=np.asarray(ao.torque), s_ctrl=np.asarray(s_c)))
            new = np.asarray(so.theta_aero)
            errs.append(np.max(np.abs(new - theta_a)))
            theta_a = new
            if errs[-1] < 1e-14:
                break
        theta_ref, _, cl_ref, cdi_ref = A.solve_coupled(W, t_ctrl, s_c, alpha, v)
        th_ref_aero = np.asarray(A.transfer_matrix(W) @ theta_ref)
        rel = float(np.max(np.abs(theta_a - th_ref_aero)) / np.max(np.abs(th_ref_aero)))
        ao = A1.apply(A1.InputSchema(alpha_tot=alpha + twist + theta_a, v=v))
        qd = float(A.divergence_q(W, s_c))
        q_ratio = 0.5 * RHO * v ** 2 / qd
        rate = (errs[-2] / errs[-3]) if len(errs) > 3 else float("nan")
        print(f"    V={v:>5.0f}  q/q_D={q_ratio:.3f}  iters={len(errs):>3d}  "
              f"observed contraction={rate:.4f}  theta rel err={rel:.2e}  "
              f"CL {float(ao.cl):.6f} vs {float(cl_ref):.6f}")
        if not (rel < 1e-11):
            FAILURES.append(f"composition at V={v}")
        if not abs(rate - q_ratio) < 0.02:
            FAILURES.append(f"contraction factor at V={v}")
    print("  the observed contraction factor is q/q_D, which is what partitioned aeroelastic")
    print("  iteration is supposed to do -- and why it stops converging at divergence")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} checks -> {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
