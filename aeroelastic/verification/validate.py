#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Check the two solvers and their coupling against closed-form results before anything is claimed.

Every target here is analytic, so a disagreement is the code's fault rather than the model's.

    .venv/bin/python aeroelastic/verification/validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AE = HERE.parent
sys.path.insert(0, str(AE / "core"))

import jax.numpy as jnp                                                   # noqa: E402
import numpy as np                                                        # noqa: E402

import aero_struct as A                                                   # noqa: E402
from aero_struct import A0, RHO, Wing                                     # noqa: E402

FAILURES: list[str] = []
RECT_DELTA: list[tuple[float, float]] = []
ELLIPTIC: dict = {}


def check(name: str, got: float, want: float, tol: float, unit: str = "") -> None:
    rel = abs(got - want) / max(abs(want), 1e-300)
    ok = rel <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52s} got {got:.8g}{unit}  "
          f"want {want:.8g}{unit}  rel {rel:.2e} (tol {tol:.0e})")
    if not ok:
        FAILURES.append(name)


def elliptic_wing(n_aero: int = 40) -> Wing:
    return Wing(n_aero=n_aero)


def _elliptic_chord(w: Wing):
    """c(t) = c0 sin(t) with the same area as the rectangular reference."""
    t, _, _ = A.aero_grid(w)
    c0 = 4.0 * w.area / (jnp.pi * w.b)
    return c0 * jnp.sin(t), c0


def _elliptic_solve(w: Wing, alpha: float):
    """Monoplane equation with a spanwise-varying chord -- the rectangular helper assumes c const."""
    t, _, _ = A.aero_grid(w)
    n = A.harmonics(w)
    c, _ = _elliptic_chord(w)
    mu = A0 * c / (4.0 * w.b)
    m = jnp.sin(jnp.outer(t, n)) * (jnp.sin(t)[:, None] + n[None, :] * mu[:, None])
    a_n = jnp.linalg.solve(m, mu * jnp.sin(t) * alpha)
    return A.coefficients(w, a_n)


def main() -> int:
    print("AERODYNAMICS -- elliptic planform has closed-form lift slope and induced drag")
    w = elliptic_wing()
    alpha = np.deg2rad(4.0)
    cl, cdi = _elliptic_solve(w, alpha)
    cl_want = A0 * alpha / (1.0 + A0 / (jnp.pi * w.ar))
    # Captured HERE, not read from `cl`/`cdi` at the end of main(). Those names are rebound twice
    # further down -- by the velocity sweep and by the mesh-convergence loop -- so the JSON was
    # recording a mesh-convergence coefficient against the elliptic reference and reporting a 69 %
    # "error" for a check that passes at 1.4e-16. Measurement code that silently records
    # a different quantity than it names is the defect this capture exists to prevent.
    ELLIPTIC["cl"] = float(cl)
    ELLIPTIC["cl_rel"] = abs(float(cl) - float(cl_want)) / float(cl_want)
    ELLIPTIC["cdi_rel"] = (abs(float(cdi) - float(cl ** 2 / (jnp.pi * w.ar)))
                           / float(cl ** 2 / (jnp.pi * w.ar)))
    check("elliptic CL = a0 alpha / (1 + a0/(pi AR))", float(cl), float(cl_want), 1e-10)
    check("elliptic CDi = CL^2/(pi AR)", float(cdi), float(cl ** 2 / (jnp.pi * w.ar)), 1e-10)

    for n in (10, 20, 40, 80):
        wn = elliptic_wing(n)
        c2, d2 = _elliptic_solve(wn, alpha)
        print(f"    N={n:<3d} CL={float(c2):.10f}  CDi={float(d2):.10f}  "
              f"e={float(c2 ** 2 / (jnp.pi * wn.ar * d2)):.10f}")

    print("\n  rectangular planform: span efficiency below 1, and CL slope below the 2-D value")
    # No invented acceptance band here. The necessary properties are checked; delta is REPORTED for
    # comparison against Glauert's published rectangular-wing table, which is the real target.
    wr = Wing()
    a_n = A.solve_aero(wr, jnp.full(wr.n_aero, alpha))
    clr, cdir = A.coefficients(wr, a_n)
    e_span = float(clr ** 2 / (jnp.pi * wr.ar * cdir))
    print(f"    AR={wr.ar:.2f}  CL={float(clr):.6f}  CDi={float(cdir):.8f}  "
          f"e={e_span:.6f}  delta={1.0 / e_span - 1.0:.6f}")
    for ar in (4.0, 6.0, 8.0, 10.0):
        wa = Wing(b=ar * 1.0, c=1.0)
        aa = A.solve_aero(wa, jnp.full(wa.n_aero, alpha))
        ca, da = A.coefficients(wa, aa)
        ea = float(ca ** 2 / (jnp.pi * wa.ar * da))
        RECT_DELTA.append((ar, 1.0 / ea - 1.0))
        print(f"      AR={ar:<5.1f} e={ea:.6f}  delta={1.0 / ea - 1.0:.6f}")
    ok = e_span < 1.0 and float(clr) < A0 * alpha
    print(f"  {'PASS' if ok else 'FAIL'}  rectangular e < 1 and CL below the 2-D value")
    if not ok:
        FAILURES.append("rectangular sanity")

    print("\nSTRUCTURE -- uniform cantilever torsion has closed-form tip rotation")
    ws = Wing(n_elem=200)
    gj = A.gj_profile(ws, jnp.zeros(3))
    k = A.stiffness(ws, gj)
    ys = A.struct_grid(ws)
    h = ys[1] - ys[0]

    t0 = 500.0                                        # N m / m, uniform distributed torque
    f = jnp.full(ws.n_elem + 1, t0 * h).at[0].set(0.5 * t0 * h).at[-1].set(0.5 * t0 * h)
    theta = jnp.linalg.solve(k, f[1:])
    check("uniform distributed torque: tip twist t0 L^2/(2 GJ)", float(theta[-1]),
          t0 * ws.semispan ** 2 / (2.0 * ws.gj_ref), 2e-4, " rad")

    ft = jnp.zeros(ws.n_elem + 1).at[-1].set(t0)
    theta_t = jnp.linalg.solve(k, ft[1:])
    check("tip torque: tip twist T0 L / GJ", float(theta_t[-1]),
          t0 * ws.semispan / ws.gj_ref, 1e-12, " rad")

    print("\nCOUPLING -- divergence dynamic pressure against the classical strip-theory result")
    # Strip theory drops the induced angle, so the coupled operator must be rebuilt without it:
    # GJ theta'' + q c^2 a0 e theta = 0, theta(0)=0, theta'(L)=0  =>  q_D = pi^2 GJ/(4 a0 e c^2 L^2)
    wd = Wing(n_aero=120, n_elem=120)
    q_strip = (jnp.pi ** 2 * wd.gj_ref
               / (4.0 * A0 * wd.e * wd.c ** 2 * wd.semispan ** 2))

    _, _, wq = A.aero_grid(wd)
    p = A.transfer_matrix(wd)[:, 1:]
    strip_torque = A0 * wd.c * wd.e * wd.c            # dt/dtheta per unit q, per unit span
    g_strip = p.T @ (wq[:, None] * strip_torque * p)
    kd = A.stiffness(wd, A.gj_profile(wd, jnp.zeros(3)))
    ev = jnp.linalg.eigvals(jnp.linalg.solve(kd, g_strip))     # K^-1 G: K is SPD, G may be singular
    q_num = 1.0 / float(jnp.max(jnp.where(jnp.abs(ev.imag) <= 1e-8 * jnp.abs(ev.real) + 1e-30,
                                          ev.real, -jnp.inf)))
    check("strip-theory divergence q_D = pi^2 GJ/(4 a0 e c^2 L^2)", q_num, float(q_strip),
          2e-3, " Pa")

    q_llt = float(A.divergence_q(wd, jnp.zeros(3)))
    print(f"    lifting-line q_D = {q_llt:.1f} Pa  vs strip theory {float(q_strip):.1f} Pa  "
          f"(ratio {q_llt / float(q_strip):.3f})")
    ok = q_llt > float(q_strip)
    print(f"  {'PASS' if ok else 'FAIL'}  induced effects RAISE q_D "
          f"(finite span reduces the effective lift slope)")
    if not ok:
        FAILURES.append("induced effects raise q_D")

    print("\n  aeroelastic amplification 1/(1 - q/q_D) on the coupled tip twist")
    wc = Wing()
    qd = float(A.divergence_q(wc, jnp.zeros(3)))
    print(f"    lifting-line q_D = {qd:.1f} Pa  "
          f"(V_div = {np.sqrt(2 * qd / RHO):.1f} m/s)")
    print(f"    {'V':>7} {'q':>9} {'q/q_D':>7} {'tip twist':>11} {'CL':>9} {'CL/CL_rigid':>12}")
    prev = -1.0
    mono = True
    for v in (20.0, 40.0, 60.0, 80.0, 100.0, 110.0):
        q = 0.5 * RHO * v ** 2
        th, _, cl, _ = A.solve_coupled(wc, jnp.zeros(3), jnp.zeros(3), alpha, v)
        _, _, clr2, _ = A.solve_rigid(wc, jnp.zeros(3), jnp.zeros(3), alpha, v)
        ratio = float(cl / clr2)
        print(f"    {v:>7.0f} {q:>9.0f} {q / qd:>7.3f} {float(th[-1]):>11.5f} "
              f"{float(cl):>9.5f} {ratio:>12.4f}")
        mono &= ratio > prev
        prev = ratio
    print(f"  {'PASS' if mono else 'FAIL'}  lift amplification increases monotonically with q")
    if not mono:
        FAILURES.append("amplification monotone")

    print("\n  zero dynamic pressure must reproduce the rigid wing exactly")
    th0, _, cl0, cdi0 = A.solve_coupled(wc, jnp.array([0.02, 0.0, -0.03]),
                                        jnp.zeros(3), alpha, 1e-6)
    _, _, clr0, cdir0 = A.solve_rigid(wc, jnp.array([0.02, 0.0, -0.03]),
                                      jnp.zeros(3), alpha, 1e-6)
    check("q -> 0: coupled CL equals rigid CL", float(cl0), float(clr0), 1e-9)
    check("q -> 0: coupled CDi equals rigid CDi", float(cdi0), float(cdir0), 1e-9)
    print(f"    tip twist at q -> 0: {float(th0[-1]):.3e} rad (want ~0)")

    print("\n  mesh convergence of the coupled solution (aero and structure refined together)")
    for na, ne in ((20, 12), (40, 24), (80, 48), (160, 96)):
        wm = Wing(n_aero=na, n_elem=ne)
        th, _, cl, cdi = A.solve_coupled(wm, jnp.zeros(3), jnp.zeros(3), alpha, 90.0)
        print(f"    N_aero={na:<4d} N_elem={ne:<4d} tip twist={float(th[-1]):.8f}  "
              f"CL={float(cl):.8f}  CDi={float(cdi):.8f}")

    import json
    (AE / "records" / "validation.json").write_text(json.dumps({
        "elliptic_cl": ELLIPTIC["cl"], "elliptic_cl_rel": ELLIPTIC["cl_rel"],
        "elliptic_cdi_rel": ELLIPTIC["cdi_rel"],
        "rect_delta": {str(ar): d for ar, d in RECT_DELTA},
        "divergence_strip_rel": abs(q_num - float(q_strip)) / float(q_strip),
        "divergence_llt_over_strip": q_llt / float(q_strip),
        "failures": FAILURES,
    }, indent=1) + "\n")
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} checks -> {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
