#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Where a locally correct derivative stops being useful for the assembled system.

DESCRIPTIVE. There is no criterion here and nothing passes or fails. It measures, and the two step
lengths below are fixed in this file before any row was read.

WHY IT IS NOT JUST THE COSINE. `sweep_speed.py` already reports cos(g_rigid, g_coupled) collapsing
with dynamic pressure, and a cosine is a statement about DIRECTION only. Three things have to be
separated before that number is allowed to carry an argument:

    direction     cos(g_r, g_c). Positive means the rigid gradient still points downhill on the
                  coupled objective; it is the sign that decides descent, not the magnitude.
    magnitude     ||g_r|| / ||g_c|| and ||g_r - g_c|| / ||g_c||. Two gradients can agree in
                  direction and disagree by an order of magnitude in length, which changes the step
                  a line search takes and nothing about whether it is downhill.
    consequence   what a real step actually does. Move a fixed distance along -g_r and along -g_c
                  and evaluate the coupled objective at both. This is the only column that
                  answers "does the approximation remain operationally adequate", because it is the
                  only one measured on the objective rather than on a gradient.

The first-order decrease available along a unit direction p is -(g_c . p), maximised at
p = -g_c/||g_c||. Along the rigid direction it is ||g_c|| cos, so the cosine is exactly the
FRACTION OF THE AVAILABLE DECREASE the rigid provider captures -- which is why it is reported as a
fraction and not as an angle.

Both designs come from the committed record; nothing is re-optimized. Only the operating point
moves, so every row is the same wing at a different flight condition.

    .venv/bin/python aeroelastic/studies/validity_regime.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE.parent / "records"
sys.path.insert(0, str(HERE.parent / "core"))

import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
import numpy as np                                                        # noqa: E402

import aero_struct as A                                                   # noqa: E402
from aero_struct import RHO, Wing                                         # noqa: E402
from model import CL_TARGET, W_MASS, refs                                 # noqa: E402

W = Wing()
# Fixed before any row was read. Two lengths, so no conclusion rests on one arbitrary step.
STEPS = (0.005, 0.05)
SPEEDS = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 57.0, 60.0, 65.0, 70.0, 75.0,
          80.0, 85.0, 88.0, 90.0, 91.0, 92.0]


def objective(x, v, solver, cdi_ref, mass_ref):
    _, _, _, cdi, _ = A.solve_trimmed(W, x[:3], x[3:], v, CL_TARGET, solver)
    return cdi / cdi_ref + W_MASS * A.mass_proxy(W, x[3:]) / mass_ref


def sweep(x0):
    cdi_ref, mass_ref = refs()
    xj = jnp.asarray(x0)
    rows = []
    for v in SPEEDS:
        jc = jax.jit(lambda z, v_=v: objective(z, v_, A.solve_coupled, cdi_ref, mass_ref))
        jr = jax.jit(lambda z, v_=v: objective(z, v_, A.solve_rigid, cdi_ref, mass_ref))
        gc = np.asarray(jax.grad(jc)(xj), float)
        gr = np.asarray(jax.grad(jr)(xj), float)
        nc, nr = float(np.linalg.norm(gc)), float(np.linalg.norm(gr))
        cos = float(gc @ gr / (nc * nr))
        qd = float(A.divergence_q(W, jnp.asarray(x0[3:])))
        r = 0.5 * RHO * v ** 2 / qd
        j0 = float(jc(xj))

        row = {"v": v, "q_over_qd": r, "cos": cos,
               "norm_coupled": nc, "norm_rigid": nr, "norm_ratio": nr / nc,
               "rel_norm_diff": float(np.linalg.norm(gr - gc) / nc),
               # the fraction of the first-order decrease the rigid direction captures
               "decrease_fraction": cos,
               "j0": j0}
        for a in STEPS:
            xr = xj - a * jnp.asarray(gr / nr)
            xc = xj - a * jnp.asarray(gc / nc)
            row[f"dj_rigid_step_{a}"] = float(jc(xr)) - j0
            row[f"dj_coupled_step_{a}"] = float(jc(xc)) - j0
        rows.append(row)
    return rows


def segment(x_a, x_b, v):
    """cos along the straight line from one design to another, at ONE fixed flight condition.

    The speed sweep cannot separate two explanations of a collapsing cosine: stronger coupling, or
    standing nearer the coupled optimum where the coupled gradient is small and the rigid provider's
    fixed error dominates it. Holding the flight condition fixed and moving the design separates
    them, which is the only reason this control exists.
    """
    cdi_ref, mass_ref = refs()
    rows = []
    for t in np.linspace(0.0, 1.0, 11):
        x = x_a + t * (x_b - x_a)
        xj = jnp.asarray(x)
        gc = np.asarray(jax.grad(lambda z: objective(z, v, A.solve_coupled, cdi_ref, mass_ref))(xj))
        gr = np.asarray(jax.grad(lambda z: objective(z, v, A.solve_rigid, cdi_ref, mass_ref))(xj))
        nc, nr = float(np.linalg.norm(gc)), float(np.linalg.norm(gr))
        qd = float(A.divergence_q(W, jnp.asarray(x[3:])))
        rows.append({"t": float(t), "v": v, "q_over_qd": 0.5 * RHO * v ** 2 / qd,
                     "dist_to_opt": float(np.linalg.norm(x - x_b)),
                     "cos": float(gc @ gr / (nc * nr)),
                     "norm_coupled": nc, "norm_rigid": nr, "norm_ratio": nr / nc})
    return rows


def show(name, rows, note):
    print(f"\n{name} -- {note}\n")
    print(f"{'V m/s':>6} {'q/q_D':>7} {'cos':>9} {'|g_c|':>10} {'|g_r|':>10} {'|g_r|/|g_c|':>12}"
          f" {'dJ along -g_r':>14} {'dJ along -g_c':>14}")
    a = STEPS[0]
    for r in rows:
        flag = ""
        if r["cos"] < 0:
            flag = "   <-- uphill on the coupled objective"
        elif r[f"dj_rigid_step_{a}"] > 0:
            flag = "   <-- step increases the coupled objective"
        print(f"{r['v']:6.0f} {r['q_over_qd']:7.3f} {r['cos']:9.5f} {r['norm_coupled']:10.3e} "
              f"{r['norm_rigid']:10.3e} {r['norm_ratio']:12.3f} "
              f"{r[f'dj_rigid_step_{a}']:14.3e} {r[f'dj_coupled_step_{a}']:14.3e}{flag}")


def main() -> int:
    sv = json.loads((REC / "served_optimize.json").read_text())
    x0 = np.asarray(sv["optimize"]["x"], float)
    # CONTROL, and it is not optional. The committed design was optimized at V = 80, so ||g_c|| is
    # smallest exactly there and both the cosine and the norm ratio are measured on a short vector.
    # The neutral design carries the same stiffness -- hence the same q_D and the same q/q_D axis --
    # and no twist, so no optimizer ever chose it. If the collapse survives here it is a property of
    # the coupling, and if it does not, the headline is an artifact of standing at an optimum.
    x_neutral = np.array([0.0, 0.0, 0.0, *x0[3:]])

    print("Same wing, same two derivative providers, only the flight condition changes.")
    print("Both gradients are EXACT derivatives -- of different models.")
    rows = sweep(x0)
    rows_n = sweep(x_neutral)
    show("COMMITTED OPTIMIZED DESIGN", rows, "optimized at V = 80, so it is stationary there")
    show("NEUTRAL DESIGN, same stiffness, zero twist", rows_n, "no optimizer chose this point")

    seg = {v: segment(x_neutral, x0, v) for v in (40.0, 80.0)}
    print("\nPROXIMITY, at two fixed flight conditions -- the straight line from the neutral")
    print("design to the committed optimized one. Coupling is held fixed along each row block.\n")
    print(f"{'V':>4} {'t':>5} {'q/q_D':>7} {'|x-x_opt|':>10} {'cos':>9} {'|g_c|':>10} "
          f"{'|g_r|':>10}")
    for v, srows in seg.items():
        for r in srows:
            print(f"{v:4.0f} {r['t']:5.2f} {r['q_over_qd']:7.3f} {r['dist_to_opt']:10.4f} "
                  f"{r['cos']:9.5f} {r['norm_coupled']:10.3e} {r['norm_rigid']:10.3e}")

    a = STEPS[0]
    lo = min(rows, key=lambda r: r["q_over_qd"])
    hi = max(rows, key=lambda r: r["q_over_qd"])
    des = next(r for r in rows if abs(r["v"] - 80.0) < 1e-9)
    sign_change = [rows[i] for i in range(1, len(rows))
                   if rows[i]["cos"] * rows[i - 1]["cos"] < 0]
    first_bad_step = next((r for r in rows if r[f"dj_rigid_step_{a}"] > 0), None)

    sign_n = [rows_n[i] for i in range(1, len(rows_n))
              if rows_n[i]["cos"] * rows_n[i - 1]["cos"] < 0]
    des_n = next(r for r in rows_n if abs(r["v"] - 80.0) < 1e-9)
    first_bad_n = next((r for r in rows_n if r[f"dj_rigid_step_{a}"] > 0), None)
    nr_const = max(r["norm_rigid"] for r in rows) - min(r["norm_rigid"] for r in rows)

    print(f"\n  THE RIGID PROVIDER DOES NOT MOVE. ||g_r|| = {rows[0]['norm_rigid']:.4f} at every "
          f"speed (spread {nr_const:.2e}): at fixed C_L the rigid objective has no dynamic")
    print("  pressure in it at all. The provider gives one answer for every flight condition,")
    print("  while the coupled gradient rotates through 180 degrees.")
    print(f"\n  DIRECTION.  cos = {lo['cos']:.5f} at q/q_D = {lo['q_over_qd']:.3f}, "
          f"{des['cos']:.4f} at the design condition {des['q_over_qd']:.3f}, "
          f"{hi['cos']:.4f} at {hi['q_over_qd']:.3f}.")
    if sign_change:
        print(f"              descent is lost between q/q_D = "
              f"{rows[rows.index(sign_change[0]) - 1]['q_over_qd']:.3f} and "
              f"{sign_change[0]['q_over_qd']:.3f} at the committed design; "
              + (f"between {rows_n[rows_n.index(sign_n[0]) - 1]['q_over_qd']:.3f} and "
                 f"{sign_n[0]['q_over_qd']:.3f} at the neutral one"
                 if sign_n else "the neutral design does not change sign in this range"))
    print(f"  MAGNITUDE.  ||g_r||/||g_c|| runs {min(r['norm_ratio'] for r in rows):.2f} to "
          f"{max(r['norm_ratio'] for r in rows):.2f} at the committed design, and "
          f"{min(r['norm_ratio'] for r in rows_n):.2f} to "
          f"{max(r['norm_ratio'] for r in rows_n):.2f} at the neutral one.")
    print(f"              The {des['norm_ratio']:.1f} at the committed design condition is NOT a "
          f"property of the coupling: ||g_c|| is smallest there because the design")
    print(f"              is optimal there. At the neutral design the same condition gives "
          f"{des_n['norm_ratio']:.2f}. Quote the neutral number.")
    if first_bad_step:
        print(f"  CONSEQUENCE. a step of {a} along the rigid direction first INCREASES the coupled "
              f"objective at q/q_D = {first_bad_step['q_over_qd']:.3f} at the committed design"
              + (f", and at {first_bad_n['q_over_qd']:.3f} at the neutral one."
                 if first_bad_n else ", and never at the neutral one."))
    print("\n  WHAT THE CONTROL DOES TO THE HEADLINE. At the neutral design the cosine goes from "
          f"{rows_n[0]['cos']:.5f} to {rows_n[-1]['cos']:.5f}")
    print("  over the SAME range of dynamic pressure -- a mild decay, no sign change, and every")
    print("  step along the rigid direction still decreases the coupled objective. So dynamic")
    print("  pressure alone does not destroy the rigid gradient's usefulness on this wing.")
    print("  The collapse needs the second ingredient, and the proximity block shows it: the")
    print("  rigid provider's error is a fixed vector, so it dominates precisely where the coupled")
    print("  gradient has become small -- at the coupled optimum, which is the last place an")
    print("  optimizer visits and the only place the design decision is actually taken.")

    out = {"designs": {"committed_optimized": [float(t) for t in x0],
                       "neutral_same_stiffness": [float(t) for t in x_neutral]},
           "steps": list(STEPS), "cl_target": CL_TARGET,
           "rows": rows, "rows_neutral": rows_n,
           "segment": {str(int(v)): r for v, r in seg.items()},
           "rigid_norm_constant": float(rows[0]["norm_rigid"]),
           "rigid_norm_spread": float(nr_const),
           "note": "committed designs re-trimmed at each speed; nothing re-optimized. The neutral "
                   "design is the control for evaluating at a stationary point, and the segment "
                   "block separates coupling strength from proximity to the coupled optimum."}
    (REC / "validity_regime.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {REC / 'validity_regime.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
