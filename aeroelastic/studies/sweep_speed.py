#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Both committed designs, re-scored across flight speed. The headline gap is a one-point number.

WHY THIS EXISTS. The submission's headline is that the aerodynamic-only design carries 19.9 % more
induced drag than the coupled one at equal lift and equal structure. That is measured at ONE
operating point -- 80 m/s, C_L = 0.50, q/q_D = 0.747 -- which leaves the obvious question open:
does the advantage survive off design?

It does not. Below about 57 m/s the ranking REVERSES and the aerodynamic-only design is the better
of the two. At low dynamic pressure the wing barely deforms, the rigid model is nearly right, and a
design tailored for a wing that twists is carrying tailoring it does not need. The coupled design
wins where the coupling matters and loses where it does not.

THE GRADIENT ANGLE ACROSS SPEED. This also measures the ANGLE between the two
gradients at each speed. The aerodynamic-only gradient is the q -> 0 limit of the coupled one: at a
tenth of the divergence pressure the two point the same way to six decimals, and at the design
condition they are nearly orthogonal. The rigid provider is not approximately right and slowly
degrading -- it is exactly right in a limit this wing does not fly in. That is why a derivative
belongs to a declared objective AT a declared operating point rather than to "the wing".

Nothing here is re-optimized. Both design vectors are read from the committed record and re-trimmed
to the same C_L at each speed, so the only thing that varies is the operating point.

    .venv/bin/python aeroelastic/studies/sweep_speed.py
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
SPEEDS = [30.0, 40.0, 50.0, 55.0, 56.0, 57.0, 58.0, 60.0, 70.0, 80.0, 90.0]


def score(x, v):
    t, s = jnp.asarray(x[:3]), jnp.asarray(x[3:])
    _, _, cl, cdi, _ = A.solve_trimmed(W, t, s, v, CL_TARGET, A.solve_coupled)
    return float(cdi), float(cl), 0.5 * RHO * v ** 2 / float(A.divergence_q(W, s))


def objective(xv, v, solver, cdi_ref, mass_ref):
    t, s = xv[:3], xv[3:]
    _, _, _, cdi, _ = A.solve_trimmed(W, t, s, v, CL_TARGET, solver)
    return cdi / cdi_ref + W_MASS * A.mass_proxy(W, s) / mass_ref


def gradient_agreement(x, v):
    """The angle between the rigid provider's gradient and the coupled one, at one speed.

    Both are exact derivatives -- of different objectives. As q -> 0 the elastic twist vanishes,
    the coupled model becomes the rigid one, and so do their derivatives. This measures where in
    between the provider stops being usable:
    the same provider is right at one operating point and nearly orthogonal at another.
    """
    cdi_ref, mass_ref = refs()
    xj = jnp.asarray(x)
    gc = np.asarray(jax.grad(lambda z: objective(z, v, A.solve_coupled, cdi_ref, mass_ref))(xj))
    gr = np.asarray(jax.grad(lambda z: objective(z, v, A.solve_rigid, cdi_ref, mass_ref))(xj))
    nc, nr = np.linalg.norm(gc), np.linalg.norm(gr)
    return (float(gc @ gr / (nc * nr)),
            float(np.linalg.norm(gr - gc) / nc))


def main() -> int:
    sv = json.loads((REC / "served_optimize.json").read_text())
    xc = np.asarray(sv["optimize"]["x"])
    xa = np.asarray(sv["aero_only"]["x"])

    rows = []
    for v in SPEEDS:
        cdi_c, cl_c, r = score(xc, v)
        cdi_a, cl_a, _ = score(xa, v)
        cos, rel = gradient_agreement(xc, v)
        rows.append({"v": v, "q_over_qd": r, "cdi_coupled": cdi_c, "cdi_aero_only": cdi_a,
                     "excess_pct": 100.0 * (cdi_a / cdi_c - 1.0),
                     "cl_coupled": cl_c, "cl_aero_only": cl_a,
                     "grad_cos_rigid_vs_coupled": cos, "grad_rel_diff": rel})

    print(f"{'V m/s':>7} {'q/q_D':>7} {'CDi coupled':>13} {'CDi aero-only':>14} {'excess %':>9}"
          f" {'cos(g_rigid,g_coupled)':>23}")
    for r in rows:
        print(f"{r['v']:7.0f} {r['q_over_qd']:7.3f} {r['cdi_coupled']:13.6f} "
              f"{r['cdi_aero_only']:14.6f} {r['excess_pct']:9.2f} "
              f"{r['grad_cos_rigid_vs_coupled']:23.6f}")

    # the crossover: the lowest tabulated speed at which the coupled design is already ahead
    ahead = [r for r in rows if r["excess_pct"] > 0]
    cross = min(r["v"] for r in ahead) if ahead else None
    below = [r for r in rows if r["excess_pct"] < 0]
    worst = min(below, key=lambda r: r["excess_pct"]) if below else None

    print(f"\n  crossover between {max(r['v'] for r in rows if r['excess_pct'] < 0):.0f} and "
          f"{cross:.0f} m/s -- below it the aerodynamic-only design is the better one")
    if worst:
        print(f"  worst case for the coupled design: {worst['excess_pct']:+.1f} % at "
              f"{worst['v']:.0f} m/s (q/q_D = {worst['q_over_qd']:.3f})")
    print(f"  every row trimmed to C_L = {CL_TARGET}; no design was re-optimized")

    design = next(r for r in rows if abs(r["v"] - 80.0) < 1e-9)
    low = next(r for r in rows if abs(r["v"] - 30.0) < 1e-9)
    print(f"\n  the two gradients agree to cos = {low['grad_cos_rigid_vs_coupled']:.6f} at "
          f"q/q_D = {low['q_over_qd']:.3f}, and to cos = "
          f"{design['grad_cos_rigid_vs_coupled']:.4f} at the design condition")
    print("  the rigid provider is exactly right in a limit this wing does not fly in")

    out = {"speeds": rows, "crossover_v": cross,
           "design_point_v": 80.0,
           "grad_cos_at_design": design["grad_cos_rigid_vs_coupled"],
           "grad_cos_at_low_q": low["grad_cos_rigid_vs_coupled"],
           "note": "both committed designs re-trimmed at each speed; nothing re-optimized"}
    (REC / "speed_sweep.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {REC / 'speed_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
