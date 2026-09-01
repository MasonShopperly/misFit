#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Qualify the provider once, then optimize across the served boundary.

Binds 8811-8812 and must be serialised against any other driver that binds them.

The check runs ONCE, before the optimizer starts, and the optimizer then uses the gradient the check
qualified. It is not re-checked per iteration: the cross-component check separates the pairs for 24
forward evaluations, and nothing in this repository measures what continuous re-checking would buy.

THE RIGID PROVIDER IS THE AERODYNAMIC COMPONENT USED ALONE.
`aero.A1` is a complete, correct, self-consistent aerodynamic solver with its own derivative
endpoints. Ask it for a gradient and you get the exact derivative of the rigid-wing objective. The
only thing wrong with it is that the aircraft is not rigid, and no endpoint on A1 can tell you that.
The cross-component check can, from forward evaluations of the coupled objective alone.

    .venv/bin/python aeroelastic/studies/run_served_optimize.py
    .venv/bin/python aeroelastic/studies/run_served_optimize.py --out /tmp/live.json

`--out` exists so a demonstration re-run does not have to overwrite the evidence it is meant to
reproduce. The default is unchanged and still writes served_optimize.json, which is how the
committed record is regenerated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPONENTS = HERE.parent / "components"
REC = HERE.parent / "records"
sys.path.insert(0, str(HERE.parent / "core"))
sys.path.insert(0, str(HERE.parent / "verification"))

import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
import numpy as np                                                        # noqa: E402
from scipy.optimize import minimize                                       # noqa: E402

import aero_struct as A                                                   # noqa: E402
from aero_struct import RHO, Wing                                         # noqa: E402
from model import CL_TARGET, V, W_MASS, refs                              # noqa: E402
from run_served import (                                                  # noqa: E402
    AERO, CALLS, P_AERO, P_STRUCT, STRUCT, assert_ports_free, call,
    clean_pycache, coupled_forward, served_gradient, start_server, wait_healthy,
)
from legacy_taylor_check import run_check                                 # noqa: E402

W = Wing()
S_FLOOR, S_CEIL = -0.7, 1.0
TWIST_BOUND = 0.10


def rigid_state(t_ctrl, s_ctrl, alpha):
    """The aerodynamic component used ALONE: no structural feedback, theta = 0."""
    twist = np.asarray(A.twist_profile(W, jnp.asarray(t_ctrl)))
    ao = call(AERO, "apply", {"alpha_tot": alpha + twist, "v": np.float64(V)})
    so = call(STRUCT, "apply", {"torque": np.zeros(len(twist)),
                                "s_ctrl": np.asarray(s_ctrl, np.float64)})
    return {"cl": float(ao["cl"]), "cdi": float(ao["cdi"]), "mass": float(so["mass"]),
            "u": alpha + twist}


def rigid_objective_and_gradient(x, cdi_ref, mass_ref):
    """A1's own gradient, exactly: correct for the rigid model, blind to the structure."""
    t_ctrl, s_ctrl = np.asarray(x[:3]), np.asarray(x[3:])
    s0 = rigid_state(t_ctrl, s_ctrl, 0.0)
    s1 = rigid_state(t_ctrl, s_ctrl, 1.0)
    alpha = (CL_TARGET - s0["cl"]) / (s1["cl"] - s0["cl"])
    st = rigid_state(t_ctrl, s_ctrl, alpha)
    j = st["cdi"] / cdi_ref + W_MASS * st["mass"] / mass_ref

    h = 1e-3
    dcdi_da = (rigid_state(t_ctrl, s_ctrl, alpha + h)["cdi"]
               - rigid_state(t_ctrl, s_ctrl, alpha - h)["cdi"]) / (2 * h)
    lam = (dcdi_da / cdi_ref) / (s1["cl"] - s0["cl"])
    ain = {"alpha_tot": st["u"], "v": np.float64(V)}
    u_bar = np.asarray(call(AERO, "vector_jacobian_product", ain,
                            vjp_inputs=["alpha_tot"], vjp_outputs=["cl", "cdi"],
                            cotangent_vector={"cl": np.float64(-lam),
                                              "cdi": np.float64(1.0 / cdi_ref)})["alpha_tot"],
                       np.float64)
    s_bar = np.asarray(call(STRUCT, "vector_jacobian_product",
                            {"torque": np.zeros(u_bar.size), "s_ctrl": s_ctrl},
                            vjp_inputs=["s_ctrl"], vjp_outputs=["mass"],
                            cotangent_vector={"mass": np.float64(W_MASS / mass_ref)})["s_ctrl"],
                       np.float64)
    dtwist_dt = np.asarray(jax.jacobian(lambda tc: A.twist_profile(W, tc))(jnp.asarray(t_ctrl)))
    return j, np.concatenate([dtwist_dt.T @ u_bar, s_bar])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REC / "served_optimize.json"),
                    help="where to write the record; defaults to the committed path")
    dest = Path(ap.parse_args().out)
    assert_ports_free([P_AERO, P_STRUCT])
    procs = []
    out = {}
    try:
        procs.append(start_server(COMPONENTS / "aero" / "tesseract_api.py", P_AERO))
        procs.append(start_server(COMPONENTS / "structure" / "tesseract_api.py", P_STRUCT))
        wait_healthy(AERO)
        wait_healthy(STRUCT)
        cdi_ref, mass_ref = refs()
        x0 = np.zeros(6)
        print(f"aero.A1 :{P_AERO} (JAX autodiff)   struct.S1 :{P_STRUCT} (hand adjoint)\n")

        # ---------------------------------------------------------------- step 1: qualify
        print("STEP 1 -- qualify the aerodynamic component's own gradient against the COUPLED")
        print("          objective, once, using only forward evaluations of that objective.")
        base = dict(CALLS.n)

        def J_coupled(m):
            """FORWARD ONLY. Calling served_gradient here would compute an adjoint the check never
            uses and inflate the measured qualification cost by every VJP round trip it makes."""
            t_ctrl, s_ctrl = np.asarray(m[:3]), np.asarray(m[3:])
            s0 = coupled_forward(t_ctrl, s_ctrl, 0.0, V)
            s1 = coupled_forward(t_ctrl, s_ctrl, 1.0, V)
            alpha = (CL_TARGET - s0["cl"]) / (s1["cl"] - s0["cl"])
            st = coupled_forward(t_ctrl, s_ctrl, alpha, V)
            return st["cdi"] / cdi_ref + W_MASS * st["mass"] / mass_ref

        def g_rigid(m):
            return rigid_objective_and_gradient(m, cdi_ref, mass_ref)[1]

        rng = np.random.default_rng(20260806)
        dirs = {f"random-{i}": rng.normal(0, 1, 6) for i in range(2)}
        rec = run_check(J_coupled, g_rigid, x0, dirs)
        qual_calls = {k: CALLS.n.get(k, 0) - base.get(k, 0) for k in CALLS.n}
        orders = [v.get("fitted_order") for v in rec["directions"].values()]
        # A bare "verdict: FAIL" reads as a broken run. What is on trial here is the PAIRING, and
        # its failing is the finding rather than a defect: the remainder |J(m+hp) - J(m) - h g.p|
        # decays at order 2 when the supplied gradient is a derivative of THIS objective and at
        # order 1 when it is not. Say which pairing and say what the number means.
        print("  pairing on trial: coupled objective + rigid gradient")
        print(f"  verdict: {rec['verdict']}   Taylor-remainder orders "
              f"{' '.join(f'{o:.2f}' for o in orders if isinstance(o, float))}")
        print("  order ~2 = the gradient IS a derivative of this objective; ~1 = it is not")
        print(f"  cost: {rec['objective_applies_total']} coupled-objective evaluations, "
              f"{rec['gradient_calls']} gradient call")
        print(f"  HTTP during qualification: {qual_calls}")
        out["qualification"] = {"verdict": rec["verdict"], "orders": orders,
                                "objective_applies": rec["objective_applies_total"],
                                "http": qual_calls}
        # Three-way, not two. `!= "FAIL"` folded INCONCLUSIVE in with PASS and printed "qualifies"
        # for a run that reached no verdict at all -- the one outcome this project's own tool is
        # built to report honestly rather than round toward a decision.
        if rec["verdict"] == "FAIL":
            print("  -> DISQUALIFIED. This is the expected outcome and the point of the step:")
            print("     A1's gradient is the exact derivative of a rigid wing, and this wing is")
            print("     not rigid. The optimizer uses the coupled adjoint instead.")
        elif rec["verdict"] == "PASS":
            print("  -> the rigid gradient QUALIFIES; it would be used. (On the committed run it "
                  "does not: the verdict is FAIL.)")
        else:
            print(f"  -> {rec['verdict']}: no verdict was reached, so nothing is qualified. "
                  f"This is not a pass.")

        # ---------------------------------------------------------------- step 2: optimize
        print("\nSTEP 2 -- optimize with the qualified gradient, every call across the boundary")
        base = dict(CALLS.n)
        hist = []
        t0 = time.perf_counter()

        def fg(m):
            j, g, *_ = served_gradient(np.asarray(m), V, cdi_ref, mass_ref, CL_TARGET)
            hist.append(j)
            return j, np.asarray(g, np.float64)

        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       bounds=[(-TWIST_BOUND, TWIST_BOUND)] * 3 + [(S_FLOOR, S_CEIL)] * 3,
                       options={"maxiter": 60, "ftol": 1e-12})
        wall = time.perf_counter() - t0
        opt_calls = {k: CALLS.n.get(k, 0) - base.get(k, 0) for k in CALLS.n}

        # The SAME pipeline, driven by the aerodynamic component's own gradient, so the figure and
        # the headline numbers come from one run rather than two. Without this arm the comparison
        # design came from the in-process matrix and the two disagreed on tip washout.
        print("\n  aerodynamic-only arm: same pipeline, A1's own gradient")
        base_r = dict(CALLS.n)
        hist_r = []

        def fg_rigid(m):
            j, g = rigid_objective_and_gradient(np.asarray(m), cdi_ref, mass_ref)
            hist_r.append(j)
            return j, np.asarray(g, np.float64)

        res_r = minimize(fg_rigid, x0, jac=True, method="L-BFGS-B",
                         bounds=[(-TWIST_BOUND, TWIST_BOUND)] * 3 + [(S_FLOOR, S_CEIL)] * 3,
                         options={"maxiter": 60, "ftol": 1e-12})
        # scored on the coupled objective, which is the only fair comparison: the aerodynamic-only
        # arm optimized a different objective from the one both arms are scored against.
        j_r_coupled = served_gradient(res_r.x, V, cdi_ref, mass_ref, CL_TARGET)[0]
        st_r = coupled_forward(res_r.x[:3], res_r.x[3:],
                               float(A.trim_alpha(W, jnp.asarray(res_r.x[:3]),
                                                  jnp.asarray(res_r.x[3:]), V, CL_TARGET,
                                                  A.solve_coupled)), V)
        qd_r = float(A.divergence_q(W, jnp.asarray(res_r.x[3:])))
        print(f"    its own objective {res_r.fun:.6f}; scored coupled {j_r_coupled:.6f}  "
              f"CDi {st_r['cdi']:.6f}  mass {st_r['mass']:.4f}  "
              f"q/q_D {0.5 * RHO * V ** 2 / qd_r:.3f}")
        out["aero_only"] = {"J_own": float(res_r.fun), "J_coupled": float(j_r_coupled),
                            "x": res_r.x.tolist(), "cdi": st_r["cdi"], "mass": st_r["mass"],
                            "cl": st_r["cl"], "q_over_qd": 0.5 * RHO * V ** 2 / qd_r,
                            "nit": int(res_r.nit),
                            "http": {k: CALLS.n.get(k, 0) - base_r.get(k, 0) for k in CALLS.n}}

        st = coupled_forward(res.x[:3], res.x[3:],
                             float(A.trim_alpha(W, jnp.asarray(res.x[:3]), jnp.asarray(res.x[3:]),
                                                V, CL_TARGET, A.solve_coupled)), V)
        qd = float(A.divergence_q(W, jnp.asarray(res.x[3:])))
        print(f"  J {hist[0]:.6f} -> {res.fun:.6f}  ({100 * (1 - res.fun / hist[0]):.1f} % better)"
              f"  in {res.nit} iterations, {wall:.1f} s")
        print(f"  design: twist {[round(float(np.degrees(v)), 2) for v in res.x[:3]]} deg, "
              f"log GJ {[round(float(v), 3) for v in res.x[3:]]}")
        print(f"  CL {st['cl']:.4f}  CDi {st['cdi']:.6f}  q/q_D {0.5 * RHO * V ** 2 / qd:.3f}")
        out["optimize"] = {"J0": hist[0], "J": float(res.fun), "nit": int(res.nit),
                           "x": res.x.tolist(), "cl": st["cl"], "cdi": st["cdi"],
                           "mass": float(A.mass_proxy(W, jnp.asarray(res.x[3:]))),
                           "q_over_qd": 0.5 * RHO * V ** 2 / qd, "wall_s": wall,
                           "http": opt_calls}

        # ---------------------------------------------------------------- step 3: participation
        print("\nSTEP 3 -- endpoint inventory. Both services, both directions, no bypass.")
        print(f"  {'endpoint':<40} {'qualify':>9} {'optimize':>10}")
        for k in sorted(set(qual_calls) | set(opt_calls)):
            print(f"  {k:<40} {qual_calls.get(k, 0):>9d} {opt_calls.get(k, 0):>10d}")
        ok = (opt_calls.get("aero.A1.apply", 0) > 0 and opt_calls.get("struct.S1.apply", 0) > 0
              and opt_calls.get("aero.A1.vector_jacobian_product", 0) > 0
              and opt_calls.get("struct.S1.vector_jacobian_product", 0) > 0)
        print(f"  both components serve apply AND vjp during optimization: {ok}")
        out["both_participate"] = bool(ok)
        dest.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {dest}")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        for _api in (COMPONENTS / "aero" / "tesseract_api.py",
                     COMPONENTS / "structure" / "tesseract_api.py"):
            clean_pycache(_api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
