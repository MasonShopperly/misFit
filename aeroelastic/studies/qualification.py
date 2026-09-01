#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Cross-component qualification, and whether a physics heuristic replaces it.

Points and thresholds were frozen BEFORE this ran; they are tabulated in
results/aeroelastic_qualification.md.

Part 1 drives `verification/legacy_taylor_check.run_check` over four pairs at six predeclared design
points, and writes the 24 `part1_checker` rows of the record. It is given callables rather than URLs
because the aeroelastic objective is not itself a served Tesseract; the criterion, the sweep and the
verdict logic are unchanged. The submission's own qualification runs the platform's
`tesseract-runtime check-gradients` against the composed `coupled.C1` Tesseract instead
(results/native_composed_check.md).

Part 2 scores four predeclared physics comparators against the committed validity map's firing set.
A comparator that discriminated as well for less would make a derivative-level check unnecessary,
and that is reported at equal prominence either way.

    .venv/bin/python aeroelastic/studies/qualification.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))
sys.path.insert(0, str(HERE.parent / "verification"))
REC = HERE.parent / "records"

import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
import numpy as np                                                        # noqa: E402

import aero_struct as A                                                   # noqa: E402
from aero_struct import RHO, Wing                                         # noqa: E402
from model import CL_TARGET, V, W_MASS, _terms, refs                      # noqa: E402
from legacy_taylor_check import run_check                                 # noqa: E402

W = Wing()
SEED = 20260806

# Frozen points: (label, region, tip twist rad, uniform log GJ)
POINTS = [
    ("before-a", "before", np.deg2rad(-1.15), 0.00),
    ("before-b", "before", np.deg2rad(+1.15), -0.50),
    ("edge", "edge", np.deg2rad(-4.58), 0.00),
    ("inside-a", "inside", np.deg2rad(-5.16), -0.50),
    ("inside-b", "inside", np.deg2rad(-5.73), -0.80),
    ("after", "after", np.deg2rad(-5.73), -1.00),
]

# Frozen comparator thresholds, each from a stated physical source
THRESH = {"dynamic-pressure ratio": 0.80, "maximum elastic twist": np.deg2rad(5.0),
          "twist cancellation ratio": 1.0, "load redistribution": 0.25}


def design(tip, s):
    return np.array([0.0, 0.5 * tip, tip, s, s, s])


def objectives(cdi_ref, mass_ref):
    """The four pairings. Each returns (J, g) callables over the 6-vector design."""
    def coupled(x):
        cdi, mass, _ = _terms(x, "coupled")
        return cdi / cdi_ref + W_MASS * mass / mass_ref

    def rigid(x):
        cdi, mass, _ = _terms(x, "rigid")
        return cdi / cdi_ref + W_MASS * mass / mass_ref

    def frozen_def(x):
        """Frozen-deformation: honour the coupled state, do not let it respond to the design."""
        t_ctrl, s_ctrl = x[:3], x[3:]
        theta, _, _, _, alpha = A.solve_trimmed(W, t_ctrl, s_ctrl, V, CL_TARGET, A.solve_coupled)
        th = jax.lax.stop_gradient(theta)
        p = A.transfer_matrix(W)
        a_n = A.solve_aero(W, alpha + A.twist_profile(W, t_ctrl) + p @ th)
        _, cdi = A.coefficients(W, a_n)
        return cdi / cdi_ref + W_MASS * A.mass_proxy(W, s_ctrl) / mass_ref

    jf = {"coupled": jax.jit(coupled), "rigid": jax.jit(rigid), "frozen": jax.jit(frozen_def)}
    gf = {k: jax.jit(jax.grad(v)) for k, v in
          {"coupled": coupled, "rigid": rigid, "frozen": frozen_def}.items()}
    return jf, gf


PAIRS = [("matched-coupled", "coupled", "coupled"),
         ("matched-rigid", "rigid", "rigid"),
         ("crossed-rigid", "coupled", "rigid"),
         ("crossed-frozen", "coupled", "frozen")]


def part1(cdi_ref, mass_ref):
    jf, gf = objectives(cdi_ref, mass_ref)
    rng = np.random.default_rng(SEED)
    dirs = {f"random-{i}": rng.normal(0, 1, 6) for i in range(3)}
    rows = []
    print("PART 1 -- the shipped cross-component check, four pairs, six frozen points")
    print(f"{'point':>9} {'region':>7} {'pair':>16} {'verdict':>13} {'orders':>26} "
          f"{'obj calls':>10}")
    for label, region, tip, s in POINTS:
        x = design(tip, s)
        for pname, jkey, gkey in PAIRS:
            J = lambda m, f=jf[jkey]: float(f(jnp.asarray(m, jnp.float64)))
            g = lambda m, f=gf[gkey]: np.asarray(f(jnp.asarray(m, jnp.float64)), np.float64)
            d = dict(dirs)
            d["steepest"] = -g(x)
            try:
                rec = run_check(J, g, x, d)
            except Exception as exc:                                   # noqa: BLE001
                print(f"{label:>9} {region:>7} {pname:>16} {'ERROR':>13}  {exc}")
                continue
            orders = [v.get("fitted_order") for v in rec["directions"].values()]
            ostr = " ".join(f"{o:.2f}" if isinstance(o, float) else "--" for o in orders)
            print(f"{label:>9} {region:>7} {pname:>16} {rec['verdict']:>13} {ostr:>26} "
                  f"{rec['objective_applies_total']:>10}")
            rows.append({"point": label, "region": region, "pair": pname,
                         "verdict": rec["verdict"], "orders": orders,
                         "objective_applies": rec["objective_applies_total"],
                         "gradient_calls": rec["gradient_calls"]})
    return rows


def comparators(x, cdi_ref, mass_ref):
    """The four predeclared physics metrics at one design."""
    t_ctrl, s_ctrl = jnp.asarray(x[:3]), jnp.asarray(x[3:])
    theta, a_n, _, cdi, alpha = A.solve_trimmed(W, t_ctrl, s_ctrl, V, CL_TARGET, A.solve_coupled)
    p = A.transfer_matrix(W)
    theta_a = np.asarray(p @ theta)
    geo = np.asarray(A.twist_profile(W, t_ctrl))
    qd = float(A.divergence_q(W, s_ctrl))
    l_c = np.asarray(A.sectional_lift(W, a_n, V))
    _, a_r, _, _, _ = A.solve_trimmed(W, t_ctrl, s_ctrl, V, CL_TARGET, A.solve_rigid)
    l_r = np.asarray(A.sectional_lift(W, a_r, V))
    denom = max(np.max(np.abs(geo)), 1e-12)
    return {
        "dynamic-pressure ratio": (0.5 * RHO * V ** 2) / qd,
        "maximum elastic twist": float(np.max(np.abs(theta_a))),
        "twist cancellation ratio": float(np.max(np.abs(theta_a)) / denom),
        "load redistribution": float(np.linalg.norm(l_c - l_r) / max(np.linalg.norm(l_r), 1e-30)),
    }


def part2():
    print("\nPART 2 -- physics comparators against the committed validity map's firing set")
    vm = json.loads((REC / "validity_map.json").read_text())
    tips, stiffs, grid = vm["tips_rad"], vm["stiffs_log"], vm["r_hat"]
    eta = vm["eta_1"]
    truth, metrics = [], {k: [] for k in THRESH}
    for i, s in enumerate(stiffs):
        for j, t in enumerate(tips):
            r = grid[i][j]
            if r is None or not np.isfinite(r):
                continue
            truth.append(bool(r < eta))
            m = comparators(design(t, s), *refs())
            for k in THRESH:
                metrics[k].append(m[k])
    truth = np.asarray(truth)
    print(f"  ground truth: {int(truth.sum())} firing of {truth.size} designs")
    print(f"\n{'comparator':>26} {'threshold':>10} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
          f"{'accuracy':>9} {'needs':>22}")
    needs = {"dynamic-pressure ratio": "coupled state only",
             "maximum elastic twist": "coupled state only",
             "twist cancellation ratio": "coupled state only",
             "load redistribution": "coupled + rigid state"}
    out = []
    for k, thr in THRESH.items():
        pred = np.asarray(metrics[k]) >= thr
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
        acc = (tp + tn) / truth.size
        print(f"{k:>26} {thr:>10.4f} {tp:>4d} {fp:>4d} {fn:>4d} {tn:>4d} {acc:>9.4f} "
              f"{needs[k]:>22}")
        out.append({"comparator": k, "threshold": float(thr), "tp": tp, "fp": fp,
                    "fn": fn, "tn": tn, "accuracy": acc})
    print("\n  The checker needs two extra objective applies per assessed direction; every")
    print("  comparator above needs only state the optimizer has already computed.")
    return out, {k: [float(v) for v in metrics[k]] for k in metrics}, truth.tolist()


def main() -> int:
    cdi_ref, mass_ref = refs()
    p1 = part1(cdi_ref, mass_ref)
    p2, metrics, truth = part2()
    (REC / "qualification.json").write_text(json.dumps(
        {"part1_checker": p1, "part2_comparators": p2,
         "comparator_values": metrics, "truth": truth}, indent=1) + "\n")
    print(f"\nwrote {REC / 'qualification.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
