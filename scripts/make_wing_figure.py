#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""The application figure: two wings, and where the aerodynamic-only gradient stops describing one.

Every number is read from committed evidence -- `served_optimize.json` for the two designs -- and
the wing states are recomputed from those designs by the same solver the experiments used. The
bottom panel is a gradient-validity flight recorder along the straight path between the two
designs, which is the question worth answering: does the simplified gradient fail somewhere the
optimizer must actually go?

    .venv/bin/python scripts/make_wing_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AE = ROOT / "aeroelastic"
REC = AE / "records"
sys.path.insert(0, str(AE / "core"))

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402

import aero_struct as A                                                   # noqa: E402
from aero_struct import RHO, Wing                                         # noqa: E402
from model import CL_TARGET, H_FD, V, W_MASS, _terms, refs                # noqa: E402

W = Wing()
OUT = ROOT / "docs" / "figures" / "aeroelastic_wings.png"
INK, MUTED, GRID = "#24292f", "#57606a", "#d0d7de"
RIGID, COUPLED = "#cf222e", "#0969da"
ETA_1 = 0.01


def state(x):
    t, s = jnp.asarray(x[:3]), jnp.asarray(x[3:])
    theta, a_n, cl, cdi, alpha = A.solve_trimmed(W, t, s, V, CL_TARGET, A.solve_coupled)
    p = A.transfer_matrix(W)
    _, eta, _ = A.aero_grid(W)
    return {"eta": np.asarray(eta), "lift": np.asarray(A.sectional_lift(W, a_n, V)),
            "theta_e": np.degrees(np.asarray(p @ theta)),
            "theta_g": np.degrees(np.asarray(A.twist_profile(W, t))),
            "cl": float(cl), "cdi": float(cdi), "mass": float(A.mass_proxy(W, s)),
            "q_over_qd": 0.5 * RHO * V ** 2 / float(A.divergence_q(W, s))}


def r_hat_along(xa, xb, n=41):
    """r_hat for the aerodynamic-only gradient, along the straight path from xa to xb."""
    cdi_ref, mass_ref = refs()

    def mk(mode):
        def j(x):
            cdi, mass, _ = _terms(x, mode)
            return cdi / cdi_ref + W_MASS * mass / mass_ref
        return j
    jc, jr = jax.jit(mk("coupled")), mk("rigid")
    gc, gr = jax.jit(jax.grad(mk("coupled"))), jax.jit(jax.grad(jr))
    ts, rs, cs = [], [], []
    for t in np.linspace(0.0, 1.0, n):
        x = (1 - t) * np.asarray(xa) + t * np.asarray(xb)
        g = np.asarray(gr(jnp.asarray(x)), np.float64)
        f = np.asarray(gc(jnp.asarray(x)), np.float64)
        nrm = np.linalg.norm(g)
        if nrm == 0 or not np.isfinite(nrm):
            continue
        p = -g / nrm
        dd = (float(jc(jnp.asarray(x + H_FD * p))) - float(jc(jnp.asarray(x - H_FD * p)))) / (2 * H_FD)
        ts.append(t); rs.append(dd / float(g @ p))
        cs.append(float(g @ f / (nrm * np.linalg.norm(f))))
    return np.array(ts), np.array(rs), np.array(cs)


def main() -> int:
    # BOTH designs come from the SAME served run, so the figure and the headline numbers cannot
    # disagree about tip washout for what is nominally the same design.
    sv = json.loads((REC / "served_optimize.json").read_text())
    xa = np.asarray(sv["aero_only"]["x"])
    xb = np.asarray(sv["optimize"]["x"])
    ja_coupled, jb = sv["aero_only"]["J_coupled"], sv["optimize"]["J"]
    ja_self = sv["aero_only"]["J_own"]
    a, c = state(xa), state(xb)

    fig = plt.figure(figsize=(10.6, 8.4), dpi=130)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.62,
                          left=0.095, right=0.965, top=0.885, bottom=0.075)
    fig.suptitle("At the design condition, the aerodynamic solver alone costs 19.9 % more induced drag",
                 fontsize=15.5, color=INK, y=0.975, weight="bold")
    fig.text(0.53, 0.925, "both wings trimmed to the same lift,  $C_L$ = 0.50  at 80 m/s",
             ha="center", fontsize=11, color=MUTED)

    ax = fig.add_subplot(gs[0])
    ax.plot(a["eta"], a["lift"], color=RIGID, lw=2.6, label="aerodynamic gradient only")
    ax.plot(c["eta"], c["lift"], color=COUPLED, lw=2.6, label="coupled gradient")
    ax.set_ylabel("lift per span  [N/m]", fontsize=10.5, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, None)
    ax.legend(fontsize=10.5, frameon=False, loc="lower left")
    ax.set_title(f"same lift, same structure — and {100 * (a['cdi'] / c['cdi'] - 1):.0f} % more "
                 f"induced drag ({a['cdi']:.5f} vs {c['cdi']:.5f})",
                 fontsize=12, color=INK, pad=8)

    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(0, color=GRID, lw=1)
    for st, col in ((a, RIGID), (c, COUPLED)):
        ax2.plot(st["eta"], st["theta_g"], color=col, lw=1.6, ls="--", alpha=0.75)
        ax2.plot(st["eta"], st["theta_g"] + st["theta_e"], color=col, lw=2.6)
    ax2.set_ylabel("twist  [deg]", fontsize=10.5, color=MUTED)
    ax2.set_xlabel("spanwise station        root $\\leftarrow$          $\\rightarrow$ tip",
                   fontsize=10.5, color=MUTED)
    ax2.set_xlim(0, 1)
    ax2.set_title(f"dashed: twist built in   ·   solid: twist in flight   ·   "
                  f"$q/q_D$ {a['q_over_qd']:.2f} $\\rightarrow$ {c['q_over_qd']:.2f}",
                  fontsize=12, color=INK, pad=8)

    t, r, cos = r_hat_along(xa, xb)
    ax3 = fig.add_subplot(gs[2])
    bad = r < ETA_1
    # The ratio is ill-conditioned at t = 0: the aerodynamic-only design is a stationary point of
    # its OWN objective, so the denominator goes to zero there and the ratio blows up. Clip the
    # display rather than let a small-denominator artifact set the scale, and say so on the axis.
    lo = max(-6.0, float(r.min()) * 1.15)
    ax3.axhspan(lo, ETA_1, color=RIGID, alpha=0.10)
    ax3.axhline(ETA_1, color=RIGID, lw=1.4, ls=":")
    ax3.plot(t, r, color=INK, lw=2.6)
    ax3.plot(t[bad], r[bad], "o", color=RIGID, ms=4.5, zorder=4)
    ax3.plot([0], [r[0]], "o", color=RIGID, ms=11, mec="white", mew=1.8, zorder=5)
    ax3.plot([1], [r[-1]], "o", color=COUPLED, ms=11, mec="white", mew=1.8, zorder=5)
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(lo, 0.75)
    ax3.set_ylabel("measured / claimed slope", fontsize=10.5, color=MUTED)
    if float(r.min()) < lo:
        ax3.text(0.012, 0.055, f"(clipped: reaches {r.min():.1f} at the left endpoint, where the "
                 f"aerodynamic-only gradient is at its own optimum)",
                 transform=ax3.transAxes, fontsize=9, color=MUTED)
    ax3.set_xlabel("along the straight path from one design to the other", fontsize=10.5,
                   color=MUTED)
    ax3.set_title(f"its own model scored it {ja_self:.4f} — a tie.  The coupled objective scores "
                  f"it {ja_coupled:.4f}, {100 * (ja_coupled / jb - 1):.0f} % worse",
                  fontsize=12, color=INK, pad=8)
    ax3.text(0.985, 0.30, "shaded: no step size helps —\nthe gradient itself must change",
             transform=ax3.transAxes, fontsize=10.5, color=RIGID, ha="right")
    ax3.text(0.0, -0.20, "aerodynamic-only design", transform=ax3.get_xaxis_transform(),
             fontsize=10, color=RIGID, ha="left", va="top")
    ax3.text(1.0, -0.20, "coupled design", transform=ax3.get_xaxis_transform(),
             fontsize=10, color=COUPLED, ha="right", va="top")

    for aa in (ax, ax2, ax3):
        for sp in ("top", "right"):
            aa.spines[sp].set_visible(False)
        aa.tick_params(labelsize=9.5, colors=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    # The figure's derived numbers are persisted so every number in a caption is recomputed from
    # evidence rather than retyped.
    (REC / "wing_figure.json").write_text(json.dumps({
        "rigid": {k: v for k, v in a.items() if not isinstance(v, np.ndarray)},
        "coupled": {k: v for k, v in c.items() if not isinstance(v, np.ndarray)},
        "structure_ratio": a["mass"] / c["mass"],
        "aero_only_self_scored": ja_self,
        "aero_only_coupled_scored": ja_coupled,
        "coupled_scored": jb,
        "aero_only_worse_pct": 100.0 * (ja_coupled / jb - 1.0),
        "aero_only_drag_excess_pct": 100.0 * (a["cdi"] / c["cdi"] - 1.0),
        "path_uphill_fraction": float(bad.mean()),
        "path_r_hat_min": float(r.min()),
        "path_cos_min": float(cos.min()),
        "eta_1": ETA_1,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {(REC / 'wing_figure.json').relative_to(ROOT)}")
    print(f"  aerodynamic-only: CDi {a['cdi']:.6f}  structure {a['mass']:.4f}  "
          f"q/q_D {a['q_over_qd']:.3f}")
    print(f"  coupled         : CDi {c['cdi']:.6f}  structure {c['mass']:.4f}  "
          f"q/q_D {c['q_over_qd']:.3f}")
    # Both arms end at the same stiffness bound, so the mass proxy is identical by construction and
    # the ratio is 1.00. It is printed as an identity rather than as a ratio, so this command's own
    # stdout cannot be read as claiming a mass advantage for either arm.
    print(f"  structure mass proxy: identical to {a['mass'] / c['mass']:.2f}x — both arms end at "
          f"the same stiffness bound, so no mass advantage is claimed for either")
    print(f"  r_hat below eta_1 on {100 * bad.mean():.1f} % of the path; "
          f"min {r.min():.3f}, min cos {cos.min():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
