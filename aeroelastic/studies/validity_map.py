#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""The 289-design sweep that locates where the rigid gradient stops being a descent direction.

Produces `validity_map.json`, the record behind the headline validity-boundary numbers: 289
sampled designs, 31 firing, 30 of those 31 with the coupled objective increasing along the rigid
descent direction.

At each design the map records three quantities.

    r_hat     the coupled objective's actual directional derivative along p, divided by the rate
              the rigid gradient predicts, where p = -g_rigid / ||g_rigid|| is the unit rigid
              descent direction. The design fires when r_hat < ETA_1 = 0.01. A negative r_hat
              means the coupled objective INCREASES along a direction the rigid model calls
              downhill.
    cos       cos(g_rigid, g_coupled), the fraction of the available first-order decrease the
              rigid direction captures. Reported alongside r_hat because direction agreement and
              realised decrease are separate questions.
    q/q_D     dynamic pressure over the design's own divergence dynamic pressure, so a row can be
              read against how close that wing is to static divergence.

The two axes control how much elastic twist competes with the twist the designer asked for: tip
geometric twist (washout is negative) against uniform log stiffness ratio. Both are fixed in this
file, as is ETA_1.

The directional derivative is a central difference with H_FD from `model.py`, so each cell costs
two coupled objective applies plus two reverse-mode gradients.

    .venv/bin/python aeroelastic/studies/validity_map.py
    .venv/bin/python aeroelastic/studies/validity_map.py --out /tmp/validity_map.json

Without `--out` nothing is written, so the committed record is only overwritten when it is named
explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))

import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
import numpy as np                                                        # noqa: E402

from model import H_FD, W_MASS, _terms, refs                              # noqa: E402

ETA_1 = 0.01
TIPS = np.linspace(-0.10, 0.06, 17)        # rad of tip geometric twist
STIFFS = np.linspace(0.6, -1.0, 17)        # log GJ ratio, uniform


def r_hat(x, cdi_ref, mass_ref):
    def mk(mode):
        def j(xx):
            cdi, mass, _ = _terms(xx, mode)
            return cdi / cdi_ref + W_MASS * mass / mass_ref
        return j
    jf = mk("coupled")
    gc = np.asarray(jax.grad(mk("rigid"))(jnp.asarray(x)), dtype=np.float64)
    gf = np.asarray(jax.grad(jf)(jnp.asarray(x)), dtype=np.float64)
    n = np.linalg.norm(gc)
    if n == 0.0 or not np.isfinite(n):
        return None, None
    p = -gc / n
    dd_i = (float(jf(jnp.asarray(x + H_FD * p))) - float(jf(jnp.asarray(x - H_FD * p)))) / (2 * H_FD)
    cos = float(gc @ gf / (n * np.linalg.norm(gf)))
    return dd_i / float(gc @ p), cos


def sweep():
    """The 17x17 grid. Returns (r_hat, cos, q/q_D) as (len(STIFFS), len(TIPS)) arrays."""
    cdi_ref, mass_ref = refs()
    grid = np.full((len(STIFFS), len(TIPS)), np.nan)
    cosg = np.full((len(STIFFS), len(TIPS)), np.nan)
    qrat = np.full((len(STIFFS), len(TIPS)), np.nan)
    for i, s in enumerate(STIFFS):
        for j, t in enumerate(TIPS):
            x = np.array([0.0, 0.5 * t, t, s, s, s])
            r, c = r_hat(x, cdi_ref, mass_ref)
            if r is not None:
                grid[i, j], cosg[i, j] = r, c
            _, _, q_ratio = _terms(jnp.asarray(x), "coupled")
            qrat[i, j] = float(q_ratio)
    return grid, cosg, qrat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="write the map here; omit to print the summary without writing")
    args = ap.parse_args()

    grid, cosg, qrat = sweep()

    print("r_hat over design space. '#' = fires (r_hat < 0.01); '.' = rigid gradient adequate")
    print(f"{'log GJ':>7} {'q/q_D':>7}  tip twist (deg) "
          + " ".join(f"{np.degrees(t):>5.1f}" for t in TIPS))
    for i, s in enumerate(STIFFS):
        marks = "".join("#" if (np.isfinite(grid[i, j]) and grid[i, j] < ETA_1) else
                        ("?" if not np.isfinite(grid[i, j]) else ".") for j in range(len(TIPS)))
        print(f"{s:>7.2f} {qrat[i, len(TIPS) // 2]:>7.3f}                  {marks}")

    fired = np.isfinite(grid) & (grid < ETA_1)
    print(f"\nfires on {fired.sum()} of {grid.size} sampled designs "
          f"({100.0 * fired.sum() / grid.size:.1f} %)")
    print(f"of those, {int((grid[fired] < 0.0).sum())} have r_hat < 0, meaning the coupled "
          f"objective increases along the rigid descent direction")
    ok = np.isfinite(grid) & ~fired
    print(f"where it does not fire: r_hat median {np.median(grid[ok]):.3f}, "
          f"min {np.min(grid[ok]):.3f}")
    print(f"where it fires:         r_hat median {np.median(grid[fired]):.3f}, "
          f"min {np.min(grid[fired]):.3f}")
    print(f"cos over the firing region: median {np.median(cosg[fired]):.4f}, "
          f"min {np.min(cosg[fired]):.4f}")

    print("\nSlice at uniform stiffness log GJ = 0 (the reference wing):")
    i0 = int(np.argmin(np.abs(STIFFS)))
    print(f"  {'tip twist':>10} {'r_hat':>9} {'cos':>8} {'fires':>6}")
    for j, t in enumerate(TIPS):
        print(f"  {np.degrees(t):>9.2f}d {grid[i0, j]:>9.4f} {cosg[i0, j]:>8.4f} "
              f"{'FIRE' if grid[i0, j] < ETA_1 else '-':>6}")

    if args.out is None:
        print("\nno --out given; nothing written")
        return 0
    args.out.write_text(json.dumps(
        {"tips_rad": TIPS.tolist(), "stiffs_log": STIFFS.tolist(),
         "r_hat": grid.tolist(), "cos": cosg.tolist(), "q_over_qd": qrat.tolist(),
         "eta_1": ETA_1}, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
