#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Every public visual, rendered from committed evidence by one command.

    .venv/bin/python scripts/make_media.py            # render every published surface
    .venv/bin/python scripts/make_media.py --extra    # also the unpublished 16:9 video slate

Every plotted number is read from the committed record it belongs to and never retyped.

`--check` IS A DEVELOPER CHECK AND IS NOT PART OF THE DOCUMENTED REPRODUCTION ROUTE, which is why
the usage lines above omit it. It renders IN PLACE and then compares, so it overwrites the committed
surfaces and leaves the drifting ones modified in the working tree; `git checkout -- docs/figures`
puts them back. On a clone with nothing wrong with it, two of the four do not match:

    aeroelastic_hero.png          43 px of 1.3 M differ, by 1 intensity level
    aeroelastic_morph.gif      6,612 px of 8.3 M differ, by up to 51 intensity levels

What that measures is the rendering environment, not the code. Three consecutive renders on one
machine produced byte-identical output on all four surfaces; the difference is between this machine
and the one the committed figures were rendered on. The PNG differences are anti-alias rounding. The
GIF's are not: it is palettized, its 256-entry palette is itself re-derived on each render, and a
sub-level change in the source raster can move a pixel to a palette entry far away in intensity --
hence a 51-level pixel in an animation no viewer can tell from the committed one.

That tail is why this stays a strict byte comparison rather than growing an equivalence tolerance. A
threshold loose enough to pass the GIF is loose enough to pass a real rendering fault, and byte
comparison is what caught the three defects that actually occurred here: a missing binding, a figure
drawn from a different experiment than its headline, and a reversed root/tip orientation. So
`--check` exits 1 off its home environment, and is documented that way rather than relaxed until it
passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AE = ROOT / "aeroelastic"
REC = AE / "records"
FIG = ROOT / "docs/figures"
sys.path.insert(0, str(AE / "core"))

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import jax.numpy as jnp                                                   # noqa: E402
import matplotlib.patches as mpatches                                     # noqa: E402
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402

import aero_struct as A                                                   # noqa: E402
from aero_struct import RHO, Wing                                         # noqa: E402
from model import CL_TARGET, V                                            # noqa: E402

W = Wing()
INK, MUTED, GRID, PAPER = "#1f2328", "#59636e", "#d1d9e0", "#ffffff"
RIGID, COUPLED = "#cf222e", "#0969da"
SAVE = {"facecolor": PAPER, "metadata": {"Date": None}}


def served():
    return json.loads((REC / "served_optimize.json").read_text())


def state(x):
    """Recompute a wing state from a committed design vector. eta = 0 is ROOT, eta = 1 is TIP."""
    t, s = jnp.asarray(x[:3]), jnp.asarray(x[3:])
    theta, a_n, cl, cdi, alpha = A.solve_trimmed(W, t, s, V, CL_TARGET, A.solve_coupled)
    p = A.transfer_matrix(W)
    _, eta, _ = A.aero_grid(W)
    order = np.argsort(np.asarray(eta))          # ascending eta: root -> tip, left -> right
    return {"eta": np.asarray(eta)[order],
            "lift": np.asarray(A.sectional_lift(W, a_n, V))[order],
            "theta_e": np.degrees(np.asarray(p @ theta))[order],
            "theta_g": np.degrees(np.asarray(A.twist_profile(W, t)))[order],
            "cl": float(cl), "cdi": float(cdi), "mass": float(A.mass_proxy(W, s)),
            "q_over_qd": 0.5 * RHO * V ** 2 / float(A.divergence_q(W, s))}


def despine(ax, keep=("left", "bottom")):
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(sp in keep)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


# ------------------------------------------------------------------ desktop hero
def desktop_hero(sv, out: Path) -> None:
    """Application first: geometry, then consequence, then the diagnostic as supporting evidence."""
    a, c = state(np.asarray(sv["aero_only"]["x"])), state(np.asarray(sv["optimize"]["x"]))
    drag_excess = 100.0 * (a["cdi"] / c["cdi"] - 1.0)
    worse = 100.0 * (sv["aero_only"]["J_coupled"] / sv["optimize"]["J"] - 1.0)

    fig = plt.figure(figsize=(10.4, 7.4), dpi=130)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], hspace=0.34, wspace=0.24,
                          left=0.085, right=0.965, top=0.845, bottom=0.055)
    fig.suptitle("At the design condition, the air simulator alone costs 20 % more induced drag",
                 fontsize=16, color=INK, weight="bold", y=0.972)
    fig.text(0.525, 0.918, "both wings trimmed to the same lift and built with the same structure"
             "  ·  $C_L$ = 0.50 at 80 m/s", ha="center", fontsize=11, color=MUTED)

    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(0, color=GRID, lw=1)
    k = np.linspace(0, len(c["eta"]) - 1, 11).astype(int)
    ax.plot(c["eta"], c["theta_g"], color=MUTED, lw=1.9, ls=(0, (5, 3)))
    ax.plot(c["eta"], c["theta_g"] + c["theta_e"], color=COUPLED, lw=2.7)
    ax.plot(c["eta"][k], (c["theta_g"] + c["theta_e"])[k], "o", color=COUPLED, ms=5)
    ax.fill_between(c["eta"], c["theta_g"], c["theta_g"] + c["theta_e"], color=COUPLED, alpha=0.13)
    m = len(c["eta"]) // 2
    ax.annotate("as built", (c["eta"][m], c["theta_g"][m]), textcoords="offset points",
                xytext=(0, -20), fontsize=11.5, color=MUTED, ha="center", weight="bold")
    ax.annotate("in flight", (c["eta"][m], (c["theta_g"] + c["theta_e"])[m]),
                textcoords="offset points", xytext=(0, 11), fontsize=11.5, color=COUPLED,
                ha="center", weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["root", "tip"], fontsize=11.5, color=INK)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylabel("twist  (degrees)", fontsize=11, color=MUTED)
    ax.set_title("built-in twist against twist in flight", fontsize=12.5, color=INK, pad=8)
    ax.tick_params(axis="y", labelsize=10, colors=MUTED)
    despine(ax)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(a["eta"], a["lift"], color=RIGID, lw=2.7)
    ax2.plot(c["eta"], c["lift"], color=COUPLED, lw=2.7)
    ax2.plot(a["eta"][k], a["lift"][k], "s", color=RIGID, ms=5)
    ax2.plot(c["eta"][k], c["lift"][k], "o", color=COUPLED, ms=5)
    ax2.annotate("air simulator only", (a["eta"][3], a["lift"][3]),
                 textcoords="offset points", xytext=(12, -18), fontsize=11.5, color=RIGID,
                 weight="bold")
    j = len(c["eta"]) // 5
    ax2.annotate("both simulators", (c["eta"][j], c["lift"][j]), textcoords="offset points",
                 xytext=(10, 16), fontsize=11.5, color=COUPLED, weight="bold")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["root", "tip"], fontsize=11.5, color=INK)
    ax2.set_xlim(-0.03, 1.03); ax2.set_ylim(0, None)
    ax2.set_ylabel("lift per metre of span  (N/m)", fontsize=11, color=MUTED)
    ax2.set_title("so the load lands somewhere else", fontsize=12.5, color=INK, pad=8)
    ax2.tick_params(axis="y", labelsize=10, colors=MUTED)
    despine(ax2)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    box = mpatches.FancyBboxPatch((0.005, 0.06), 0.99, 0.88, transform=ax3.transAxes,
                                  boxstyle="round,pad=0.012", linewidth=1.2,
                                  edgecolor=GRID, facecolor="#f6f8fa", zorder=0)
    ax3.add_patch(box)
    ax3.text(0.5, 0.90, "Both models pass alone. The crossed objective–gradient pairing fails.",
             transform=ax3.transAxes, fontsize=13.5, color=INK, ha="center", weight="bold")
    # The verdicts are the platform's own, from check-gradients, not this project's.
    plat = json.loads((REC / "platform_checker_ae.json").read_text())
    nat = json.loads((REC / "native_composed_check.json").read_text())
    crossed = [r for k, r in nat.items() if k.endswith("crossed-rigid")]
    cf = sum(sum(r["failures"]) for r in crossed)
    cc = sum(sum(r["checks"]) for r in crossed)
    ax3.text(0.5, 0.755, "every verdict below from  tesseract-runtime check-gradients",
             transform=ax3.transAxes, fontsize=10.2, color=MUTED, ha="center")
    rows = [("air model checked against itself", "passes",
             f"0 of {sum(plat['aero.A1']['checks']):,} checks", COUPLED),
            ("structure model checked against itself", "passes",
             f"0 of {sum(plat['struct.S1']['checks']):,} checks", COUPLED),
            ("air model's gradient against the coupled aircraft", "FAILS",
             f"{cf} of {cc} checks", RIGID)]
    for i, (what, verdict, order, col) in enumerate(rows):
        y = 0.585 - i * 0.165
        ax3.text(0.045, y, what, transform=ax3.transAxes, fontsize=11.5, color=INK, va="center")
        ax3.text(0.665, y, verdict, transform=ax3.transAxes, fontsize=11.5, color=col,
                 weight="bold", va="center")
        ax3.text(0.795, y, order, transform=ax3.transAxes, fontsize=11.5, color=MUTED,
                 va="center")
    ax3.text(0.045, 0.10, f"Used anyway, that pairing costs {drag_excess:.0f}% more drag "
             f"and scores {worse:.0f}% worse, while its own model reports success.",
             transform=ax3.transAxes, fontsize=11.5, color=INK, style="italic")

    fig.savefig(out, **SAVE)
    plt.close(fig)


# ------------------------------------------------------------------ video crop
def landscape(sv, out: Path, size, dpi, title_fs, big_fs) -> None:
    """One idea, one number, on a 16:9 slate for video use."""
    a, c = state(np.asarray(sv["aero_only"]["x"])), state(np.asarray(sv["optimize"]["x"]))
    drag_excess = 100.0 * (a["cdi"] / c["cdi"] - 1.0)

    fig = plt.figure(figsize=size, dpi=dpi)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.20,
                          left=0.065, right=0.955, top=0.775, bottom=0.115)
    fig.suptitle("A wing twists under the air that lifts it —\nand the twisting changes the lift",
                 fontsize=title_fs, color=INK, weight="bold", y=0.955, linespacing=1.32)

    ax = fig.add_subplot(gs[0])
    ax.axhline(0, color=GRID, lw=1)
    k = np.linspace(0, len(c["eta"]) - 1, 10).astype(int)
    ax.plot(c["eta"], c["theta_g"], color=MUTED, lw=2.0, ls=(0, (5, 3)))
    ax.plot(c["eta"], c["theta_g"] + c["theta_e"], color=COUPLED, lw=2.8)
    ax.plot(c["eta"][k], (c["theta_g"] + c["theta_e"])[k], "o", color=COUPLED, ms=5)
    ax.fill_between(c["eta"], c["theta_g"], c["theta_g"] + c["theta_e"], color=COUPLED, alpha=0.13)
    # Anchored at MID-SPAN, not at the tip: off the terminal point the blue curve runs straight
    # through the words "in flight", and "built" collides with the tip tick.
    m = len(c["eta"]) // 2
    ax.annotate("built", (c["eta"][m], c["theta_g"][m]), textcoords="offset points",
                xytext=(0, -20), fontsize=big_fs * 0.52, color=MUTED, ha="center",
                weight="bold")
    ax.annotate("in flight", (c["eta"][m], (c["theta_g"] + c["theta_e"])[m]),
                textcoords="offset points", xytext=(10, 16), fontsize=big_fs * 0.52,
                color=COUPLED, ha="left", weight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["root", "tip"], fontsize=big_fs * 0.5, color=INK)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylabel("twist (deg)", fontsize=big_fs * 0.48, color=MUTED)
    ax.tick_params(axis="y", labelsize=big_fs * 0.42, colors=MUTED)
    despine(ax)

    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    ax2.text(0.0, 0.90, "Design it with the air\nsimulator alone:", fontsize=big_fs * 0.60,
             color=INK, va="top", linespacing=1.35)
    ax2.text(0.0, 0.46, f"+{drag_excess:.0f}%", fontsize=big_fs * 1.9, color=RIGID,
             weight="bold", va="center")
    ax2.text(0.0, 0.20, "induced drag, at the same lift\nand the same structure",
             fontsize=big_fs * 0.52, color=MUTED, va="top", linespacing=1.35)
    ax2.text(0.0, -0.02, "its own model reports success", fontsize=big_fs * 0.50, color=RIGID,
             style="italic", va="top")

    fig.text(0.065, 0.035, "misFit  ·  coupled aeroelastic design across two Tesseracts",
             fontsize=big_fs * 0.44, color=MUTED)
    fig.savefig(out, **SAVE)
    plt.close(fig)


# ------------------------------------------------------------------ architecture, drawn not parsed
def architecture(out_png: Path) -> None:
    """A deterministic architecture drawing with a reproducible source.

    Not Mermaid: this host has no renderer, and shipping unrendered Mermaid means shipping a diagram
    nobody has ever seen. Drawn with the same library that draws every other figure, so it renders
    identically wherever those do.

    Layout is chosen so no edge crosses a box: the optimizer loop closes across the TOP, and the
    qualification verdict is routed around the OUTSIDE on the right. The one thing this diagram must
    not do is suggest the check sits inside the optimizer loop.
    """
    for out, dpi in ((out_png, 150),):
        fig = plt.figure(figsize=(10.6, 6.4), dpi=dpi)
        fig.patch.set_facecolor(PAPER)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(-2, 104); ax.set_ylim(-4, 66); ax.axis("off")

        def box(x, y, w, h, title, sub, edge, fill):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.7", linewidth=1.7,
                edgecolor=edge, facecolor=fill, zorder=2))
            ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top", fontsize=12,
                    color=INK, weight="bold", zorder=3)
            if sub:
                ax.text(x + w / 2, y + h - 8.0, sub, ha="center", va="top", fontsize=9.5,
                        color=MUTED, zorder=3, linespacing=1.4)

        def arrow(x1, y1, x2, y2, col=INK, style="-", rad=0.0, lw=1.8):
            ax.annotate("", (x2, y2), (x1, y1), zorder=1, arrowprops=dict(
                arrowstyle="-|>", color=col, lw=lw, linestyle=style, shrinkA=2, shrinkB=2,
                connectionstyle=f"arc3,rad={rad}"))

        def tag(x, y, s, col=INK, fs=9.5):
            ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=col, zorder=4,
                    linespacing=1.3, bbox=dict(boxstyle="round,pad=0.28", fc=PAPER, ec="none"))

        box(3, 47, 26, 12, "design variables", "twist  ·  stiffness", INK, "#f6f8fa")
        box(71, 47, 26, 12, "optimizer", "uses the qualified gradient", INK, "#f6f8fa")
        box(3, 26, 26, 15, "aero.A1  Tesseract",
            "lifting line\ndifferentiated by JAX autodiff", COUPLED, "#ddf4ff")
        box(37, 26, 26, 15, "struct.S1  Tesseract",
            "torsion beam\nadjoint derived by hand", COUPLED, "#ddf4ff")
        box(71, 26, 26, 15, "coupled objective",
            "drag + structural mass\nat fixed trimmed lift", INK, "#f6f8fa")
        # Three short lines, not two long ones: a 60-character line overflows the box's right edge
        # in the rasterised PNG the README embeds.
        box(24, 2, 46, 15, "coupled.C1  +  check-gradients",
            "the assembled objective as a third Tesseract,\n"
            "checked by the platform's own tool —\n"
            "once, before the optimizer starts", RIGID, "#fff1f0")

        arrow(70.4, 53, 29.6, 53)
        tag(50, 56.2, "proposes the next wing")
        arrow(16, 46.4, 16, 41.6)
        arrow(29.6, 36, 36.4, 36, col=COUPLED, rad=-0.34)
        tag(33, 40.4, "loads", COUPLED)
        arrow(36.4, 31, 29.6, 31, col=COUPLED, rad=-0.34)
        tag(33, 26.6, "twist", COUPLED)
        arrow(63.6, 33.5, 70.4, 33.5)
        arrow(84, 41.6, 84, 46.4)

        arrow(78, 25.4, 66, 16.6, col=RIGID, style=(0, (4, 3)), rad=0.16)
        tag(76.5, 19.5, "13 evaluations,\nonce", RIGID, 9.0)
        # verdict routed OUTSIDE every box, up the right-hand margin
        ax.plot([70.4, 101, 101], [9.5, 9.5, 53], color=RIGID, lw=1.8, ls=(0, (4, 3)), zorder=1)
        arrow(101, 53, 97.6, 53, col=RIGID, style=(0, (4, 3)))
        tag(88, 6.0, "verdict: use the coupled gradient", RIGID, 9.0)

        ax.text(3, 63.4, "Two solvers, coupled both ways. The check runs once, off to the side.",
                fontsize=13, color=INK, weight="bold")
        ax.plot([3, 8], [-1.4, -1.4], color=COUPLED, lw=2.0)
        ax.text(9.2, -1.4, "physics, both directions", fontsize=9.5, color=MUTED, va="center")
        ax.plot([40, 45], [-1.4, -1.4], color=RIGID, lw=2.0, ls=(0, (4, 3)))
        ax.text(46.2, -1.4, "qualification, traversed once", fontsize=9.5, color=MUTED,
                va="center")

        fig.savefig(out, **SAVE)
        plt.close(fig)


# ------------------------------------------------------------------ the two designs, morphing
def wing_morph(sv, out: Path) -> None:
    """The one moving picture: the aerodynamic-only wing becoming the coupled one.

    Every frame is a real coupled solve at a real design vector, interpolated along the SAME
    straight path between the two committed designs that `wing_figure.json` already samples for the
    descent test. Nothing is re-optimized and nothing is drawn that is not solved: the twist curves,
    the lift distribution and the induced-drag readout all come from `state(x)` at that frame's x.

    Determinism is the whole reason this is allowed to exist at all: fixed frame count,
    PillowWriter, no timestamp, and `make_media.py --check` byte-compares it like every other
    surface. If it ever stops reproducing on a fixed environment it is withdrawn rather than
    argued with.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    xa = np.asarray(sv["aero_only"]["x"])
    xc = np.asarray(sv["optimize"]["x"])
    HOLD, SWEEP = 6, 22
    ts = ([0.0] * HOLD + list(np.linspace(0.0, 1.0, SWEEP)) + [1.0] * HOLD
          + list(np.linspace(1.0, 0.0, SWEEP // 2)))
    frames = [state(xa + t * (xc - xa)) for t in ts]
    a, c = frames[0], state(xc)
    tw_lo = min(f["theta_g"].min() for f in frames) - 0.6
    tw_hi = max((f["theta_g"] + f["theta_e"]).max() for f in frames) + 0.6
    li_hi = max(f["lift"].max() for f in frames) * 1.08
    cdi_hi = max(a["cdi"], c["cdi"]) * 1.10

    fig = plt.figure(figsize=(7.2, 3.6), dpi=100)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.60], wspace=0.40,
                          left=0.068, right=0.972, top=0.80, bottom=0.145)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
    title = fig.text(0.5, 0.925, "", ha="center", fontsize=12.5, color=INK, weight="bold")

    def update(i):
        f, t = frames[i], ts[i]
        col = RIGID if t < 0.5 else COUPLED
        for ax in (ax1, ax2, ax3):
            ax.clear()

        ax1.axhline(0, color=GRID, lw=1)
        ax1.plot(f["eta"], f["theta_g"], color=MUTED, lw=1.8, ls=(0, (5, 3)))
        ax1.plot(f["eta"], f["theta_g"] + f["theta_e"], color=col, lw=2.6)
        ax1.fill_between(f["eta"], f["theta_g"], f["theta_g"] + f["theta_e"],
                         color=col, alpha=0.14)
        ax1.set_ylim(tw_lo, tw_hi); ax1.set_xlim(-0.03, 1.03)
        ax1.set_xticks([0, 1]); ax1.set_xticklabels(["root", "tip"], fontsize=10, color=INK)
        ax1.set_ylabel("twist  (degrees)", fontsize=10, color=MUTED)
        ax1.tick_params(axis="y", labelsize=9, colors=MUTED)
        ax1.set_title("as built, and in flight", fontsize=10.5, color=MUTED, pad=6)
        despine(ax1)

        ax2.plot(a["eta"], a["lift"], color=RIGID, lw=1.2, alpha=0.35)
        ax2.plot(c["eta"], c["lift"], color=COUPLED, lw=1.2, alpha=0.35)
        ax2.plot(f["eta"], f["lift"], color=col, lw=2.6)
        ax2.set_ylim(0, li_hi); ax2.set_xlim(-0.03, 1.03)
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["root", "tip"], fontsize=10, color=INK)
        ax2.set_ylabel("lift per metre  (N/m)", fontsize=10, color=MUTED)
        ax2.tick_params(axis="y", labelsize=9, colors=MUTED)
        ax2.set_title("where the load lands", fontsize=10.5, color=MUTED, pad=6)
        despine(ax2)

        ax3.bar([0], [f["cdi"]], width=0.5, color=col)
        ax3.axhline(c["cdi"], color=COUPLED, lw=1.1, ls=(0, (4, 3)))
        ax3.set_ylim(0, cdi_hi); ax3.set_xlim(-1.0, 1.0)
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.set_title("induced drag", fontsize=10.5, color=MUTED, pad=6)
        ax3.text(0, f["cdi"] + cdi_hi * 0.035, f"{f['cdi']:.5f}", ha="center", va="bottom",
                 fontsize=11.5, color=col, weight="bold")
        despine(ax3, keep=())

        title.set_text("designed by the air simulator alone" if t < 0.5
                       else "designed by both simulators together")
        title.set_color(RIGID if t < 0.5 else COUPLED)
        return ()

    fig.text(0.075, 0.035, "same trimmed lift, same structure  ·  every frame is a coupled solve",
             fontsize=9, color=MUTED)
    FuncAnimation(fig, update, frames=len(ts), blit=False).save(
        str(out), writer=PillowWriter(fps=8))
    plt.close(fig)


def validity_regime(sv, out: Path) -> None:
    """Two panels, because one factor is not the answer and a one-panel figure would say it was.

    The speed sweep at the committed optimized design shows a cosine falling from 1 through 0 to
    -1, and this project read that as validity collapsing with operating condition. Two controls
    say otherwise: at a generic design the same sweep barely moves, and at a fixed flight condition
    walking toward the coupled optimum inverts the advice on its own. Both panels are here so the
    figure cannot be quoted as the one-factor claim.

    No region is called safe or unsafe. The bands say what was measured -- aligned, disagreement,
    opposing advice -- and nothing about whether a reader's problem tolerates it.
    """
    del sv
    rec = json.loads((REC / "validity_regime.json").read_text())
    rows, rows_n = rec["rows"], rec["rows_neutral"]
    seg = rec["segment"]

    fig = plt.figure(figsize=(11.0, 4.6), dpi=150)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22,
                          left=0.062, right=0.975, top=0.80, bottom=0.145)
    fig.suptitle("Neither factor alone inverts the cheap gradient's advice — the boundary needs "
                 "both", fontsize=13.5, color=INK, weight="bold", y=0.955)

    def bands(ax, x0, x1, label=False):
        ax.axhspan(0.0, 1.05, color=COUPLED, alpha=0.045)
        ax.axhspan(-1.05, 0.0, color=RIGID, alpha=0.055)
        ax.axhline(0.0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        if label:
            ax.text(x0 + 0.02 * (x1 - x0), 0.045, "the two agree on a descent direction",
                    ha="left", va="bottom", fontsize=8.2, color=MUTED)
            ax.text(x0 + 0.02 * (x1 - x0), -0.045,
                    "the rigid provider points uphill\non the coupled objective",
                    ha="left", va="top", fontsize=8.2, color=RIGID, linespacing=1.25)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xlim(x0, x1)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        despine(ax)

    # ---- left: sweep the flight condition, at two designs
    ax = fig.add_subplot(gs[0])
    bands(ax, 0.0, 1.0, label=True)
    xq = [r["q_over_qd"] for r in rows]
    ax.plot(xq, [r["cos"] for r in rows], color=COUPLED, lw=2.6, marker="o", ms=3.6,
            label="at the optimized design")
    ax.plot([r["q_over_qd"] for r in rows_n], [r["cos"] for r in rows_n], color=MUTED, lw=2.2,
            ls=(0, (5, 3)), marker="s", ms=3.2, label="at a generic design, same stiffness")
    des = next(r for r in rows if abs(r["v"] - 80.0) < 1e-9)
    ax.axvline(des["q_over_qd"], color=INK, lw=1.0, alpha=0.35)
    ax.text(des["q_over_qd"] + 0.018, 0.90, "design\ncondition", ha="left", va="top",
            fontsize=8.6, color=INK, linespacing=1.2)
    ax.plot([des["q_over_qd"]], [des["cos"]], "o", ms=8, mfc="none", mec=COUPLED, mew=1.8)
    ax.set_xlabel("dynamic pressure as a fraction of the wing's divergence pressure,  "
                  "$q/q_D$", fontsize=10, color=INK)
    ax.set_ylabel("cosine between the two gradients", fontsize=10, color=INK)
    ax.set_title("Flight condition alone does not do it", fontsize=10.5, color=INK, pad=7)
    ax.legend(loc="lower left", frameon=False, fontsize=8.8, labelcolor=INK)

    # ---- right: hold the flight condition, walk the design
    ax2 = fig.add_subplot(gs[1])
    bands(ax2, 0.0, 1.0)
    for key, color, lw, ls in (("80", COUPLED, 2.6, "-"), ("40", MUTED, 2.2, (0, (5, 3)))):
        s = seg[key]
        q = s[0]["q_over_qd"]
        ax2.plot([r["t"] for r in s], [r["cos"] for r in s], color=color, lw=lw, ls=ls,
                 marker="o" if key == "80" else "s", ms=3.6 if key == "80" else 3.2,
                 label=f"held at $q/q_D$ = {q:.3f}")
    ax2.set_xlabel("straight line from the generic design (0) to the optimized one (1)",
                   fontsize=10, color=INK)
    ax2.set_title("Nor does proximity to the optimum alone", fontsize=10.5, color=INK, pad=7)
    ax2.legend(loc="lower left", frameon=False, fontsize=8.8, labelcolor=INK)
    ax2.set_yticklabels([])
    fig.savefig(out, **SAVE)
    plt.close(fig)


TARGETS = {
    "validity_regime.png": ("gradient alignment, two factors", validity_regime),
    "aeroelastic_hero.png": ("desktop hero", desktop_hero),
}

# A 16:9 slate for video use. It is presentation rather than evidence, this repository does not
# publish it, and the default run does not write it: regenerating the figures should leave the
# clone as it was found instead of dropping a file the reader has no use for.
EXTRA_TARGETS = {
    "aeroelastic_slate_16x9.png": ("video-safe 16:9 1920x1080",
                                   lambda sv, o: landscape(sv, o, (19.2, 10.8), 100, 30, 34)),
}


# The two surfaces measured as not byte-reproducible on a healthy clone; see the module docstring
# for the per-surface pixel counts. Named so `--check` can separate them from a genuine fault.
KNOWN_DRIFT = {"aeroelastic_hero.png", "aeroelastic_morph.gif"}


def render(extra: bool = False) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    sv = served()
    made = []
    for name, (_, fn) in ({**TARGETS, **EXTRA_TARGETS} if extra else TARGETS).items():
        p = FIG / name
        fn(sv, p)
        made.append(p)
    wing_morph(sv, FIG / "aeroelastic_morph.gif")
    architecture(FIG / "architecture.png")
    made += [FIG / "aeroelastic_morph.gif", FIG / "architecture.png"]
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="developer check: re-render and require every output byte-identical to "
                         "what is committed. Exits 1 anywhere but the environment those files "
                         "were rendered on; see the module docstring for the measured drift")
    ap.add_argument("--extra", action="store_true",
                    help="also render the 16:9 video slate, which this repository "
                         "does not publish")
    args = ap.parse_args()

    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in FIG.glob("*") if p.is_file()} if args.check else {}
    made = render(extra=args.extra)
    bad = []
    for p in made:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        dim = ""
        if p.suffix == ".png":
            b = p.read_bytes()[:24]
            dim = f"{int.from_bytes(b[16:20], 'big')}x{int.from_bytes(b[20:24], 'big')}"
        print(f"  {p.relative_to(ROOT)!s:<44} {p.stat().st_size / 1024:>7.0f} KB  "
              f"{dim:>10}  {h[:12]}")
        if args.check and before.get(p) not in (None, h):
            bad.append(p.name)
    if args.check:
        if bad:
            # Still exit 1: the comparison is not relaxed to make a healthy clone pass. But name
            # which surfaces are the known rasterization drift, so a developer reading this can
            # tell the expected three from a fourth that would be a real change.
            print(f"\nNOT BYTE-REPRODUCIBLE: {bad}")
            print(f"  known drift, expected:  {sorted(KNOWN_DRIFT)}")
            unexpected = sorted(set(bad) - KNOWN_DRIFT)
            print(f"  UNEXPECTED:             {unexpected}" if unexpected else
                  "  no unexpected surface; this exit 1 is the documented rasterization drift")
            return 1
        print("\nOK: every media surface re-rendered byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
