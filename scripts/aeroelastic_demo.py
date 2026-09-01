#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""The application demo: qualify the gradient providers, then show the wing that was optimized.

    .venv/bin/python scripts/aeroelastic_demo.py            # replay committed evidence, ~1 s
    .venv/bin/python scripts/aeroelastic_demo.py --live     # serve, qualify and optimize, ~18 min

The default REPLAYS the committed run and says so on every line it prints. `--live` starts both
Tesseracts on 8811-8812 and recomputes everything -- BOTH the qualification, which is now the
platform's own check-gradients against the composed Tesseract, and the optimization it authorises.
Nothing here presents a replay as a live computation.

`--live` writes its record to a scratch file and COMPARES it, field by field, against the committed
one, leaving the working tree untouched either way. Fields are reported as EXACT MATCH,
NUMERICALLY CONSISTENT or FAILED, and only FAILED sets a non-zero exit status. See TOLERANCE below
for why the middle category has to exist.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AE = ROOT / "aeroelastic"
REC = AE / "records"
B, R, N = "\033[1m", "\033[0m", "\033[2m"


def rule(title: str) -> None:
    print(f"\n{B}{title}{R}\n" + "─" * 76)


def replay(served_path: Path | None = None, native_path: Path | None = None) -> int:
    """Print the demo from a served-optimize record.

    `served_path` selects WHICH record sections 3 and 4 come from; `native_path` selects the
    qualification record section 2 reads. Everything else -- the component-local platform verdicts
    and the regime study -- comes from committed studies produced by other drivers, which a live
    re-run does not reproduce. Tagging those as live would splice two runs into one table, so each
    section is tagged with where its own numbers came from, individually.
    """
    live = served_path is not None
    served = json.loads((served_path or REC / "served_optimize.json").read_text())
    fig = json.loads((REC / "wing_figure.json").read_text())
    regime = json.loads((REC / "validity_regime.json").read_text())
    tag = f"{N}[replayed from committed evidence]{R}"
    now = f"{N}[recomputed live, just now]{R}" if live else tag

    rule("1 · The coupled problem")
    print("  A flexible wing is loaded by the air, deforms, and the deformation changes the load.")
    print("  Two Tesseracts hold the two halves and neither can produce the flying shape alone:")
    print("    aero.A1     lifting-line aerodynamics, differentiated by JAX autodiff")
    print("    struct.S1   cantilever torsion, differentiated by a hand-derived adjoint")

    rule("2 · Qualify the pairing — with the platform's own checker  "
         + (now if native_path else tag))
    plat = json.loads((REC / "platform_checker_ae.json").read_text())
    ps = plat["_summary"]
    print(f"  Component-local first. `tesseract-runtime check-gradients` on each service against")
    print(f"  finite differences of its own apply: {ps['total_failures']} failures / "
          f"{ps['total_checks']:,} checks across both components.")
    print(f"  {B}Both providers are correct. That is the platform's verdict, not ours.{R}")

    native = json.loads((native_path or REC / "native_composed_check.json").read_text())
    print(f"\n  The assembled objective lives in neither component, so it is exposed as a third")
    print(f"  Tesseract — coupled.C1, which holds no physics and routes every evaluation to the")
    print(f"  other two over HTTP. The SAME platform checker is then pointed at that:")
    label = {"matched-coupled": "coupled objective + coupled gradient",
             "matched-rigid": "rigid objective   + rigid gradient  ",
             "crossed-rigid": "coupled objective + rigid gradient  "}
    for pair in ("matched-coupled", "matched-rigid", "crossed-rigid"):
        rows = [r for k, r in native.items() if k.endswith("/" + pair)]
        if not rows:
            continue
        ok = all(v == "passed" for r in rows for v in r["verdicts"].values())
        nf, nc = sum(sum(r["failures"]) for r in rows), sum(sum(r["checks"]) for r in rows)
        print(f"  {'✓' if ok else '✗'} {label[pair]:<48s} {'passed' if ok else 'FAILED':<6s} "
              f"{nf:>4d} / {nc} checks")
    print(f"\n  {B}Only the pairing fails.{R} The aerodynamic-only gradient is not broken — it is")
    print("  the exact derivative of a model this aircraft does not fly, and no component-local")
    print("  check can see that, because the objective it is wrong about is in neither component.")
    per_run = sum(sum(r["checks"]) for k, r in native.items()
                  if k.endswith("/crossed-rigid")) // 3
    print(f"  Cost: {per_run} checks per pairing at each of three design points, covering all six")
    print("  design components on all three derivative endpoints. Zero lines of our numerical code:")
    print("  the tool is the platform's, and coupled.C1 is what makes it applicable here.")

    rule("3 · Optimize with the gradient that qualified  " + now)
    o = served["optimize"]
    print(f"  objective      {o['J0']:.4f}  →  {o['J']:.4f}   "
          f"({100 * (1 - o['J'] / o['J0']):.1f} % better in {o['nit']} iterations)")
    print(f"  trimmed lift   C_L = {o['cl']:.4f}          (held fixed — not bought by "
          f"carrying less)")
    print(f"  induced drag   C_Di = {o['cdi']:.6f}")
    a = served["aero_only"]
    print(f"  washout        {abs(o['x'][0] - o['x'][2]) * 57.29578:.1f}° root to tip"
          f"   (aerodynamic-only: {abs(a['x'][0] - a['x'][2]) * 57.29578:.1f}°)")
    print(f"  divergence     q/q_D = {o['q_over_qd']:.3f}")
    print(f"\n  {B}The same pipeline driven by the aerodynamic component's own gradient:{R}")
    print(f"  its own model scored that wing {a['J_own']:.4f} — effectively a tie.")
    print(f"  the two simulators together score it {a['J_coupled']:.4f}, "
          f"{fig['aero_only_worse_pct']:.1f} % worse,")
    print(f"  carrying {fig['aero_only_drag_excess_pct']:.1f} % more induced drag at the same "
          f"lift and the same structure.")

    rule("4 · Both services, both directions  " + now)
    h = o["http"]
    for k in ("aero.A1.apply", "aero.A1.vector_jacobian_product",
              "struct.S1.apply", "struct.S1.vector_jacobian_product"):
        print(f"  {k:<42s} {h[k]:>7,d} calls")
    print("\n  The forward state is a fixed point between the two services; the gradient is that")
    print("  loop transposed. Neither has a route that avoids the other.")

    rule("5 · What sets the boundary, and what none of this claims")
    nrows, seg = regime["rows_neutral"], regime["segment"]["80"]
    print(f"  Four predeclared physics heuristics were scored against the 289-design map and all")
    print(f"  four were beaten by a constant \"never fires\". The pairing is therefore qualified")
    print(f"  once, on the assembled objective, at the condition being flown.")
    print(f"\n  And the trigger is not the flight condition alone. At a design no optimizer chose,")
    print(f"  the two gradients still agree to cos = {min(r['cos'] for r in nrows):.5f} at "
          f"99 % of the divergence pressure.")
    print(f"  Hold the condition fixed and walk toward the coupled optimum instead and the cosine")
    print(f"  reaches {min(r['cos'] for r in seg):.5f}: the approximate provider's error is a fixed "
          f"vector, so it")
    print(f"  dominates where the coupled gradient has become small — the endgame, not the start.")
    print("\n  Reduced-order model — lifting line plus torsion beam, static aeroelasticity only.")
    print("  One wing, one objective, six design variables. Not a benchmark and not a law.")

    print(f"\n  figure  docs/figures/validity_regime.png")
    print(f"  report  results/native_composed_check.md · results/validity_regime.md")
    if not live:
        print(f"\n  {N}re-run live with:  .venv/bin/python scripts/aeroelastic_demo.py --live{R}")
    return 0


# `wall_s` is how long this machine took. It is the one field exempted from comparison outright:
# it is a stopwatch reading, and it MUST differ between two honest runs.
TIMING_FIELDS = {"wall_s"}

# TOLERANCE. Two honest runs of this code on this data need not produce the same bits. Reduction
# order in float64 BLAS is not fixed across runs, so a re-run drifts in the last few significant
# digits. Comparing by equality turns that drift into a nominal failure, which is why these
# thresholds exist and why a numerically consistent run exits 0.
#
# rtol 1e-6 sits between the two things it has to separate. Measured drift on this record set lands
# in the 9th to 13th significant digit, three or more orders of magnitude inside it. Every number
# reported from these records is quoted to at most 5 significant digits, so a difference large
# enough to matter to a reader is at least two orders of magnitude outside it.
RTOL = 1e-6
# atol governs the quantities that are zero for reporting purposes: the coupled gradient at the
# optimized design, which is stationary, and the residual metrics that measure agreement with the
# monolithic solve. A relative test on those compares noise against noise.
#
# It is set from the noise floor of the instrument that produces them, not from their printed
# magnitude. They are central differences at eps = 1e-4 on an objective of order 1, so their
# absolute noise is about eps_machine / h = 2e-12; a re-run of this record moves them by 4e-12 to
# 1.6e-11. 1e-9 sits ~2 orders above that floor and ~3 orders below 1.7e-06, the smallest quantity
# this project reports, so it cannot absorb a change in any number that appears in the results.
#
# The first value tried here was 1e-12, reasoned from the printed magnitudes rather than from how
# they are computed. It sat below the finite-difference noise floor and failed three such fields on
# a re-run that had reproduced everything else.
ATOL = 1e-9


def classify(was, now) -> str:
    """EXACT, CONSISTENT or FAILED for one field.

    Integers, booleans and strings are compared by equality with no tolerance at all: they are
    call counts, failure counts, exit codes and verdicts. A count that moves is a different run,
    not a rounding difference, and rounding a verdict toward agreement is the one thing this
    comparison must never do.
    """
    if was == now and type(was) is type(now):
        return "EXACT"
    if isinstance(was, bool) or isinstance(now, bool):
        return "FAILED"
    if isinstance(was, int) and isinstance(now, int):
        return "FAILED"
    if isinstance(was, (int, float)) and isinstance(now, (int, float)):
        return "CONSISTENT" if math.isclose(now, was, rel_tol=RTOL, abs_tol=ATOL) else "FAILED"
    return "FAILED"


def deviation(was, now) -> str:
    if not isinstance(was, (int, float)) or isinstance(was, bool):
        return ""
    d = abs(now - was)
    return f"{d / abs(was):.1e} relative" if was else f"{d:.1e} absolute"


CASES = [
    # identical values, of each kind that appears in these records
    (1.0, 1.0, "EXACT", "identical float"),
    ("PASS", "PASS", "EXACT", "identical verdict"),
    (True, True, "EXACT", "identical flag"),
    # float drift in the last digits is what the tolerance exists to accept
    (0.9336175307578589, 0.9336175307578123, "CONSISTENT", "objective, last-digit drift"),
    (1.0, 1.0 + 5e-7, "CONSISTENT", "just inside rtol"),
    (1.0, 1.0 + 5e-5, "FAILED", "just outside rtol"),
    (0.9336, 0.9341, "FAILED", "objective moved in the 4th digit"),
    # near-zero fields, at the magnitudes measured on an actual re-run of these records: the
    # stationary coupled gradient and the facade-vs-monolithic residuals. atol has to admit these.
    (-1.6893064724854412e-06, -1.6892975907012442e-06, "CONSISTENT", "stationary gradient, 8.9e-12"),
    (1.1308498581996673e-06, 1.1308343150773226e-06, "CONSISTENT", "stationary gradient, 1.6e-11"),
    (1.3827873900333843e-09, 1.3786652683219282e-09, "CONSISTENT", "monolithic residual, 4.1e-12"),
    (0.0, 1e-13, "CONSISTENT", "noise against an exact zero"),
    # ... without absorbing a change big enough to alter a reported number, or a sign flip
    (0.0, 1e-3, "FAILED", "a real value appearing where zero was"),
    (0.0, 1e-7, "FAILED", "two orders above atol"),
    (-1.689e-06, 1.689e-06, "FAILED", "sign flip in the stationary gradient"),
    (-1.7e-06, -1.8e-06, "FAILED", "near-zero value that actually moved"),
    # discrete fields get no tolerance at all
    (462, 461, "FAILED", "failure count"),
    (879, 878, "FAILED", "call count"),
    ("PASS", "FAIL", "FAILED", "verdict"),
    (True, False, "FAILED", "flag"),
    (1, True, "FAILED", "a bool is not the integer it equals"),
    (0, False, "FAILED", "a bool is not the integer it equals"),
]


def _selftest_classify() -> int:
    """The comparison decides whether a reproduction passed, so it is tested rather than trusted."""
    bad = [(was, now, want, got, why) for was, now, want, why in CASES
           if (got := classify(was, now)) != want]
    for was, now, want, got, why in bad:
        print(f"  WRONG  {why}: classify({was!r}, {now!r}) = {got}, expected {want}")
    print(f"comparison self-test: {len(CASES) - len(bad)} of {len(CASES)} cases pass "
          f"(rtol {RTOL:g}, atol {ATOL:g})")
    return 1 if bad else 0


def flatten(node, prefix=""):
    """Every leaf of the record, as dotted paths, so the comparison cannot skip a nested field."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, node


def compare(live_path: Path, committed_path: Path) -> int:
    """Field by field against the committed record. Returns the number of real disagreements.

    Every field that is not bit-identical is printed with its deviation, whichever side of the
    tolerance it falls on. The tolerance changes the exit status, never the visibility.
    """
    a = dict(flatten(json.loads(live_path.read_text())))
    b = dict(flatten(json.loads(committed_path.read_text())))
    only = sorted(set(a) ^ set(b))
    exact, timing, consistent, failed = 0, [], [], []
    for k in sorted(set(a) & set(b)):
        if k.split(".")[-1] in TIMING_FIELDS:
            timing.append((k, b[k], a[k]))
            continue
        verdict = classify(b[k], a[k])
        if verdict == "EXACT":
            exact += 1
        elif verdict == "CONSISTENT":
            consistent.append((k, b[k], a[k]))
        else:
            failed.append((k, b[k], a[k]))
    comparable = exact + len(consistent) + len(failed)

    rule("Live against committed")
    print(f"  EXACT MATCH             {exact} of {comparable} fields, bit for bit across the network")
    print(f"  NUMERICALLY CONSISTENT  {len(consistent)}"
          f"   (within rtol {RTOL:g}, atol {ATOL:g})")
    print(f"  FAILED                  {len(failed) + len(only)}")
    for k, was, now in timing:
        print(f"  {N}{k}: {was} -> {now}   (wall-clock; exempt, this is a stopwatch reading){R}")
    for k, was, now in consistent:
        print(f"  {N}CONSISTENT  {k}: committed {was!r}, live {now!r}   "
              f"({deviation(was, now)}){R}")
    for k, was, now in failed:
        print(f"  {B}FAILED      {k}: committed {was!r}, live {now!r}   "
              f"({deviation(was, now)}){R}")
    for k in only:
        print(f"  {B}FAILED      {k}: present in only one of the two records{R}")

    if failed or only:
        print(f"\n  {B}{len(failed) + len(only)} field(s) failed. The committed evidence and this "
              f"machine do not agree.{R}")
    elif consistent:
        print(f"\n  {B}Reproduced. {exact} fields bit-identical and {len(consistent)} within "
              f"tolerance; no field disagrees materially.{R}")
    else:
        print(f"\n  {B}Reproduced exactly. Every comparable field is bit-identical.{R}")
    print(f"  {N}the committed record was not modified; the live one is at {live_path}{R}")
    return len(failed) + len(only)


def live() -> int:
    """Both live drivers, serialised: they bind the same ports 8811-8812.

    The qualification the demo now shows is the platform's own checker against the composed
    Tesseract, so a live run has to execute THAT and not only the optimization it authorises.
    Both records go to scratch and are field-compared against the committed ones; the working
    tree is untouched either way.
    """
    print(f"{B}LIVE{R} — starting both Tesseracts and recomputing everything (~18 min on 16 cores; the composed check is most of it)\n",
          flush=True)
    with tempfile.TemporaryDirectory() as td:
        nat = Path(td) / "native_composed_check.live.json"
        import os
        rc = subprocess.call([sys.executable, str(AE / "verification" / "run_native_composed_check.py")],
                             env=dict(os.environ, MISFIT_NATIVE_OUT=str(nat)))
        if rc != 0 or not nat.is_file():
            print(f"\n{B}LIVE QUALIFICATION FAILED{R} (exit {rc}). Committed evidence untouched.")
            return rc or 1
        bad = compare(nat, REC / "native_composed_check.json")

        out = Path(td) / "served_optimize.live.json"
        rc = subprocess.call([sys.executable, str(AE / "studies" / "run_served_optimize.py"),
                              "--out", str(out)])
        if rc != 0 or not out.is_file():
            print(f"\n{B}LIVE RUN FAILED{R} (exit {rc}). The committed evidence is untouched.")
            return rc or 1
        bad += compare(out, REC / "served_optimize.json")
        print()
        replay(out, nat)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true",
                    help="serve the components and recompute instead of replaying")
    ap.add_argument("--self-test", action="store_true",
                    help="check the live-comparison semantics against planted cases and exit")
    args = ap.parse_args()
    if args.self_test:
        return _selftest_classify()
    return live() if args.live else replay()


if __name__ == "__main__":
    sys.exit(main())
