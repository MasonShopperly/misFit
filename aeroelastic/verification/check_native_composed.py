#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Gate check: the composed-Tesseract record is internally consistent and untampered.

WHY THIS EXISTS. The submission's active qualification route is now the platform's own
`tesseract-runtime check-gradients`, run against `coupled.C1` -- the facade that exposes the
assembled aeroelastic objective as one Tesseract. Every published claim about that route quotes
`aeroelastic/records/native_composed_check.json`, written by a driver that binds ports
and is deliberately never auto-run. A record nobody re-derives is a record
that can rot, so everything checkable without re-executing is checked here:

    * coverage -- three points x three pairings x three endpoints, no gaps;
    * the invocation parameters match the values frozen below;
    * exit codes, verdicts and failure counts are mutually consistent;
    * both matched pairings passed everywhere and the crossed pairing failed everywhere, which is
      the claim; if that ever inverts, this says so before a document does;
    * every invocation recorded NON-ZERO HTTP calls to BOTH components -- a facade that answered
      without asking A1 and S1 would have tested nothing;
    * the design point is float-equal to the committed optimized design, not approximately;
    * the recorded tesseract-core version equals the requirements.txt pin;
    * each recorded rejection is RE-DERIVED under the platform's own criterion
      |a - b| <= atol + rtol|b| from the stored floats, rather than trusted.

SELF-TEST. Thirteen planted defects, every one of which must produce at least one problem, or this check
fails itself.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AE = HERE.parent
ROOT = AE.parent
REC = AE / "records" / "native_composed_check.json"
OPT = AE / "records" / "served_optimize.json"
ENDPOINTS = ("jacobian", "jacobian_vector_product", "vector_jacobian_product")
PAIRINGS = ("matched-coupled", "matched-rigid", "crossed-rigid")
POINTS = ("before-a", "inside-a", "design")
# Frozen before the composed component existed.
FROZEN = {"seed": 20260806, "eps": 1e-4, "rtol": 0.1, "max_evals": 60}

# The wall times the Cost table of results/native_composed_check.md prints, to the precision it
# prints them at. Held here so the report cannot drift from the record it cites.
REPORTED_WALL_S = {"design/matched-coupled": 74.9,
                   "design/crossed-rigid": 25.9,
                   "design/matched-rigid": 1.8}
ATOL = 1e-8                      # the platform's np.allclose atol, not configurable from the CLI
N_DESIGN = 6


def pinned_runtime_version() -> str | None:
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"tesseract-core\[runtime\]==(\S+)", line.strip())
        if m:
            return m.group(1)
    return None


def validate(rec: dict, pin: str, opt_x: list[float]) -> list[str]:
    p: list[str] = []
    s = rec.get("_summary")
    if not isinstance(s, dict):
        return ["record has no _summary; it is structurally incomplete"]

    for k, want in FROZEN.items():
        if s.get(k) != want:
            p.append(f"parameter {k}={s.get(k)!r} != the frozen plan's {want!r}")
    if s.get("tesseract_core") != pin:
        p.append(f"recorded tesseract-core {s.get('tesseract_core')!r} != requirements pin {pin!r}")
    if s.get("coverage_indices_per_endpoint") != [list(range(N_DESIGN))] * len(ENDPOINTS):
        p.append("the platform's sampling did not cover all six design indices on every endpoint")

    keys = {f"{pt}/{pr}" for pt in POINTS for pr in PAIRINGS}
    missing = sorted(keys - set(rec))
    if missing:
        p.append(f"missing invocations: {missing}")

    for key in sorted(keys & set(rec)):
        r = rec[key]
        v = r.get("verdicts", {})
        if set(v) != set(ENDPOINTS):
            p.append(f"{key}: endpoints {sorted(v)} != all three")
            continue
        nf, nc = sum(r.get("failures", [])), sum(r.get("checks", []))
        if nc != FROZEN["max_evals"] * len(ENDPOINTS):
            p.append(f"{key}: {nc} checks != {FROZEN['max_evals']} x {len(ENDPOINTS)}")
        all_pass = all(x == "passed" for x in v.values())
        if all_pass is not (nf == 0):
            p.append(f"{key}: verdicts {sorted(set(v.values()))} inconsistent with {nf} failures")
        if r.get("exit") != (0 if all_pass else 1):
            p.append(f"{key}: exit {r.get('exit')!r} inconsistent with verdicts")
        a1 = sum(n for k, n in r.get("http_calls", {}).items() if k.startswith("aero"))
        s1 = sum(n for k, n in r.get("http_calls", {}).items() if k.startswith("struct"))
        if a1 == 0 or s1 == 0:
            p.append(f"{key}: {a1} aero and {s1} struct calls -- the facade answered without "
                     f"asking a component, so nothing about composition was tested")
        # The report quotes a wall time per invocation. Nothing used to read this field, so a
        # retyped wall column could contradict the record and still pass -- which is exactly what
        # happened. Report and record now have to agree to the printed precision.
        w = r.get("wall_s")
        if not isinstance(w, (int, float)) or w <= 0:
            p.append(f"{key}: wall_s {w!r} is not a positive number")
        elif key in REPORTED_WALL_S and round(float(w), 1) != REPORTED_WALL_S[key]:
            p.append(f"{key}: results/native_composed_check.md quotes "
                     f"{REPORTED_WALL_S[key]} s but the record holds {w} s")
        want_pass = not key.endswith("crossed-rigid")
        if all_pass is not want_pass:
            p.append(f"{key}: {'passed' if all_pass else 'failed'}, but the committed claim is "
                     f"that matched pairings pass and the crossed pairing fails")
        # Re-derive each recorded rejection under the platform's own criterion.
        fi = r.get("failing_indices", {})
        if want_pass and fi:
            p.append(f"{key}: a passing pairing recorded {len(fi)} rejected indices")
        for idx, d in fi.items():
            if not idx.isdigit() or not 0 <= int(idx) < N_DESIGN:
                p.append(f"{key}: rejected index {idx!r} is not a design index")
                continue
            a, b = d.get("declared"), d.get("finite_difference")
            if not isinstance(a, float) or not isinstance(b, float):
                p.append(f"{key}[{idx}]: non-numeric rejection record")
                continue
            if abs(a - b) <= ATOL + FROZEN["rtol"] * abs(b):
                p.append(f"{key}[{idx}]: recorded as rejected, but {a:.6e} vs {b:.6e} SATISFIES "
                         f"the platform's own |a-b| <= atol + rtol|b|")
        if not want_pass and not fi:
            p.append(f"{key}: failed but recorded no rejected index; the failure is unsupported")

    d = rec.get("design/matched-coupled", {}).get("x")
    if d != opt_x:
        p.append("the 'design' point is not float-equal to the committed optimized design")
    for key in sorted(keys & set(rec)):
        if rec[key].get("x") != rec[f"{rec[key]['point']}/matched-coupled"].get("x"):
            p.append(f"{key}: evaluated at a different design from its own matched control")

    # The composed check compares C1's derivatives against finite differences of C1's OWN apply,
    # so without this the whole experiment is circular: a wrong fixed point or a wrong transpose
    # would be certified with confidence and the 462 failures would be measuring C1's bugs.
    fac = s.get("facade_vs_monolithic")
    if not isinstance(fac, dict) or set(fac) != set(POINTS):
        p.append("no facade-vs-monolithic validation for all three points; the composed check "
                 "cannot be distinguished from a measurement of the facade's own defects")
    else:
        for pt, r in fac.items():
            worst = max(r.values()) if isinstance(r, dict) and r else 1.0
            if not isinstance(worst, float) or worst > 1e-7:
                p.append(f"{pt}: facade disagrees with the monolithic solve by {worst!r}")

    if s.get("verdict") != "NATIVE_DISCRIMINATES":
        p.append(f"summary verdict {s.get('verdict')!r} != NATIVE_DISCRIMINATES")
    if s.get("matched_all_passed") is not True or s.get("crossed_all_failed") is not True:
        p.append("summary booleans disagree with the claim the documents make")
    return p


def self_test(rec: dict, pin: str, opt_x: list[float]) -> list[str]:
    def survives(fn) -> bool:
        r = copy.deepcopy(rec)
        fn(r)
        return not validate(r, pin, opt_x)

    cases = {
        "crossed pairing silently flipped to passing":
            lambda r: r["design/crossed-rigid"]["verdicts"].update(
                {e: "passed" for e in ENDPOINTS}),
        "matched pairing failure hidden":
            lambda r: r["before-a/matched-coupled"]["verdicts"].update(jacobian="failed"),
        "HTTP calls zeroed -- facade answered alone":
            lambda r: r["inside-a/crossed-rigid"].update(http_calls={}),
        "struct.S1 never asked":
            lambda r: r["design/matched-coupled"].update(
                http_calls={"aero.A1.apply": 10, "_total": 10}),
        "an endpoint dropped from the record":
            lambda r: r["inside-a/matched-rigid"]["verdicts"].pop("jacobian"),
        "failure count edited without the verdict":
            lambda r: r["before-a/crossed-rigid"].update(failures=[0, 0, 0]),
        "a rejection that does not violate the criterion":
            lambda r: r["before-a/crossed-rigid"]["failing_indices"]["0"].update(
                declared=1.0, finite_difference=1.0),
        "design point moved off the committed optimum":
            lambda r: r["design/matched-coupled"]["x"].__setitem__(0, 0.123),
        "tesseract-core version drifted from the pin":
            lambda r: r["_summary"].update(tesseract_core="9.9.9"),
        "tolerance loosened after the fact":
            lambda r: r["_summary"].update(rtol=0.9),
        "facade validation dropped":
            lambda r: r["_summary"].pop("facade_vs_monolithic"),
        "facade silently disagrees with the monolithic solve":
            lambda r: r["_summary"]["facade_vs_monolithic"]["design"].update(j_rel=1e-3),
        "a wall time in the report contradicts the record":
            lambda r: r["design/crossed-rigid"].update(wall_s=28.6),
    }
    return [f"SELF-TEST: planted defect not caught: {n}" for n, fn in cases.items() if survives(fn)]


def main() -> int:
    pin = pinned_runtime_version()
    if pin is None:
        print("FAIL: no tesseract-core pin in requirements.txt")
        return 1
    for f in (REC, OPT):
        if not f.is_file():
            print(f"FAIL: missing {f.relative_to(ROOT)}")
            return 1
    rec = json.loads(REC.read_text())
    opt_x = [float(v) for v in json.loads(OPT.read_text())["optimize"]["x"]]

    problems = validate(rec, pin, opt_x) + self_test(rec, pin, opt_x)
    if problems:
        print(f"FAIL: {len(problems)} problem(s) with the composed-Tesseract record")
        for pr in problems:
            print(f"  {pr}")
        return 1
    calls = sum(sum(n for k, n in r["http_calls"].items() if not k.startswith("_"))
                for r in rec.values() if isinstance(r, dict) and "http_calls" in r)
    print(f"OK: composed-Tesseract record consistent -- 3 points x 3 pairings x 3 endpoints, "
          f"{calls} component HTTP calls, every rejection re-derived under the platform's own "
          f"criterion, 13 planted defects all caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
