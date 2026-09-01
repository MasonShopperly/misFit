#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""The platform's own gradient checker, run on the two components this submission ships.

`tesseract-runtime check-gradients` compares each derivative endpoint of ONE `tesseract_api.py`
against central finite differences of THAT SAME module's `apply`. That is the correct design for
the question it asks, and the question is about one component at a time.

WHAT A PASS HERE MEANS, AND WHAT IT DOES NOT. It means each component's declared derivative is a
derivative of its own `apply`. That is component-local correctness, and it is exactly what this
project claims for both. It says nothing about whether either derivative is a derivative of the
ASSEMBLED coupled objective -- the checker cannot ask that here, because the objective is not
inside either module. Both components passing while the crossed pairing fails is not a
contradiction; it is the result. The assembled question is asked in
`run_native_composed_check.py`, against the composed Tesseract `coupled.C1`.

DETERMINISM. `--seed` is fixed, so the sampled indices are reproducible. Counts are set by the
declared shapes, not by the draw. No timestamp or duration enters the record.

COVERAGE. Exactly two components and exactly three endpoints each must be reported, or this writes
NO record and exits 1 rather than tabulating whatever it happened to find.

    .venv/bin/python aeroelastic/verification/run_platform_checker_ae.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AE = HERE.parent
ROOT = AE.parent
COMPONENTS = AE / "components"
RUNTIME = Path(sys.executable).parent / "tesseract-runtime"
SEED = 20260806
ENDPOINTS = ("jacobian", "jacobian_vector_product", "vector_jacobian_product")

# Fixed, declared payloads. Shapes come from each component's InputSchema; the values are an
# ordinary operating point, not a tuned one.
CASES = [
    ("aero.A1", "aero", {
        "alpha_tot": [0.05 + 0.01 * ((i * 37 % 19) - 9) / 9.0 for i in range(40)],
        "v": 80.0}),
    ("struct.S1", "structure", {
        "torque": [50.0 * ((i * 53 % 23) - 11) / 11.0 for i in range(40)],
        "s_ctrl": [-0.2, 0.0, 0.2]}),
]

RESULT = re.compile(r"Gradient check for (\w+) (passed|failed)")
COUNTS = re.compile(r"\((\d+) failures / (\d+) checks\)")
ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def run(component: str, payload: dict) -> dict:
    env = dict(os.environ,
               TESSERACT_API_PATH=str(COMPONENTS / component / "tesseract_api.py"),
               TESSERACT_RUNTIME_CHECK_GRADIENTS_SHOW_PROGRESS="0")
    r = subprocess.run([str(RUNTIME), "check-gradients", json.dumps({"inputs": payload}),
                        "--seed", str(SEED)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    text = ANSI.sub("", r.stdout + r.stderr)
    verdicts = dict(RESULT.findall(text))
    counts = COUNTS.findall(text)
    return {"exit": r.returncode, "verdicts": verdicts,
            "failures": [int(a) for a, _ in counts], "checks": [int(b) for _, b in counts]}


def clean(component: str) -> None:
    """Importing a component's api module leaves a __pycache__ inside its build context, which a
    copytree would carry into an image. A driver that leaves the tree dirty is not finished."""
    import shutil
    cache = COMPONENTS / component / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache)


def main() -> int:
    out, problems = {}, []
    for label, component, payload in CASES:
        try:
            res = run(component, payload)
        finally:
            clean(component)
        out[label] = {"component": f"aeroelastic/components/{component}",
                      "seed": SEED, "payload": payload, **res}
        if set(res["verdicts"]) != set(ENDPOINTS):
            problems.append(f"{label}: reported {sorted(res['verdicts'])}, expected {list(ENDPOINTS)}")
        if len(res["checks"]) != len(ENDPOINTS):
            problems.append(f"{label}: {len(res['checks'])} count lines, expected {len(ENDPOINTS)}")

    if len(out) != 2:
        problems.append(f"{len(out)} components reported, expected 2")

    for label, rec in out.items():
        tot_f, tot_c = sum(rec["failures"]), sum(rec["checks"])
        print(f"  {label:<10s} exit {rec['exit']}   "
              + "  ".join(f"{e}={rec['verdicts'].get(e, '?')}" for e in ENDPOINTS)
              + f"   {tot_f} failures / {tot_c} checks")

    if problems:
        print("\nCOVERAGE SHORTFALL -- no record written:")
        for p in problems:
            print(f"  {p}")
        return 1

    total_f = sum(sum(r["failures"]) for r in out.values())
    total_c = sum(sum(r["checks"]) for r in out.values())
    out["_summary"] = {"components": 2, "endpoints_each": len(ENDPOINTS),
                       "total_failures": total_f, "total_checks": total_c,
                       "all_passed": all(v == "passed" for r in out.values()
                                         for v in r["verdicts"].values())}
    (AE / "records" / "platform_checker_ae.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n  the platform's own checker: {total_f} failures / {total_c} checks across "
          f"2 components x {len(ENDPOINTS)} endpoints")
    print("  component-local correctness only. It cannot ask whether either derivative is a")
    print("  derivative of the ASSEMBLED coupled objective -- that objective is in neither module.")
    print(f"\nwrote {AE / 'records' / 'platform_checker_ae.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
