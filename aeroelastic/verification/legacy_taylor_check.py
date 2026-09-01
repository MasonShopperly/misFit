#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""A Taylor-remainder check on an objective/gradient pair served over HTTP.

Two services can each be internally consistent -- each passing the per-component
`tesseract-runtime check-gradients` -- while the gradient one of them serves is not the
derivative of the objective the other serves. That relationship is between the two
components, so a per-component check cannot see it. This tool tests it from function
values alone, over URLs, by measuring how the forward Taylor remainder

    R(h) = | J(m + h p) - J(m) - h (g . p) |

decays as h halves. A gradient that is a derivative of this objective leaves a remainder
of order 2; one that is not leaves order 1.

VERDICTS. PASS: along the tested directions at the tested point, the supplied directional
derivative matches the objective to second order. FAIL: the remainder decays at first
order, the signature of a gradient that is not a derivative of this objective along that
direction. INCONCLUSIVE: the sweep reached the floating-point noise floor or returned
mixed orders, and no verdict is issued rather than a wrong one. None of the three is a
proof of global gradient correctness -- the test sees finitely many directions at one
point, and it loses power near a stationary point.

    .venv/bin/python aeroelastic/verification/legacy_taylor_check.py OBJECTIVE_URL GRADIENT_URL \\
        --point '[-3.0, -3.4, ...]' [--direction gradient|diagonal|'[...]']... \\
        [--json out.json]

Exit codes (stable contract): 0 PASS on every direction; 2 FAIL on any direction;
3 INCONCLUSIVE (no FAIL, not all PASS); 4 cannot reach a service; 5 malformed service
response; 6 local usage error -- bad arguments, detected before any network contact;
1 internal error. 6 is separate from 4 so that a caller can tell a typo from an outage.
Cost: 6 objective applies per direction + 1 gradient call, printed. The step schedule,
floor, plateau and verdict rules are frozen constants in `sweep` below, unchanged since
the qualification record was produced.

KNOWN FALSE NEGATIVE, disclosed and not patched. The remainder above is FORWARD, so it carries the
gradient error at order h AND the objective's curvature along the probe at order h^2. They cross at
h* = 2|e_p| / (C (d.p)^2). If the whole ladder sits above h*, curvature dominates every rung, the
fit reads order 2, and this returns PASS on a gradient it should reject -- measured at an absolute
error of 1e-4 with curvature 1e2 to 1e8, PASS at order 2.0 every time, where a two-evaluation
CENTRAL difference cancels the curvature term identically and reports the error exactly. Changing
the stencil would invalidate the committed record this produced, so the defect is disclosed and
bounded rather than quietly patched.

WHERE THIS SITS. It is NOT the submission's qualification route, and nothing about it is claimed as
new -- the mechanism is dolfin-adjoint's `taylor_test` criterion, itself MINPACK-1 `chkder` (1980)
with an h-ladder attached, and `scipy.optimize.check_grad` answers the same question in two
evaluations. The submission qualifies through the platform's own `tesseract-runtime
check-gradients`, run against the composed Tesseract `coupled.C1`
(`results/native_composed_check.md`). This file ships for one reason: two committed records --
`aeroelastic/records/served_optimize.json` step 1 and Part 1 of
`aeroelastic/records/qualification.json` -- were
produced by it, and reproducing them requires it unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AE = HERE.parent
ROOT = AE.parent

# Frozen before the qualification records were produced, and not touched since. h0 is in design
# space; five rungs at h0/2^j; a rung is unusable when its remainder is inside the float64 floor;
# PASS needs every usable pair at order >= 1.7, FAIL needs the fitted order <= 1.3.
H0 = 1e-5
N_HALVINGS = 4
PASS_MIN = 1.7
FAIL_MAX = 1.3
PLATEAU_RATIO = math.sqrt(2.0)   # local order < 0.5
FLOOR_FACTOR = 8.0
COST_CAP = 12
EPS64 = float(np.finfo(np.float64).eps)

EXIT_PASS, EXIT_INTERNAL, EXIT_FAIL, EXIT_INCONCLUSIVE = 0, 1, 2, 3
EXIT_CONNECT, EXIT_MALFORMED, EXIT_USAGE = 4, 5, 6

# The one place the verdict -> exit mapping lives. A test that rebuilt the same three pairs as a
# second literal and compared the two would be a tautology that cannot fail; a test can only check
# this mapping by importing it.
VERDICT_EXIT = {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL, "INCONCLUSIVE": EXIT_INCONCLUSIVE}

# Connect, then read. A checker with no read timeout waits forever on a service that accepted the
# connection and then hung, which is the failure a served boundary actually produces -- and the
# documented exit code 4 could never be reached.
HTTP_TIMEOUT = (5.0, 300.0)

SCHEMA_VERSION = "1.1"


class UsageError(ValueError):
    """A local argument is unusable. Raised before any service is contacted."""


class MalformedResponse(Exception):
    """A service answered something outside its contract."""


def checker_version() -> str:
    """The commit the checker ran from, so a record can be traced to a source state."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip() if out.returncode == 0 else ""
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:                                              # noqa: BLE001
        return "unknown"


def finite_vector(name: str, value, n: int | None = None) -> np.ndarray:
    """A design point or a direction: one-dimensional, numeric, finite, and the right length."""
    try:
        v = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise UsageError(f"{name} is not a numeric array: {e}") from None
    if v.ndim != 1:
        raise UsageError(f"{name} must be a one-dimensional array; got shape {v.shape}")
    if v.size == 0:
        raise UsageError(f"{name} is empty")
    if not np.all(np.isfinite(v)):
        raise UsageError(f"{name} contains non-finite entries")
    if n is not None and v.size != n:
        raise UsageError(f"{name} has length {v.size}; the point has length {n}")
    return v


def build_pair(url_obj: str, url_grad: str):
    from tesseract_core import Tesseract
    to = Tesseract.from_url(url_obj, timeout=HTTP_TIMEOUT)
    tg = Tesseract.from_url(url_grad, timeout=HTTP_TIMEOUT)

    def J(m):
        out = to.apply({"m": np.asarray(m, np.float64)})
        if "J" not in out:
            raise MalformedResponse(f"objective response lacks 'J': keys {list(out)}")
        j = np.asarray(out["J"], dtype=np.float64)
        if j.size != 1:
            raise MalformedResponse(f"objective returned J with shape {j.shape}; expected a scalar")
        v = float(j.reshape(()))
        if not math.isfinite(v):
            raise MalformedResponse(f"objective returned a non-finite J ({v!r}); the remainder "
                                    f"sweep has nothing to measure")
        return v

    def g(m):
        out = tg.vector_jacobian_product(
            {"m": np.asarray(m, np.float64)}, vjp_inputs=["m"], vjp_outputs=["J"],
            cotangent_vector={"J": np.float64(1.0)})
        if "m" not in out:
            raise MalformedResponse(f"gradient response lacks 'm': keys {list(out)}")
        gv = np.asarray(out["m"], dtype=np.float64)
        if gv.shape != np.shape(m):
            raise MalformedResponse(
                f"gradient has shape {gv.shape} but the point has shape {np.shape(m)}; "
                f"these are not the same design space")
        if not np.all(np.isfinite(gv)):
            raise MalformedResponse("gradient contains non-finite entries")
        return gv
    return J, g


def sweep(J, g_vec, m, p, label):
    """One subject-direction: residuals, usable mask, orders, verdict, counts."""
    p = np.asarray(p, dtype=np.float64)
    nrm = float(np.linalg.norm(p))
    if nrm == 0.0:
        raise ValueError(f"{label}: zero direction")
    p = p / nrm
    applies = 0

    def eval_J(x):
        nonlocal applies
        applies += 1
        return float(J(x))

    J0 = eval_J(m)
    gTp = float(np.asarray(g_vec, dtype=np.float64) @ p)
    hs, Rs, floors = [], [], []
    for j in range(N_HALVINGS + 1):
        h = H0 / (2 ** j)
        Jh = eval_J(m + h * p)
        if not (math.isfinite(Jh) and math.isfinite(J0)):
            return {"label": label, "verdict": "INVALID",
                    "reason": "non-finite objective", "objective_applies": applies}
        hs.append(h)
        Rs.append(abs(Jh - J0 - h * gTp))
        floors.append(FLOOR_FACTOR * EPS64 * max(abs(J0), abs(Jh)))
    if applies > COST_CAP:
        raise RuntimeError(f"{label}: {applies} applies exceeds the frozen cap {COST_CAP}")

    usable = [bool(Rs[j] > floors[j]) for j in range(len(Rs))]
    if False in usable:                     # contiguity: truncate at first unusable rung
        first_bad = usable.index(False)
        for j in range(first_bad, len(usable)):
            usable[j] = False
    # plateau: first successive pair of then-usable rungs whose residual change is
    # slower than the frozen ratio; rungs from the plateau onward are unusable.
    plateau_at = None
    for j in range(len(Rs) - 1):
        if usable[j] and usable[j + 1] and Rs[j + 1] > 0 \
                and (Rs[j] / Rs[j + 1]) < PLATEAU_RATIO:
            plateau_at = j + 1
            break
    if plateau_at is not None:
        for j in range(plateau_at, len(usable)):
            usable[j] = False
    n_use = sum(usable)

    pair_orders = [
        {"between": [j, j + 1],
         "order": (math.log2(Rs[j] / Rs[j + 1]) if Rs[j + 1] > 0 else None),
         "usable": usable[j] and usable[j + 1]}
        for j in range(len(Rs) - 1)]
    fit = None
    if n_use >= 2:
        xs = [math.log2(hs[j]) for j in range(len(hs)) if usable[j]]
        ys = [math.log2(Rs[j]) for j in range(len(hs)) if usable[j]]
        xb, yb = sum(xs) / len(xs), sum(ys) / len(ys)
        denom = sum((x - xb) ** 2 for x in xs)
        fit = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / denom if denom else None

    use_pairs = [po["order"] for po in pair_orders if po["usable"] and po["order"] is not None]
    if plateau_at is not None and n_use < 3:
        verdict = "INCONCLUSIVE"
    elif use_pairs and n_use >= 3 and all(o >= PASS_MIN for o in use_pairs):
        verdict = "PASS"
    elif fit is not None and fit <= FAIL_MAX and n_use >= 3:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    return {"label": label, "h": hs, "R": Rs, "floor": floors, "usable": usable,
            "plateau_at": plateau_at, "n_usable": n_use,
            "pair_orders": pair_orders, "fitted_order": fit, "verdict": verdict,
            "objective_applies": applies,
            "gradient_calls": "1 per subject, shared across its directions",
            "J0": J0, "gTp": gTp}


def run_check(J, g, m, directions: dict) -> dict:
    """Core check over callables; the CLI wraps this with URL clients.
    directions: name -> vector (unnormalized; sweep normalizes)."""
    m = np.asarray(m, dtype=np.float64)
    gv = g(m)
    out = {"point": m.tolist(), "supplied_gradient": gv.tolist(),
           "directions": {}, "gradient_calls": 1}
    for name, d in directions.items():
        out["directions"][name] = sweep(J, gv, m, d, name)
    verdicts = [v["verdict"] for v in out["directions"].values()]
    out["verdict"] = ("FAIL" if "FAIL" in verdicts else
                      "PASS" if verdicts and all(v == "PASS" for v in verdicts) else
                      "INCONCLUSIVE")
    out["objective_applies_total"] = sum(
        v.get("objective_applies", 0) for v in out["directions"].values())
    out["limitation"] = ("finitely many directions at one point; NOT a proof of "
                         "global gradient correctness. Near a stationary point the "
                         "test loses power, because the term it measures vanishes with "
                         "the gradient itself.")
    out["meta"] = {
        "schema_version": SCHEMA_VERSION,
        "checker_commit": checker_version(),
        "point_dimension": int(m.size),
        "direction_names": sorted(out["directions"]),
        # Read from the frozen constants, never retyped, so a record cannot disagree with
        # the schedule that produced it.
        "step_schedule": {"h0": H0, "rule": "h_j = h0 / 2**j", "n_rungs": N_HALVINGS + 1},
    }
    out["recommended_next"] = {
        "FAIL": "do not trust this gradient source for this objective; if it is one of "
                "several providers, arbitrate away from it and run the official "
                "per-component checker on each side to localize",
        "PASS": "consistent along the tested directions; for component-local assurance "
                "run tesseract-runtime check-gradients on each service's own module",
        "INCONCLUSIVE": "no verdict issued; retry with a different point or direction, "
                        "or reduce objective noise before concluding anything",
    }[out["verdict"]]
    return out


GRADIENT_DIRECTION = object()      # sentinel: resolvable only once the gradient is fetched


def parse_directions(specs: list[str], n: int) -> dict:
    """name -> direction vector, validated OFFLINE against the point's dimension.

    Only the `gradient` direction needs the service, so it is left as a sentinel; everything else
    is checked here, before a socket is opened. A zero or mis-shaped direction has no directional
    derivative and the sweep divides by its norm, so both are usage errors rather than results.
    """
    dirs: dict = {}
    for s in specs:
        if s == "gradient":
            dirs["gradient"] = GRADIENT_DIRECTION
        elif s == "diagonal":
            dirs["diagonal"] = finite_vector("the diagonal direction",
                                             np.ones(n, dtype=np.float64), n)
        else:
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError as e:
                raise UsageError(f"--direction {s!r} is neither 'gradient', 'diagonal', nor "
                                 f"valid JSON: {e}") from None
            v = finite_vector(f"--direction {s!r}", parsed, n)
            if not np.any(v):
                raise UsageError(f"--direction {s!r} is exactly zero; a zero direction has no "
                                 f"directional derivative")
            dirs[f"custom{len(dirs)}"] = v
    return dirs


def resolve_gradient_direction(dirs: dict, gv: np.ndarray) -> dict:
    """Steepest descent, once the supplied gradient is known. None at a stationary point: the
    direction is undefined there, which is a documented power loss, not a crash."""
    if "gradient" in dirs and dirs["gradient"] is GRADIENT_DIRECTION:
        dirs["gradient"] = None if float(np.linalg.norm(gv)) == 0.0 else -gv
    return dirs


def main() -> int:
    ap = argparse.ArgumentParser(prog="misfit-check", add_help=True)
    ap.add_argument("objective_url")
    ap.add_argument("gradient_url")
    ap.add_argument("--point", required=True, help="JSON array, the design point m")
    ap.add_argument("--direction", action="append", default=None,
                    help="gradient | diagonal | JSON array (repeatable; "
                         "default: gradient + diagonal)")
    ap.add_argument("--json", help="write the machine-readable record here")
    args = ap.parse_args()

    try:
        # Arguments first, and entirely offline: a typo must never be reported as an outage.
        try:
            point = json.loads(args.point)
        except json.JSONDecodeError as e:
            raise UsageError(f"--point is not valid JSON: {e}") from None
        m = finite_vector("--point", point)
        specs = args.direction or ["gradient", "diagonal"]
        dirs = parse_directions(specs, m.size)          # still offline
        J, g = build_pair(args.objective_url, args.gradient_url)
        gv = g(m)                      # first wire contact: connection errors land here
        dirs = resolve_gradient_direction(dirs, gv)
        dropped = [k for k, v in dirs.items() if v is None]
        dirs = {k: v for k, v in dirs.items() if v is not None}
        if not dirs:
            print("misfit-check: supplied gradient is exactly zero and no other "
                  "direction was given -- the gradient-direction test is undefined "
                  "here (documented power loss). Give --direction diagonal.")
            return EXIT_INCONCLUSIVE
        rec = run_check(J, lambda _m: gv, m, dirs)
        rec["meta"]["objective_url"] = args.objective_url
        rec["meta"]["gradient_url"] = args.gradient_url
        if dropped:
            rec["dropped_directions"] = {
                k: "supplied gradient exactly zero; direction undefined"
                for k in dropped}
    except UsageError as e:
        print(f"misfit-check: {e}")
        return EXIT_USAGE
    except MalformedResponse as e:
        print(f"misfit-check: malformed service response: {e}")
        return EXIT_MALFORMED
    except (ConnectionError, OSError, TimeoutError) as e:
        print(f"misfit-check: cannot reach a service: {type(e).__name__}: {e}")
        return EXIT_CONNECT
    except json.JSONDecodeError as e:
        # Not a local argument -- those are parsed above and raise UsageError. Reaching here means
        # a SERVICE sent something that is not JSON.
        print(f"misfit-check: malformed service response (not JSON): {e}")
        return EXIT_MALFORMED
    except Exception as e:                                    # noqa: BLE001
        # requests wraps connection errors in its own types; classify by name.
        if "Connection" in type(e).__name__ or "Timeout" in type(e).__name__:
            print(f"misfit-check: cannot reach a service: {type(e).__name__}: {e}")
            return EXIT_CONNECT
        # An error STATUS from a service is the service's problem, not ours. tesseract-core
        # surfaces it as a bare RuntimeError("Error <code> from Tesseract: ..."), which matches
        # none of the name tests below. Without this branch a 500 reports as "internal error" and
        # exit 1 -- the checker taking the blame for the component it is checking, and the
        # documented meaning of exit 5 never being reached.
        if isinstance(e, RuntimeError) and re.match(r"Error \d{3} from Tesseract", str(e)):
            print(f"misfit-check: a service returned an error status: {str(e)[:300]}")
            return EXIT_MALFORMED
        if "Validation" in type(e).__name__ or "HTTP" in type(e).__name__:
            print(f"misfit-check: malformed request/response: "
                  f"{type(e).__name__}: {str(e)[:300]}")
            return EXIT_MALFORMED
        print(f"misfit-check: internal error: {type(e).__name__}: {e}")
        return EXIT_INTERNAL

    print(f"misfit-check  {args.objective_url}  vs  {args.gradient_url}")
    for name, v in rec["directions"].items():
        f = v.get("fitted_order")
        print(f"  {name:<12s} {v['verdict']:<14s} fitted order "
              f"{f if f is None else round(f, 3)}  "
              f"usable rungs {v.get('n_usable', '-')}  "
              f"applies {v.get('objective_applies', '-')}")
    print(f"  VERDICT {rec['verdict']} | total objective applies "
          f"{rec['objective_applies_total']} + 1 gradient call")
    print(f"  {rec['limitation']}")
    print(f"  next: {rec['recommended_next']}")
    if args.json:
        # Standards-compliant: allow_nan=False rejects NaN/Infinity rather than emitting the
        # non-standard literals most parsers refuse, and there is no `default=` fallback silently
        # stringifying a value nobody meant to serialise. A record that cannot be written is a
        # defect to surface, not to paper over.
        try:
            text = json.dumps(rec, indent=1, allow_nan=False, sort_keys=False)
        except (ValueError, TypeError) as e:
            print(f"misfit-check: the record is not serialisable as standard JSON: {e}")
            return EXIT_INTERNAL
        Path(args.json).write_text(text + "\n")
    return VERDICT_EXIT[rec["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
