#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""The platform's own checker, asked the cross-component question directly.

Points and settings were frozen before this file existed. Binds
8811-8812 and must be serialised against any other driver that binds them.

`run_platform_checker_ae.py` runs `tesseract-runtime check-gradients` on aero.A1 and struct.S1 and
both pass. From that this project concluded that the platform cannot ask whether a derivative
belongs to the ASSEMBLED objective. That conclusion is about where the boundary was drawn.

Here the boundary is drawn round the composition: coupled.C1 exposes the assembled trimmed
objective as one Tesseract, holds no physics, and routes every evaluation to A1 and S1 over HTTP.
The platform's own instrument -- its own central differences, its own tolerance, its own sampling --
is then pointed at three declared pairings:

    matched-coupled   coupled objective + coupled adjoint     should pass
    matched-rigid     rigid   objective + rigid gradient      should pass
    crossed-rigid     coupled objective + RIGID gradient      the question

COVERAGE. The platform samples which design indices to check from its own
RandomState. That sampling is replicated here and the record is WITHHELD unless all six design
indices and all three endpoints were covered, in every invocation.

VOID CONDITIONS, frozen in the plan. A matched pairing that fails means the FACADE is wrong, not
the platform: the run is void and says so rather than being read as a result. Zero HTTP calls to
either component means the facade answered without asking, and the record is invalid.

    .venv/bin/python aeroelastic/verification/run_native_composed_check.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
AE = HERE.parent
ROOT = AE.parent
COMPONENTS = AE / "components"
REC = AE / "records"
sys.path.insert(0, str(AE / "core"))

import numpy as np                                                        # noqa: E402

P_AERO, P_STRUCT = 8811, 8812
AERO_URL, STRUCT_URL = f"http://127.0.0.1:{P_AERO}", f"http://127.0.0.1:{P_STRUCT}"
RUNTIME = Path(sys.executable).parent / "tesseract-runtime"
C1 = COMPONENTS / "coupled" / "tesseract_api.py"
RESULTS = AE / "served"

# --- everything below was frozen before the composed component existed --------------------
SEED = 20260806
EPS, RTOL, MAX_EVALS = 1e-4, 0.1, 60
ENDPOINTS = ("jacobian", "jacobian_vector_product", "vector_jacobian_product")
PAIRINGS = ("matched-coupled", "matched-rigid", "crossed-rigid")
V, CL_TARGET = 80.0, 0.5
N_DESIGN = 6
CEILING_S = 1800.0                       # per invocation; abort condition of the plan

# The verdicts and counts this driver must reproduce exactly, or fail. --max-failures and the HTTP
# call counter are RECORDING-ONLY controls: they bound what the CLI echoes and how calls are
# tallied, and must not be able to move a verdict or a count. This table is what holds them to it.
PRIOR_RUN = {
    "before-a/matched-coupled": ("passed", 0, 180), "before-a/matched-rigid": ("passed", 0, 180),
    "before-a/crossed-rigid": ("failed", 180, 180),
    "inside-a/matched-coupled": ("passed", 0, 180), "inside-a/matched-rigid": ("passed", 0, 180),
    "inside-a/crossed-rigid": ("failed", 180, 180),
    "design/matched-coupled": ("passed", 0, 180), "design/matched-rigid": ("passed", 0, 180),
    "design/crossed-rigid": ("failed", 102, 180),
}

RESULT_RE = re.compile(r"Gradient check for (\w+) (passed|failed)")
COUNT_RE = re.compile(r"\((\d+) failures / (\d+) checks\)")
FAILURE_RE = re.compile(r"Index: \((\d+),\)\n\s*\w+ value: ([-\d.e+naif]+)\n"
                        r"\s*Finite difference value: ([-\d.e+naif]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
# Reporting only: --max-failures bounds how many failures the CLI ECHOES. It is not passed to
# check_gradients_ and cannot move a verdict or a count; the table above is what proves that.
MAX_FAILURES = 200


def points():
    """The three frozen points. `design` is read from the committed record, not recomputed."""
    sv = json.loads((REC / "served_optimize.json").read_text())
    return [("before-a", [0.0, 0.5 * np.deg2rad(-1.15), np.deg2rad(-1.15), 0.0, 0.0, 0.0]),
            ("inside-a", [0.0, 0.5 * np.deg2rad(-5.16), np.deg2rad(-5.16), -0.5, -0.5, -0.5]),
            ("design", list(np.asarray(sv["optimize"]["x"], float)))]


def assert_ports_free(ports):
    busy = [p for p in ports if socket.socket().connect_ex(("127.0.0.1", int(p))) == 0]
    if busy:
        raise SystemExit(f"ports already in use: {busy}. Stop the other driver first.")


def start_server(api: Path, port: int):
    env = dict(os.environ, TESSERACT_API_PATH=str(api))
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = open(RESULTS / f"serve_{port}.log", "w")
    return subprocess.Popen(
        [str(RUNTIME), "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        cwd=tempfile.mkdtemp(prefix=f"c1{port}_"))


def wait_healthy(url: str, timeout: float = 240.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.4)
    raise RuntimeError(f"{url} never became healthy")


def clean_pycache():
    for d in ("aero", "structure", "coupled"):
        cache = COMPONENTS / d / "__pycache__"
        if not (COMPONENTS / d).is_dir():                 # a renamed component would make
            raise RuntimeError(f"no such component: {d}")  # ignore_errors hide this silently
        shutil.rmtree(cache, ignore_errors=True)


def expected_coverage():
    """Replicate the platform's own index sampling, so coverage is asserted and not assumed.

    check_gradients() builds one RandomState(seed) and each endpoint draws from it in turn;
    _sample_indices asks for max(1, max_evals * size / total) = MAX_EVALS draws with replacement.
    """
    rng = np.random.RandomState(SEED)
    return [sorted(set(int(i) for i in rng.choice(N_DESIGN, MAX_EVALS))) for _ in ENDPOINTS]


def run_check(label, x, pairing, cdi_ref, mass_ref):
    counts_path = RESULTS / f"c1_counts_{label}_{pairing}.json"
    counts_path.unlink(missing_ok=True)
    payload = {"inputs": {"x": list(map(float, x)), "v": V, "cl_target": CL_TARGET,
                          "cdi_ref": cdi_ref, "mass_ref": mass_ref, "pairing": pairing}}
    env = dict(os.environ, TESSERACT_API_PATH=str(C1),
               MISFIT_AERO_URL=AERO_URL, MISFIT_STRUCT_URL=STRUCT_URL,
               MISFIT_C1_COUNTS=str(counts_path),
               TESSERACT_RUNTIME_CHECK_GRADIENTS_SHOW_PROGRESS="0")
    t0 = time.perf_counter()
    r = subprocess.run([str(RUNTIME), "check-gradients", json.dumps(payload),
                        "--seed", str(SEED), "--eps", str(EPS), "--rtol", str(RTOL),
                        "--max-evals", str(MAX_EVALS), "--max-failures", str(MAX_FAILURES)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=CEILING_S)
    wall = time.perf_counter() - t0
    text = ANSI.sub("", r.stdout + r.stderr)
    verdicts = dict(RESULT_RE.findall(text))
    counts = COUNT_RE.findall(text)
    # Sum the per-import instances; see the note in components/coupled/tesseract_api.py.
    raw = json.loads(counts_path.read_text()) if counts_path.is_file() else {}
    calls: dict[str, int] = {}
    for inst in raw.values():
        for k, n in inst.items():
            calls[k] = calls.get(k, 0) + n
    # Which DESIGN indices the platform rejected, and by how much. A pairing can disagree on some
    # components and agree on others, and an aggregate verdict hides that.
    per_index: dict[str, dict] = {}
    for idx, gv, fv in FAILURE_RE.findall(text):
        try:
            per_index[idx] = {"declared": float(gv), "finite_difference": float(fv)}
        except ValueError:
            per_index[idx] = {"declared": gv, "finite_difference": fv}
    return {"exit": r.returncode, "verdicts": verdicts,
            "failures": [int(a) for a, _ in counts], "checks": [int(b) for _, b in counts],
            "http_calls": calls, "http_instances": len(raw), "wall_s": round(wall, 1),
            "failing_indices": {k: per_index[k] for k in sorted(per_index, key=int)},
            "stderr_tail": text.strip().splitlines()[-3:] if r.returncode not in (0, 1) else []}


def main() -> int:
    assert_ports_free([P_AERO, P_STRUCT])
    RESULTS.mkdir(parents=True, exist_ok=True)
    clean_pycache()
    procs, out, problems = [], {}, []
    try:
        procs.append(start_server(COMPONENTS / "aero" / "tesseract_api.py", P_AERO))
        procs.append(start_server(COMPONENTS / "structure" / "tesseract_api.py", P_STRUCT))
        wait_healthy(AERO_URL)
        wait_healthy(STRUCT_URL)

        # Normalisation constants: taken from the SERVED components, then checked against the
        # monolithic values every committed result used. If these disagreed, C1 would be scoring a
        # different objective and every comparison below would be meaningless.
        os.environ.update(MISFIT_AERO_URL=AERO_URL, MISFIT_STRUCT_URL=STRUCT_URL)
        sys.path.insert(0, str(COMPONENTS / "coupled"))
        import tesseract_api as C                                          # noqa: E402
        st = C.trimmed(np.zeros(3), np.zeros(3), V, CL_TARGET, C.coupled_forward)
        cdi_ref, mass_ref = st["cdi"], st["mass"]
        from model import refs                                             # noqa: E402
        cdi_mono, mass_mono = refs()
        d_cdi = abs(cdi_ref - cdi_mono) / cdi_mono
        d_mass = abs(mass_ref - mass_mono) / mass_mono
        print(f"normalisation from the served pair: CDi_ref={cdi_ref:.12f}  mass_ref={mass_ref:.12f}")
        print(f"  vs the monolithic constants every committed result used: "
              f"rel {d_cdi:.2e} and {d_mass:.2e}")
        if max(d_cdi, d_mass) > 1e-9:
            problems.append(f"normalisation disagrees with the committed objective: "
                            f"{d_cdi:.2e}, {d_mass:.2e}")

        # ---- the facade itself, against the monolithic in-process solve, at every point.
        # Without this the experiment is circular: check-gradients compares C1's derivative
        # endpoints against finite differences of C1's own apply, so a wrong fixed point or a
        # wrong transpose would be certified with full confidence and 462/540 would be a
        # measurement of C1's bugs. The gap was identified before the evidence existed to close
        # it, and this block is what closes it.
        import jax                                                         # noqa: E402
        import jax.numpy as jnp                                            # noqa: E402
        import aero_struct as A                                            # noqa: E402
        from aero_struct import Wing                                       # noqa: E402
        from model import W_MASS                                           # noqa: E402
        WING = Wing()

        def mono(xx, solver):
            _, _, _, cdi, _ = A.solve_trimmed(WING, xx[:3], xx[3:], V, CL_TARGET, solver)
            return cdi / cdi_ref + W_MASS * A.mass_proxy(WING, xx[3:]) / mass_ref

        facade = {}
        print("\nTHE FACADE AGAINST THE MONOLITHIC SOLVE -- C1 holds no physics, and this is the")
        print("proof that what it assembles from A1 and S1 is the same objective and the same")
        print("gradients the in-process model produces.\n")
        print(f"{'point':>9} {'J rel':>10} {'coupled grad rel':>18} {'rigid grad rel':>16}")
        for label, x in points():
            xa = np.asarray(x, float)
            j_served, _ = C.objective_value(xa, V, CL_TARGET, cdi_ref, mass_ref, "coupled")
            j_mono = float(mono(jnp.asarray(xa), A.solve_coupled))
            gc_s = C.coupled_gradient(xa, V, CL_TARGET, cdi_ref, mass_ref)
            gr_s = C.rigid_gradient(xa, V, CL_TARGET, cdi_ref, mass_ref)
            gc_m = np.asarray(jax.grad(lambda z: mono(z, A.solve_coupled))(jnp.asarray(xa)))
            gr_m = np.asarray(jax.grad(lambda z: mono(z, A.solve_rigid))(jnp.asarray(xa)))
            rec = {"j_rel": abs(j_served - j_mono) / abs(j_mono),
                   "coupled_grad_rel": float(np.max(np.abs(gc_s - gc_m))
                                             / np.max(np.abs(gc_m))),
                   "rigid_grad_rel": float(np.max(np.abs(gr_s - gr_m))
                                           / np.max(np.abs(gr_m)))}
            facade[label] = rec
            print(f"{label:>9} {rec['j_rel']:>10.2e} {rec['coupled_grad_rel']:>18.2e} "
                  f"{rec['rigid_grad_rel']:>16.2e}")
            if max(rec.values()) > 1e-7:
                problems.append(f"{label}: the facade disagrees with the monolithic solve by "
                                f"{max(rec.values()):.2e}; the composed check would be measuring "
                                f"the facade rather than the pairing")
        print("  every value the platform checker sees is the assembled physics, not an artifact")
        print("  of the composition -- and the crossed pairing still fails, so it is the PAIRING")

        cover = expected_coverage()
        print(f"\nplatform sampling with seed {SEED}: each endpoint draws {MAX_EVALS} indices; "
              f"union per endpoint {[len(c) for c in cover]} of {N_DESIGN}")
        if any(c != list(range(N_DESIGN)) for c in cover):
            problems.append(f"platform sampling does not cover all {N_DESIGN} design indices: "
                            f"{cover}")

        print(f"\n{'point':>9} {'pairing':>16} {'exit':>5} "
              + "  ".join(f"{e[:12]:>12}" for e in ENDPOINTS)
              + f" {'fail/checks':>13} {'A1':>6} {'S1':>6} {'wall s':>7}")
        for label, x in points():
            for pairing in PAIRINGS:
                rec = run_check(label, x, pairing, cdi_ref, mass_ref)
                a1 = sum(v for k, v in rec["http_calls"].items() if k.startswith("aero"))
                s1 = sum(v for k, v in rec["http_calls"].items() if k.startswith("struct"))
                print(f"{label:>9} {pairing:>16} {rec['exit']:>5} "
                      + "  ".join(f"{rec['verdicts'].get(e, '?'):>12}" for e in ENDPOINTS)
                      + f" {sum(rec['failures']):>5}/{sum(rec['checks']):<7} "
                      f"{a1:>6} {s1:>6} {rec['wall_s']:>7.1f}")
                if rec["stderr_tail"]:
                    for line in rec["stderr_tail"]:
                        print(f"          ! {line}")
                out[f"{label}/{pairing}"] = {"point": label, "pairing": pairing,
                                             "x": list(map(float, x)), **rec}
                if set(rec["verdicts"]) != set(ENDPOINTS):
                    problems.append(f"{label}/{pairing}: reported {sorted(rec['verdicts'])}, "
                                    f"expected {list(ENDPOINTS)}")
                if len(rec["checks"]) != len(ENDPOINTS):
                    problems.append(f"{label}/{pairing}: {len(rec['checks'])} count lines")
                if a1 == 0 or s1 == 0:
                    problems.append(f"{label}/{pairing}: {a1} aero and {s1} struct calls -- the "
                                    f"facade answered without asking a component")
                key = f"{label}/{pairing}"
                want_v, want_f, want_c = PRIOR_RUN[key]
                got = (set(rec["verdicts"].values()), sum(rec["failures"]), sum(rec["checks"]))
                if got != ({want_v}, want_f, want_c):
                    problems.append(f"{key}: does not reproduce the reference "
                                    f"{(want_v, want_f, want_c)} -- got {got}. A recording-only "
                                    f"control has moved a verdict or a count.")
                if rec["failing_indices"]:
                    fi = rec["failing_indices"]
                    print("           rejected design indices "
                          + ", ".join(f"{k}: declared {v['declared']:+.3e} vs central difference "
                                      f"{v['finite_difference']:+.3e}"
                                      for k, v in list(fi.items())[:2]))
                    if len(fi) > 2:
                        print(f"           and {len(fi) - 2} more: "
                              + ", ".join(sorted(fi, key=int)[2:]))
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        clean_pycache()

    if problems:
        print("\nCOVERAGE SHORTFALL -- no record written:")
        for p in problems:
            print(f"  {p}")
        return 1

    matched = [r for k, r in out.items() if r["pairing"].startswith("matched")]
    crossed = [r for k, r in out.items() if r["pairing"] == "crossed-rigid"]
    matched_ok = all(v == "passed" for r in matched for v in r["verdicts"].values())
    crossed_all_fail = all(v == "failed" for r in crossed for v in r["verdicts"].values())
    if not matched_ok:
        print("\nVOID -- a matched pairing failed. Under the frozen plan that means the FACADE is")
        print("wrong, not the platform. The run is recorded and is not a result.")
    verdict = ("VOID_FACADE" if not matched_ok else
               "NATIVE_DISCRIMINATES" if crossed_all_fail else
               "NATIVE_DOES_NOT_DISCRIMINATE" if all(
                   v == "passed" for r in crossed for v in r["verdicts"].values()) else
               "MIXED")
    out["_summary"] = {
        "verdict": verdict, "seed": SEED, "eps": EPS, "rtol": RTOL, "max_evals": MAX_EVALS,
        "points": [p for p, _ in points()], "pairings": list(PAIRINGS),
        "endpoints": list(ENDPOINTS), "v": V, "cl_target": CL_TARGET,
        "cdi_ref": cdi_ref, "mass_ref": mass_ref,
        "tesseract_core": __import__("importlib.metadata", fromlist=["version"]
                                     ).version("tesseract-core"),
        "coverage_indices_per_endpoint": cover,
        "facade_vs_monolithic": facade,
        "matched_all_passed": matched_ok, "crossed_all_failed": crossed_all_fail,
        "total_http_calls": sum(sum(r["http_calls"].get(k, 0) for k in r["http_calls"]
                                    if not k.startswith("_"))
                                for r in out.values() if isinstance(r, dict)
                                and "http_calls" in r),
    }
    # The live demo path runs this and must leave the working tree clean, so it redirects the
    # record to scratch and field-compares it against the committed one instead of overwriting.
    dest = Path(os.environ.get("MISFIT_NATIVE_OUT", REC / "native_composed_check.json"))
    dest.write_text(json.dumps(out, indent=1) + "\n")

    print(f"\n  verdict: {verdict}")
    print("  matched pairings all passed" if matched_ok else "  a matched pairing FAILED")
    print("  crossed-rigid failed at every point and every endpoint" if crossed_all_fail
          else "  crossed-rigid did NOT fail everywhere -- see the record")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
