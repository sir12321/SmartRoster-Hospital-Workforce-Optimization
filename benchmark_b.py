"""Benchmark the optimisation solvers in optimising the cost/.

Part B asks for the *best* roster, not merely a valid one, so a run is scored
on the fairness objective

    sum over nurses of  3*(M^2 + A^2 + E^2) - (M + A + E)^2

which is zero when every nurse's shift types are perfectly balanced and grows
as they skew.  Lower is better.  Each result is compared against the bundled
reference solution for the same instance:

    MATCHED       same objective as the reference
    BETTER        strictly lower than the reference
    WORSE         valid, but higher than the reference
    MISSED        gave up on an instance the reference solves
    INVALID       produced a roster that breaks a hard rule

Usage:
    python benchmark_b.py --suite suite_003
    python benchmark_b.py --solvers "optimising the cost/annealing_objective.py" --suite suite_002
"""

import argparse
import concurrent.futures
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKER = ROOT / "tests" / "data"

sys.path.insert(0, str(ROOT / "tests"))
import verifier  # noqa: E402

SUITES = ["suite_001", "suite_002", "suite_003", "suite_004"]


def reference(suite, case):
    """(objective, feasible) of the bundled model solution, or (None, None)."""
    path = CHECKER / "model-solutions" / suite / f"{case.stem}.json"
    if not path.is_file():
        return None, None
    try:
        sol = json.loads(path.read_text())
    except Exception:
        return None, None
    if sol == {}:
        return None, False
    try:
        inst = verifier.read_input(case)
        if not verifier.verify_solution(inst, sol):
            return None, None
        return verifier.calculate_objective(inst, sol), True
    except Exception:
        return None, None


def run_one(task):
    solver, suite, case, extra, hard = task
    with open(case, newline="", encoding="utf-8") as fh:
        budget = float(next(csv.DictReader(fh))["T"])
    limit = hard if hard else budget + extra
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "o.json"
        started = time.perf_counter()
        status, obj = None, None
        try:
            proc = subprocess.run(
                [sys.executable, str(solver), str(case), str(out)],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, timeout=limit)
            if proc.returncode:
                status = "ERROR"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
        elapsed = time.perf_counter() - started

        if status is None:
            if not out.is_file():
                status = "NO_OUTPUT"
            else:
                try:
                    inst = verifier.read_input(case)
                    sol = verifier.read_solution(out)
                    if sol == {}:
                        status = "GAVE_UP"
                    elif verifier.verify_solution(inst, sol):
                        obj = verifier.calculate_objective(inst, sol)
                        status = "VALID"
                    else:
                        status = "INVALID"
                except Exception:
                    status = "INVALID"

    ref_obj, ref_feasible = reference(suite, case)
    if status == "VALID":
        if ref_obj is None:
            verdict = "BETTER" if ref_feasible is False else "PASS"
        elif obj < ref_obj:
            verdict = "BETTER"
        elif obj == ref_obj:
            verdict = "MATCHED"
        else:
            verdict = "WORSE"
    elif status == "GAVE_UP":
        verdict = "OK_INFEASIBLE" if ref_feasible is False else "MISSED"
    else:
        verdict = status
    return dict(solver=solver.name, suite=suite, case=case.name, status=status,
                verdict=verdict, objective=obj, reference=ref_obj,
                seconds=elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvers", nargs="*", default=[])
    ap.add_argument("--suite", action="append", default=None)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout-extra", type=float, default=2.0)
    ap.add_argument("--hard-timeout", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    solvers = ([Path(s) if Path(s).is_absolute() else ROOT / s for s in args.solvers]
               or sorted(x for x in (ROOT / "optimising the cost").glob("*.py")
                         if not x.name.startswith("_")))
    suites = args.suite or SUITES
    cases = []
    for suite in suites:
        found = sorted((CHECKER / "test-cases" / suite).glob("*.csv"),
                       key=lambda p: (len(p.stem), p.stem))[::args.stride]
        if args.sample:
            found = found[:args.sample]
        cases.extend((suite, c) for c in found)

    tasks = [(s, suite, case, args.timeout_extra, args.hard_timeout)
             for s in solvers for suite, case in cases]
    print(f"{len(solvers)} solver(s) x {len(cases)} case(s) = {len(tasks)} runs", flush=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, res in enumerate(ex.map(run_one, tasks), 1):
            results.append(res)
            if i % 25 == 0:
                print(f"  ...{i}/{len(tasks)}", flush=True)

    print()
    hdr = (f"{'solver':<32} {'suite':<11} {'BETTER':>7} {'MATCH':>6} {'WORSE':>6} "
           f"{'MISS':>5} {'INVAL':>6} {'mean gap %':>11} {'mean s':>7}")
    print(hdr)
    print("-" * len(hdr))
    for solver in solvers:
        for suite in suites:
            sub = [r for r in results if r["solver"] == solver.name and r["suite"] == suite]
            if not sub:
                continue
            gaps = [(r["objective"] - r["reference"]) / r["reference"] * 100
                    for r in sub
                    if r["objective"] is not None and r["reference"]]
            gap = sum(gaps) / len(gaps) if gaps else 0.0
            print(f"{solver.name:<32} {suite:<11} "
                  f"{sum(r['verdict']=='BETTER' for r in sub):>7} "
                  f"{sum(r['verdict']=='MATCHED' for r in sub):>6} "
                  f"{sum(r['verdict']=='WORSE' for r in sub):>6} "
                  f"{sum(r['verdict']=='MISSED' for r in sub):>5} "
                  f"{sum(r['verdict']=='INVALID' for r in sub):>6} "
                  f"{gap:>10.1f}% "
                  f"{sum(r['seconds'] for r in sub)/len(sub):>7.2f}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\nraw results -> {args.out}")
    bad = [r for r in results if r["verdict"] in ("INVALID", "ERROR", "NO_OUTPUT")]
    if bad:
        print(f"\n{len(bad)} correctness failure(s); first few:")
        for r in bad[:8]:
            print(f"  {r['solver']} {r['suite']}/{r['case']} {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
