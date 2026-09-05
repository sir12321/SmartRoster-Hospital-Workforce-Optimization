"""Benchmark any solver in finding_solutions/ against the bundled test suites.

Usage:
    python benchmark.py --solvers finding_solutions/network_flow_matching.py --suite suite_003
    python benchmark.py --all --suite suite_001 --sample 200 --jobs 16
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


def model_verdict(suite, case):
    """FEASIBLE / INFEASIBLE / UNKNOWN according to the bundled model solution."""
    path = CHECKER / "model-solutions" / suite / f"{case.stem}.json"
    if not path.is_file():
        return "UNKNOWN"
    try:
        text = json.loads(path.read_text())
    except Exception:
        return "UNKNOWN"
    if text == {}:
        return "INFEASIBLE"
    return "FEASIBLE"


def run_one(task):
    solver, suite, case, timeout_extra, hard_timeout = task
    with open(case, newline="", encoding="utf-8") as fh:
        budget = float(next(csv.DictReader(fh))["T"])
    limit = hard_timeout if hard_timeout else budget + timeout_extra
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(solver), str(case), str(out)],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, timeout=limit)
            status = None if proc.returncode == 0 else "RUNTIME_ERROR"
            err = proc.stderr.strip()[-200:]
        except subprocess.TimeoutExpired:
            status, err = "TIMEOUT", ""
        elapsed = time.perf_counter() - started

        expected = model_verdict(suite, case)
        if status is None:
            if not out.is_file():
                status, err = "NO_OUTPUT", err
            else:
                try:
                    instance = verifier.read_input(case)
                    solution = verifier.read_solution(out)
                    if solution == {}:
                        # claimed infeasible
                        status = "CLAIM_INFEASIBLE"
                    elif verifier.verify_solution(instance, solution):
                        status = "VALID"
                    else:
                        status = "INVALID"
                except Exception as exc:
                    status, err = "EXC", f"{type(exc).__name__}: {exc}"

    # correctness verdict against the reference
    if status == "VALID":
        verdict = "OK" if expected in ("FEASIBLE", "UNKNOWN") else "WRONG_SAT"
    elif status == "CLAIM_INFEASIBLE":
        if expected == "INFEASIBLE":
            verdict = "OK"
        elif expected == "FEASIBLE":
            verdict = "MISSED"     # gave up on a solvable instance
        else:
            verdict = "OK"
    elif status == "TIMEOUT":
        verdict = "TIMEOUT"
    else:
        verdict = "ERROR"
    return dict(solver=solver.name, suite=suite, case=case.name, status=status,
                verdict=verdict, seconds=elapsed, budget=budget, expected=expected, err=err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvers", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="benchmark every solver in solutions/")
    ap.add_argument("--suite", action="append", default=None)
    ap.add_argument("--sample", type=int, default=0, help="use only the first K cases per suite")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth case")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout-extra", type=float, default=2.0)
    ap.add_argument("--hard-timeout", type=float, default=0.0,
                    help="override the per-case limit entirely (seconds)")
    ap.add_argument("--out", type=Path, default=None, help="write raw per-case results as JSON")
    args = ap.parse_args()

    if args.all:
        # skip the shared library and any private helpers
        solvers = sorted(x for x in (ROOT / "finding_solutions").glob("*.py")
                         if not x.name.startswith("_"))
    else:
        solvers = [Path(s) if Path(s).is_absolute() else ROOT / s for s in args.solvers]
    if not solvers:
        ap.error("no solvers selected")

    suites = args.suite or SUITES
    cases = []
    for suite in suites:
        found = sorted((CHECKER / "test-cases" / suite).glob("*.csv"),
                       key=lambda p: (len(p.stem), p.stem))
        found = found[::args.stride]
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
            if i % 50 == 0:
                print(f"  ...{i}/{len(tasks)}", flush=True)

    print()
    header = f"{'solver':<34} {'suite':<11} {'OK':>5} {'MISS':>5} {'WSAT':>5} {'TO':>4} {'ERR':>4} {'mean_s':>8} {'max_s':>7}"
    print(header)
    print("-" * len(header))
    summary = {}
    for solver in solvers:
        for suite in suites:
            sub = [r for r in results if r["solver"] == solver.name and r["suite"] == suite]
            if not sub:
                continue
            row = dict(
                ok=sum(r["verdict"] == "OK" for r in sub),
                missed=sum(r["verdict"] == "MISSED" for r in sub),
                wrong=sum(r["verdict"] == "WRONG_SAT" for r in sub),
                timeout=sum(r["verdict"] == "TIMEOUT" for r in sub),
                error=sum(r["verdict"] == "ERROR" for r in sub),
                n=len(sub),
                mean=sum(r["seconds"] for r in sub) / len(sub),
                mx=max(r["seconds"] for r in sub))
            summary[(solver.name, suite)] = row
            print(f"{solver.name:<34} {suite:<11} {row['ok']:>5} {row['missed']:>5} "
                  f"{row['wrong']:>5} {row['timeout']:>4} {row['error']:>4} "
                  f"{row['mean']:>8.3f} {row['mx']:>7.2f}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\nraw results -> {args.out}")

    bad = [r for r in results if r["verdict"] in ("WRONG_SAT", "ERROR")]
    if bad:
        print(f"\n{len(bad)} correctness failure(s); first few:")
        for r in bad[:10]:
            print(f"  {r['solver']} {r['suite']}/{r['case']} {r['status']} {r['err'][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
