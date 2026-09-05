"""Compare the *fairness objective* of the rosters different solvers produce.

Every solver here targets feasibility only, so any valid roster is an equally
correct answer to the stated problem.  They nonetheless differ a lot in how
evenly they spread work, and the verifier's objective

    sum over nurses of  3*(M^2 + A^2 + E^2) - (M + A + E)^2

measures exactly that (lower is better; it is minimised when a nurse's shift
types are balanced).  This script reports it so the trade-off between "finds
an answer fast" and "finds a good answer" is visible.
"""

import argparse
import concurrent.futures
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKER = ROOT / "tests" / "data"

sys.path.insert(0, str(ROOT / "tests"))
import verifier  # noqa: E402


def run(task):
    solver, case, timeout = task
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "o.json"
        try:
            subprocess.run([sys.executable, str(solver), str(case), str(out)],
                           cwd=ROOT, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            return solver.name, case.name, None
        if not out.is_file():
            return solver.name, case.name, None
        try:
            inst = verifier.read_input(case)
            sol = verifier.read_solution(out)
            if sol == {} or not verifier.verify_solution(inst, sol):
                return solver.name, case.name, None
            return solver.name, case.name, verifier.calculate_objective(inst, sol)
        except Exception:
            return solver.name, case.name, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="suite_003")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=35.0)
    args = ap.parse_args()

    solvers = sorted(x for x in (ROOT / "finding_solutions").glob("*.py")
                     if not x.name.startswith("_"))
    cases = sorted((CHECKER / "test-cases" / args.suite).glob("*.csv"))
    tasks = [(s, c, args.timeout) for s in solvers for c in cases]

    scores = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for name, case, obj in ex.map(run, tasks):
            scores.setdefault(name, {})[case] = obj

    # the best objective anyone achieved on each case, as the reference
    best = {}
    for name, per in scores.items():
        for case, obj in per.items():
            if obj is None:
                continue
            if case not in best or obj < best[case]:
                best[case] = obj

    print(f"suite {args.suite}: fairness objective, lower is better")
    print(f"{'solver':<38} {'solved':>7} {'mean obj':>10} {'vs best':>9}")
    print("-" * 68)
    rows = []
    for name in sorted(scores):
        vals = [(c, o) for c, o in scores[name].items() if o is not None]
        if not vals:
            rows.append((name, 0, None, None))
            continue
        mean = sum(o for _, o in vals) / len(vals)
        # average excess over the best anyone found, on the cases this solver solved
        gaps = [o - best[c] for c, o in vals if c in best]
        rows.append((name, len(vals), mean, sum(gaps) / len(gaps) if gaps else 0.0))
    rows.sort(key=lambda r: (-r[1], r[3] if r[3] is not None else 1e18))
    for name, n, mean, gap in rows:
        if mean is None:
            print(f"{name:<38} {n:>7} {'-':>10} {'-':>9}")
        else:
            print(f"{name:<38} {n:>7} {mean:>10.1f} {gap:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
