import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verifier  # noqa: E402

FEASIBILITY_SOLVERS = sorted(
    p for p in (ROOT / "finding_solutions").glob("*.py") if not p.name.startswith("_")
)
OPTIMISATION_SOLVERS = sorted(
    p for p in (ROOT / "optimising the cost").glob("*.py") if not p.name.startswith("_")
)


# suite_001 has 1000 cases; sampling keeps the suite fast while still
# exercising every suite. The other suites are small enough to run in full.
SAMPLE_STRIDE = {"suite_001": 25}


def cases(suite):
    found = sorted((DATA / "test-cases" / suite).glob("*.csv"),
                    key=lambda p: (len(p.stem), p.stem))
    return found[::SAMPLE_STRIDE.get(suite, 1)]


def model_solution(suite, case):
    path = DATA / "model-solutions" / suite / f"{case.stem}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def reference_feasible(suite, case):
    solution = model_solution(suite, case)
    if solution is None:
        return None
    return solution != {}


def reference_objective(suite, case):
    solution = model_solution(suite, case)
    if not solution:
        return None
    instance = verifier.read_input(case)
    if not verifier.verify_solution(instance, solution):
        return None
    return verifier.calculate_objective(instance, solution)


def run_solver(solver, case, timeout_extra=2.0):
    with open(case, newline="", encoding="utf-8") as fh:
        budget = float(next(csv.DictReader(fh))["T"])
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "out.json"
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(solver), str(case), str(output)],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=budget + timeout_extra,
        )
        elapsed = time.perf_counter() - started
        if proc.returncode:
            pytest.fail(f"{solver.name} exited {proc.returncode}: {proc.stderr.strip()[-300:]}")
        if not output.is_file():
            pytest.fail(f"{solver.name} produced no output for {case.name}")
        solution = verifier.read_solution(output)
        return solution, elapsed


@pytest.fixture(scope="session")
def instance_cache():
    cache = {}

    def get(case):
        if case not in cache:
            cache[case] = verifier.read_input(case)
        return cache[case]

    return get
