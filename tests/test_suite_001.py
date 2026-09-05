"""Feasibility solvers against a stride-25 sample of suite_001 (40 of 1000 cases).

The full 1000-case corpus is vendored under tests/data/ and used directly by
benchmark.py for the full run; this suite samples it to stay fast under pytest.
"""
import pytest

from _suite_helpers import check_feasibility_case, feasibility_params

SUITE = "suite_001"


@pytest.mark.parametrize("solver, case", feasibility_params(SUITE))
def test_feasibility(solver, case):
    check_feasibility_case(SUITE, solver, case)
