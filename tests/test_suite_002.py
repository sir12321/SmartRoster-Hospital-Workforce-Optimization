"""Feasibility solvers against the full suite_002 (24 cases)."""
import pytest

from _suite_helpers import check_feasibility_case, feasibility_params

SUITE = "suite_002"


@pytest.mark.parametrize("solver, case", feasibility_params(SUITE))
def test_feasibility(solver, case):
    check_feasibility_case(SUITE, solver, case)
