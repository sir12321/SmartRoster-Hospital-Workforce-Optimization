"""Feasibility and optimisation solvers against the full suite_004 (3 cases)."""
import pytest

from _suite_helpers import (
    check_feasibility_case,
    check_optimisation_case,
    feasibility_params,
    optimisation_params,
)

SUITE = "suite_004"


@pytest.mark.parametrize("solver, case", feasibility_params(SUITE))
def test_feasibility(solver, case):
    check_feasibility_case(SUITE, solver, case)


@pytest.mark.parametrize("solver, case", optimisation_params(SUITE))
def test_optimisation(solver, case):
    check_optimisation_case(SUITE, solver, case)
