import pytest

from conftest import (
    FEASIBILITY_SOLVERS,
    OPTIMISATION_SOLVERS,
    cases,
    reference_feasible,
    reference_objective,
    run_solver,
    verifier,
)


def feasibility_params(suite):
    return [
        pytest.param(solver, case, id=f"{solver.stem}-{case.stem}")
        for solver in FEASIBILITY_SOLVERS
        for case in cases(suite)
    ]


def optimisation_params(suite):
    return [
        pytest.param(solver, case, id=f"{solver.stem}-{case.stem}")
        for solver in OPTIMISATION_SOLVERS
        for case in cases(suite)
    ]


def check_feasibility_case(suite, solver, case):
    expected_feasible = reference_feasible(suite, case)
    solution, _ = run_solver(solver, case)

    if solution == {}:
        if expected_feasible is True:
            pytest.fail(f"{solver.name} gave up on solvable instance {case.name}")
        return

    instance = verifier.read_input(case)
    valid = verifier.verify_solution(instance, solution)
    if expected_feasible is False:
        assert not valid, f"{solver.name} claims feasible but reference says infeasible: {case.name}"
    else:
        assert valid, f"{solver.name} produced an invalid roster for {case.name}"


def check_optimisation_case(suite, solver, case):
    ref_obj = reference_objective(suite, case)
    solution, _ = run_solver(solver, case)

    if solution == {}:
        if ref_obj is not None:
            pytest.fail(f"{solver.name} gave up on solvable instance {case.name}")
        return

    instance = verifier.read_input(case)
    assert verifier.verify_solution(instance, solution), \
        f"{solver.name} produced an invalid roster for {case.name}"

    if ref_obj is not None:
        objective = verifier.calculate_objective(instance, solution)
        assert objective <= ref_obj, (
            f"{solver.name} scored {objective} which is worse than the "
            f"reference {ref_obj} on {case.name}"
        )
