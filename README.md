# SmartRoster — Hospital Workforce Optimization

_Manya Jain_

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Algorithms](https://img.shields.io/badge/Algorithms-20%20Solvers-orange.svg)](#headline-results)
[![Benchmark](https://img.shields.io/badge/Benchmark-1%2C036%20Instances-brightgreen.svg)](#benchmarking)
[![Problem](<https://img.shields.io/badge/Problem-Nurse%20Scheduling%20(NSP)-red.svg>)](https://en.wikipedia.org/wiki/Nurse_scheduling_problem)
[![Field](https://img.shields.io/badge/Field-Operations%20Research-purple.svg)](https://en.wikipedia.org/wiki/Operations_research)
[![Optimization](https://img.shields.io/badge/Optimization-CSP%20%26%20Metaheuristics-informational.svg)]()

**20 algorithms for hospital shift rostering, benchmarked head-to-head on 1,036
instances.** A constraint-satisfaction and combinatorial-optimisation study:
which techniques actually earn their keep on a hard scheduling problem, and
which impressive-sounding ones do not.

Five solvers reach **208/208** on the benchmark; the weakest reaches 150/208.
None ever emits an invalid roster. The single biggest improvement in the
project came not from a better search, but from correcting one unsound bound.

---

## The problem

Assign each of `N` nurses one code per day across `D` days:
`M` morning · `A` afternoon · `E` evening · `R` rest · `B` double shift
(morning + afternoon, surgical nurses only, costs 2 budget units).

Eight hard rules must hold simultaneously — exact daily coverage (`#M+#B = m`,
`#A+#B = a`, `#E = e`), forbidden shift transitions, at most 5 consecutive
working days, per-nurse workload caps, leave days, and at least one double
shift on every surgical day. **Instances can be genuinely infeasible** (301 of
1,036 are), so correctly answering "impossible" is part of the problem.

Two variants:

- **Feasibility** — find any valid roster, or prove none exists. _18 solvers._
- **Optimisation** — among valid rosters, minimise
  `Σ 3(M² + A² + E²) − (M + A + E)²`, which rewards giving each nurse a
  balanced _mix_ of shift types. _2 solvers._

---

## Headline results

| Rank | Solver                          | Technique                           | Solved      | Mean time |
| ---- | ------------------------------- | ----------------------------------- | ----------- | --------- |
| 1    | `constraint_propagation_luby`   | CP + Luby restarts + nogood caching | **208/208** | 0.05 s    |
| 2    | `simulated_annealing`           | Annealing over row-feasible rosters | **208/208** | 0.12 s    |
| 3    | `branch_and_bound_setwise`      | Set-wise branching + global bounds  | **208/208** | 0.12 s    |
| 4    | `portfolio_randomised_restarts` | Two-phase portfolio + randomisation | **208/208** | 0.13 s    |
| 5    | `network_flow_matching`         | Max-flow per day + DFS across days  | **208/208** | 0.13 s    |
| …    |                                 |                                     |             |           |
| 17   | `smt_encoding_z3`               | Declarative SMT, solved by z3       | 172/208     | 2.78 s    |
| 18   | `dynamic_programming_exact`     | Exact DP over aggregated states     | 150/208     | 2.45 s    |

Full table, per-suite breakdowns and analysis: **[REPORT.md](REPORT.md)**.

---

## Six findings

**1. Feasibility never needed search.** A handful of arithmetic necessary
conditions — capacity vs. demand, double-shift supply vs. surgical days, rest
ceilings — decide **all 301** infeasible instances, with **zero** false claims
on the 735 feasible ones. The 30% of the corpus that looks like it demands an
impossibility proof is settled before any search begins.

**2. An unsound bound is worse than a missing one.** A natural-looking capacity
check ignored that a double shift covers _two_ demand units while occupying
_one_ calendar day. It under-counted surgical nurses and pruned valid
schedules — including the reference solution's own path on a tight instance.
Fixing it took one solver from **4/12 to 12/12** on the hard suites and from
29 s to 35 ms. It failed silently, as a wrong answer rather than a slow one.

**3. Branch on sets, not on individuals.** Solvers that choose _which set of
nurses_ fills a quota beat those that assign nurses one at a time, because
interchangeable nurses otherwise generate `k!` redundant branches. Every solver
in the top eight branches set-wise; every solver in the bottom seven does not.
This is the largest structural effect in the study.

The control is telling: adding adjacent-pair symmetry breaking to a per-nurse
solver changed its score by **exactly zero** (195 → 195). Patching a symmetric
search is no substitute for not generating the symmetry.

**4. Strategy dominates representation.** Swapping domain lists for bitsets —
identical search, better data structures — bought 3 cases (179 → 182).
Switching the _branching decision itself_ from individual nurses to sets bought
21 (187 → 208, best-of-family to best-of-family). Optimising the representation
of a bad search is a rounding error next to fixing the search.

**5. General-purpose tools lost on encoding cost.** The SMT encoding expands an
`N=50, D=30` instance into ~40,000 assertions and 1.8 MB of text that must be
generated, parsed and only then solved — inside the same time budget everything
else gets. It scores 0/5 on the dense suite. The exact DP can produce genuine
infeasibility _proofs_, which nothing else here can — but on this corpus it
produced **zero**, because finding 1 had already settled every such case.

**6. Speed and quality are different objectives.** Among solvers that all reach
208/208, the rosters they produce differ by **57%** on the fairness objective.
A solver tuned to find _any_ answer fast reliably finds a lopsided one.

### On the optimisation variant

Both solvers match the reference objective on almost every feasible instance —
201/202 and 198/201 — and **neither ever beats it**, suggesting the reference is
at or near optimal and that both are saturating the objective.

They differ in the time/quality trade: iterated local search misses on 1 case
but runs **2–6× slower**, because it periodically tears out and rebuilds part
of the roster — escaping basins that the annealer's single-cell moves cannot
reach. The annealer misses on 3, by 0.1–0.3% of objective, for a fraction of
the runtime.

---

## The 20 solvers, and what each one does

Grouped by family. Score is out of 208 runs (all of `suite_002/003/004` plus a
1-in-5 sample of `suite_001`).

### Systematic backtracking — assign one nurse at a time

All five walk the grid day by day, nurse by nurse, and backtrack on failure.
They differ in what they know before committing to a code. Their shared
weakness: interchangeable nurses generate `k!` equivalent branches.

| Solver                           | Score | What it does                                                                                                                                                                                                      |
| -------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `backtracking_forward_checking`  | 180   | The baseline. Three cheap prunes before each nurse — total capacity, per-shift starvation, surgical-`B` availability — and codes tried in order of how badly the day still needs them (least-constraining-value). |
| `backtracking_bitset_domains`    | 180   | Identical search, but each nurse's legal codes are a 5-bit mask recomputed once per day instead of a rebuilt list. Pure constant-factor change — same tree, ~4% faster, same score.                               |
| `backtracking_value_ordering`    | 195   | Adds a scarcity signal: if a nurse's remaining budget no longer covers every day they could work, rest is tried _first_ rather than last. +15 over the baseline, for almost no cost.                              |
| `backtracking_symmetry_breaking` | 195   | Adds a rank-order rule between _adjacent_ interchangeable nurses. Scores identically to the solver it extends — see finding 3.                                                                                    |
| `backtracking_dynamic_ordering`  | 187   | Re-sorts nurses (least loaded first) at every day boundary and biases surgical nurses to conserve budget for future `B` shifts. The strongest per-nurse variant, and still below every set-wise one.              |

### MRV — choose the most constrained nurse next

| Solver                   | Score | What it does                                                                                                                                     |
| ------------------------ | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mrv_heuristic_search`   | 179   | Textbook minimum-remaining-values: at each step pick the nurse with fewest legal codes left. Domains rebuilt as lists at every node.             |
| `mrv_bitset_incremental` | 182   | Same search with bitmask domains and popcount to find the minimum. The +3 over the list version is the whole value of the representation change. |

MRV is the right instinct at the wrong granularity here — choosing _which nurse_
to assign next still explores orderings of nurses that are interchangeable.

### Set-wise branching — choose which _set_ fills a quota

The structural fix. Order within a chosen set is never explored because it is
never represented.

| Solver                        | Score   | What it does                                                                                                                                                                                                                                                                                                                                                                                                            |
| ----------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `beam_search`                 | 206     | Fixes how many `B` shifts a day gets, then picks which set of nurses fills each quota, least-loaded first. The beam caps how many candidate sets are tried per quota.                                                                                                                                                                                                                                                   |
| `branch_and_bound_setwise`    | **208** | Adds MRV over _quotas_ (fill whichever of B/M/A/E has least slack first) and global bounds over the whole remaining horizon, not just tomorrow.                                                                                                                                                                                                                                                                         |
| `constraint_propagation_luby` | **208** | Fastest solver at 0.05 s. Hall-style counting propagation prunes a day before branching; a nogood cache keyed on a symmetry-canonical state signature stops re-refuting equivalent states; Luby-scheduled restarts reshuffle the nurse order. Small pools are enumerated exhaustively, large ones sampled — and it only caches a nogood when the subtree was genuinely exhaustive, since "no luck" is not "impossible". |

### Polynomial subproblem — decide the day instead of searching it

| Solver                  | Score   | What it does                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `network_flow_matching` | **208** | Models each day as bipartite max-flow: source → nurse (cap 1) → eligible code (cap 1) → sink (cap = quota). A saturating flow exists iff the day is coverable, so days are _decided_, not searched. Backtracking happens only across days and over how many `B` shifts to place. Since a flow returns _some_ matching with no view of tomorrow, it retries under different nurse orderings to get genuinely different matchings. |

### Local search — repair a roster instead of building one

Both keep every per-nurse rule invariant and let only daily coverage break, so
the cost function is coverage error alone.

| Solver                       | Score   | What it does                                                                                                                                                                                                                              |
| ---------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `local_search_min_conflicts` | 207     | Greedy seed, then repeatedly retimes a cell in a broken day to whatever most reduces cost. Tabu list plus occasional random-walk steps to escape plateaus, and a pair-swap move that redistributes budget without changing column counts. |
| `simulated_annealing`        | **208** | Same state space, same cost function, same moves — accepts a worsening move with probability `exp(-Δ/T)`, geometric cooling, reheat on stagnation. The acceptance rule alone is the difference between 207 and 208.                       |

### Decomposition and declarative

| Solver                       | Score | What it does                                                                                                                                                                                                                                                                                                   |
| ---------------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `column_generation_patterns` | 196   | Chooses a whole legal _pattern_ per nurse (a length-`D` string valid by construction), so row rules are free and all difficulty sits in the covering problem. Patterns are priced against residual demand — the reduced-cost intuition from branch-and-price — then repaired by repricing one nurse at a time. |
| `dynamic_programming_exact`  | 150   | Aggregates nurses into _profiles_ — `(surgical, budget spent, streak, previous code, remaining leave)` — so the DP state is a multiset over profiles, not an assignment to individuals. Exhausting it is a genuine infeasibility proof. Guards abandon the DP when the profile space explodes.                 |
| `smt_encoding_z3`            | 172   | Declares every rule as SMT-LIB assertions and hands the lot to z3. No search strategy of our own — the point is to see what a general-purpose solver does when simply told the rules.                                                                                                                          |

### Portfolios

| Solver                          | Score   | What it does                                                                                                                                                                                                                                      |
| ------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `portfolio_two_phase`           | 207     | Runs beam search under a short slice of the budget, falls back to per-nurse DFS with full lookahead for the remainder. The two failure modes are near-complementary.                                                                              |
| `portfolio_randomised_restarts` | **208** | Same two phases plus randomisation — the fallback shuffles its nurse order with decaying probability, so a failed attempt does not retry the same doomed ordering. Fast, but produces the _least balanced_ rosters of any solver (see finding 6). |

### Optimisation variant

Both construct a feasible roster first, then improve it under the remaining
budget. Neither ever beat the reference objective, suggesting it is at or near
optimal on these instances.

| Solver                  | Result                 | What it does                                                                                                                                                                                                                                                                           |
| ----------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `annealing_objective`   | matched ref on 198/201 | Annealing directly on the fairness objective, with several move classes (single retiming, pairwise swap, multi-day rewrite), geometric cooling and reheats. The objective decomposes per nurse, so evaluating a move is O(1) — which is what makes the move count high enough to work. |
| `iterated_local_search` | matched ref on 201/202 | Alternates hill-climbing with _perturbation_: tear out part of the roster and rebuild it with an MRV fill, then climb again, keeping the incumbent best. Reconstruction escapes basins single-cell moves cannot — at 2–6× the runtime.                                                 |

---

## What did not work, and why

Reported as results rather than tuned away — a technique failing on a class of
instances is information about the problem's structure.

| Approach            | Outcome     | Why                                                                                      |
| ------------------- | ----------- | ---------------------------------------------------------------------------------------- |
| SMT / z3            | 172/208     | Encoding and parsing dominate the budget; 0/5 on dense instances                         |
| Exact DP            | 150/208     | State space explodes with distinct leave patterns; its proof capability was never needed |
| Column generation   | 196/208     | Exact column sums + independently sampled patterns strand demand on dense instances      |
| Min-conflicts       | 207/208     | Greedy seed overspends budget; repair moves that would free it are themselves blocked    |
| Per-nurse MRV / DFS | 179–187/208 | Budget spent re-deriving equivalent orderings of interchangeable nurses                  |

One instructive near-miss: min-conflicts and simulated annealing use the
**same** state space, cost function and move set, differing only in the
acceptance rule — and score 207 vs. 208. On this problem the acceptance rule,
not the neighbourhood, escapes local optima.

Also measured: fixing a genuine transition-table bug (`A → B` is legal, three
solvers omitted it) changed exactly **one** case in a 200-instance sample. It
was a real incompleteness, but those solvers were already losing on time — a
correctness bug and a performance bug are not the same thing, and only one of
them showed up in the score.

---

## Layout

```
finding_solutions/     18 feasibility solvers + _common.py (shared parsing, bounds)
optimising the cost/    2 optimisation solvers
tests/                  pytest suites (vendored verifier + a sampled test corpus)
benchmark.py            solve rate, timing, verdict classification
benchmark_b.py          objective value vs. reference
compare_quality.py      fairness objective of any solver's rosters
REPORT.md               full write-up
```

Every solver is standalone and stdlib-only (`smt_encoding_z3.py` also wants a
`z3` binary):

```bash
python finding_solutions/constraint_propagation_luby.py instance.csv out.json
```

## Tests

`tests/` is a self-contained pytest suite: it vendors the verifier and the full
1,036-instance corpus under `tests/data/` (suite_001 is sampled at test time,
stride 25, to stay fast; suite_002/003/004 run in full). Each test runs one
solver against one instance as a subprocess and checks the result against the
bundled reference solution.

```bash
python -m pytest tests/
python -m pytest tests/test_suite_003.py -k constraint_propagation_luby
```

## Benchmarking

The benchmark scripts use the same vendored verifier and corpus under
`tests/data/` — nothing to clone or set up separately:

```bash
python benchmark.py --all --suite suite_001 --stride 5 --jobs 15
python benchmark_b.py --suite suite_002 --jobs 15
python compare_quality.py --suite suite_003
```

The solvers themselves are standalone and need none of this.

The harness distinguishes outcomes a pass/fail runner conflates:

| Verdict     | Meaning                                                      |
| ----------- | ------------------------------------------------------------ |
| `OK`        | valid roster, or correctly identified an infeasible instance |
| `MISSED`    | gave up on a solvable instance — _weak_                      |
| `WRONG_SAT` | emitted a rule-violating roster — _broken_                   |

That distinction matters: a harness that only asks "is the output valid" would
score a solver that always answers "impossible" as largely correct. **No solver
here ever produced a `WRONG_SAT`** — they differ only in how often they give up,
and how fast.

Every number in this repository was produced by that external verifier, so the
results reflect an independent ground truth rather than self-agreement.
