# SmartRoster — methods and results

*Manya Jain*

A comparative study of eighteen feasibility algorithms and two optimisation
algorithms on the same hospital shift-rostering problem, measured on 1,036
instances against an independent verifier.

---

## 1. Problem

Assign every nurse one code per day over a `D`-day horizon.

| Code | Meaning | Budget cost |
| --- | --- | --- |
| `M` | morning | 1 |
| `A` | afternoon | 1 |
| `E` | evening | 1 |
| `R` | rest | 0 |
| `B` | morning **and** afternoon, surgical nurses only | 2 |

**Hard rules.** A roster is valid when every one of these holds:

| # | Rule |
| --- | --- |
| H1 | one code per nurse-day; `B` only for nurses `< N_s` |
| H2 | a day marked `L` must be `R` |
| H3 | no `M`/`B` after `M`/`B`, and none after `E` |
| H4 | the day after a `B` is `R` or `E` |
| H5 | never six consecutive worked days |
| H6 | each day exactly: `#M + #B = m`, `#A + #B = a`, `#E = e` |
| H7 | each `S` day has ≥1 `B`; each `G` day has none |
| H8 | per nurse, worked days + one extra per `B` ≤ `K` |

**Two decision problems.** *Part A* asks only for a valid roster, or `{}` if
none exists. *Part B* keeps every hard rule and additionally minimises

```
objective = Σ_nurses  3·(M² + A² + E²) − (M + A + E)²
```

which is zero when a nurse's three shift-type counts are equal and grows as
they skew. It rewards giving each nurse a *balanced mix* of shift types, not
merely a fair total workload.

### Two rules that are easy to get wrong

Both cost real solve rate in this study.

**`A → B` is legal.** H3 and H4 forbid `M/B → M/B`, `E → M/B`, and
`B → anything but R/E`. Nothing forbids an afternoon followed by a double
shift. Three of the original solvers encoded a transition table that omitted
it, making them unable to represent some legal schedules.

**A `B` covers two demand units but occupies one calendar day.** Capacity
bounds of the form "remaining demand ≤ Σ min(budget left, days available)"
*under-count* surgical nurses, because one of their days can absorb two units
of demand. A bound that is wrong in this direction prunes valid schedules —
see §5.2.

---

## 2. The corpus

| Suite | Instances | Feasible | Infeasible | `N` | `D` | `K` | demand/day | budget `T` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `suite_001` | 1000 | 700 | 300 | 3–50 | 1–30 | 0–31 | 0–16 | 3–10 s |
| `suite_002` | 24 | 24 | 0 | 40–50 | 30 | 4–21 | 3–9 | 20 s |
| `suite_003` | 9 | 8 | 1 | 9–50 | 6–30 | 2–22 | 3–30 | 30 s |
| `suite_004` | 3 | 3 | 0 | 14–50 | 13–30 | 11–16 | 11–19 | 30–60 s |

`suite_001` is broad and mostly easy but is 30% infeasible, so it tests
*deciding* as much as solving. `suite_002` and `suite_004` are the dense
large instances. `suite_003` mixes a few genuinely tight instances with easy
ones.

## 3. Method

`benchmark.py` runs each solver as a subprocess with the instance's own time
budget `T` (plus 2 s of slack), validates the output with the bundled
`tests/verifier.py`, and classifies the outcome:

| Verdict | Meaning |
| --- | --- |
| `OK` | valid roster, or correctly reported an infeasible instance |
| `MISSED` | returned `{}` for an instance that is in fact solvable |
| `WRONG_SAT` | returned a roster that violates a hard rule |
| `TIMEOUT` | wall clock exceeded with nothing written |

This split matters. A `MISSED` is a solver being *weak*; a `WRONG_SAT` is a
solver being *broken*. A pass/fail harness that only asks "did it produce
valid output" conflates the two, and would rate a solver that always answers
`{}` as substantially correct.

Results below use every case of `suite_002`, `suite_003` and `suite_004`, and
a deterministic 1-in-5 sample of `suite_001` (200 cases), for 208 runs per
solver. Machine: 16 cores, Python 3.14.

---

## 4. Part A results

Ordered by total solved. Per-suite cells are `solved / attempted`.

| Solver | s001 | s002 | s003 | s004 | total | mean s | max s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `constraint_propagation_luby.py` | 200/200 | 5/5 | 2/2 | 1/1 | **208/208** | 0.05 | 0.1 |
| `simulated_annealing.py` | 200/200 | 5/5 | 2/2 | 1/1 | **208/208** | 0.12 | 1.2 |
| `branch_and_bound_setwise.py` | 200/200 | 5/5 | 2/2 | 1/1 | **208/208** | 0.12 | 0.4 |
| `portfolio_randomised_restarts.py` | 200/200 | 5/5 | 2/2 | 1/1 | **208/208** | 0.13 | 0.3 |
| `network_flow_matching.py` | 200/200 | 5/5 | 2/2 | 1/1 | **208/208** | 0.13 | 0.5 |
| `portfolio_two_phase.py` | 200/200 | 5/5 | 2/2 | 0/1 | 207/208 | 0.25 | 28.6 |
| `local_search_min_conflicts.py` | 200/200 | 5/5 | 2/2 | 0/1 | 207/208 | 0.27 | 29.3 |
| `beam_search.py` | 200/200 | 5/5 | 1/2 | 0/1 | 206/208 | 0.21 | 11.1 |
| `column_generation_patterns.py` | 192/200 | 2/5 | 2/2 | 0/1 | 196/208 | 0.95 | 29.2 |
| `backtracking_symmetry_breaking.py` | 191/200 | 3/5 | 1/2 | 0/1 | 195/208 | 0.97 | 28.1 |
| `backtracking_value_ordering.py` | 191/200 | 3/5 | 1/2 | 0/1 | 195/208 | 0.98 | 28.1 |
| `backtracking_dynamic_ordering.py` | 186/200 | 0/5 | 1/2 | 0/1 | 187/208 | 1.24 | 28.6 |
| `mrv_bitset_incremental.py` | 178/200 | 3/5 | 1/2 | 0/1 | 182/208 | 1.56 | 28.2 |
| `backtracking_bitset_domains.py` | 180/200 | 0/5 | 0/2 | 0/1 | 180/208 | 1.72 | 28.1 |
| `backtracking_forward_checking.py` | 180/200 | 0/5 | 0/2 | 0/1 | 180/208 | 1.78 | 28.1 |
| `mrv_heuristic_search.py` | 175/200 | 3/5 | 1/2 | 0/1 | 179/208 | 1.65 | 28.3 |
| `smt_encoding_z3.py` | 171/200 | 0/5 | 1/2 | 0/1 | 172/208 | 2.78 | 28.5 |
| `dynamic_programming_exact.py` | 149/200 | 1/5 | 0/2 | 0/1 | 150/208 | 2.45 | 29.1 |

**No solver ever produced a `WRONG_SAT` or a `TIMEOUT`.** Every failure in the
table is a `MISSED` — the solver ran out of time and honestly reported `{}`.
The spread is entirely in solve rate and speed.

Five solvers clear the corpus. The gap between them and the bottom of the
table is a factor of ~1.4 in solve rate and ~50 in mean time.

---

## 5. What actually mattered

### 5.1 Feasibility never needed search

`_common.infeasible_reason` applies a handful of arithmetic necessary
conditions: demand exceeding headcount, a surgical day with no available
surgical nurse, total capacity below total demand (with H5's rest ceiling
folded in), and B-supply below the number of surgical days.

> These checks decide **301 of 301** infeasible instances in the corpus, with
> **zero** false claims on the 735 feasible ones.

This reframes the problem. 30% of `suite_001` looks like it demands a solver
capable of *proving* impossibility; in practice arithmetic settles all of it
before any search starts. Every solver here calls these checks first, which is
why none of them ever mistakes a hard instance for an impossible one — and why
the exact DP's ability to produce real proofs (§5.5) turns out to be worth
nothing on this data.

### 5.2 A wrong bound is worse than no bound

While building `constraint_propagation_luby.py` its lookahead used the natural
capacity bound

```
remaining demand  ≤  Σ_i min(K − worked_i , workable_days_i)
```

This is **wrong**, and wrong in the dangerous direction. A `B` consumes one
calendar day but covers two demand units, so the right-hand side under-counts
what surgical nurses can still deliver. On `suite_003/test8` — 3 surgical
nurses, `K=2`, surgical days at both ends — the bound computed `cap=2 < need=3`
after day 4 and pruned the *reference solution's own path*.

The instance is feasible, and the solver reported `{}`.

Crediting the double coverage:

```python
cap += min(Σ_surgical min((K − worked_i)//2, workable_i),
           surgical_days_remaining × min(m, a))
```

took that solver from **4/12 to 12/12** on `suite_003` + `suite_004`, and from
a 29-second average to 35 milliseconds. It is the single largest improvement in
the project, and it came from fixing a bound rather than from any change of
search strategy.

The lesson generalises: an unsound prune is far more damaging than a missing
one, because it converts "slow" into "wrong" silently.

### 5.3 Branch on sets, not on nurses

The clearest structural divide in the results table is *what a branching
decision commits to*.

- **Per-nurse branching** (`backtracking_*`, `mrv_*`): pick a nurse, pick their
  code, recurse. Nurses that are interchangeable — same class, same budget,
  same streak, same history — generate `k!` distinct branches for the same
  underlying schedule.
- **Set-wise branching** (`beam_search`, `branch_and_bound_setwise`,
  `constraint_propagation_luby`): pick a quota, then pick *which set* of nurses
  fills it. Order within the set is never explored because it is never
  represented.

Every solver in the top eight branches on sets or avoids explicit branching
altogether; every solver in the bottom seven branches on nurses.

Adjacent-pair symmetry breaking is the instructive control here.
`backtracking_symmetry_breaking` adds a rank-ordering rule between neighbouring
interchangeable nurses on top of `backtracking_value_ordering` — and scores
**exactly the same, 195/208**. Cancelling symmetry between *adjacent* nurses in
the sweep addresses a vanishing fraction of a `k!`-way redundancy; only
branching on sets removes it structurally. Trying to patch a symmetric search
is not a substitute for not generating the symmetry.

### 5.4 Representation is a constant factor; strategy is not

`mrv_heuristic_search` and `mrv_bitset_incremental` run the identical search
and differ only in whether domains are Python lists or 5-bit masks. The bitset
version is faster and solves 3 more cases (179 → 182). The same comparison
holds for `backtracking_forward_checking` vs `backtracking_bitset_domains`
(180 → 180).

Against that, switching the branching decision from individual nurses to sets
moves the best of the per-nurse family (187) to the best of the set-wise family
(208) — 21 cases. Optimising the representation of a bad search buys single
digits; changing what a branch commits to buys tens.

### 5.5 Approaches that did not pay off

These are findings, not defects. Each says something about the fit between a
technique and this problem's structure.

**SMT (`smt_encoding_z3.py`) — 172/208.** Declaring the rules and handing them
to z3 is the least code and the most general approach. It is also dominated by
encoding cost: an `N=50, D=30` instance expands to roughly **40,000 assertions
and 1.8 MB** of SMT-LIB text, which must be generated, serialised, parsed, and
only then solved, inside the same wall clock everything else gets. The pairwise
at-most-one encoding of "one code per nurse-day" and the sliding six-day rest
windows dominate that bulk. It solves 0/5 on `suite_002`. A native API using a
cardinality encoding instead of pairwise clauses would fare better; as a
text-interface encoding, it does not.

**Exact DP (`dynamic_programming_exact.py`) — 150/208, last place.** Nurses are
aggregated into *profiles* — `(surgical, budget spent, streak, previous code,
remaining leave pattern)` — so the DP state is a multiset over profiles rather
than an assignment to individuals. Exhausting that state space is a genuine
infeasibility proof, which no other solver here can offer.

That guarantee is worth nothing on this corpus. The profile count explodes with
the number of distinct leave patterns, so it abandons most large instances; and
on every instance where it *could* have proved infeasibility, §5.1's arithmetic
had already done so. Measured directly: over 120 instances not already settled
by the pre-checks, it solved 77, gave up on 42, and produced **zero**
infeasibility proofs.

Its value is negative information — it marks the boundary where exhaustive
reasoning stops being affordable, which the heuristic solvers otherwise hide.

**Column generation (`column_generation_patterns.py`) — 196/208.** Choosing a
whole legal *pattern* per nurse makes the row rules (H2–H5, H8) free by
construction, concentrating all difficulty in the covering problem. It is
strong on mid-size instances (9/9 on `suite_003`) and weak exactly where
coverage binds hardest: 2/5 on `suite_002`. Because H6 demands *exact* column
sums and patterns are sampled independently, dense instances strand demand that
no single pattern swap repairs. Making the tallies incremental raised
`suite_002` from 2/5 to 15/24 on the full suite; a targeted repair that focused
on broken days made it *worse* and was reverted.

**Min-conflicts (`local_search_min_conflicts.py`) — 207/208.** Strong overall,
but it fails on tight-`K` instances in a specific way: the greedy seed
overspends budget early, and every repair move that would free budget is itself
blocked by the row rules. A pair-swap move (exchanging two nurses' codes within
a day, which leaves column counts untouched but redistributes budget) helped
marginally.

The instructive comparison is with `simulated_annealing.py`, which uses the
**same** state space, the **same** cost function and the **same** move set, and
differs only in the acceptance rule — and scores 208/208. On this problem the
acceptance rule, not the neighbourhood, is what escapes the local optima.

**The `A → B` fix barely moved the numbers.** Correcting the transition table in
the three affected solvers changed **one** case in a 200-instance sample. It is
a real incompleteness — those solvers could not express certain legal schedules
— but they were already losing on time, so it never became the binding
constraint. A correctness bug and a performance bug are different things, and
only one of them shows up in an aggregate score.

### 5.6 Schedule quality is not correlated with solve rate

Part A treats every valid roster as equally correct, but the rosters differ
sharply in fairness. Measuring the Part B objective on `suite_003`:

| Solver | cases solved | mean objective |
| --- | --- | --- |
| `simulated_annealing.py` | 8 | 724 |
| `column_generation_patterns.py` | 8 | 827 |
| `constraint_propagation_luby.py` | 8 | 834 |
| `network_flow_matching.py` | 8 | 869 |
| `branch_and_bound_setwise.py` | 8 | 938 |
| `portfolio_two_phase.py` | 8 | 1072 |
| `portfolio_randomised_restarts.py` | 8 | 1313 |

(Lower is better. Solvers that solved fewer cases are omitted — their means are
computed over an easier subset and are not comparable.)

`portfolio_randomised_restarts` and `constraint_propagation_luby` both solve
every instance, but the former's rosters score **57% worse**. A solver tuned to
find *any* answer fast tends to find a lopsided one: it assigns the first
eligible nurse repeatedly rather than spreading shift types around. This is
precisely the gap Part B exists to close.

---

## 6. Part B results

Part B keeps all hard rules and minimises the fairness objective. Both solvers
construct a feasible roster first, then improve it under the remaining budget.

| Solver | Suite | matched ref | better | worse | mean gap | mean s |
| --- | --- | --- | --- | --- | --- | --- |
| `annealing_objective.py` | `suite_002` | 22/24 | 0 | 2 | +0.3% | 7.9 |
| `annealing_objective.py` | `suite_003` | 8/8 | 0 | 0 | 0.0% | 1.4 |
| `annealing_objective.py` | `suite_004` | 3/3 | 0 | 0 | 0.0% | 6.5 |
| `iterated_local_search.py` | `suite_002` | 24/24 | 0 | 0 | 0.0% | 18.4 |
| `iterated_local_search.py` | `suite_003` | 8/8 | 0 | 0 | 0.0% | 11.5 |
| `iterated_local_search.py` | `suite_004` | 3/3 | 0 | 0 | 0.0% | 41.2 |
| `annealing_objective.py` | `suite_001` (250) | 173/174 | 0 | 1 | +0.1% | 2.7 |
| `iterated_local_search.py` | `suite_001` (250) | 174/175 | 0 | 1 | 0.0% | 3.9 |

Aggregated over every feasible instance run (277 runs each, of which 75 are
infeasible and correctly reported as such):

| Solver | matched reference | worse | better |
| --- | --- | --- | --- |
| `annealing_objective.py` | 198/201 | 3 | 0 |
| `iterated_local_search.py` | 201/202 | 1 | 0 |

Both reach the reference objective almost everywhere; **neither ever beats
it**, which suggests the reference is at or near optimal on these instances and
that both solvers are effectively saturating the objective.

The difference is the time/quality trade. `iterated_local_search` falls short
of the reference on just 1 of 202 feasible instances but takes **2–6× longer**;
`annealing_objective` is consistently faster and pays for it on 3, by a mean
gap of 0.1–0.3% of objective. Which is preferable depends entirely on whether
the time budget is binding — at these gap sizes, most of the time it is not.

The mechanism behind the difference mirrors §5.5: annealing explores with a
cooling schedule and single-cell moves, while iterated local search
periodically *tears out and rebuilds* part of the roster. Reconstruction
escapes basins that single-cell moves cannot reach — at proportionate cost.

---

## 7. Conclusions

1. **Check the cheap arithmetic first.** Necessary conditions settled every
   infeasible instance in the corpus. No search was ever required to prove
   impossibility.
2. **Get bounds right before making them tight.** One unsound capacity bound
   cost more solve rate than every heuristic in the project gained, and it
   failed silently — as a wrong answer, not a slow one.
3. **Branch on sets, not individuals.** Eliminating permutation symmetry
   structurally beat every attempt to cope with it heuristically.
4. **Strategy dominates representation.** Bitsets bought single-digit
   improvements; changing what a branch commits to bought tens.
5. **General-purpose tools lose on encoding cost.** The SMT and exact-DP
   approaches are the most powerful in principle and the weakest here.
6. **Speed and quality are separate objectives.** The fastest feasibility
   solvers produce some of the least balanced rosters.

## 8. Reproducing

The verifier and the full 1,036-instance corpus are vendored under
`tests/data/`, so reproducing these numbers needs no external setup:

```bash
python benchmark.py --all --suite suite_001 --stride 5 --jobs 15   # feasibility
python benchmark_b.py --suite suite_002 --jobs 15                  # optimisation
python compare_quality.py --suite suite_003                        # roster quality
```

The solvers in `finding_solutions/` and `optimising the cost/` are standalone
and stdlib-only. Each takes an instance CSV and an output path:

```bash
python finding_solutions/constraint_propagation_luby.py instance.csv out.json
```

Reported numbers use every case of `suite_002`, `suite_003` and `suite_004`,
plus a deterministic 1-in-5 sample of `suite_001`, on 16 cores under Python
3.14.

For a quick pytest-based check instead, see `tests/`
(`python -m pytest tests/`) — it samples `suite_001` at stride 25 to stay
fast, running the other suites in full.

Most stochastic solvers fix an explicit seed; `portfolio_randomised_restarts`
and `iterated_local_search` do not, so their individual runs vary slightly.
Because every solver here is anytime — it returns the best roster found within
the instance's own time budget — solve rates near the margin can move by a case
or two between runs, and timings depend on machine load. The ranking and the
size of the effects discussed below were stable across repeated runs.
