"""Simulated annealing over row-feasible rosters.

Idea
    Same state space as the min-conflicts solver - every candidate roster
    already satisfies all the per-nurse row rules, and only the daily coverage
    quotas may be violated - but a different way of moving through it.

    Min-conflicts is greedy: it takes the best move it can see and needs a
    tabu list and random walk steps to escape plateaus.  Annealing instead
    accepts a worsening move with probability exp(-delta / T) and cools T on a
    geometric schedule, so early on it wanders freely and late on it behaves
    like hill climbing.  Reheating on stagnation gives it several descents
    inside one time budget.

    The comparison is the point: the two share a neighbourhood and a cost
    function and differ only in the acceptance rule, which isolates the effect
    of that rule on this problem.

Completeness
    Incomplete and stochastic.  {} is only ever emitted as a claim of
    infeasibility by the analytic pre-checks, never from a failed descent.
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (ALLOWED_AFTER, COST, MAX_CONSECUTIVE, empty_output,
                     infeasible_reason, parse_input, write_roster)


class Annealer:
    def __init__(self, inst, deadline, seed=99991):
        self.inst = inst
        self.deadline = deadline
        self.rng = random.Random(seed)
        self.grid = [["R"] * inst.D for _ in range(inst.N)]
        self.worked = [0] * inst.N
        self.cnt = {c: [0] * inst.D for c in ("M", "A", "E", "B", "R")}

    # ---- row invariant ------------------------------------------------------
    def row_ok(self, n, d, code):
        inst = self.inst
        if inst.on_leave[n][d]:
            return code == "R"
        if code == "B" and (n >= inst.Ns or not inst.is_surg_day[d]):
            return False
        row = self.grid[n]
        if self.worked[n] - COST[row[d]] + COST[code] > inst.K:
            return False
        if d > 0 and code not in ALLOWED_AFTER[row[d - 1]]:
            return False
        if d + 1 < inst.D and row[d + 1] not in ALLOWED_AFTER[code]:
            return False
        if code != "R":
            saved = row[d]
            row[d] = code
            run = 0
            ok = True
            lo = max(0, d - MAX_CONSECUTIVE)
            hi = min(inst.D - 1, d + MAX_CONSECUTIVE)
            for j in range(lo, hi + 1):
                run = 0 if row[j] == "R" else run + 1
                if run > MAX_CONSECUTIVE:
                    ok = False
                    break
            row[d] = saved
            if not ok:
                return False
        return True

    def place(self, n, d, code):
        old = self.grid[n][d]
        if old == code:
            return
        self.worked[n] += COST[code] - COST[old]
        self.cnt[old][d] -= 1
        self.cnt[code][d] += 1
        self.grid[n][d] = code

    # ---- cost ---------------------------------------------------------------
    def day_cost(self, d):
        inst = self.inst
        cb = self.cnt["B"][d]
        cost = (abs(self.cnt["M"][d] + cb - inst.m)
                + abs(self.cnt["A"][d] + cb - inst.a)
                + abs(self.cnt["E"][d] - inst.e))
        if inst.is_surg_day[d]:
            if cb == 0:
                cost += 3
        elif cb:
            cost += 3 * cb
        return cost

    def total_cost(self):
        return sum(self.day_cost(d) for d in range(self.inst.D))

    def seed_roster(self):
        inst = self.inst
        self.grid = [["R"] * inst.D for _ in range(inst.N)]
        self.worked = [0] * inst.N
        self.cnt = {c: [0] * inst.D for c in ("M", "A", "E", "B", "R")}
        for d in range(inst.D):
            self.cnt["R"][d] = inst.N
        for d in range(inst.D):
            wanted = []
            if inst.is_surg_day[d] and inst.m and inst.a:
                wanted.append("B")
            nb = 1 if wanted else 0
            wanted += ["M"] * (inst.m - nb) + ["A"] * (inst.a - nb) + ["E"] * inst.e
            taken = set()
            for code in wanted:
                best, best_key = None, None
                for n in range(inst.N):
                    if n in taken or not self.row_ok(n, d, code):
                        continue
                    key = (self.worked[n], self.rng.random())
                    if best_key is None or key < best_key:
                        best, best_key = n, key
                if best is not None:
                    self.place(best, d, code)
                    taken.add(best)
        return self.total_cost()

    # ---- annealing ----------------------------------------------------------
    def anneal(self, cost):
        inst = self.inst
        temp = self.t0
        best = cost
        since_best = 0
        steps = 0
        while cost > 0:
            steps += 1
            if not (steps & 255):
                if time.time() > self.deadline:
                    raise TimeoutError
            n = self.rng.randrange(inst.N)
            d = self.rng.randrange(inst.D)
            cur = self.grid[n][d]
            code = self.rng.choice(("M", "A", "E", "R", "B"))
            if code == cur or not self.row_ok(n, d, code):
                continue

            before = self.day_cost(d)
            self.place(n, d, code)
            delta = self.day_cost(d) - before

            if delta <= 0 or self.rng.random() < math.exp(-delta / temp):
                cost += delta                      # accept
            else:
                self.place(n, d, cur)              # reject, roll back

            if cost < best:
                best, since_best = cost, 0
            else:
                since_best += 1

            temp *= self.cooling
            if temp < self.t_min or since_best > self.stagnation:
                # reheat: a fresh descent from the current point
                temp = self.t0
                since_best = 0
        return cost

    def run(self):
        inst = self.inst
        self.t0 = 1.5
        self.t_min = 0.02
        self.cooling = 1.0 - 1.0 / max(400.0, 20.0 * inst.N * inst.D)
        self.stagnation = max(3000, 40 * inst.N * inst.D)
        while time.time() < self.deadline:
            cost = self.seed_roster()
            if cost == 0 or self.anneal(cost) == 0:
                return self.grid
        return None


def main():
    if len(sys.argv) < 3:
        print("usage: simulated_annealing.py <input_csv_path> <output_json_path>")
        return 1
    inp, outp = sys.argv[1], sys.argv[2]
    inst = parse_input(inp)
    deadline = time.time() + max(1.0, inst.T - 1.0)

    if infeasible_reason(inst) is not None:
        empty_output(outp)
        return 0

    try:
        grid = Annealer(inst, deadline).run()
    except TimeoutError:
        grid = None
    if grid is None:
        empty_output(outp)
    else:
        write_roster(grid, outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
