"""Pattern (column) based construction with a coverage LP-style repair.

Idea
    Instead of choosing a code per nurse-day, choose a whole *pattern* per
    nurse - a legal length-D string over {M,A,E,R,B} that already respects
    every row rule (leave marks, transitions, the 5-in-a-row cap, the budget
    K).  The roster is then feasible iff the chosen multiset of patterns sums,
    column by column, to the required (m, a, e) coverage with a B on every
    surgical day.

    This is the decomposition behind branch-and-price for rostering: the
    "pricing" step generates promising patterns, the "master" step picks a
    combination that meets coverage.  Solving the master exactly is an integer
    program, so here it is done by a residual-demand greedy - patterns are
    scored by how much of the *still uncovered* demand they absorb, which is
    the same reduced-cost intuition, followed by a repair pass that reprices
    and swaps individual nurses' patterns while coverage is still short.

Why it is interesting
    Row rules become free: a pattern is legal by construction, so the search
    never wastes time rediscovering the transition table.  The whole
    difficulty is concentrated in the covering problem.  The weakness is the
    mirror image - patterns are generated per nurse without knowing what the
    others will pick, so on tight instances the greedy master can strand
    demand that no single pattern swap can repair.

Completeness
    Incomplete.  Infeasibility is reported only from the analytic pre-checks.
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (ALLOWED_AFTER, COST, MAX_CONSECUTIVE, empty_output,
                     infeasible_reason, parse_input, write_roster)


class PatternSolver:
    def __init__(self, inst, deadline, seed=7):
        self.inst = inst
        self.deadline = deadline
        self.rng = random.Random(seed)

    def gen_pattern(self, n, weights):
        """Sample one legal pattern for nurse n, biased by `weights`.

        weights[d][code] says how badly day d still wants `code`; sampling
        proportionally to that is the pricing heuristic - it is what makes a
        newly generated column likely to be useful to the master problem."""
        inst = self.inst
        row = []
        prev = ""
        used = 0
        streak = 0
        for d in range(inst.D):
            if inst.on_leave[n][d]:
                row.append("R")
                prev, streak = "R", 0
                continue
            options = []
            for code in ALLOWED_AFTER[prev]:
                if code == "R":
                    continue
                if code == "B" and (n >= inst.Ns or not inst.is_surg_day[d]):
                    continue
                if used + COST[code] > inst.K:
                    continue
                if streak >= MAX_CONSECUTIVE:
                    continue
                w = weights[d].get(code, 0.0)
                if w > 0:
                    options.append((code, w))
            # always allow resting
            total = sum(w for _, w in options)
            if not options or total <= 0 or self.rng.random() < self.rest_bias:
                row.append("R")
                prev, streak = "R", 0
                continue
            pick = self.rng.random() * total
            acc = 0.0
            chosen = options[-1][0]
            for code, w in options:
                acc += w
                if pick <= acc:
                    chosen = code
                    break
            row.append(chosen)
            used += COST[chosen]
            streak += 1
            prev = chosen
        return row

    def tally(self, grid):
        """Column counts of the whole grid, computed once per rebuild."""
        inst = self.inst
        cnt = [dict(M=0, A=0, E=0, B=0) for _ in range(inst.D)]
        for n in range(inst.N):
            row = grid[n]
            for d in range(inst.D):
                c = row[d]
                if c != "R":
                    cnt[d][c] += 1
        return cnt

    def apply_row(self, cnt, row, sign):
        for d, c in enumerate(row):
            if c != "R":
                cnt[d][c] += sign

    def deficit_from(self, cnt):
        inst = self.inst
        total = 0
        for d in range(inst.D):
            c = cnt[d]
            cb = c["B"]
            total += abs(inst.m - (c["M"] + cb))
            total += abs(inst.a - (c["A"] + cb))
            total += abs(inst.e - c["E"])
            if inst.is_surg_day[d]:
                if cb == 0:
                    total += 3
            elif cb:
                total += 3 * cb
        return total

    def weights_from_cnt(self, cnt):
        inst = self.inst
        weights = []
        for d in range(inst.D):
            c = cnt[d]
            cb = c["B"]
            w = {"M": max(0.0, inst.m - (c["M"] + cb)),
                 "A": max(0.0, inst.a - (c["A"] + cb)),
                 "E": max(0.0, inst.e - c["E"])}
            w["B"] = (min(w["M"], w["A"]) + 2.0) if (inst.is_surg_day[d] and cb == 0) else 0.0
            weights.append(w)
        return weights

    def coverage(self, grid):
        """Per-day shortfall of the current assignment."""
        inst = self.inst
        short = []
        for d in range(inst.D):
            cm = ca = ce = cb = 0
            for n in range(inst.N):
                c = grid[n][d]
                if c == "M":
                    cm += 1
                elif c == "A":
                    ca += 1
                elif c == "E":
                    ce += 1
                elif c == "B":
                    cb += 1
            short.append((inst.m - (cm + cb), inst.a - (ca + cb),
                          inst.e - ce, cb, inst.is_surg_day[d]))
        return short

    def deficit(self, grid):
        """Scalar badness: unmet demand plus surplus plus B-rule breaches."""
        total = 0
        for dm, da, de, cb, surg in self.coverage(grid):
            total += abs(dm) + abs(da) + abs(de)
            if surg and cb == 0:
                total += 3
            if not surg and cb:
                total += 3 * cb
        return total

    def weights_from(self, grid, skip=None):
        """Residual demand, used as the pricing signal for new patterns."""
        inst = self.inst
        weights = []
        for d in range(inst.D):
            cm = ca = ce = cb = 0
            for n in range(inst.N):
                if n == skip:
                    continue
                c = grid[n][d]
                if c == "M":
                    cm += 1
                elif c == "A":
                    ca += 1
                elif c == "E":
                    ce += 1
                elif c == "B":
                    cb += 1
            w = {
                "M": max(0.0, inst.m - (cm + cb)),
                "A": max(0.0, inst.a - (ca + cb)),
                "E": max(0.0, inst.e - ce),
            }
            # a B is worth having exactly when the day is surgical and has none
            w["B"] = (min(w["M"], w["A"]) + 2.0) if (inst.is_surg_day[d] and cb == 0) else 0.0
            weights.append(w)
        return weights

    def build(self):
        inst = self.inst
        grid = [["R"] * inst.D for _ in range(inst.N)]
        cnt = [dict(M=0, A=0, E=0, B=0) for _ in range(inst.D)]
        # nurses that can cover B go first: surgical days are the scarce resource
        order = sorted(range(inst.N), key=lambda n: (n >= inst.Ns, self.rng.random()))
        for n in order:
            weights = self.weights_from_cnt(cnt)
            best, best_score = None, None
            for _ in range(self.samples):
                cand = self.gen_pattern(n, weights)
                score = 0.0
                for d, code in enumerate(cand):
                    if code != "R":
                        score += weights[d].get(code, 0.0)
                if best_score is None or score > best_score:
                    best, best_score = cand, score
            grid[n] = best
            self.apply_row(cnt, best, +1)
        self.cnt = cnt
        return grid

    def repair(self, grid):
        """Reprice one nurse at a time against the residual demand and keep
        the new pattern when it lowers the overall deficit."""
        inst = self.inst
        cnt = self.cnt
        best_def = self.deficit_from(cnt)
        stall = 0
        while best_def > 0 and stall < self.patience:
            if time.time() > self.deadline:
                raise TimeoutError
            n = self.rng.randrange(inst.N)
            saved = grid[n]
            self.apply_row(cnt, saved, -1)          # price nurse n out
            weights = self.weights_from_cnt(cnt)
            improved = False
            for _ in range(self.samples):
                cand = self.gen_pattern(n, weights)
                self.apply_row(cnt, cand, +1)
                cur = self.deficit_from(cnt)
                if cur < best_def:
                    best_def, saved = cur, cand
                    improved = True
                    self.apply_row(cnt, cand, -1)
                    if best_def == 0:
                        break
                else:
                    self.apply_row(cnt, cand, -1)
            grid[n] = saved
            self.apply_row(cnt, saved, +1)          # price the winner back in
            stall = 0 if improved else stall + 1
        return best_def

    def run(self):
        inst = self.inst
        self.patience = max(200, 8 * inst.N)
        while time.time() < self.deadline:
            # each outer round re-rolls the sampling temperature
            self.samples = 4 + self.rng.randrange(12)
            self.rest_bias = self.rng.uniform(0.02, 0.35)
            grid = self.build()
            if self.deficit_from(self.cnt) == 0 or self.repair(grid) == 0:
                return grid
        return None


def main():
    if len(sys.argv) < 3:
        print("usage: column_generation_patterns.py <input_csv_path> <output_json_path>")
        return 1
    inp, outp = sys.argv[1], sys.argv[2]
    inst = parse_input(inp)
    deadline = time.time() + max(1.0, inst.T - 1.0)

    if infeasible_reason(inst) is not None:
        empty_output(outp)
        return 0

    try:
        grid = PatternSolver(inst, deadline).run()
    except TimeoutError:
        grid = None
    if grid is None:
        empty_output(outp)
    else:
        write_roster(grid, outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
