"""Local search: construct a roster that already satisfies the per-nurse rules,
then repair the coverage violations by min-conflicts hill climbing.

Idea
    Split the hard rules into two groups.

      *Row* rules  - the ones about a single nurse's own timeline: leave days,
      the transition table (H3/H4), the 5-in-a-row cap (H5) and the budget K
      (H8).  These are made *invariant*: the initial roster satisfies them and
      every move is rejected unless it keeps satisfying them.

      *Column* rules - the daily quotas m, a, e and the surgical B rule
      (H6/H7).  These are allowed to be violated and drive the cost function
      cost = sum over days of |#M+#B - m| + |#A+#B - a| + |#E - e| + B-penalty.

    A move retimes a single nurse-day (or swaps a pair) to whatever code most
    reduces the cost, with sideways moves allowed and a tabu list plus random
    restarts to escape plateaus - the classic min-conflicts recipe.

Completeness
    None: this is a stochastic incomplete method.  It never reports
    infeasibility from its own failure - only the analytic pre-checks may do
    that - so a timeout yields {} without any claim of a proof.

    It is included because it scales differently from the systematic solvers:
    its cost per move is independent of D, so it degrades gracefully on wide
    instances where DFS thrashes.  The results table shows where that pays off
    and where it does not.
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (COST, MAX_CONSECUTIVE, ALLOWED_AFTER, empty_output,
                     infeasible_reason, parse_input, write_roster)


class LocalSearch:
    def __init__(self, inst, deadline, seed=12345):
        self.inst = inst
        self.deadline = deadline
        self.rng = random.Random(seed)
        self.grid = [["R"] * inst.D for _ in range(inst.N)]
        self.worked = [0] * inst.N
        # per-day tallies, kept in step with the grid so day_cost is O(1)
        self.cnt = {c: [0] * inst.D for c in ("M", "A", "E", "B", "R")}

    # ---- row feasibility (kept invariant) -----------------------------------
    def row_ok(self, n, d, code):
        """Could nurse n legally hold `code` on day d, given the rest of their
        row as it currently stands?  Checks leave, both transition directions,
        the budget and the 6-day rest window."""
        inst = self.inst
        if inst.on_leave[n][d]:
            return code == "R"
        if code == "B" and n >= inst.Ns:
            return False
        if code == "B" and not inst.is_surg_day[d]:
            return False
        row = self.grid[n]
        # budget, accounting for what this day currently costs
        if self.worked[n] - COST[row[d]] + COST[code] > inst.K:
            return False
        # transitions on both sides
        if d > 0 and code not in ALLOWED_AFTER[row[d - 1]]:
            return False
        if d + 1 < inst.D and row[d + 1] not in ALLOWED_AFTER[code]:
            return False
        # H5: no 6 consecutive working days through this cell
        if code != "R":
            saved = row[d]
            row[d] = code
            ok = True
            lo = max(0, d - MAX_CONSECUTIVE)
            hi = min(inst.D - 1, d + MAX_CONSECUTIVE)
            run = 0
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

    # ---- column cost --------------------------------------------------------
    def day_cost(self, d):
        inst = self.inst
        cm = self.cnt["M"][d]
        ca = self.cnt["A"][d]
        ce = self.cnt["E"][d]
        cb = self.cnt["B"][d]
        cost = abs(cm + cb - inst.m) + abs(ca + cb - inst.a) + abs(ce - inst.e)
        if inst.is_surg_day[d] and cb == 0:
            cost += 3          # a surgical day with no B is a hard miss
        if not inst.is_surg_day[d]:
            cost += cb * 3     # B is illegal on a general day
        return cost

    def total_cost(self):
        return sum(self.day_cost(d) for d in range(self.inst.D))

    # ---- construction -------------------------------------------------------
    def seed_roster(self):
        """Greedy start: walk the days, filling each quota with whichever
        legal nurse is least loaded so far.  Rows stay feasible throughout."""
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
            wanted += ["M"] * (inst.m - (1 if "B" in wanted else 0))
            wanted += ["A"] * (inst.a - (1 if "B" in wanted else 0))
            wanted += ["E"] * inst.e
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

    # ---- repair -------------------------------------------------------------
    def repair(self, max_steps):
        """Min-conflicts: repeatedly pick a broken day and retime one cell in
        it so the total cost drops as much as possible."""
        inst = self.inst
        cost = self.total_cost()
        tabu = {}
        step = 0
        while cost > 0 and step < max_steps:
            step += 1
            if not (step & 255) and time.time() > self.deadline:
                raise TimeoutError
            broken = [d for d in range(inst.D) if self.day_cost(d) > 0]
            if not broken:
                break
            d = self.rng.choice(broken)
            before = self.day_cost(d)

            best = None
            best_delta = 0
            candidates = list(range(inst.N))
            self.rng.shuffle(candidates)
            for n in candidates:
                cur = self.grid[n][d]
                for code in ("B", "M", "A", "E", "R"):
                    if code == cur or not self.row_ok(n, d, code):
                        continue
                    if tabu.get((n, d, code), 0) > step:
                        continue
                    saved = cur
                    self.place(n, d, code)
                    delta = self.day_cost(d) - before
                    self.place(n, d, saved)
                    if best is None or delta < best_delta:
                        best, best_delta = (n, code), delta
                        if delta < 0:
                            break
                if best is not None and best_delta < 0:
                    break

            if best is None or best_delta >= 0:
                # No single retiming helps.  Try swapping two nurses' codes on
                # this day: that keeps the column tallies fixed but moves
                # budget between rows, which is what unsticks a tight roster.
                if self.swap_move(d):
                    cost = self.total_cost()
                    continue
            if best is None:
                continue
            n, code = best
            # accept improving and sideways moves; take a random walk step
            # occasionally so plateaus do not trap the search
            if best_delta > 0 and self.rng.random() > 0.02:
                continue
            old = self.grid[n][d]
            self.place(n, d, code)
            tabu[(n, d, old)] = step + 8 + self.rng.randrange(8)
            cost += best_delta
        return cost

    def swap_move(self, d):
        """Exchange the codes of two nurses on day d, if both rows allow it.
        Column counts are unchanged, so the day cost is unchanged - the point
        is purely to redistribute budget and streaks between rows."""
        inst = self.inst
        idx = list(range(inst.N))
        self.rng.shuffle(idx)
        for a in idx[:24]:
            ca = self.grid[a][d]
            for b in idx[:24]:
                if a == b:
                    continue
                cb = self.grid[b][d]
                if ca == cb:
                    continue
                # tentatively vacate both cells so row_ok sees a clean slate
                self.place(a, d, "R")
                self.place(b, d, "R")
                ok = self.row_ok(a, d, cb) and self.row_ok(b, d, ca)
                if ok:
                    self.place(a, d, cb)
                    self.place(b, d, ca)
                    return True
                self.place(a, d, ca)
                self.place(b, d, cb)
        return False

    def run(self):
        inst = self.inst
        steps = max(2000, 60 * inst.N * inst.D)
        while time.time() < self.deadline:
            self.seed_roster()
            if self.repair(steps) == 0:
                return self.grid
        return None


def main():
    if len(sys.argv) < 3:
        print("usage: local_search_min_conflicts.py <input_csv_path> <output_json_path>")
        return 1
    inp, outp = sys.argv[1], sys.argv[2]
    inst = parse_input(inp)
    deadline = time.time() + max(1.0, inst.T - 1.0)

    if infeasible_reason(inst) is not None:
        empty_output(outp)
        return 0

    searcher = LocalSearch(inst, deadline)
    try:
        grid = searcher.run()
    except TimeoutError:
        grid = None
    if grid is None:
        empty_output(outp)
    else:
        write_roster(grid, outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
