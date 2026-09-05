"""Day-by-day search where each day is solved as a bipartite matching / flow.

Idea
    Fix the number of B shifts on a day.  What remains is: choose which nurses
    take M, which take A, which take E, and which rest, subject to each nurse's
    own eligibility for each code.  That is exactly a bipartite degree
    constrained subgraph problem - nurses on one side, the (m, a, e) slots on
    the other - and it is solvable in polynomial time by max flow.

        source -> nurse      capacity 1        (a nurse works one code a day)
        nurse  -> code       capacity 1        iff that code is legal for them
        code   -> sink       capacity = quota  (m - b, a - b, e, and b)

    A saturating flow exists iff the day can be covered at all, so a day is
    never *searched*: it is decided.  Backtracking then happens only across
    days, over the choice of how many B shifts to place and, when a day has
    many valid matchings, over which one to keep.

Why it is interesting
    The per-day combinatorics vanish - no enumerating subsets of 50 nurses.
    The cost is that a matching only knows about *today*: it happily picks the
    nurses that today's flow finds first, with no view of tomorrow's budget.
    Tomorrow's damage is handled by re-running the flow with a cost bias
    (least-loaded nurses preferred) and by backtracking when a later day turns
    out to be unmatchable.

Completeness
    The per-day flow is exact.  The cross-day search is bounded by a node
    limit, so exhausting it is not a proof of infeasibility - only the
    analytic pre-checks report {} as a claim.
"""

import os
import random
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (ALLOWED_AFTER, COST, MAX_CONSECUTIVE, empty_output,
                     infeasible_reason, parse_input, write_roster)

sys.setrecursionlimit(100000)


class Dinic:
    """Dinic max flow - enough for the small per-day graphs used here."""

    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add(self, u, v, cap):
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])

    def bfs(self, s, t):
        self.level = [-1] * self.n
        self.level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v, cap, _ in self.graph[u]:
                if cap > 0 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    q.append(v)
        return self.level[t] >= 0

    def dfs(self, u, t, f):
        if u == t:
            return f
        while self.it[u] < len(self.graph[u]):
            edge = self.graph[u][self.it[u]]
            v, cap = edge[0], edge[1]
            if cap > 0 and self.level[v] == self.level[u] + 1:
                d = self.dfs(v, t, min(f, cap))
                if d > 0:
                    edge[1] -= d
                    self.graph[v][edge[2]][1] += d
                    return d
            self.it[u] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0
        while self.bfs(s, t):
            self.it = [0] * self.n
            while True:
                f = self.dfs(s, t, 1 << 30)
                if f == 0:
                    break
                flow += f
        return flow


class FlowSolver:
    def __init__(self, inst, deadline):
        self.inst = inst
        self.deadline = deadline
        self.grid = [["R"] * inst.D for _ in range(inst.N)]
        self.worked = [0] * inst.N
        self.streak = [0] * inst.N
        self.prev = [""] * inst.N
        self.nodes = 0
        self.rng = random.Random(20240501)

    def eligible(self, n, d, code):
        inst = self.inst
        if inst.on_leave[n][d]:
            return code == "R"
        if code == "B" and (n >= inst.Ns or not inst.is_surg_day[d]):
            return False
        if code != "R" and self.streak[n] >= MAX_CONSECUTIVE:
            return False
        if self.worked[n] + COST[code] > inst.K:
            return False
        return code in ALLOWED_AFTER[self.prev[n]]

    def match_day(self, d, nb, order):
        """Exact per-day assignment via max flow.

        `order` biases which nurses the flow reaches first: the graph is built
        with the least loaded nurses' edges added first, so Dinic tends to
        saturate them, which spreads the workload without any extra machinery.
        Returns the list of codes for the day, or None when the day cannot be
        covered with exactly nb B shifts."""
        inst = self.inst
        codes = ["B", "M", "A", "E"]
        quota = {"B": nb, "M": inst.m - nb, "A": inst.a - nb, "E": inst.e}
        if any(q < 0 for q in quota.values()):
            return None

        # node ids: 0 = source, 1..N = nurses, N+1..N+4 = codes, N+5 = sink
        S, T = 0, inst.N + 5
        flow = Dinic(inst.N + 6)
        for n in order:
            flow.add(S, 1 + n, 1)
        code_node = {c: inst.N + 1 + i for i, c in enumerate(codes)}
        for n in order:
            for c in codes:
                if quota[c] and self.eligible(n, d, c):
                    flow.add(1 + n, code_node[c], 1)
        need = 0
        for c in codes:
            if quota[c]:
                flow.add(code_node[c], T, quota[c])
                need += quota[c]

        if flow.max_flow(S, T) != need:
            return None

        # read the assignment back off the saturated nurse->code edges
        day = ["R"] * inst.N
        for n in range(inst.N):
            for v, cap, _ in flow.graph[1 + n]:
                if v in code_node.values() and cap == 0:
                    for c, node in code_node.items():
                        if node == v:
                            day[n] = c
                            break
                    break
        return day

    def lookahead_ok(self, d):
        """Same capacity bound the other solvers use, with the B shift
        correctly credited for covering two demand units in one day."""
        inst = self.inst
        nxt = d + 1
        if nxt >= inst.D:
            return True
        remaining = (inst.D - nxt) * inst.daily_demand
        cap = sum(min(inst.K - self.worked[i], inst.workable_from[i][nxt])
                  for i in range(inst.N))
        n_surg = inst.surg_from[nxt]
        if n_surg:
            extra = sum(min((inst.K - self.worked[i]) // 2, inst.workable_from[i][nxt])
                        for i in range(inst.Ns))
            cap += min(extra, n_surg * min(inst.m, inst.a))
        if cap < remaining:
            return False
        if n_surg:
            bcap = sum(min((inst.K - self.worked[i]) // 2, inst.workable_from[i][nxt])
                       for i in range(inst.Ns))
            if bcap < n_surg:
                return False
        return True

    def solve(self, d):
        inst = self.inst
        if d == inst.D:
            return True
        if time.time() > self.deadline:
            raise TimeoutError
        self.nodes += 1
        if self.nodes > self.node_cap:
            raise TimeoutError

        lo, hi = (1, min(inst.m, inst.a)) if inst.is_surg_day[d] else (0, 0)
        for nb in range(lo, hi + 1):
          # A max-flow returns *some* saturating assignment, not a
          # tomorrow-aware one.  Re-running it under different nurse
          # orderings yields genuinely different matchings, which is how this
          # solver backtracks *within* a day rather than only across days.
          for attempt in range(self.match_retries):
            if attempt == 0:
                order = sorted(range(inst.N), key=lambda n: (self.worked[n], self.streak[n]))
            elif attempt == 1:
                # favour nurses whose remaining budget is large relative to
                # the days they can still work
                order = sorted(range(inst.N), key=lambda n: (
                    -(inst.K - self.worked[n]) / (inst.workable_from[n][d] or 1),
                    self.streak[n]))
            else:
                order = list(range(inst.N))
                self.rng.shuffle(order)
            day = self.match_day(d, nb, order)
            if day is None:
                continue
            saved_w = self.worked[:]
            saved_s = self.streak[:]
            saved_p = self.prev[:]
            for n in range(inst.N):
                self.grid[n][d] = day[n]
                self.worked[n] += COST[day[n]]
                self.streak[n] = 0 if day[n] == "R" else self.streak[n] + 1
                self.prev[n] = day[n]
            if self.lookahead_ok(d) and self.solve(d + 1):
                return True
            self.worked[:] = saved_w
            self.streak[:] = saved_s
            self.prev[:] = saved_p
            for n in range(inst.N):
                self.grid[n][d] = "R"
        return False

    def run(self):
        self.node_cap = 200000
        self.match_retries = 6
        try:
            if self.solve(0):
                return self.grid
        except TimeoutError:
            pass
        return None


def main():
    if len(sys.argv) < 3:
        print("usage: network_flow_matching.py <input_csv_path> <output_json_path>")
        return 1
    inp, outp = sys.argv[1], sys.argv[2]
    inst = parse_input(inp)
    deadline = time.time() + max(1.0, inst.T - 1.0)

    if infeasible_reason(inst) is not None:
        empty_output(outp)
        return 0

    grid = FlowSolver(inst, deadline).run()
    if grid is None:
        empty_output(outp)
    else:
        write_roster(grid, outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
