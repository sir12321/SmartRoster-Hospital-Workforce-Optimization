"""Constraint propagation with randomised restarts (Luby sequence).

Idea
    Model the roster as a CSP over N*D variables with domains {M,A,E,R,B}.
    Search day by day; before branching on a day, run a fixpoint propagation
    loop that repeatedly (a) prunes each nurse's domain from the transition
    rule, the leave marks, the budget K and the 5-in-a-row rule, and (b) does
    a counting / Hall-style check that the day's exact quotas m, a, e can still
    be met by the surviving domains.  Any domain wipe-out fails the day early.

    Because a wrong early decision is expensive and the search is not
    systematic over nurse orderings, the whole search restarts on a Luby
    schedule with a reshuffled nurse order and a fresh RNG.  Restarts are what
    makes this robust on the instances where a fixed ordering thrashes.

Completeness
    Within one restart the day-level search is exhaustive over the quota
    splits it enumerates, but the restart cap means a failure to find a roster
    is not a proof of infeasibility.  Infeasibility is therefore only ever
    reported from the sound analytic pre-checks in _common.
"""

import itertools
import json
import math
import os
import random
import sys
import time

sys.setrecursionlimit(100000)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (BIT_A, BIT_B, BIT_E, BIT_M, BIT_R, ALLOWED_MASK, COST,
                     MAX_CONSECUTIVE, empty_output, infeasible_reason,
                     parse_input, write_roster)

POPCOUNT = [bin(i).count("1") for i in range(32)]


def luby(i):
    """Term i (0-indexed) of the Luby restart sequence 1,1,2,1,1,2,4,1,1,2,...

    Luby, Sinclair and Zuckerman (1993):
        t_k = 2^(j-1)                 when k = 2^j - 1
        t_k = t_(k - 2^(j-1) + 1)     when 2^(j-1) <= k < 2^j - 1
    Iterative so it costs nothing and cannot blow the stack."""
    k = i + 1
    while True:
        j = k.bit_length()
        if k == (1 << j) - 1:
            return 1 << (j - 1)
        k = k - (1 << (j - 1)) + 1


class Solver:
    def __init__(self, inst, deadline, seed):
        self.inst = inst
        self.deadline = deadline
        self.rng = random.Random(seed)
        N, D = inst.N, inst.D
        self.grid = [["R"] * D for _ in range(N)]
        self.worked = [0] * N          # budget units spent (B counts 2)
        self.streak = [0] * N          # consecutive worked days ending yesterday
        self.prev = [""] * N
        self.ticks = 0
        self.budget_ticks = 0
        # Nogood cache: day-entry states already proven to lead nowhere.
        # The state that matters is, per nurse, (budget left, streak, prev
        # code) - two searches reaching the same tuple have identical futures.
        self.nogood = set()
        # True while the current subtree has enumerated rather than sampled
        self.exhaustive = True

    # ---- domain construction -------------------------------------------------
    def base_mask(self, n, d):
        """Codes nurse n could legally take on day d given history alone."""
        inst = self.inst
        if inst.on_leave[n][d]:
            return BIT_R
        if self.streak[n] >= MAX_CONSECUTIVE:
            return BIT_R
        left = inst.K - self.worked[n]
        if left <= 0:
            return BIT_R
        mask = ALLOWED_MASK[self.prev[n]]
        if left < 2 or n >= inst.Ns or not inst.is_surg_day[d]:
            mask &= ~BIT_B
        return mask | BIT_R

    def propagate(self, d, masks, need_m, need_a, need_e, need_b):
        """Counting check: can the surviving domains still hit the quotas?

        Returns False when the day is provably unsatisfiable.  This is the
        Hall-style bound - for each shift we count how many nurses can still
        take it, and we also bound the total number of duty slots available."""
        cap_m = cap_a = cap_e = cap_b = 0
        duty = 0
        for n in range(self.inst.N):
            mk = masks[n]
            if mk & BIT_M:
                cap_m += 1
            if mk & BIT_A:
                cap_a += 1
            if mk & BIT_E:
                cap_e += 1
            if mk & BIT_B:
                cap_b += 1
            if mk & ~BIT_R:
                duty += 1
        # a B covers one M and one A simultaneously
        if cap_m + cap_b < need_m or cap_a + cap_b < need_a or cap_e < need_e:
            return False
        if need_b > cap_b:
            return False
        # every duty slot needs a distinct nurse (a B occupies one nurse for two)
        if duty < need_m + need_a + need_e - min(cap_b, min(need_m, need_a)):
            return False
        return True

    def lookahead_ok(self, d):
        """Global capacity bound over the days still to be rostered.

        Careful with B: it consumes two budget units and one calendar day, but
        it covers two demand units (an M and an A at once).  A bound that
        counts calendar days alone therefore *undercounts* what surgical
        nurses can still deliver and will prune valid schedules on tight
        instances - so the surgical slack is added back explicitly."""
        inst = self.inst
        nxt = d + 1
        if nxt >= inst.D:
            return True
        remaining = (inst.D - nxt) * inst.daily_demand

        cap = 0
        for i in range(inst.N):
            cap += min(inst.K - self.worked[i], inst.workable_from[i][nxt])
        # Each B still placeable adds one unit of coverage beyond the
        # day-count bound.  At most one B per remaining surgical day is useful
        # for this bound, and each needs 2 spare budget units.
        n_surg_days = inst.surg_from[nxt]
        if n_surg_days:
            extra = 0
            for i in range(inst.Ns):
                spare = inst.K - self.worked[i]
                extra += min(spare // 2, inst.workable_from[i][nxt])
            cap += min(extra, n_surg_days * min(inst.m, inst.a))
        if cap < remaining:
            return False

        if n_surg_days:
            bcap = 0
            for i in range(inst.Ns):
                bcap += min((inst.K - self.worked[i]) // 2, inst.workable_from[i][nxt])
            if bcap < n_surg_days:
                return False
        return True

    # ---- search --------------------------------------------------------------
    def state_key(self, d):
        """A canonical signature of the state entering day d.

        Two nurses are interchangeable from here on when they agree on
        (budget left, current streak, yesterday's code) AND on their whole
        future leave pattern and surgical status.  Sorting the per-nurse
        tuples inside each class therefore collapses that permutation
        symmetry, which is what makes the nogood cache small enough to pay
        for itself - without it the same day is re-refuted once per ordering."""
        inst = self.inst
        surg, gen = [], []
        for n in range(inst.N):
            sig = (inst.K - self.worked[n], self.streak[n], self.prev[n],
                   inst.leaves[n * inst.D + d:(n + 1) * inst.D])
            (surg if n < inst.Ns else gen).append(sig)
        surg.sort()
        gen.sort()
        return (d, tuple(surg), tuple(gen))

    def solve_day(self, d, order):
        if time.time() > self.deadline:
            raise TimeoutError
        self.budget_ticks += 1
        if self.budget_ticks > self.tick_cap:
            raise RestartSignal
        inst = self.inst
        if d == inst.D:
            return True

        key = self.state_key(d)
        if key in self.nogood:
            return False

        masks = [self.base_mask(n, d) for n in range(inst.N)]
        # how many B shifts to place today: >=1 on a surgical day, 0 otherwise
        if inst.is_surg_day[d]:
            lo, hi = 1, min(inst.m, inst.a, sum(1 for n in range(inst.N) if masks[n] & BIT_B))
        else:
            lo = hi = 0
        if hi < lo:
            return False

        saved_exhaustive = self.exhaustive
        self.exhaustive = True
        for nb in range(lo, hi + 1):
            if not self.propagate(d, masks, inst.m, inst.a, inst.e, nb):
                continue
            if self.assign(d, order, 0, inst.m - nb, inst.a - nb, inst.e, nb, masks):
                self.exhaustive = saved_exhaustive
                return True
        # Only a search that actually enumerated every option proves this state
        # dead.  When any group was sampled rather than enumerated, "no luck"
        # is not "impossible", and caching it would lose real solutions.
        if self.exhaustive:
            self.nogood.add(key)
        self.exhaustive = saved_exhaustive and self.exhaustive
        return False

    def assign(self, d, order, idx, rem_m, rem_a, rem_e, rem_b, masks):
        """Fill day d by choosing, for each quota in turn, which nurses take it.

        Branching on a whole shift group at once (rather than nurse by nurse)
        collapses the permutation symmetry between interchangeable nurses: the
        order within a group never matters, only the set does.  Nurses are
        offered least-loaded first so the roster stays balanced, and only a
        bounded number of alternative sets is tried per group, which is what
        keeps the branching factor sane at N=50."""
        inst = self.inst
        if time.time() > self.deadline:
            raise TimeoutError
        self.budget_ticks += 1
        if self.budget_ticks > self.tick_cap:
            raise RestartSignal

        quotas = (("B", rem_b, BIT_B), ("M", rem_m, BIT_M),
                  ("A", rem_a, BIT_A), ("E", rem_e, BIT_E))
        # pick the quota with the least slack (fail-first / MRV over groups)
        target = None
        best_slack = None
        for code, need, bit in quotas:
            if need <= 0:
                continue
            pool = [n for n in order
                    if masks[n] & bit and self.grid[n][d] == "R"
                    and self.worked[n] + COST[code] <= inst.K]
            if len(pool) < need:
                return False
            slack = len(pool) - need
            if best_slack is None or slack < best_slack:
                best_slack, target = slack, (code, need, pool)

        if target is None:
            # every quota met - close the day off
            if not self.lookahead_ok(d):
                return False
            saved_streak = self.streak[:]
            saved_prev = self.prev[:]
            for n in range(inst.N):
                code = self.grid[n][d]
                self.streak[n] = 0 if code == "R" else self.streak[n] + 1
                self.prev[n] = code
            if self.solve_day(d + 1, order):
                return True
            self.streak[:] = saved_streak
            self.prev[:] = saved_prev
            return False

        code, need, pool = target
        cost = COST[code]
        # least loaded first; jitter breaks ties differently on each restart
        pool.sort(key=lambda n: (self.worked[n], self.streak[n], self.jitter[n]))

        # Candidate sets: the greedy (least loaded) set first, then randomised
        # draws from a widened prefix of the pool.  Sampling rather than
        # enumerating C(|pool|, need) is what keeps this tractable at N=50,
        # and the restart loop supplies the diversity a systematic search
        # would have given.
        window = min(len(pool), need + self.alt_width)
        n_subsets = math.comb(len(pool), need) if need <= len(pool) else 0
        if n_subsets <= self.exhaustive_cap:
            # Small pool: enumerate every subset.  Tight little instances are
            # solved (or genuinely refuted) here rather than by luck.
            candidates = list(itertools.combinations(pool, need))
        else:
            self.exhaustive = False
            seen = set()
            candidates = []
            greedy = tuple(pool[:need])
            candidates.append(greedy)
            seen.add(greedy)
            tries = 0
            while len(candidates) < self.set_samples and tries < self.set_samples * 4:
                tries += 1
                pick = tuple(sorted(self.rng.sample(pool[:window], need)))
                if pick not in seen:
                    seen.add(pick)
                    candidates.append(pick)

        for chosen in candidates:
            for n in chosen:
                self.grid[n][d] = code
                self.worked[n] += cost
            if self.assign(d, order, idx,
                           rem_m - (need if code == "M" else 0),
                           rem_a - (need if code == "A" else 0),
                           rem_e - (need if code == "E" else 0),
                           rem_b - (need if code == "B" else 0), masks):
                return True
            for n in chosen:
                self.grid[n][d] = "R"
                self.worked[n] -= cost
        return False

    def attempt(self, tick_cap):
        """One restart. Returns the grid, or None if it hit the node cap."""
        inst = self.inst
        self.tick_cap = tick_cap
        self.budget_ticks = 0
        self.grid = [["R"] * inst.D for _ in range(inst.N)]
        self.worked = [0] * inst.N
        self.streak = [0] * inst.N
        self.prev = [""] * inst.N
        order = list(range(inst.N))
        self.rng.shuffle(order)
        # per-restart tie-break noise, so restarts explore different sets
        self.jitter = [self.rng.random() for _ in range(inst.N)]
        self.alt_width = 2 + self.rng.randrange(6)
        self.set_samples = 3 + self.rng.randrange(4)
        # enumerate subsets outright while that is cheap
        self.exhaustive_cap = 120
        self.exhaustive = True
        try:
            if self.solve_day(0, order):
                return self.grid
        except RestartSignal:
            return None
        return None


class RestartSignal(Exception):
    pass


def main():
    if len(sys.argv) < 3:
        print("usage: constraint_propagation_luby.py <input_csv_path> <output_json_path>")
        return 1
    inp, outp = sys.argv[1], sys.argv[2]
    inst = parse_input(inp)
    start = time.time()
    deadline = start + max(1.0, inst.T - 1.0)

    reason = infeasible_reason(inst)
    if reason is not None:
        empty_output(outp)
        return 0

    solver = Solver(inst, deadline, seed=0xC0FFEE)
    unit = 4000 + 40 * inst.N * inst.D      # nodes in one Luby unit
    i = 0
    try:
        while time.time() < deadline:
            grid = solver.attempt(luby(i) * unit)
            if grid is not None:
                write_roster(grid, outp)
                return 0
            i += 1
    except TimeoutError:
        pass
    empty_output(outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
