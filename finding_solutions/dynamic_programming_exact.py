"""Exact dynamic programming over day columns, with nurses aggregated by state.

Idea
    Sweep the calendar one day at a time.  The only thing the future needs to
    know about a nurse is their *profile*:

        (surgical?, budget already spent, current consecutive-work streak,
         yesterday's code, their remaining leave pattern)

    Two nurses sharing a profile are completely interchangeable from here on.
    So instead of tracking which nurse does what, the DP state is a
    *multiset*: how many nurses sit in each profile.  A day's transition picks
    how many nurses move from each profile to each code, subject to the quotas
    m, a, e and the surgical B rule.

    This is the same collapse that symmetry breaking approximates, but done
    exactly and once, rather than rediscovered at every node.  When the profile
    count stays small - few distinct leave patterns, small K, short horizons -
    the whole instance is decided outright, and a failure here is a genuine
    proof of infeasibility rather than a timeout.

Completeness
    Genuinely complete *when it finishes*: exploring the state space without
    reaching day D proves the instance infeasible.  The state space grows very
    fast with distinct leave patterns and with K, so a guard abandons the DP
    once the frontier exceeds a cap; past that point it reports {} without any
    infeasibility claim beyond what the analytic pre-checks establish.

    That trade - exact but only on the easy end of the range - is the reason
    this solver is in the study: it marks the boundary where exhaustive
    reasoning stops being affordable, which the heuristic solvers otherwise
    obscure.
"""

import os
import sys
import time
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (ALLOWED_AFTER, COST, MAX_CONSECUTIVE, empty_output,
                     infeasible_reason, parse_input, write_roster)

CODES = ("M", "A", "E", "R", "B")


class ExactDP:
    """Profile-aggregated DP.  Returns a roster, "infeasible", or "gave_up"."""

    def __init__(self, inst, deadline, state_cap=200000):
        self.inst = inst
        self.deadline = deadline
        self.state_cap = state_cap
        self.seen = {}
        self.gave_up = False

    def leave_key(self, n, d):
        """A nurse's remaining leave pattern from day d, as a bitstring."""
        inst = self.inst
        return inst.leaves[n * inst.D + d: (n + 1) * inst.D]

    def initial_state(self):
        inst = self.inst
        profiles = {}
        for n in range(inst.N):
            key = (n < inst.Ns, 0, 0, "", self.leave_key(n, 0))
            profiles[key] = profiles.get(key, 0) + 1
        return tuple(sorted(profiles.items()))

    def options(self, profile, d):
        """Codes available to a nurse in this profile on day d."""
        inst = self.inst
        surg, spent, streak, prev, leave = profile
        if leave and leave[0] == "L":
            return ("R",)
        out = []
        for code in ALLOWED_AFTER[prev]:
            if code == "R":
                continue
            if code == "B" and (not surg or not inst.is_surg_day[d]):
                continue
            if spent + COST[code] > inst.K:
                continue
            if streak >= MAX_CONSECUTIVE:
                continue
            out.append(code)
        out.append("R")
        return tuple(out)

    def advance(self, profile, code, d):
        surg, spent, streak, _prev, leave = profile
        return (surg, spent + COST[code],
                0 if code == "R" else streak + 1,
                code, leave[1:])

    def solve(self, state, d, trail):
        """Depth-first over day transitions, memoised on (state, day)."""
        inst = self.inst
        if d == inst.D:
            return trail
        if time.time() > self.deadline:
            raise TimeoutError
        key = (state, d)
        if key in self.seen:
            return None
        if len(self.seen) > self.state_cap:
            self.gave_up = True
            raise GaveUp
        self.seen[key] = True

        profiles = list(state)
        # enumerate how many of each profile take each code
        results = []
        self._enumerate(profiles, 0, d, {}, [0, 0, 0, 0], results)
        for choice in results:
            if time.time() > self.deadline:
                raise TimeoutError
            nxt = {}
            for (profile, _count), per_code in zip(profiles, choice):
                for code, k in per_code.items():
                    if not k:
                        continue
                    np = self.advance(profile, code, d)
                    nxt[np] = nxt.get(np, 0) + k
            got = self.solve(tuple(sorted(nxt.items())), d + 1,
                             trail + [choice])
            if got is not None:
                return got
        return None

    def _enumerate(self, profiles, i, d, _acc, counts, out, partial=None):
        """Split each profile's nurses across the codes it may take, keeping
        the running (m, a, e, b) tallies within the day's quotas."""
        inst = self.inst
        if partial is None:
            partial = []
        cm, ca, ce, cb = counts
        if time.time() > self.deadline:
            raise TimeoutError
        if i == len(profiles):
            if cm + cb == inst.m and ca + cb == inst.a and ce == inst.e:
                if inst.is_surg_day[d] == (cb > 0):
                    out.append(list(partial))
            return
        if len(out) >= self.branch_cap:
            return
        # prune: even giving every remaining nurse to a quota cannot overshoot
        if cm + cb > inst.m or ca + cb > inst.a or ce > inst.e:
            return

        profile, count = profiles[i]
        codes = self.options(profile, d)
        # C(count + parts - 1, parts - 1) splits exist; refuse to enumerate a
        # profile that would explode rather than silently burning the budget
        if count > 24 or len(codes) ** 2 * count > 4000:
            self.gave_up = True
            raise GaveUp
        for split in self._splits(count, len(codes)):
            per_code = {}
            nm, na, ne, nb = cm, ca, ce, cb
            for code, k in zip(codes, split):
                if k:
                    per_code[code] = k
                    if code == "M":
                        nm += k
                    elif code == "A":
                        na += k
                    elif code == "E":
                        ne += k
                    elif code == "B":
                        nb += k
            partial.append(per_code)
            self._enumerate(profiles, i + 1, d, _acc, [nm, na, ne, nb], out, partial)
            partial.pop()
            if len(out) >= self.branch_cap:
                return

    @staticmethod
    @lru_cache(maxsize=None)
    def _splits(total, parts):
        """All ways to write `total` as an ordered sum of `parts` naturals."""
        if parts == 1:
            return ((total,),)
        acc = []
        for first in range(total + 1):
            for rest in ExactDP._splits(total - first, parts - 1):
                acc.append((first,) + rest)
        return tuple(acc)

    def tractable(self):
        """Refuse instances whose profile space obviously cannot be enumerated.

        The frontier is bounded by (distinct leave patterns) x (K+1 budgets) x
        (6 streaks) x (5 previous codes); when that product is already huge at
        day 0 the DP will only burn the whole time budget before giving up, so
        it is better to say so immediately."""
        inst = self.inst
        patterns = len({inst.leaves[n * inst.D:(n + 1) * inst.D] for n in range(inst.N)})
        rough = patterns * (inst.K + 1) * 6 * 5
        return rough <= 200000 and inst.N <= 50 and inst.D <= 30

    def run(self):
        inst = self.inst
        self.branch_cap = 4000
        if not self.tractable():
            return None, "gave_up"
        state = self.initial_state()
        try:
            trail = self.solve(state, 0, [])
        except GaveUp:
            return None, "gave_up"
        except (TimeoutError, RecursionError):
            return None, "gave_up"
        if trail is None:
            # the DP explored every reachable state: a real proof
            return None, "infeasible"
        return self.rebuild(trail), "ok"

    def rebuild(self, trail):
        """Turn the per-day profile splits back into concrete nurse rows."""
        inst = self.inst
        grid = [["R"] * inst.D for _ in range(inst.N)]
        # live nurses, each carrying its current profile
        live = {}
        for n in range(inst.N):
            key = (n < inst.Ns, 0, 0, "", self.leave_key(n, 0))
            live.setdefault(key, []).append(n)

        for d, choice in enumerate(trail):
            profiles = sorted(live.keys())
            nxt = {}
            for profile, per_code in zip(profiles, choice):
                pool = live[profile]
                pos = 0
                for code, k in per_code.items():
                    for _ in range(k):
                        n = pool[pos]
                        pos += 1
                        grid[n][d] = code
                        np = self.advance(profile, code, d)
                        nxt.setdefault(np, []).append(n)
            live = nxt
        return grid


class GaveUp(Exception):
    pass


def main():
    if len(sys.argv) < 3:
        print("usage: dynamic_programming_exact.py <input_csv_path> <output_json_path>")
        return 1
    inp, outp = sys.argv[1], sys.argv[2]
    inst = parse_input(inp)
    deadline = time.time() + max(1.0, inst.T - 1.0)

    if infeasible_reason(inst) is not None:
        empty_output(outp)
        return 0

    sys.setrecursionlimit(100000)
    grid, verdict = ExactDP(inst, deadline).run()
    if verdict == "ok":
        write_roster(grid, outp)
    else:
        # "infeasible" here is a proof; "gave_up" is not.  Both surface as {},
        # which is all the output format can express.
        empty_output(outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
