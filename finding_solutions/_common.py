"""Shared plumbing for every roster solver in this directory.

Every solver is a standalone script:  python <solver>.py <input.csv> <output.json>
so this module only holds what is genuinely common: input parsing, the shift
transition table derived from the problem rules, cheap necessary-condition
feasibility tests, and output writing.

Shift alphabet
    M  morning        A  afternoon      E  evening
    R  rest           B  "both" M+A double shift, surgical nurses only

Hard rules (H1-H8)
    H1  every nurse gets exactly one code per day; B only for nurses < N_s
    H2  a day marked L (leave) must be R
    H3  no M/B immediately after M/B, and no M/B immediately after E
    H4  after a B the next day must be R or E
    H5  no 6 consecutive working days (some R in every window of 6)
    H6  per day: #M + #B == m, #A + #B == a, #E == e
    H7  a surgical day (S) needs at least one B; a general day (G) allows none
    H8  per nurse, sum over days of (1 per worked day + 1 extra for B) <= K
"""

import csv
import json

SHIFTS = ("M", "A", "E", "R", "B")

# Bit encoding, used by the mask based solvers.
BIT_M, BIT_A, BIT_E, BIT_B, BIT_R = 1, 2, 4, 8, 16
BIT = {"M": BIT_M, "A": BIT_A, "E": BIT_E, "B": BIT_B, "R": BIT_R}

# Cost toward the per nurse budget K.  A B shift burns two units.
COST = {"M": 1, "A": 1, "E": 1, "R": 0, "B": 2}

# Which code may follow which, straight from H3/H4.  Note A -> B is legal:
# forgetting that is the single most common way to make a solver incomplete.
ALLOWED_AFTER = {
    "":  ("M", "A", "E", "R", "B"),
    "R": ("M", "A", "E", "R", "B"),
    "M": ("A", "E", "R"),
    "A": ("M", "A", "E", "R", "B"),
    "E": ("A", "E", "R"),
    "B": ("E", "R"),
}
ALLOWED_MASK = {k: sum(BIT[s] for s in v) for k, v in ALLOWED_AFTER.items()}

MAX_CONSECUTIVE = 5   # H5: at most 5 worked days in a row


class Instance:
    """A parsed problem instance plus the derived tables solvers keep asking for."""

    __slots__ = ("N", "D", "Ns", "Ng", "m", "a", "e", "T", "days", "K", "leaves",
                 "on_leave", "is_surg_day", "surg_from", "workable_from", "leaves_from")

    def __init__(self, N, D, Ns, Ng, m, a, e, T, days, K, leaves):
        self.N, self.D, self.Ns, self.Ng = N, D, Ns, Ng
        self.m, self.a, self.e = m, a, e
        self.T, self.days, self.K, self.leaves = T, days, K, leaves

        self.on_leave = [[leaves[i * D + d] == "L" for d in range(D)] for i in range(N)]
        self.is_surg_day = [days[d] == "S" for d in range(D)]

        # surg_from[d] = number of surgical days in [d, D)
        self.surg_from = [0] * (D + 1)
        for d in range(D - 1, -1, -1):
            self.surg_from[d] = self.surg_from[d + 1] + (1 if self.is_surg_day[d] else 0)

        # workable_from[i][d] = days in [d, D) nurse i is not on leave
        # leaves_from[i][d]   = days in [d, D) nurse i IS on leave
        self.workable_from = [[0] * (D + 1) for _ in range(N)]
        self.leaves_from = [[0] * (D + 1) for _ in range(N)]
        for i in range(N):
            wf, lf, ol = self.workable_from[i], self.leaves_from[i], self.on_leave[i]
            for d in range(D - 1, -1, -1):
                wf[d] = wf[d + 1] + (0 if ol[d] else 1)
                lf[d] = lf[d + 1] + (1 if ol[d] else 0)

    @property
    def daily_demand(self):
        return self.m + self.a + self.e


def parse_input(path):
    with open(path, "r", newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    return Instance(
        N=int(row["N"]), D=int(row["D"]), Ns=int(row["N_s"]), Ng=int(row["N_g"]),
        m=int(row["m"]), a=int(row["a"]), e=int(row["e"]), T=float(row["T"]),
        days=row["days"], K=int(row["K"]), leaves=row["leaves"])


def infeasible_reason(inst):
    """Cheap necessary conditions.  Returns a reason string, or None if we
    cannot rule the instance out this way.  Never claims infeasible wrongly:
    every test here is a genuine implication of the hard rules."""
    N, D, Ns, K = inst.N, inst.D, inst.Ns, inst.K
    m, a, e = inst.m, inst.a, inst.e
    demand = inst.daily_demand

    if D == 0:
        return None
    if demand > N:
        return "a single day needs more nurses than exist"
    if m + a + e > 0 and N == 0:
        return "no nurses available at all"

    # A surgical day needs a B, and B is an M and an A at once.
    for d in range(D):
        if inst.is_surg_day[d]:
            if Ns == 0:
                return "surgical day with no surgical nurses"
            if m == 0 or a == 0:
                return "surgical day requires a B shift but m or a is zero"
            # at least one surgical nurse must actually be on duty that day
            if not any(not inst.on_leave[i][d] for i in range(Ns)):
                return f"every surgical nurse is on leave on surgical day {d}"

    # Per day, enough non-leave nurses must exist to cover the demand.  A B
    # shift lets one surgical nurse cover both an M and an A slot.
    for d in range(D):
        free = sum(1 for i in range(N) if not inst.on_leave[i][d])
        free_surg = sum(1 for i in range(Ns) if not inst.on_leave[i][d])
        if free + min(free_surg, min(m, a)) < demand:
            return f"day {d} has too few available nurses"

    # Global budget: total slot-units needed vs. total units available, where a
    # nurse is additionally capped by H5 (at most 5 of every 6 days worked) and
    # by their own leave days.
    need = D * demand
    cap = 0
    for i in range(N):
        workable = inst.workable_from[i][0]
        rest_cap = D - D // 6            # H5 ceiling on worked days
        cap += min(K, workable, rest_cap)
    if cap < need:
        return "total nurse capacity below total demand"

    # B supply: every surgical day consumes at least one B, and a B costs 2
    # budget units and forces the following day to R or E.
    n_surg = inst.surg_from[0]
    if n_surg:
        b_cap = 0
        for i in range(Ns):
            workable = inst.workable_from[i][0]
            b_cap += min(K // 2, workable, (D + 1) // 2)
        if b_cap < n_surg:
            return "not enough surgical B capacity for all surgical days"
    return None


def empty_output(path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({}, fh)


def write_roster(grid, path):
    """grid[i][d] -> the JSON object the verifier expects."""
    solution = {}
    for i, row in enumerate(grid):
        for d, code in enumerate(row):
            solution[f"N{i}_{d}"] = code
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(solution, fh, indent=1)
    return solution


def objective(inst, grid):
    """The Part-B fairness objective; lower is better.  Reported here so the
    feasibility solvers can still be compared on schedule quality."""
    total = 0
    for i in range(inst.N):
        mo = sum(1 for d in range(inst.D) if grid[i][d] in ("M", "B"))
        af = sum(1 for d in range(inst.D) if grid[i][d] in ("A", "B"))
        ev = sum(1 for d in range(inst.D) if grid[i][d] == "E")
        tot = mo + af + ev
        total += 3 * (mo * mo + af * af + ev * ev) - tot * tot
    return total
