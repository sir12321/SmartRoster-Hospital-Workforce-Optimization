"""Minimum-remaining-values search over nurses, with explicit domain lists.

Instead of a fixed nurse sweep, at each step this picks the *most constrained*
unassigned nurse - the one with the fewest legal codes left today - which is
the textbook MRV heuristic.  Domains are built as Python lists from the
transition table and the day's outstanding demand.

What it teaches: MRV is the right instinct and the wrong granularity here.
Choosing which nurse to assign next still explores orderings of interchangeable
nurses, so the symmetry it is fighting is one it cannot see.  Rebuilding the
domain lists at every node is also expensive - the bitmask variant does the
same search several times faster.
"""

import sys
import csv
import json
import time

'''
R->R
R->M
R->A
R->E
R->B
A->M
A->A
A->E
A->R
M->A
M->E
M->R
E->E
E->A
E->R
B->E
B->R

count -> R = 0,0,0
         B = 1,1,0
         M = 1,0,0
         A = 0,1,0
         E = 0,0,1
'''

ALLOWED_AFTER = {
    '': ['M', 'A', 'E', 'B'],
    'R': ['M', 'A', 'E', 'B'],
    'M': ['A', 'E'],
    'A': ['M', 'A', 'E', 'B'],
    'E': ['A', 'E'],
    'B': ['E'],
}

SHIFT_COST = {'M': 1, 'A': 1, 'E': 1, 'R': 0, 'B': 2}

MAX_CONSECUTIVE = 5

def parse_input(input_csv):
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)

        N = int(row['N'])
        D = int(row['D'])
        N_s = int(row['N_s'])
        N_g = int(row['N_g'])
        m = int(row['m'])
        a = int(row['a'])
        e = int(row['e'])
        T = float(row['T'])
        days = row['days']
        max_shifts = int(row['K'])
        leaves = row['leaves']
    return N, D, N_s, N_g, m, a, e, T, days, max_shifts, leaves

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: mrv_heuristic_search.py <input_csv_path> <output_json_path>")
        sys.exit(1)

    sys.setrecursionlimit(5000)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    N, D, Ns, Ng, req_m, req_a, req_e, T, days, max_shifts, leaves = parse_input(input_file)

    start_time = time.time()
    deadline = start_time + max(1.0, T - 2.0)
    tick = [0]

    is_on_leave = [[False] * D for _ in range(N)]
    for i in range(N):
        for j in range(D):
            if leaves[i * D + j] == 'L':
                is_on_leave[i][j] = True

    roster = [[''] * D for _ in range(N)]
    shifts_worked = [0] * N
    consecutive_work = [0] * N

    def build_domain(n, d, rem_m, rem_a, rem_e, nurses_left):
        if is_on_leave[n][d]:
            return ['R']
        if consecutive_work[n] >= MAX_CONSECUTIVE:
            return ['R']

        budget_left = max_shifts - shifts_worked[n]
        if budget_left <= 0:
            return ['R']

        prev_shift = roster[n][d - 1] if d > 0 else ''

        valid_shifts = []
        for s in ALLOWED_AFTER[prev_shift]:
            if SHIFT_COST[s] > budget_left:
                continue
            if s == 'B':
                if n < Ns and days[d] == 'S' and rem_m > 0 and rem_a > 0:
                    valid_shifts.append((s, rem_m + rem_a))
            elif s == 'M':
                if rem_m > 0: valid_shifts.append((s, rem_m))
            elif s == 'A':
                if rem_a > 0: valid_shifts.append((s, rem_a))
            elif s == 'E':
                if rem_e > 0: valid_shifts.append((s, rem_e))

        valid_shifts.sort(key=lambda x: x[1], reverse=True)
        domain = [s[0] for s in valid_shifts]

        domain.append('R')

        return domain

    def backtrack(d, pending, rem_m, rem_a, rem_e, b_count):
        tick[0] += 1
        if not (tick[0] & 1023):
            if time.time() > deadline:
                raise TimeoutError

        if d == D:
            return True

        if not pending:
            if rem_m != 0 or rem_a != 0 or rem_e != 0:
                return False
            if days[d] == 'S' and b_count == 0:
                return False
            return backtrack(d + 1, list(range(N)), req_m, req_a, req_e, 0)

        nurses_left = len(pending)
        surgical_left = sum(1 for i in pending if i < Ns)

        if (rem_m + rem_a + rem_e) > (surgical_left * 2) + (nurses_left - surgical_left):
            return False
        if rem_m > nurses_left or rem_a > nurses_left or rem_e > nurses_left:
            return False
        if days[d] == 'S' and b_count == 0 and surgical_left == 0:
            return False

        best_n = -1
        best_domain = None
        for n in pending:
            dom = build_domain(n, d, rem_m, rem_a, rem_e, nurses_left)
            if not dom:
                return False
            if len(dom) == 1:
                best_n, best_domain = n, dom
                break
            if best_domain is None or len(dom) < len(best_domain):
                best_n, best_domain = n, dom

        rest = [i for i in pending if i != best_n]

        for shift in best_domain:
            shift_cost = SHIFT_COST[shift]
            prev_streak = consecutive_work[best_n]

            roster[best_n][d] = shift
            shifts_worked[best_n] += shift_cost
            consecutive_work[best_n] = 0 if shift == 'R' else prev_streak + 1

            nm = rem_m - (1 if shift in ['M', 'B'] else 0)
            na = rem_a - (1 if shift in ['A', 'B'] else 0)
            ne = rem_e - (1 if shift == 'E' else 0)
            nb = b_count + (1 if shift == 'B' else 0)

            if backtrack(d, rest, nm, na, ne, nb):
                return True

            roster[best_n][d] = ''
            shifts_worked[best_n] -= shift_cost
            consecutive_work[best_n] = prev_streak

        return False

    try:
        success = backtrack(0, list(range(N)), req_m, req_a, req_e, 0)
    except (TimeoutError, RecursionError):
        success = False

    if success:
        solution = {}
        for i in range(N):
            for j in range(D):
                solution[f"N{i}_{j}"] = roster[i][j]
        with open(output_file, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        with open(output_file, 'w') as f:
            json.dump({}, f)
