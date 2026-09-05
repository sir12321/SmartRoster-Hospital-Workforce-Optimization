"""Chronological DFS with per-nurse domains held as bitmasks.

The same search as the forward-checking baseline, with the representation
changed: each nurse's legal codes for the current day are a 5-bit mask,
recomputed once per day and intersected with what the day still needs.  That
turns the innermost domain test from a list build into a couple of integer
operations.

What it teaches: a large constant-factor speedup that does not change which
instances are solvable.  It moves the timeout boundary a little; it does not
move the asymptotics, because the search tree is exactly the same shape.
"""

import sys
import csv
import json
import time

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

BIT_M, BIT_A, BIT_E, BIT_B, BIT_R = 1, 2, 4, 8, 16
BIT = {'M': BIT_M, 'A': BIT_A, 'E': BIT_E, 'B': BIT_B, 'R': BIT_R}
ALLOWED_MASK = {k: sum(BIT[s] for s in v) for k, v in ALLOWED_AFTER.items()}

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
        print("usage: backtracking_bitset_domains.py <input_csv_path> <output_json_path>")
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
    base = [0] * N

    def compute_base(d):
        surgical_day = days[d] == 'S'
        for n in range(N):
            if is_on_leave[n][d] or consecutive_work[n] >= MAX_CONSECUTIVE:
                base[n] = BIT_R
                continue
            budget_left = max_shifts - shifts_worked[n]
            if budget_left <= 0:
                base[n] = BIT_R
                continue
            mask = ALLOWED_MASK[roster[n][d - 1] if d > 0 else '']
            if budget_left < 2 or not (n < Ns and surgical_day):
                mask &= ~BIT_B
            base[n] = mask | BIT_R

    def backtrack(d, n, rem_m, rem_a, rem_e, b_count):
        tick[0] += 1
        if not (tick[0] & 1023):
            if time.time() > deadline:
                raise TimeoutError

        if d == D:
            return True

        if n == N:
            if rem_m != 0 or rem_a != 0 or rem_e != 0:
                return False
            if days[d] == 'S' and b_count == 0:
                return False
            nd = d + 1
            if nd == D:
                return True
            saved = base[:]
            compute_base(nd)
            if backtrack(nd, 0, req_m, req_a, req_e, 0):
                return True
            base[:] = saved
            return False

        nurses_left = N - n
        surgical_left = max(0, Ns - n)

        if (rem_m + rem_a + rem_e) > (surgical_left * 2) + (nurses_left - surgical_left):
            return False
        if rem_m > nurses_left or rem_a > nurses_left or rem_e > nurses_left:
            return False
        if days[d] == 'S' and b_count == 0 and surgical_left == 0:
            return False

        avail = 0
        if rem_m > 0: avail |= BIT_M
        if rem_a > 0: avail |= BIT_A
        if rem_e > 0: avail |= BIT_E
        if rem_m > 0 and rem_a > 0: avail |= BIT_B
        avail |= BIT_R

        mask = base[n] & avail
        if mask == 0:
            return False

        order = []
        if mask & BIT_B: order.append(('B', rem_m + rem_a))
        if mask & BIT_M: order.append(('M', rem_m))
        if mask & BIT_A: order.append(('A', rem_a))
        if mask & BIT_E: order.append(('E', rem_e))
        order.sort(key=lambda x: x[1], reverse=True)
        domain = [x[0] for x in order]
        if mask & BIT_R:
            domain.append('R')

        for shift in domain:
            shift_cost = SHIFT_COST[shift]
            prev_streak = consecutive_work[n]

            roster[n][d] = shift
            shifts_worked[n] += shift_cost
            consecutive_work[n] = 0 if shift == 'R' else prev_streak + 1

            nm = rem_m - (1 if shift in ('M', 'B') else 0)
            na = rem_a - (1 if shift in ('A', 'B') else 0)
            ne = rem_e - (1 if shift == 'E' else 0)
            nb = b_count + (1 if shift == 'B' else 0)

            if backtrack(d, n + 1, nm, na, ne, nb):
                return True

            roster[n][d] = ''
            shifts_worked[n] -= shift_cost
            consecutive_work[n] = prev_streak

        return False

    compute_base(0)
    try:
        success = backtrack(0, 0, req_m, req_a, req_e, 0)
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
