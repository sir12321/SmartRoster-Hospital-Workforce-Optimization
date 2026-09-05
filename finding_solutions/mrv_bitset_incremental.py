"""MRV search with incremental bitmask domains.

The MRV solver with its domain lists replaced by 5-bit masks that are computed
once per day and then only intersected, never rebuilt.  Popcount picks the most
constrained nurse.

What it teaches: isolates the cost of the representation from the cost of the
strategy.  Against mrv_heuristic_search.py the search is identical and only
the bookkeeping differs, so the gap between them is pure constant factor - and
it is not enough to rescue the approach on the large suites.
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

BIT_M, BIT_A, BIT_E, BIT_B, BIT_R = 1, 2, 4, 8, 16
BIT = {'M': BIT_M, 'A': BIT_A, 'E': BIT_E, 'B': BIT_B, 'R': BIT_R}

ALLOWED_MASK = {
    k: sum(BIT[s] for s in v) for k, v in ALLOWED_AFTER.items()
}

POPCOUNT = [bin(i).count('1') for i in range(32)]

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
        print("usage: mrv_bitset_incremental.py <input_csv_path> <output_json_path>")
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

    def base_mask(n, d):
        if is_on_leave[n][d]:
            return BIT_R
        if consecutive_work[n] >= MAX_CONSECUTIVE:
            return BIT_R

        budget_left = max_shifts - shifts_worked[n]
        if budget_left <= 0:
            return BIT_R

        prev_shift = roster[n][d - 1] if d > 0 else ''
        mask = ALLOWED_MASK[prev_shift]

        if budget_left < 2:
            mask &= ~BIT_B
        if not (n < Ns and days[d] == 'S'):
            mask &= ~BIT_B

        return mask | BIT_R

    def backtrack(d, pending, base, rem_m, rem_a, rem_e, b_count):
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
            nd = d + 1
            if nd == D:
                return True
            nbase = [base_mask(i, nd) for i in range(N)]
            return backtrack(nd, list(range(N)), nbase, req_m, req_a, req_e, 0)

        nurses_left = len(pending)
        surgical_left = 0
        for i in pending:
            if i < Ns:
                surgical_left += 1

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

        best_n = -1
        best_mask = 0
        best_size = 99
        for n in pending:
            mask = base[n] & avail
            if mask == 0:
                return False
            size = POPCOUNT[mask]
            if size == 1:
                best_n, best_mask = n, mask
                break
            if size < best_size:
                best_n, best_mask, best_size = n, mask, size

        rest = [i for i in pending if i != best_n]

        order = []
        if best_mask & BIT_B: order.append(('B', rem_m + rem_a))
        if best_mask & BIT_M: order.append(('M', rem_m))
        if best_mask & BIT_A: order.append(('A', rem_a))
        if best_mask & BIT_E: order.append(('E', rem_e))
        order.sort(key=lambda x: x[1], reverse=True)
        domain = [x[0] for x in order]
        if best_mask & BIT_R:
            domain.append('R')

        for shift in domain:
            shift_cost = SHIFT_COST[shift]
            prev_streak = consecutive_work[best_n]

            roster[best_n][d] = shift
            shifts_worked[best_n] += shift_cost
            consecutive_work[best_n] = 0 if shift == 'R' else prev_streak + 1

            nm = rem_m - (1 if shift in ('M', 'B') else 0)
            na = rem_a - (1 if shift in ('A', 'B') else 0)
            ne = rem_e - (1 if shift == 'E' else 0)
            nb = b_count + (1 if shift == 'B' else 0)

            if backtrack(d, rest, base, nm, na, ne, nb):
                return True

            roster[best_n][d] = ''
            shifts_worked[best_n] -= shift_cost
            consecutive_work[best_n] = prev_streak

        return False

    try:
        success = backtrack(0, list(range(N)), [base_mask(i, 0) for i in range(N)], req_m, req_a, req_e, 0)
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
