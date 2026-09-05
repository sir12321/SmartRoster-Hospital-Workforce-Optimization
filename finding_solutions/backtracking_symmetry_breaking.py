"""Chronological DFS with adjacent-nurse symmetry breaking.

Identical to the rest-saturation solver except for one rule: when nurse n and
nurse n-1 are indistinguishable - same class, same leave status, same budget
spent, same streak, same previous code, same remaining workload - their codes
are forced into a fixed rank order.  That removes the swap symmetry between
them without removing any genuinely different schedule.

What it teaches: symmetry breaking helps, but only between *adjacent* nurses in
the sweep, so it catches a small slice of the permutation symmetry.  Solvers
that branch on whole groups (see branch_and_bound_setwise.py) eliminate
all of it structurally, and do much better.
"""

import sys
import csv
import json
import time

RANK = {'B': 4, 'M': 3, 'A': 2, 'E': 1, 'R': 0, '': 99}

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
        print("usage: backtracking_symmetry_breaking.py <input_csv_path> <output_json_path>")
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

    workable_from = [[0] * (D + 1) for _ in range(N)]
    for i in range(N):
        for j in range(D - 1, -1, -1):
            workable_from[i][j] = workable_from[i][j + 1] + (0 if is_on_leave[i][j] else 1)

    roster = [[''] * D for _ in range(N)]
    shifts_worked = [0] * N
    consecutive_work = [0] * N

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
            
            return backtrack(d + 1, 0, req_m, req_a, req_e, 0)

        nurses_left = N - n
        surgical_left = max(0, Ns - n)
        
        if (rem_m + rem_a + rem_e) > (surgical_left * 2) + (nurses_left - surgical_left):
            return False
            
        if rem_m > nurses_left or rem_a > nurses_left or rem_e > nurses_left:
            return False
            
        if days[d] == 'S' and b_count == 0 and surgical_left == 0:
            return False
            
        if is_on_leave[n][d]:
            domain = ['R']
        else:
            valid_shifts = []
            
            if n < Ns and days[d] == 'S' and (rem_m > 0 and rem_a > 0):
                valid_shifts.append(('B', rem_m + rem_a))
                
            if rem_m > 0: valid_shifts.append(('M', rem_m))
            if rem_a > 0: valid_shifts.append(('A', rem_a))
            if rem_e > 0: valid_shifts.append(('E', rem_e))
            
            budget_n = max_shifts - shifts_worked[n]
            workable_n = workable_from[n][d]

            valid_shifts.sort(key=lambda x: x[1], reverse=True)
            
            domain = [s[0] for s in valid_shifts]

            domain.append('R')
            if not (budget_n >= workable_n and workable_n > 0) and len(domain) > 1:
                if (rem_m + rem_a + rem_e) < nurses_left:
                    domain.insert(0, domain.pop())

        twin = -1
        if n > 0:
            p = n - 1
            if (p < Ns) == (n < Ns) and is_on_leave[p][d] == is_on_leave[n][d]:
                if shifts_worked[p] == shifts_worked[n] and consecutive_work[p] == consecutive_work[n]:
                    if (roster[p][d - 1] if d > 0 else '') == (roster[n][d - 1] if d > 0 else ''):
                        if workable_from[p][d] == workable_from[n][d]:
                            twin = p

        for shift in domain:
            if twin >= 0 and RANK[shift] < RANK[roster[twin][d]]:
                continue
            shift_cost = 2 if shift == 'B' else (0 if shift == 'R' else 1)
            if shifts_worked[n] + shift_cost > max_shifts: 
                continue
            
            if d > 0:
                prev_shift = roster[n][d-1]
                
                if shift in ['M', 'B']:
                    if prev_shift in ['M', 'B']: continue
                    if prev_shift == 'E': continue
                
                if prev_shift == 'B' and shift in ['M', 'A', 'B']: 
                    continue

            prev_streak = consecutive_work[n]
            if shift != 'R' and prev_streak >= 5: continue
            
            roster[n][d] = shift
            shifts_worked[n] += shift_cost
            consecutive_work[n] = 0 if shift == 'R' else prev_streak + 1
            
            nm = rem_m - (1 if shift in ['M', 'B'] else 0)
            na = rem_a - (1 if shift in ['A', 'B'] else 0)
            ne = rem_e - (1 if shift == 'E' else 0)
            nb = b_count + (1 if shift == 'B' else 0)
            
            if backtrack(d, n + 1, nm, na, ne, nb):
                return True
                
            roster[n][d] = ''
            shifts_worked[n] -= shift_cost
            consecutive_work[n] = prev_streak
            
        return False

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