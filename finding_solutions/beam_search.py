"""Beam search over per-day shift groups rather than per-nurse assignments.

The first solver here to branch the right way round: for a given day it fixes
how many B shifts to place, then chooses *which set of nurses* fills each
quota, taking the least-loaded candidates first.  Sets, not sequences - so the
permutation symmetry between interchangeable nurses never enters the tree.

The beam is the cap on how many candidate sets are tried per quota, which keeps
the branching factor bounded at the cost of completeness.

What it teaches: this single change - group branching instead of nurse
branching - is the largest effect in the whole study, and everything that
follows builds on it.
"""

import sys
import csv
import json
import time
import itertools

def solve():
    if len(sys.argv) < 3:
        sys.exit(1)
        
    sys.setrecursionlimit(5000)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        N = int(row['N'])
        D = int(row['D'])
        Ns = int(row['N_s'])
        req_m = int(row['m'])
        req_a = int(row['a'])
        req_e = int(row['e'])
        T = float(row['T'])
        days = row['days']
        max_shifts = int(row['K'])
        leaves = row['leaves']

    start_time = time.time()
    
    is_on_leave = [[False] * D for _ in range(N)]
    for i in range(N):
        for j in range(D):
            if leaves[i * D + j] == 'L':
                is_on_leave[i][j] = True

    roster = [[''] * D for _ in range(N)]
    shifts_worked = [0] * N
    consecutive_work = [0] * N
    
    # --- CONSTANT TIME CONSTRAINT CHECKS ---
    def can_B(n, d):
        if n >= Ns: return False
        if shifts_worked[n] + 2 > max_shifts: return False
        if consecutive_work[n] >= 5: return False
        prev = roster[n][d-1] if d > 0 else ''
        if prev in ['M', 'E', 'B']: return False
        return True

    def can_M(n, d):
        if shifts_worked[n] + 1 > max_shifts: return False
        if consecutive_work[n] >= 5: return False
        prev = roster[n][d-1] if d > 0 else ''
        if prev in ['M', 'E', 'B']: return False
        return True

    def can_A(n, d):
        if shifts_worked[n] + 1 > max_shifts: return False
        if consecutive_work[n] >= 5: return False
        prev = roster[n][d-1] if d > 0 else ''
        if prev == 'B': return False
        return True

    def can_E(n, d):
        if shifts_worked[n] + 1 > max_shifts: return False
        if consecutive_work[n] >= 5: return False
        return True

    def apply_rest(d, current_used):
        old_streaks = []
        for n in range(N):
            if not current_used[n]:
                roster[n][d] = 'R'
                old_streaks.append((n, consecutive_work[n]))
                consecutive_work[n] = 0
        return old_streaks

    def revert_rest(d, old_streaks):
        for n, old_cw in old_streaks:
            roster[n][d] = ''
            consecutive_work[n] = old_cw

    # --- CORE DFS SOLVER ---
    def backtrack(d):
        if d == D: return True
        if time.time() - start_time > min(T, 1): return False
        
        # O(1) Global capacity prune
        shifts_needed = (D - d) * (req_m + req_a + req_e)
        rem_cap = sum(max_shifts - shifts_worked[i] for i in range(N))
        if rem_cap < shifts_needed:
            return False

        # Build valid shift quotas for the day
        shift_sets = []
        if days[d] == 'G':
            shift_sets.append([('M', req_m), ('A', req_a), ('E', req_e)])
        else:
            for b in range(1, min(req_m, req_a) + 1):
                shift_sets.append([('B', b), ('M', req_m - b), ('A', req_a - b), ('E', req_e)])

        for sset in shift_sets:
            def assign_groups(idx, current_used):
                if time.time() - start_time > min(T, 10.5): return False
                
                # Base Case: Daily quotas filled, bulk-assign R to everyone else
                if idx == len(sset):
                    old_streaks = apply_rest(d, current_used)
                    if backtrack(d + 1): return True
                    revert_rest(d, old_streaks)
                    return False
                    
                shift, count = sset[idx]
                if count == 0:
                    return assign_groups(idx + 1, current_used)
                    
                # Find all legally capable nurses for this specific shift type
                capable = []
                for n in range(N):
                    if current_used[n] or is_on_leave[n][d]: continue
                    if shift == 'B' and can_B(n, d): capable.append(n)
                    elif shift == 'M' and can_M(n, d): capable.append(n)
                    elif shift == 'A' and can_A(n, d): capable.append(n)
                    elif shift == 'E' and can_E(n, d): capable.append(n)
                    
                if len(capable) < count:
                    return False
                    
                # Heuristic: Prioritize nurses with the lowest shift counts to balance load
                capable.sort(key=lambda x: shifts_worked[x])
                
                # Beam Search Cutoff: Try at most 1000 combinations to prevent Symmetry Thrashing
                comb_counter = 0
                for subset in itertools.combinations(capable, count):
                    comb_counter += 1
                    if comb_counter > 1000: break
                        
                    for n in subset:
                        roster[n][d] = shift
                        shifts_worked[n] += (2 if shift == 'B' else 1)
                        consecutive_work[n] += 1
                        current_used[n] = True
                        
                    if assign_groups(idx + 1, current_used):
                        return True
                        
                    for n in subset:
                        roster[n][d] = ''
                        shifts_worked[n] -= (2 if shift == 'B' else 1)
                        consecutive_work[n] -= 1
                        current_used[n] = False
                        
                return False
                
            if assign_groups(0, [False]*N):
                return True
                
        return False

    success = backtrack(0)
    
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

if __name__ == '__main__':
    solve()