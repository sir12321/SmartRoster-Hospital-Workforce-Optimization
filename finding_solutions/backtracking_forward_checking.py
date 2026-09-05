"""Chronological DFS with forward checking and a least-constraining-value order.

The baseline of this project, and the most direct reading of the problem:
walk the grid day by day, nurse by nurse, assigning a code and backtracking on
failure.  Three cheap forward checks fire before each nurse is considered - a
total-capacity bound, a per-shift starvation bound, and the surgical-B check -
and the codes are tried in order of how badly the day still needs them.

What it teaches: the value ordering is doing real work, but the *variable*
ordering (a fixed nurse sweep) is not.  Every permutation of interchangeable
nurses is a distinct branch here, and that is what eventually sinks it on the
larger suites.
"""

import sys
import csv
import json
import time

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
        print("usage: backtracking_forward_checking.py <input_csv_path> <output_json_path>")
        sys.exit(1)
        
    # Increase recursion depth to handle maximum possible grid size (50 nurses * 30 days = 1500 depth)
    sys.setrecursionlimit(5000)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    N, D, Ns, Ng, req_m, req_a, req_e, T, days, max_shifts, leaves = parse_input(input_file)
    
    start_time = time.time()
    deadline = start_time + max(1.0, T - 2.0)
    tick = [0]

    # convert to 2D boolean array to check if nurse is avalable on a day (fast lookups)
    is_on_leave = [[False] * D for _ in range(N)]
    for i in range(N):
        for j in range(D):
            if leaves[i * D + j] == 'L':
                is_on_leave[i][j] = True

    # Global state trackers for constant-time constraint checking
    roster = [[''] * D for _ in range(N)]
    shifts_worked = [0] * N
    consecutive_work = [0] * N

    def backtrack(d, n, rem_m, rem_a, rem_e, b_count):
        """
        Recursive DFS to assign shifts day by day, nurse by nurse.
        d: current day index
        n: current nurse index
        rem_m, rem_a, rem_e: remaining M, A, E slots needed for the current day
        b_count: number of 'B' shifts assigned on the current day
        """
        # Time budget check (checked once per day to save overhead)
        tick[0] += 1
        if not (tick[0] & 1023):
            if time.time() > deadline:
                raise TimeoutError

        # Base Case 1: Reached the end of the roster successfully
        if d == D:
            return True
            
        # Base Case 2: Reached the end of the current day
        if n == N:
            # Check H4: Exact coverage must be met
            if rem_m != 0 or rem_a != 0 or rem_e != 0:
                return False
            # Check H7: Surgical days must have at least one 'B' shift
            if days[d] == 'S' and b_count == 0:
                return False
            
            # Proceed to the next day, resetting daily requirements
            return backtrack(d + 1, 0, req_m, req_a, req_e, 0)

            
# --- ENHANCED FORWARD CHECKING (Fail Fast) ---
        nurses_left = N - n
        surgical_left = max(0, Ns - n)
        
        # Prune 1: Total capacity check
        if (rem_m + rem_a + rem_e) > (surgical_left * 2) + (nurses_left - surgical_left):
            return False
            
        # Prune 2: Specific shift starvation check
        # If we need more of a specific shift than there are nurses left, it's a dead end.
        if rem_m > nurses_left or rem_a > nurses_left or rem_e > nurses_left:
            return False
            
        # Prune 3: Surgical 'B' shift requirement check
        if days[d] == 'S' and b_count == 0 and surgical_left == 0:
            return False
            
        # --- LCV DOMAIN GENERATION (Least Constraining Value) ---
        if is_on_leave[n][d]:
            domain = ['R']
        else:
            # Build a list of valid shifts paired with their current "demand" or "need"
            valid_shifts = []
            
            if n < Ns and days[d] == 'S' and (rem_m > 0 and rem_a > 0):
                # 'B' satisfies both M and A, making it highly valuable if both are needed
                valid_shifts.append(('B', rem_m + rem_a))
                
            if rem_m > 0: valid_shifts.append(('M', rem_m))
            if rem_a > 0: valid_shifts.append(('A', rem_a))
            if rem_e > 0: valid_shifts.append(('E', rem_e))
            
            # Sort shifts by highest demand first (LCV heuristic)
            valid_shifts.sort(key=lambda x: x[1], reverse=True)
            
            # Extract just the shift characters
            domain = [s[0] for s in valid_shifts]
            
            # Always append 'R' at the end as a fallback, unless exact coverage is already met
            domain.append('R')

        # --- ITERATE OVER VALID SHIFTS ---
        for shift in domain:
            # H8 Check: Max K shifts per nurse
            # 'B' counts as 2 shifts; 'R' is 0; others are 1
            shift_cost = 2 if shift == 'B' else (0 if shift == 'R' else 1)
            if shifts_worked[n] + shift_cost > max_shifts: 
                continue
            
#  Look-back Checks (H2, H3, H6)
            if d > 0:
                prev_shift = roster[n][d-1]
                
                # If today's shift requires the morning (M or B)
                if shift in ['M', 'B']:
                    if prev_shift in ['M', 'B']: continue  # H2: No consecutive mornings
                    if prev_shift == 'E': continue         # H3: No morning after evening
                
                # H6: If yesterday was B, today cannot touch Morning or Afternoon
                if prev_shift == 'B' and shift in ['M', 'A', 'B']: 
                    continue

            # 4. H5 Check: Max 5 consecutive working days
            prev_streak = consecutive_work[n]
            if shift != 'R' and prev_streak >= 5: continue
            
            # --- APPLY STATE ---
            roster[n][d] = shift
            shifts_worked[n] += shift_cost
            consecutive_work[n] = 0 if shift == 'R' else prev_streak + 1
            
            nm = rem_m - (1 if shift in ['M', 'B'] else 0)
            na = rem_a - (1 if shift in ['A', 'B'] else 0)
            ne = rem_e - (1 if shift == 'E' else 0)
            nb = b_count + (1 if shift == 'B' else 0)
            
            # --- RECURSE ---
            if backtrack(d, n + 1, nm, na, ne, nb):
                return True
                
            # --- BACKTRACK (Revert state) ---
            roster[n][d] = ''
            shifts_worked[n] -= shift_cost
            consecutive_work[n] = prev_streak
            
        return False

    # Start the DFS search
    try:
        success = backtrack(0, 0, req_m, req_a, req_e, 0)
    except (TimeoutError, RecursionError):
        success = False
    
    # Format the output based on search result
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