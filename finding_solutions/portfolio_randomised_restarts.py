"""Two-phase hybrid with randomised restarts in the fallback search.

The two-phase solver plus randomisation: the fallback DFS shuffles its nurse
order with decaying probability as it descends, so a failed attempt does not
retry the same doomed ordering.  It also front-loads a set of analytic
feasibility pre-checks so hopeless instances are rejected without any search.

What it teaches: randomisation is a cheap substitute for the diversity a
restart schedule provides.  Compare with constraint_propagation_luby.py,
which does the same job deliberately with a Luby sequence.
"""

import csv
import itertools
import json
import random
import sys
import time

sys.setrecursionlimit(20000)

BIT_M, BIT_A, BIT_E, BIT_B, BIT_R = 1, 2, 4, 8, 16
BIT = {'M': BIT_M, 'A': BIT_A, 'E': BIT_E, 'B': BIT_B, 'R': BIT_R}
SHIFT_COST = {'M': 1, 'A': 1, 'E': 1, 'R': 0, 'B': 2}
RANK = {'B': 4, 'M': 3, 'A': 2, 'E': 1, 'R': 0}

ALLOWED_AFTER = {
    '': ['M', 'A', 'E', 'B'],
    'R': ['M', 'A', 'E', 'B'],
    'M': ['A', 'E'],
    'A': ['M', 'A', 'E', 'B'],
    'E': ['A', 'E'],
    'B': ['E'],
}
ALLOWED_MASK = {k: sum(BIT[s] for s in v) for k, v in ALLOWED_AFTER.items()}
MAX_CONSECUTIVE = 5

def parse_input(input_csv):
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        N = int(row['N'])
        D = int(row['D'])
        Ns = int(row['N_s'])
        Ng = int(row['N_g'])
        req_m = int(row['m'])
        req_a = int(row['a'])
        req_e = int(row['e'])
        T = float(row['T'])
        days = row['days']
        max_shifts = int(row['K'])
        leaves = row['leaves']
    return N, D, Ns, Ng, req_m, req_a, req_e, T, days, max_shifts, leaves

def solve_roster(input_file, output_file):
    N, D, Ns, Ng, req_m, req_a, req_e, T, days, max_shifts, leaves = parse_input(input_file)

    start_time = time.time()
    deadline = start_time + max(1.0, T - 1.5)

    total_shifts_needed = D * (req_m + req_a + req_e)
    total_max_cap = N * max_shifts
    if total_max_cap < total_shifts_needed:
        with open(output_file, 'w') as f: json.dump({}, f)
        return

    future_s_days = [sum(1 for dd in range(d, D) if days[dd] == 'S') for d in range(D + 1)]
    if future_s_days[0] > 0 and Ns == 0:
        with open(output_file, 'w') as f: json.dump({}, f)
        return

    is_on_leave = [[leaves[i * D + j] == 'L' for j in range(D)] for i in range(N)]
    workable_from = [[sum(0 if is_on_leave[i][dd] else 1 for dd in range(d, D)) for d in range(D + 1)] for i in range(N)]

    for d in range(D):
        if days[d] == 'S':
            if not any(not is_on_leave[i][d] for i in range(Ns)):
                with open(output_file, 'w') as f: json.dump({}, f)
                return
            
    num_s_days = future_s_days[0]
    if num_s_days > 0:
        max_b_supply = Ns * min(max_shifts // 2, (D + 1) // 2)
        if max_b_supply < num_s_days:
            with open(output_file, 'w') as f: json.dump({}, f)
            return

    max_workable_days = D - D // 6
    rest_capped_supply = N * min(max_shifts, max_workable_days)
    if rest_capped_supply < total_shifts_needed:
        with open(output_file, 'w') as f: json.dump({}, f)
        return

    phase1_deadline = start_time + min(T * 0.05, 1)

    cb_roster = [[''] * D for _ in range(N)]
    cb_shifts_worked = [0] * N
    cb_consecutive_work = [0] * N

    def can_B_cb(n, d):
        if n >= Ns: return False
        if cb_shifts_worked[n] + 2 > max_shifts: return False
        if cb_consecutive_work[n] >= 5: return False
        prev = cb_roster[n][d - 1] if d > 0 else ''
        if prev in ['M', 'E', 'B']: return False
        return True

    def can_M_cb(n, d):
        if cb_shifts_worked[n] + 1 > max_shifts: return False
        if cb_consecutive_work[n] >= 5: return False
        prev = cb_roster[n][d - 1] if d > 0 else ''
        if prev in ['M', 'E', 'B']: return False
        return True

    def can_A_cb(n, d):
        if cb_shifts_worked[n] + 1 > max_shifts: return False
        if cb_consecutive_work[n] >= 5: return False
        prev = cb_roster[n][d - 1] if d > 0 else ''
        if prev == 'B': return False
        return True

    def can_E_cb(n, d):
        if cb_shifts_worked[n] + 1 > max_shifts: return False
        if cb_consecutive_work[n] >= 5: return False
        return True

    def apply_rest_cb(d, current_used):
        old_streaks = []
        for n in range(N):
            if not current_used[n]:
                cb_roster[n][d] = 'R'
                old_streaks.append((n, cb_consecutive_work[n]))
                cb_consecutive_work[n] = 0
        return old_streaks

    def revert_rest_cb(d, old_streaks):
        for n, old_cw in old_streaks:
            cb_roster[n][d] = ''
            cb_consecutive_work[n] = old_cw

    def backtrack_cb(d):
        if d == D: return True
        if time.time() > phase1_deadline: return False

        shifts_needed = (D - d) * (req_m + req_a + req_e)
        rem_cap = sum(max_shifts - cb_shifts_worked[i] for i in range(N))
        if rem_cap < shifts_needed:
            return False

        b_cap = sum((max_shifts - cb_shifts_worked[i]) // 2 for i in range(Ns))
        if b_cap < future_s_days[d]:
            return False

        shift_sets = []
        if days[d] == 'G':
            shift_sets.append([('M', req_m), ('A', req_a), ('E', req_e)])
        else:
            for b in range(1, min(req_m, req_a) + 1):
                shift_sets.append([('B', b), ('M', req_m - b), ('A', req_a - b), ('E', req_e)])

        for sset in shift_sets:
            def assign_groups_cb(idx, current_used):
                if time.time() > phase1_deadline: return False

                if idx == len(sset):
                    rem_b_cap = sum((max_shifts - cb_shifts_worked[i]) // 2 for i in range(Ns))
                    if rem_b_cap < future_s_days[d + 1]:
                        return False
                    old_streaks = apply_rest_cb(d, current_used)
                    if backtrack_cb(d + 1): return True
                    revert_rest_cb(d, old_streaks)
                    return False

                shift, count = sset[idx]
                if count == 0:
                    return assign_groups_cb(idx + 1, current_used)

                capable = []
                for n in range(N):
                    if current_used[n] or is_on_leave[n][d]: continue
                    if shift == 'B' and can_B_cb(n, d): capable.append(n)
                    elif shift == 'M' and can_M_cb(n, d): capable.append(n)
                    elif shift == 'A' and can_A_cb(n, d): capable.append(n)
                    elif shift == 'E' and can_E_cb(n, d): capable.append(n)

                def nurse_score_cb(n, s):
                    score = (cb_shifts_worked[n] / max(1, max_shifts)) * 100.0 + cb_consecutive_work[n] * 3.0
                    rem_s = future_s_days[d + 1]
                    if n < Ns and s in ('M', 'A', 'E'):
                        if rem_s > 0:
                            rem_b_cap = sum((max_shifts - cb_shifts_worked[i]) // 2 for i in range(Ns))
                            if rem_b_cap <= rem_s + 2:
                                b_left = max_shifts - cb_shifts_worked[n]
                                if b_left % 2 == 0:
                                    score += 50.0
                            scarcity = rem_s / max(1, rem_b_cap)
                            if scarcity > 0.8:
                                score += 30.0 * scarcity
                    if d + 1 < D and is_on_leave[n][d + 1]:
                        score -= 5.0
                    if d + 1 < D and days[d + 1] == 'S' and n < Ns:
                        if s in ('M', 'E', 'B'):
                            score += 30.0
                    return score

                capable.sort(key=lambda x: nurse_score_cb(x, shift))
                pool = capable[:count + 8]

                comb_counter = 0
                for subset in itertools.combinations(pool, count):
                    comb_counter += 1
                    if comb_counter > 15: break

                    for n in subset:
                        cb_roster[n][d] = shift
                        cb_shifts_worked[n] += (2 if shift == 'B' else 1)
                        cb_consecutive_work[n] += 1
                        current_used[n] = True

                    if assign_groups_cb(idx + 1, current_used):
                        return True

                    for n in subset:
                        cb_roster[n][d] = ''
                        cb_shifts_worked[n] -= (2 if shift == 'B' else 1)
                        cb_consecutive_work[n] -= 1
                        current_used[n] = False

                return False

            if assign_groups_cb(0, [False] * N):
                return True

        return False

    cb_success = backtrack_cb(0)
    if cb_success:
        solution = {f"N{i}_{j}": cb_roster[i][j] for i in range(N) for j in range(D)}
        with open(output_file, 'w') as f:
            json.dump(solution, f, indent=4)
        return

    roster = [[''] * D for _ in range(N)]
    shifts_worked = [0] * N
    consecutive_work = [0] * N
    base = [0] * N
    tick = [0]

    def compute_base(d):
        surgical_day = (days[d] == 'S')
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

    def backtrack(d, idx, rem_m, rem_a, rem_e, b_count, order_n):
        tick[0] += 1
        if not (tick[0] & 1023):
            if time.time() > deadline:
                raise TimeoutError

        if d == D: return True

        if idx == N:
            if rem_m != 0 or rem_a != 0 or rem_e != 0: return False
            if days[d] == 'S' and b_count == 0: return False

            nd = d + 1
            if nd == D: return True

            rem_shifts_needed = (D - nd) * (req_m + req_a + req_e)
            if sum(max_shifts - shifts_worked[i] for i in range(N)) < rem_shifts_needed:
                return False

            rem_s = future_s_days[nd]
            if rem_s > 0 and sum(max_shifts - shifts_worked[i] for i in range(Ns)) < rem_s * 2:
                return False

            saved = base[:]
            compute_base(nd)
            depth = nd * N
            epsilon = 0.8 * (1.0 - (depth / (D * N)))
            if random.random() < epsilon:
                next_order = list(range(N))
                random.shuffle(next_order)
            else:
                next_order = sorted(range(N), key=lambda x: (shifts_worked[x], consecutive_work[x]))
            if backtrack(nd, 0, req_m, req_a, req_e, 0, next_order): return True
            base[:] = saved
            return False

        n = order_n[idx]
        nurses_left = N - idx
        surgical_left = sum(1 for k in range(idx, N) if order_n[k] < Ns)

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
        if mask == 0: return False

        twin = -1
        if idx > 0:
            p = order_n[idx - 1]
            if (p < Ns) == (n < Ns) and is_on_leave[p][d] == is_on_leave[n][d]:
                if shifts_worked[p] == shifts_worked[n] and consecutive_work[p] == consecutive_work[n]:
                    if (roster[p][d - 1] if d > 0 else '') == (roster[n][d - 1] if d > 0 else ''):
                        if workable_from[p][d] == workable_from[n][d]:
                            twin = p

        def shift_priority(s):
            weight = 0
            if s == 'B': weight = rem_m + rem_a + 20
            elif s == 'M': weight = rem_m
            elif s == 'A': weight = rem_a
            elif s == 'E': weight = rem_e

            rem_s = future_s_days[d + 1]
            if rem_s > 0 and n < Ns:
                scarcity = rem_s / max(1, Ns)
                if scarcity > 1.2:
                    if s in ('M', 'A', 'E'): weight -= int(15 * scarcity)
                    if d + 1 < D and days[d + 1] == 'S' and s in ('M', 'E', 'B'): weight -= int(25 * scarcity)

            if d + 1 < D and is_on_leave[n][d + 1] and s != 'R': weight += 10
            return weight

        order = []
        if mask & BIT_B: order.append(('B', shift_priority('B')))
        if mask & BIT_M: order.append(('M', shift_priority('M')))
        if mask & BIT_A: order.append(('A', shift_priority('A')))
        if mask & BIT_E: order.append(('E', shift_priority('E')))
        order.sort(key=lambda x: x[1], reverse=True)

        domain = [x[0] for x in order]
        if mask & BIT_R:
            domain.append('R')
            budget_n = max_shifts - shifts_worked[n]
            workable_n = workable_from[n][d]
            if not (budget_n >= workable_n and workable_n > 0) and len(domain) > 1:
                if (rem_m + rem_a + rem_e) < nurses_left:
                    domain.insert(0, domain.pop())

        for shift in domain:
            if twin >= 0 and RANK[shift] > RANK[roster[twin][d]]: continue
            shift_cost = SHIFT_COST[shift]
            if shifts_worked[n] + shift_cost > max_shifts: continue

            prev_streak = consecutive_work[n]
            roster[n][d] = shift
            shifts_worked[n] += shift_cost
            consecutive_work[n] = 0 if shift == 'R' else prev_streak + 1

            nm = rem_m - (1 if shift in ('M', 'B') else 0)
            na = rem_a - (1 if shift in ('A', 'B') else 0)
            ne = rem_e - (1 if shift == 'E' else 0)
            nb = b_count + (1 if shift == 'B' else 0)

            if backtrack(d, idx + 1, nm, na, ne, nb, order_n): return True

            roster[n][d] = ''
            shifts_worked[n] -= shift_cost
            consecutive_work[n] = prev_streak

        return False

    success = False
    tick[0] = 0
    compute_base(0)
    
    epsilon = 0.8
    if random.random() < epsilon:
        init_order = list(range(N))
        random.shuffle(init_order)
    else:
        init_order = sorted(range(N), key=lambda x: (shifts_worked[x], consecutive_work[x]))
        
    try:
        success = backtrack(0, 0, req_m, req_a, req_e, 0, init_order)
    except (TimeoutError, RecursionError):
        success = False

    if success:
        solution = {f"N{i}_{j}": roster[i][j] for i in range(N) for j in range(D)}
        with open(output_file, 'w') as f:
            json.dump(solution, f, indent=4)
    else:
        with open(output_file, 'w') as f:
            json.dump({}, f)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: portfolio_randomised_restarts.py <input_csv_path> <output_json_path>")
        sys.exit(1)
    solve_roster(sys.argv[1], sys.argv[2])