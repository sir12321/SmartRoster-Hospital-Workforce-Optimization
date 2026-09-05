"""Simulated annealing on the fairness objective, over a feasible starting roster.

The optimisation problem
    Part B keeps every hard rule from the feasibility problem and adds a
    quality measure to minimise:

        sum over nurses of  3*(M^2 + A^2 + E^2) - (M + A + E)^2

    which is zero exactly when a nurse's morning/afternoon/evening counts are
    equal, and grows quadratically as they skew.  Minimising it means spreading
    each *type* of shift evenly within each nurse's own workload - a nurse who
    works only mornings is penalised even if their total load is fair.

Approach
    Two stages.  First a feasibility search (the same day-by-day construction
    used in Part A) produces any valid roster at all, under a short slice of
    the time budget.  Everything after that is annealing: propose a local
    change, accept it outright if it lowers the objective, and otherwise accept
    it with probability exp(-delta / temperature).

    The temperature follows a geometric schedule from T_init down to T_min
    across the remaining budget, with reheats when the search goes too long
    without improving.  Several move classes are used - retiming a single
    nurse-day, swapping two nurses on a day, and larger multi-day
    rewrites - so the neighbourhood is not limited to single-cell edits.

    Every proposal is checked against the hard rules before it is scored, so
    the search never leaves the feasible region and the incumbent is always a
    legal roster.

Why the delta is computed per nurse
    The objective decomposes over nurses, so a move touching nurse n changes
    only n's term.  Evaluating a proposal is therefore O(1) rather than O(N),
    which is what makes the move count high enough for annealing to work
    inside the time budget.
"""

import sys
sys.setrecursionlimit(20000)
import itertools
import csv, json, time, random, math

BIT_M, BIT_A, BIT_E, BIT_B, BIT_R = 1, 2, 4, 8, 16
BIT = {'M': BIT_M, 'A': BIT_A, 'E': BIT_E, 'B': BIT_B, 'R': BIT_R}
SHIFT_COST = {'M': 1, 'A': 1, 'E': 1, 'R': 0, 'B': 2}
ALLOWED_AFTER = {
    '': ['M', 'A', 'E', 'B', 'R'],
    'R': ['M', 'A', 'E', 'B', 'R'],
    'M': ['A', 'E', 'R'],
    'A': ['M', 'A', 'E', 'B', 'R'],
    'E': ['A', 'E', 'R'],
    'B': ['E', 'R'],
}
ALLOWED_MASK = {
    '': BIT_M | BIT_A | BIT_E | BIT_B | BIT_R,
    'R': BIT_M | BIT_A | BIT_E | BIT_B | BIT_R,
    'M': BIT_A | BIT_E | BIT_R,
    'A': BIT_M | BIT_A | BIT_E | BIT_B | BIT_R,
    'E': BIT_A | BIT_E | BIT_R,
    'B': BIT_E | BIT_R
}

MAX_CONSECUTIVE = 5
RANK = {'R': 0, 'M': 1, 'A': 2, 'E': 3, 'B': 4}

def parse_input(input_csv):
    with open(input_csv, 'r') as f:
        header = f.readline().strip().split(',')
        vals = f.readline().strip().split(',')
    
    N = int(vals[0])
    D = int(vals[1])
    Ns = int(vals[2])
    Ng = int(vals[3])
    req_m = int(vals[4])
    req_a = int(vals[5])
    req_e = int(vals[6])
    T = float(vals[7])
    days = vals[8]
    max_shifts = int(vals[9])
    leaves = vals[10]
    return N, D, Ns, Ng, req_m, req_a, req_e, T, days, max_shifts, leaves

def get_initial_roster(N, D, Ns, Ng, req_m, req_a, req_e, T, days, max_shifts, leaves, start_time, deadline, is_on_leave, workable_from, future_s_days):
    total_shifts_needed = D * (req_m + req_a + req_e)
    total_max_cap = N * max_shifts
    if total_max_cap < total_shifts_needed:
        return None

    if future_s_days[0] > 0 and Ns == 0:
        return None

    for d in range(D):
        if days[d] == 'S':
            if not any(not is_on_leave[i][d] for i in range(Ns)):
                return None

    num_s_days = future_s_days[0]
    if num_s_days > 0:
        max_b_supply = Ns * min(max_shifts // 2, (D + 1) // 2)
        if max_b_supply < num_s_days:
            return None

    max_workable_days = D - D // 6
    rest_capped_supply = N * min(max_shifts, max_workable_days)
    if rest_capped_supply < total_shifts_needed:
        return None

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
        return cb_roster

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

    import random
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
        return roster
    return None


def solve(input_file, output_file):
    random.seed(42)
    N, D, Ns, Ng, req_m, req_a, req_e, T, days, max_shifts, leaves = parse_input(input_file)
    target_cost = 0.0
    start_time = time.time()
    deadline = start_time + T * 0.95
    
    future_s_days = [sum(1 for dd in range(d, D) if days[dd] == 'S') for d in range(D + 1)]
    is_on_leave = [[leaves[i * D + j] == 'L' for j in range(D)] for i in range(N)]
    workable_from = [[sum(0 if is_on_leave[i][dd] else 1 for dd in range(d, D)) for d in range(D + 1)] for i in range(N)]
    

    init_deadline = start_time + min(1.5, T * 0.20)
    roster = get_initial_roster(N, D, Ns, Ng, req_m, req_a, req_e, T, days, max_shifts, leaves, start_time, init_deadline, is_on_leave, workable_from, future_s_days)
    
    if not roster:
        with open(output_file, 'w') as f: json.dump({}, f)
        return
    is_on_leave = [[leaves[i * D + j] == 'L' for j in range(D)] for i in range(N)]
    
    m_cnt = [0] * N
    a_cnt = [0] * N
    e_cnt = [0] * N
    shifts_worked_sa = [0] * N
    for n in range(N):
        for d in range(D):
            s = roster[n][d]
            if s == 'M': m_cnt[n] += 1; shifts_worked_sa[n] += 1
            elif s == 'A': a_cnt[n] += 1; shifts_worked_sa[n] += 1
            elif s == 'E': e_cnt[n] += 1; shifts_worked_sa[n] += 1
            elif s == 'B': m_cnt[n] += 1; a_cnt[n] += 1; shifts_worked_sa[n] += 2
            
    def calc_total_cost():
        c = 0
        for i in range(N):
            s = m_cnt[i] + a_cnt[i] + e_cnt[i]
            c += (3 * m_cnt[i] - s)**2 + (3 * a_cnt[i] - s)**2 + (3 * e_cnt[i] - s)**2
        return c / 3.0
        
    best_cost = calc_total_cost()
    current_cost = best_cost
    best_roster = [list(r) for r in roster]
    
    def cost_of(s): return 2 if s == 'B' else (0 if s == 'R' else 1)
    def can_shift_sa(n, new_s, old_s, d, current_roster, current_sw):
        if new_s == old_s: return True
        if is_on_leave[n][d] and new_s != 'R': return False
        if current_sw + cost_of(new_s) - cost_of(old_s) > max_shifts: return False
        if new_s == 'B' and n >= Ns: return False
        if new_s == 'B' and days[d] != 'S': return False
        prev_s = current_roster[n][d-1] if d > 0 else ''
        if new_s not in ALLOWED_AFTER[prev_s]: return False
        next_s = current_roster[n][d+1] if d < D - 1 else ''
        if next_s and next_s not in ALLOWED_AFTER[new_s]: return False
        if new_s != 'R' and old_s == 'R':
            streak = 1
            cd = d - 1
            while cd >= 0 and current_roster[n][cd] != 'R': streak += 1; cd -= 1
            cd = d + 1
            while cd < D and current_roster[n][cd] != 'R': streak += 1; cd += 1
            if streak > 5: return False
        return True
        
    def delta_cost(n, old_s, new_s):
        om, oa, oe = m_cnt[n], a_cnt[n], e_cnt[n]
        omu = (om + oa + oe) / 3.0
        oc = 3.0 * ((om - omu)**2 + (oa - omu)**2 + (oe - omu)**2)
        nm = om - (1 if old_s in ('M','B') else 0) + (1 if new_s in ('M','B') else 0)
        na = oa - (1 if old_s in ('A','B') else 0) + (1 if new_s in ('A','B') else 0)
        ne = oe - (1 if old_s == 'E' else 0) + (1 if new_s == 'E' else 0)
        nmu = (nm + na + ne) / 3.0
        nc = 3.0 * ((nm - nmu)**2 + (na - nmu)**2 + (ne - nmu)**2)
        return nc - oc, nm, na, ne
        
    def is_valid_sched(n, sched):
        for dd in range(D):
            if is_on_leave[n][dd] and sched[dd] != 'R':
                return False
            if sched[dd] == 'B' and (n >= Ns or days[dd] != 'S'):
                return False
        for dd in range(1, D):
            if sched[dd] not in ALLOWED_AFTER[sched[dd - 1]]:
                return False
        streak = 0
        for dd in range(D):
            if sched[dd] != 'R':
                streak += 1
                if streak > 5:
                    return False
            else:
                streak = 0
        sw = sum(cost_of(sched[dd]) for dd in range(D))
        if sw > max_shifts:
            return False
        return True

    sa_start_time = time.time()
    total_sa_time = max(0.1, deadline - sa_start_time)
    T_init = 8.0
    T_min = 0.02
    temp = T_init
    iters = 0
    last_improve_time = time.time()
    iters_since_improvement = 0
    roster_sa = [list(r) for r in best_roster]
    
    n_range = range(N)
    surgical_days = [dd for dd in range(D) if days[dd] == 'S']
    has_surgical = len(surgical_days) > 0 and Ns >= 1 and N >= 2

    active_swaps = []
    if has_surgical:
        active_swaps.append('merge_split')
    if D >= 2:
        active_swaps.append('cross_day_trade')
        active_swaps.append('cross_day_bridge')
    active_swaps.append('cycle3')
    active_swaps.append('consolidation')
    active_swaps.append('rebalancing')
    active_swaps.append('standard')

    curr_swap_idx = 0
    swap_fails = 0
    SWAP_TIMEOUT = max(1000, N * D * 2)
    stagnant_cycles = 0
    total_reheats = 0
    
    if best_cost <= target_cost + 0.001:
        pass
        
    while time.time() < deadline and best_cost > target_cost + 0.001:
        iters += 1
        iters_since_improvement += 1
        
        if not (iters & 2047):
            curr_t = time.time()
            time_limit = deadline - start_time
            if (curr_t - last_improve_time) > time_limit * 0.50:
                break
                
        elapsed_ratio = min(1.0, max(0.0, (time.time() - sa_start_time) / total_sa_time))
        temp = T_init * ((T_min / T_init) ** elapsed_ratio)
        
        current_swap = active_swaps[curr_swap_idx]
        cost_before_step = best_cost

        if current_swap == 'merge_split':
            d = random.choice(surgical_days)
            
            if random.random() < 0.5:
                m_on_d = [n for n in range(N) if roster_sa[n][d] == 'M']
                a_on_d = [n for n in range(N) if roster_sa[n][d] == 'A']
                if m_on_d and a_on_d:
                    random.shuffle(m_on_d)
                    random.shuffle(a_on_d)
                    merged = False
                    for nb in m_on_d:
                        if merged: break
                        if nb < Ns and can_shift_sa(nb, 'B', 'M', d, roster_sa, shifts_worked_sa[nb]):
                            for nr in a_on_d:
                                if nr == nb: continue
                                if not can_shift_sa(nr, 'R', 'A', d, roster_sa, shifts_worked_sa[nr]): continue
                                dc1, nm1, na1, ne1 = delta_cost(nb, 'M', 'B')
                                dc2, nm2, na2, ne2 = delta_cost(nr, 'A', 'R')
                                td = dc1 + dc2
                                if td < 0 or random.random() < math.exp(-td / max(0.001, temp)):
                                    roster_sa[nb][d] = 'B'
                                    roster_sa[nr][d] = 'R'
                                    shifts_worked_sa[nb] += 1
                                    shifts_worked_sa[nr] -= 1
                                    m_cnt[nb], a_cnt[nb], e_cnt[nb] = nm1, na1, ne1
                                    m_cnt[nr], a_cnt[nr], e_cnt[nr] = nm2, na2, ne2
                                    current_cost += td
                                    if current_cost < best_cost - 0.001:
                                        best_cost = current_cost
                                        best_roster = [list(r) for r in roster_sa]
                                        iters_since_improvement = 0
                                        last_improve_time = time.time()
                                    merged = True
                                break
                    if not merged:
                        for nb in a_on_d:
                            if merged: break
                            if nb < Ns and can_shift_sa(nb, 'B', 'A', d, roster_sa, shifts_worked_sa[nb]):
                                for nr in m_on_d:
                                    if nr == nb: continue
                                    if not can_shift_sa(nr, 'R', 'M', d, roster_sa, shifts_worked_sa[nr]): continue
                                    dc1, nm1, na1, ne1 = delta_cost(nb, 'A', 'B')
                                    dc2, nm2, na2, ne2 = delta_cost(nr, 'M', 'R')
                                    td = dc1 + dc2
                                    if td < 0 or random.random() < math.exp(-td / max(0.001, temp)):
                                        roster_sa[nb][d] = 'B'
                                        roster_sa[nr][d] = 'R'
                                        shifts_worked_sa[nb] += 1
                                        shifts_worked_sa[nr] -= 1
                                        m_cnt[nb], a_cnt[nb], e_cnt[nb] = nm1, na1, ne1
                                        m_cnt[nr], a_cnt[nr], e_cnt[nr] = nm2, na2, ne2
                                        current_cost += td
                                        if current_cost < best_cost - 0.001:
                                            best_cost = current_cost
                                            best_roster = [list(r) for r in roster_sa]
                                            iters_since_improvement = 0
                                            last_improve_time = time.time()
                                        merged = True
                                    break
                    if not merged:
                        resting_surg = [n for n in range(Ns) if roster_sa[n][d] == 'R' and can_shift_sa(n, 'B', 'R', d, roster_sa, shifts_worked_sa[n])]
                        if resting_surg:
                            random.shuffle(resting_surg)
                            for ns in resting_surg:
                                if merged: break
                                for nm in m_on_d:
                                    if merged: break
                                    if not can_shift_sa(nm, 'R', 'M', d, roster_sa, shifts_worked_sa[nm]): continue
                                    for na in a_on_d:
                                        if na == nm: continue
                                        if not can_shift_sa(na, 'R', 'A', d, roster_sa, shifts_worked_sa[na]): continue
                                        dc_s, nm_s, na_s, ne_s = delta_cost(ns, 'R', 'B')
                                        dc_m, nm_m, na_m, ne_m = delta_cost(nm, 'M', 'R')
                                        dc_a, nm_a, na_a, ne_a = delta_cost(na, 'A', 'R')
                                        td = dc_s + dc_m + dc_a
                                        if td < 0 or random.random() < math.exp(-td / max(0.001, temp)):
                                            roster_sa[ns][d] = 'B'
                                            roster_sa[nm][d] = 'R'
                                            roster_sa[na][d] = 'R'
                                            shifts_worked_sa[ns] += 2
                                            shifts_worked_sa[nm] -= 1
                                            shifts_worked_sa[na] -= 1
                                            m_cnt[ns], a_cnt[ns], e_cnt[ns] = nm_s, na_s, ne_s
                                            m_cnt[nm], a_cnt[nm], e_cnt[nm] = nm_m, na_m, ne_m
                                            m_cnt[na], a_cnt[na], e_cnt[na] = nm_a, na_a, ne_a
                                            current_cost += td
                                            if current_cost < best_cost - 0.001:
                                                best_cost = current_cost
                                                best_roster = [list(r) for r in roster_sa]
                                                iters_since_improvement = 0
                                                last_improve_time = time.time()
                                            merged = True
                                            break
            else:
                b_on_d = [n for n in range(N) if roster_sa[n][d] == 'B']
                r_on_d = [n for n in range(N) if roster_sa[n][d] == 'R']
                if len(b_on_d) >= 2 and r_on_d:
                    random.shuffle(b_on_d)
                    random.shuffle(r_on_d)
                    split_done = False
                    for nb in b_on_d:
                        if split_done: break
                        if can_shift_sa(nb, 'M', 'B', d, roster_sa, shifts_worked_sa[nb]):
                            for nr in r_on_d:
                                if nr == nb: continue
                                if not can_shift_sa(nr, 'A', 'R', d, roster_sa, shifts_worked_sa[nr]): continue
                                dc1, nm1, na1, ne1 = delta_cost(nb, 'B', 'M')
                                dc2, nm2, na2, ne2 = delta_cost(nr, 'R', 'A')
                                td = dc1 + dc2
                                if td < 0 or random.random() < math.exp(-td / max(0.001, temp)):
                                    roster_sa[nb][d] = 'M'
                                    roster_sa[nr][d] = 'A'
                                    shifts_worked_sa[nb] -= 1
                                    shifts_worked_sa[nr] += 1
                                    m_cnt[nb], a_cnt[nb], e_cnt[nb] = nm1, na1, ne1
                                    m_cnt[nr], a_cnt[nr], e_cnt[nr] = nm2, na2, ne2
                                    current_cost += td
                                    if current_cost < best_cost - 0.001:
                                        best_cost = current_cost
                                        best_roster = [list(r) for r in roster_sa]
                                        iters_since_improvement = 0
                                        last_improve_time = time.time()
                                    split_done = True
                                break
                    if not split_done:
                        for nb in b_on_d:
                            if split_done: break
                            if can_shift_sa(nb, 'A', 'B', d, roster_sa, shifts_worked_sa[nb]):
                                for nr in r_on_d:
                                    if nr == nb: continue
                                    if not can_shift_sa(nr, 'M', 'R', d, roster_sa, shifts_worked_sa[nr]): continue
                                    dc1, nm1, na1, ne1 = delta_cost(nb, 'B', 'A')
                                    dc2, nm2, na2, ne2 = delta_cost(nr, 'R', 'M')
                                    td = dc1 + dc2
                                    if td < 0 or random.random() < math.exp(-td / max(0.001, temp)):
                                        roster_sa[nb][d] = 'A'
                                        roster_sa[nr][d] = 'M'
                                        shifts_worked_sa[nb] -= 1
                                        shifts_worked_sa[nr] += 1
                                        m_cnt[nb], a_cnt[nb], e_cnt[nb] = nm1, na1, ne1
                                        m_cnt[nr], a_cnt[nr], e_cnt[nr] = nm2, na2, ne2
                                        current_cost += td
                                        if current_cost < best_cost - 0.001:
                                            best_cost = current_cost
                                            best_roster = [list(r) for r in roster_sa]
                                            iters_since_improvement = 0
                                            last_improve_time = time.time()
                                        split_done = True
                                    break
        elif current_swap == 'cross_day_bridge':
            d1, d2 = random.sample(range(D), 2)
            carriers = [n for n in range(N) if roster_sa[n][d1] in ('M', 'A', 'E') and roster_sa[n][d2] == 'R' and not is_on_leave[n][d2]]
            if carriers:
                nA = random.choice(carriers)
                s = roster_sa[nA][d1]
                sched_A = roster_sa[nA][:]
                sched_A[d1] = 'R'
                sched_A[d2] = s
                if is_valid_sched(nA, sched_A):
                    r_on_d1 = [n for n in range(N) if n != nA and roster_sa[n][d1] == 'R' and not is_on_leave[n][d1]]
                    s_on_d2 = [n for n in range(N) if n != nA and roster_sa[n][d2] == s]
                    if r_on_d1 and s_on_d2:
                        def n1_ben(n):
                            sn = m_cnt[n] + a_cnt[n] + e_cnt[n]
                            return sn - 3*m_cnt[n] if s == 'M' else (sn - 3*a_cnt[n] if s == 'A' else sn - 3*e_cnt[n])
                        n1 = max(r_on_d1, key=n1_ben) if random.random() < 0.7 else random.choice(r_on_d1)
                        def n2_ben(n):
                            sn = m_cnt[n] + a_cnt[n] + e_cnt[n]
                            return 3*m_cnt[n] - sn if s == 'M' else (3*a_cnt[n] - sn if s == 'A' else 3*e_cnt[n] - sn)
                        n2 = max(s_on_d2, key=n2_ben) if random.random() < 0.7 else random.choice(s_on_d2)
                        if n1 != n2:
                            sched_1 = roster_sa[n1][:]
                            sched_1[d1] = s
                            if is_valid_sched(n1, sched_1):
                                sched_2 = roster_sa[n2][:]
                                sched_2[d2] = 'R'
                                if is_valid_sched(n2, sched_2):
                                    dc1, nm1, na1, ne1 = delta_cost(n1, 'R', s)
                                    dc2, nm2, na2, ne2 = delta_cost(n2, s, 'R')
                                    total_delta = dc1 + dc2
                                    if total_delta < 0 or random.random() < math.exp(-total_delta / max(0.001, temp)):
                                        roster_sa[nA][d1] = 'R'
                                        roster_sa[nA][d2] = s
                                        roster_sa[n1][d1] = s
                                        roster_sa[n2][d2] = 'R'
                                        shifts_worked_sa[n1] += cost_of(s)
                                        shifts_worked_sa[n2] -= cost_of(s)
                                        m_cnt[n1], a_cnt[n1], e_cnt[n1] = nm1, na1, ne1
                                        m_cnt[n2], a_cnt[n2], e_cnt[n2] = nm2, na2, ne2
                                        current_cost += total_delta
                                        if current_cost < best_cost - 0.001:
                                            best_cost = current_cost
                                            best_roster = [list(r) for r in roster_sa]
                                            iters_since_improvement = 0
                                            last_improve_time = time.time()

        elif current_swap == 'cross_day_trade':
            n1_cand = -1
            for _ in range(4):
                c = random.randint(0, N - 1)
                sc = m_cnt[c] + a_cnt[c] + e_cnt[c]
                if sc > 0 and ((3*m_cnt[c]-sc)**2 + (3*a_cnt[c]-sc)**2 + (3*e_cnt[c]-sc)**2) > 0:
                    n1_cand = c
                    break
            if n1_cand >= 0 and random.random() < 0.70:
                work_days = [dd for dd in range(D) if roster_sa[n1_cand][dd] != 'R']
                rest_days = [dd for dd in range(D) if roster_sa[n1_cand][dd] == 'R' and not is_on_leave[n1_cand][dd]]
                if work_days and rest_days:
                    d1 = random.choice(work_days)
                    d2 = random.choice(rest_days)
                else:
                    d1, d2 = random.sample(range(D), 2)
            else:
                d1, d2 = random.sample(range(D), 2)

            s1_allowed = ('M', 'A', 'E', 'B') if days[d1] == 'S' else ('M', 'A', 'E')
            s2_allowed = ('M', 'A', 'E', 'B') if days[d2] == 'S' else ('M', 'A', 'E')
            w1 = [n for n in range(N) if roster_sa[n][d1] in s1_allowed and roster_sa[n][d2] == 'R' and not is_on_leave[n][d2]]
            w2 = [n for n in range(N) if roster_sa[n][d1] == 'R' and not is_on_leave[n][d1] and roster_sa[n][d2] in s2_allowed]
            if w1 and w2:
                n1 = n1_cand if n1_cand in w1 else random.choice(w1)
                n2 = random.choice(w2)
                if n1 != n2:
                    s1 = roster_sa[n1][d1]
                    s2 = roster_sa[n2][d2]
                    h7_ok = True
                    if days[d1] == 'S' and s1 == 'B' and s2 != 'B':
                        if sum(1 for n in range(N) if roster_sa[n][d1] == 'B') < 2: h7_ok = False
                    if days[d2] == 'S' and s2 == 'B' and s1 != 'B':
                        if sum(1 for n in range(N) if roster_sa[n][d2] == 'B') < 2: h7_ok = False
                    if h7_ok:
                        sched_1 = roster_sa[n1][:]; sched_1[d1] = 'R'; sched_1[d2] = s2
                        sched_2 = roster_sa[n2][:]; sched_2[d1] = s1; sched_2[d2] = 'R'
                        if is_valid_sched(n1, sched_1) and is_valid_sched(n2, sched_2):
                            dc1, nm1, na1, ne1 = delta_cost(n1, s1, s2)
                            dc2, nm2, na2, ne2 = delta_cost(n2, s2, s1)
                            total_delta = dc1 + dc2
                            if total_delta < 0 or random.random() < math.exp(-total_delta / max(0.001, temp)):
                                roster_sa[n1][d1] = 'R'; roster_sa[n1][d2] = s2
                                roster_sa[n2][d1] = s1; roster_sa[n2][d2] = 'R'
                                shifts_worked_sa[n1] += cost_of(s2) - cost_of(s1)
                                shifts_worked_sa[n2] += cost_of(s1) - cost_of(s2)
                                m_cnt[n1], a_cnt[n1], e_cnt[n1] = nm1, na1, ne1
                                m_cnt[n2], a_cnt[n2], e_cnt[n2] = nm2, na2, ne2
                                current_cost += total_delta
                                if current_cost < best_cost - 0.001:
                                    best_cost = current_cost
                                    best_roster = [list(r) for r in roster_sa]
                                    iters_since_improvement = 0
                                    last_improve_time = time.time()

        elif current_swap == 'cycle3':
            d = random.randint(0, D - 1)
            m_on_d = [n for n in range(N) if roster_sa[n][d] == 'M']
            a_on_d = [n for n in range(N) if roster_sa[n][d] == 'A']
            e_on_d = [n for n in range(N) if roster_sa[n][d] == 'E']
            if m_on_d and a_on_d and e_on_d:
                n1 = random.choice(m_on_d)
                n2 = random.choice(a_on_d)
                n3 = random.choice(e_on_d)
                if len({n1, n2, n3}) == 3:
                    rot_done = False
                    if (can_shift_sa(n1, 'A', 'M', d, roster_sa, shifts_worked_sa[n1]) and
                        can_shift_sa(n2, 'E', 'A', d, roster_sa, shifts_worked_sa[n2]) and
                        can_shift_sa(n3, 'M', 'E', d, roster_sa, shifts_worked_sa[n3])):
                        dc1, nm1, na1, ne1 = delta_cost(n1, 'M', 'A')
                        dc2, nm2, na2, ne2 = delta_cost(n2, 'A', 'E')
                        dc3, nm3, na3, ne3 = delta_cost(n3, 'E', 'M')
                        total_delta = dc1 + dc2 + dc3
                        if total_delta < 0 or random.random() < math.exp(-total_delta / max(0.001, temp)):
                            roster_sa[n1][d] = 'A'
                            roster_sa[n2][d] = 'E'
                            roster_sa[n3][d] = 'M'
                            m_cnt[n1], a_cnt[n1], e_cnt[n1] = nm1, na1, ne1
                            m_cnt[n2], a_cnt[n2], e_cnt[n2] = nm2, na2, ne2
                            m_cnt[n3], a_cnt[n3], e_cnt[n3] = nm3, na3, ne3
                            current_cost += total_delta
                            if current_cost < best_cost - 0.001:
                                best_cost = current_cost
                                best_roster = [list(r) for r in roster_sa]
                                iters_since_improvement = 0
                                last_improve_time = time.time()
                            rot_done = True

                    if not rot_done:
                        if (can_shift_sa(n1, 'E', 'M', d, roster_sa, shifts_worked_sa[n1]) and
                            can_shift_sa(n2, 'M', 'A', d, roster_sa, shifts_worked_sa[n2]) and
                            can_shift_sa(n3, 'A', 'E', d, roster_sa, shifts_worked_sa[n3])):
                            dc1, nm1, na1, ne1 = delta_cost(n1, 'M', 'E')
                            dc2, nm2, na2, ne2 = delta_cost(n2, 'A', 'M')
                            dc3, nm3, na3, ne3 = delta_cost(n3, 'E', 'A')
                            total_delta = dc1 + dc2 + dc3
                            if total_delta < 0 or random.random() < math.exp(-total_delta / max(0.001, temp)):
                                roster_sa[n1][d] = 'E'
                                roster_sa[n2][d] = 'M'
                                roster_sa[n3][d] = 'A'
                                m_cnt[n1], a_cnt[n1], e_cnt[n1] = nm1, na1, ne1
                                m_cnt[n2], a_cnt[n2], e_cnt[n2] = nm2, na2, ne2
                                m_cnt[n3], a_cnt[n3], e_cnt[n3] = nm3, na3, ne3
                                current_cost += total_delta
                                if current_cost < best_cost - 0.001:
                                    best_cost = current_cost
                                    best_roster = [list(r) for r in roster_sa]
                                    iters_since_improvement = 0
                                    last_improve_time = time.time()

        elif current_swap == 'consolidation':
            best_var = -1
            n1 = -1
            for _ in range(4):
                c = random.randint(0, N - 1)
                sc = m_cnt[c] + a_cnt[c] + e_cnt[c]
                if sc == 0: continue
                vc = (3*m_cnt[c]-sc)**2 + (3*a_cnt[c]-sc)**2 + (3*e_cnt[c]-sc)**2
                if vc > best_var:
                    best_var = vc
                    n1 = c
            if n1 >= 0 and best_var > 0:
                sc = m_cnt[n1] + a_cnt[n1] + e_cnt[n1]
                avg = sc / 3.0
                excess = []
                if m_cnt[n1] > avg + 0.01: excess.append('M')
                if a_cnt[n1] > avg + 0.01: excess.append('A')
                if e_cnt[n1] > avg + 0.01: excess.append('E')
                if 'M' in excess and 'A' in excess and any(roster_sa[n1][dd] == 'B' for dd in range(D)):
                    excess.append('B')
                if excess:
                    target_shift = random.choice(excess)
                    days_with = [dd for dd in range(D) if roster_sa[n1][dd] == target_shift]
                    if days_with:
                        d = random.choice(days_with)
                        cand_pool = range(Ns) if target_shift == 'B' else range(N)
                        r_nurses = [n for n in cand_pool if n != n1 and roster_sa[n][d] == 'R'
                                    and can_shift_sa(n, target_shift, 'R', d, roster_sa, shifts_worked_sa[n])]
                        if r_nurses and can_shift_sa(n1, 'R', target_shift, d, roster_sa, shifts_worked_sa[n1]):
                            def cons_benefit(n):
                                sn = m_cnt[n] + a_cnt[n] + e_cnt[n]
                                if target_shift == 'M': return sn - 3*m_cnt[n]
                                elif target_shift == 'A': return sn - 3*a_cnt[n]
                                elif target_shift == 'E': return sn - 3*e_cnt[n]
                                else: return 2 * sn - 3 * (m_cnt[n] + a_cnt[n])
                            n2 = max(r_nurses, key=cons_benefit)
                            dc1, nm1, na1, ne1 = delta_cost(n1, target_shift, 'R')
                            dc2, nm2, na2, ne2 = delta_cost(n2, 'R', target_shift)
                            total_delta = dc1 + dc2
                            if total_delta < 0 or random.random() < math.exp(-total_delta / max(0.001, temp)):
                                roster_sa[n1][d] = 'R'
                                roster_sa[n2][d] = target_shift
                                shifts_worked_sa[n1] += cost_of('R') - cost_of(target_shift)
                                shifts_worked_sa[n2] += cost_of(target_shift) - cost_of('R')
                                m_cnt[n1], a_cnt[n1], e_cnt[n1] = nm1, na1, ne1
                                m_cnt[n2], a_cnt[n2], e_cnt[n2] = nm2, na2, ne2
                                current_cost += total_delta
                                if current_cost < best_cost - 0.001:
                                    best_cost = current_cost
                                    best_roster = [list(r) for r in roster_sa]
                                    iters_since_improvement = 0
                                    last_improve_time = time.time()

        elif current_swap == 'rebalancing':
            best_var = -1
            n1 = -1
            for _ in range(4):
                c = random.randint(0, N - 1)
                sc = m_cnt[c] + a_cnt[c] + e_cnt[c]
                if sc == 0: continue
                vc = (3*m_cnt[c]-sc)**2 + (3*a_cnt[c]-sc)**2 + (3*e_cnt[c]-sc)**2
                if vc > best_var:
                    best_var = vc
                    n1 = c
            if n1 >= 0 and best_var > 0:
                sc = m_cnt[n1] + a_cnt[n1] + e_cnt[n1]
                avg = sc / 3.0
                deficit = []
                if m_cnt[n1] < avg - 0.01: deficit.append('M')
                if a_cnt[n1] < avg - 0.01: deficit.append('A')
                if e_cnt[n1] < avg - 0.01: deficit.append('E')
                if deficit:
                    want_shift = random.choice(deficit)
                    rest_days = [dd for dd in range(D) if roster_sa[n1][dd] == 'R' and not is_on_leave[n1][dd]]
                    if rest_days:
                        d = random.choice(rest_days)
                        donors = [n for n in range(N) if n != n1 and roster_sa[n][d] == want_shift
                                  and can_shift_sa(n, 'R', want_shift, d, roster_sa, shifts_worked_sa[n])]
                        if donors and can_shift_sa(n1, want_shift, 'R', d, roster_sa, shifts_worked_sa[n1]):
                            def rebal_score(n):
                                sn = m_cnt[n] + a_cnt[n] + e_cnt[n]
                                if want_shift == 'M': return 3*m_cnt[n] - sn
                                elif want_shift == 'A': return 3*a_cnt[n] - sn
                                else: return 3*e_cnt[n] - sn
                            n2 = max(donors, key=rebal_score)
                            dc1, nm1, na1, ne1 = delta_cost(n1, 'R', want_shift)
                            dc2, nm2, na2, ne2 = delta_cost(n2, want_shift, 'R')
                            total_delta = dc1 + dc2
                            if total_delta < 0 or random.random() < math.exp(-total_delta / max(0.001, temp)):
                                roster_sa[n1][d] = want_shift
                                roster_sa[n2][d] = 'R'
                                shifts_worked_sa[n1] += cost_of(want_shift) - cost_of('R')
                                shifts_worked_sa[n2] += cost_of('R') - cost_of(want_shift)
                                m_cnt[n1], a_cnt[n1], e_cnt[n1] = nm1, na1, ne1
                                m_cnt[n2], a_cnt[n2], e_cnt[n2] = nm2, na2, ne2
                                current_cost += total_delta
                                if current_cost < best_cost - 0.001:
                                    best_cost = current_cost
                                    best_roster = [list(r) for r in roster_sa]
                                    iters_since_improvement = 0
                                    last_improve_time = time.time()

        elif current_swap == 'standard':
            d = random.randint(0, D - 1)
            if random.random() < 0.75 and N >= 4:
                c1, c2, c3, c4 = random.sample(n_range, 4)
                s1 = m_cnt[c1] + a_cnt[c1] + e_cnt[c1]; v1 = (3*m_cnt[c1]-s1)**2 + (3*a_cnt[c1]-s1)**2 + (3*e_cnt[c1]-s1)**2
                s2 = m_cnt[c2] + a_cnt[c2] + e_cnt[c2]; v2 = (3*m_cnt[c2]-s2)**2 + (3*a_cnt[c2]-s2)**2 + (3*e_cnt[c2]-s2)**2
                n1 = c1 if v1 > v2 else c2
                s3 = m_cnt[c3] + a_cnt[c3] + e_cnt[c3]; v3 = (3*m_cnt[c3]-s3)**2 + (3*a_cnt[c3]-s3)**2 + (3*e_cnt[c3]-s3)**2
                s4 = m_cnt[c4] + a_cnt[c4] + e_cnt[c4]; v4 = (3*m_cnt[c4]-s4)**2 + (3*a_cnt[c4]-s4)**2 + (3*e_cnt[c4]-s4)**2
                n2 = c3 if v3 > v4 else c4
                if n1 == n2: n1, n2 = random.sample(n_range, 2)
            elif N >= 2:
                n1, n2 = random.sample(n_range, 2)
            else:
                n1, n2 = 0, 0

            s1, s2 = roster_sa[n1][d], roster_sa[n2][d]
            if s1 != s2 and can_shift_sa(n1, s2, s1, d, roster_sa, shifts_worked_sa[n1]) and can_shift_sa(n2, s1, s2, d, roster_sa, shifts_worked_sa[n2]):
                dc1, nm1, na1, ne1 = delta_cost(n1, s1, s2)
                dc2, nm2, na2, ne2 = delta_cost(n2, s2, s1)
                total_delta = dc1 + dc2

                if total_delta < 0 or random.random() < math.exp(-total_delta / max(0.001, temp)):
                    roster_sa[n1][d], roster_sa[n2][d] = s2, s1
                    shifts_worked_sa[n1] += cost_of(s2) - cost_of(s1)
                    shifts_worked_sa[n2] += cost_of(s1) - cost_of(s2)
                    m_cnt[n1], a_cnt[n1], e_cnt[n1] = nm1, na1, ne1
                    m_cnt[n2], a_cnt[n2], e_cnt[n2] = nm2, na2, ne2
                    current_cost += total_delta

                    if current_cost < best_cost - 0.001:
                        best_cost = current_cost
                        best_roster = [list(r) for r in roster_sa]
                        iters_since_improvement = 0
                        last_improve_time = time.time()

        if best_cost < cost_before_step - 0.001:
            swap_fails = 0
            stagnant_cycles = 0
        else:
            swap_fails += 1
            if swap_fails >= SWAP_TIMEOUT:
                curr_swap_idx = (curr_swap_idx + 1) % len(active_swaps)
                swap_fails = 0
                if curr_swap_idx == 0:
                    stagnant_cycles += 1
                    if stagnant_cycles >= 2:
                        if time.time() - last_improve_time < min(T * 0.4, 5.0) and total_reheats < 10:
                            total_reheats += 1
                            temp = min(20.0, temp * 1.5)
                            stagnant_cycles = 0
                            roster_sa = [list(r) for r in best_roster]
                            current_cost = best_cost
                            for i in range(N):
                                m_cnt[i] = sum(1 for d_idx in range(D) if roster_sa[i][d_idx] in ('M', 'B'))
                                a_cnt[i] = sum(1 for d_idx in range(D) if roster_sa[i][d_idx] in ('A', 'B'))
                                e_cnt[i] = sum(1 for d_idx in range(D) if roster_sa[i][d_idx] == 'E')
                                shifts_worked_sa[i] = sum(cost_of(roster_sa[i][d_idx]) for d_idx in range(D))
                        else:
                            break
                
    roster_final = [list(r) for r in best_roster]
    for i in range(N):
        m_cnt[i] = sum(1 for d_idx in range(D) if roster_final[i][d_idx] in ('M', 'B'))
        a_cnt[i] = sum(1 for d_idx in range(D) if roster_final[i][d_idx] in ('A', 'B'))
        e_cnt[i] = sum(1 for d_idx in range(D) if roster_final[i][d_idx] == 'E')
        shifts_worked_sa[i] = sum(cost_of(roster_final[i][d_idx]) for d_idx in range(D))

    cleanup_deadline = min(start_time + T * 0.98, time.time() + 1.5)
    improved = True
    while improved and time.time() < cleanup_deadline and best_cost > target_cost + 0.001:
        improved = False
        for d in range(D):
            if improved or time.time() >= cleanup_deadline: break
            for n1 in range(N):
                if improved: break
                s1 = roster_final[n1][d]
                for n2 in range(n1 + 1, N):
                    s2 = roster_final[n2][d]
                    if s1 != s2 and can_shift_sa(n1, s2, s1, d, roster_final, shifts_worked_sa[n1]) and can_shift_sa(n2, s1, s2, d, roster_final, shifts_worked_sa[n2]):
                        dc1, nm1, na1, ne1 = delta_cost(n1, s1, s2)
                        dc2, nm2, na2, ne2 = delta_cost(n2, s2, s1)
                        if dc1 + dc2 < -0.001:
                            roster_final[n1][d], roster_final[n2][d] = s2, s1
                            shifts_worked_sa[n1] += cost_of(s2) - cost_of(s1)
                            shifts_worked_sa[n2] += cost_of(s1) - cost_of(s2)
                            m_cnt[n1], a_cnt[n1], e_cnt[n1] = nm1, na1, ne1
                            m_cnt[n2], a_cnt[n2], e_cnt[n2] = nm2, na2, ne2
                            best_cost += dc1 + dc2
                            best_roster = [list(r) for r in roster_final]
                            improved = True
                            break

        if not improved and has_surgical and time.time() < cleanup_deadline:
            for d in surgical_days:
                if improved: break
                m_on_d = [n for n in range(N) if roster_final[n][d] == 'M']
                a_on_d = [n for n in range(N) if roster_final[n][d] == 'A']
                resting_surg = [n for n in range(Ns) if roster_final[n][d] == 'R' and can_shift_sa(n, 'B', 'R', d, roster_final, shifts_worked_sa[n])]
                if resting_surg and m_on_d and a_on_d:
                    for ns in resting_surg:
                        if improved: break
                        for nm in m_on_d:
                            if improved or not can_shift_sa(nm, 'R', 'M', d, roster_final, shifts_worked_sa[nm]): continue
                            for na in a_on_d:
                                if na == nm or not can_shift_sa(na, 'R', 'A', d, roster_final, shifts_worked_sa[na]): continue
                                dc_s, nm_s, na_s, ne_s = delta_cost(ns, 'R', 'B')
                                dc_m, nm_m, na_m, ne_m = delta_cost(nm, 'M', 'R')
                                dc_a, nm_a, na_a, ne_a = delta_cost(na, 'A', 'R')
                                if dc_s + dc_m + dc_a < -0.001:
                                    roster_final[ns][d] = 'B'
                                    roster_final[nm][d] = 'R'
                                    roster_final[na][d] = 'R'
                                    shifts_worked_sa[ns] += 2
                                    shifts_worked_sa[nm] -= 1
                                    shifts_worked_sa[na] -= 1
                                    m_cnt[ns], a_cnt[ns], e_cnt[ns] = nm_s, na_s, ne_s
                                    m_cnt[nm], a_cnt[nm], e_cnt[nm] = nm_m, na_m, ne_m
                                    m_cnt[na], a_cnt[na], e_cnt[na] = nm_a, na_a, ne_a
                                    best_cost += dc_s + dc_m + dc_a
                                    best_roster = [list(r) for r in roster_final]
                                    improved = True
                                    break

        if not improved and time.time() < cleanup_deadline:
            for d in range(D):
                if improved: break
                m_on_d = [n for n in range(N) if roster_final[n][d] == 'M']
                a_on_d = [n for n in range(N) if roster_final[n][d] == 'A']
                e_on_d = [n for n in range(N) if roster_final[n][d] == 'E']
                for n1 in m_on_d:
                    if improved: break
                    for n2 in a_on_d:
                        if improved: break
                        for n3 in e_on_d:
                            if (can_shift_sa(n1, 'A', 'M', d, roster_final, shifts_worked_sa[n1]) and
                                can_shift_sa(n2, 'E', 'A', d, roster_final, shifts_worked_sa[n2]) and
                                can_shift_sa(n3, 'M', 'E', d, roster_final, shifts_worked_sa[n3])):
                                dc1, nm1, na1, ne1 = delta_cost(n1, 'M', 'A')
                                dc2, nm2, na2, ne2 = delta_cost(n2, 'A', 'E')
                                dc3, nm3, na3, ne3 = delta_cost(n3, 'E', 'M')
                                if dc1 + dc2 + dc3 < -0.001:
                                    roster_final[n1][d] = 'A'; roster_final[n2][d] = 'E'; roster_final[n3][d] = 'M'
                                    m_cnt[n1], a_cnt[n1], e_cnt[n1] = nm1, na1, ne1
                                    m_cnt[n2], a_cnt[n2], e_cnt[n2] = nm2, na2, ne2
                                    m_cnt[n3], a_cnt[n3], e_cnt[n3] = nm3, na3, ne3
                                    best_cost += dc1 + dc2 + dc3
                                    best_roster = [list(r) for r in roster_final]
                                    improved = True
                                    break
                            if (can_shift_sa(n1, 'E', 'M', d, roster_final, shifts_worked_sa[n1]) and
                                can_shift_sa(n2, 'M', 'A', d, roster_final, shifts_worked_sa[n2]) and
                                can_shift_sa(n3, 'A', 'E', d, roster_final, shifts_worked_sa[n3])):
                                dc1, nm1, na1, ne1 = delta_cost(n1, 'M', 'E')
                                dc2, nm2, na2, ne2 = delta_cost(n2, 'A', 'M')
                                dc3, nm3, na3, ne3 = delta_cost(n3, 'E', 'A')
                                if dc1 + dc2 + dc3 < -0.001:
                                    roster_final[n1][d] = 'E'; roster_final[n2][d] = 'M'; roster_final[n3][d] = 'A'
                                    m_cnt[n1], a_cnt[n1], e_cnt[n1] = nm1, na1, ne1
                                    m_cnt[n2], a_cnt[n2], e_cnt[n2] = nm2, na2, ne2
                                    m_cnt[n3], a_cnt[n3], e_cnt[n3] = nm3, na3, ne3
                                    best_cost += dc1 + dc2 + dc3
                                    best_roster = [list(r) for r in roster_final]
                                    improved = True
                                    break
                
    solution = {f"N{i}_{j}": best_roster[i][j] for i in range(N) for j in range(D)}
    with open(output_file, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == '__main__':
    solve(sys.argv[1], sys.argv[2])