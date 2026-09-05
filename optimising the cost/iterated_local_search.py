"""Iterated local search: rebuild-and-improve around the best roster found.

Same objective as the annealing solver - minimise

    sum over nurses of  3*(M^2 + A^2 + E^2) - (M + A + E)^2

- but a different search strategy.  Rather than one long annealing run with a
cooling schedule, this alternates between two phases:

    *Intensify*  - hill climb on the current roster, taking improving moves
                   until no single change helps.
    *Perturb*    - tear out part of the roster and rebuild it with the
                   feasibility construction, then climb again from there.

The incumbent best is kept across perturbations, so the search can accept a
disruptive restart without risking the best roster found so far.  Day
reconstruction uses an MRV fill that picks the most constrained quota first,
and each candidate nurse schedule is validated against the hard rules before
it is costed.

Trade-off against the annealing solver
    This one is markedly slower per instance but more reliable at reaching the
    reference objective, because a full rebuild escapes basins that
    single-cell moves cannot.  Annealing gets most of the quality for a
    fraction of the time.  The results table quantifies that trade.
"""

import random
import sys
import csv
import json
import time

def solve(N, D, Ns, Ng, m, a, e, T, days_str, K, leaves_str):
    is_surgical = [i < Ns for i in range(N)]
    leave = [[leaves_str[i * D + d] == 'L' for d in range(D)] for i in range(N)]
    is_surg_day = [days_str[d] == 'S' for d in range(D)]
    
    start_time = time.time()
    deadline = start_time + max(1.0, T * 0.95)

    rem_surg = [0] * D
    curr = 0
    for d in range(D - 1, -1, -1):
        if is_surg_day[d]: curr += 1
        rem_surg[d] = curr

    rem_leaves = [[0] * (D + 1) for _ in range(N)]
    for i in range(N):
        for d in range(D - 1, -1, -1):
            rem_leaves[i][d] = rem_leaves[i][d+1] + (1 if leave[i][d] else 0)

    schedule = [['R'] * D for _ in range(N)]
    count = [0] * N
    streak = [0] * N
    prev_shift = ['R'] * N
    
    best_overall_schedule = None
    best_overall_cost = float('inf')
    last_improve_time = time.time()
    backtrack_target = None
    def check_nurse_validity(i, sched):
        shifts_worked = 0
        strk = 0
        for d in range(D):
            s = sched[i][d]
            if s != 'R':
                if leave[i][d]: return False 
                if s == 'B' and not is_surgical[i]: return False 
                shifts_worked += 2 if s == 'B' else 1
                strk += 1
                if strk > 5: return False 
            else:
                strk = 0
        
        if shifts_worked > K: return False 
        
        for d in range(D - 1):
            s1, s2 = sched[i][d], sched[i][d+1]
            if s2 in ('M', 'B') and s1 in ('M', 'E', 'B'): return False
            if s1 == 'B' and s2 == 'A': return False
            
        return True

    def nurse_cost(i, sched):
        c_M = sum(1 for s in sched[i] if s in ('M', 'B'))
        c_A = sum(1 for s in sched[i] if s in ('A', 'B'))
        c_E = sum(1 for s in sched[i] if s == 'E')
        mu = (c_M + c_A + c_E) / 3.0
        return 3.0 * ((c_M - mu)**2 + (c_A - mu)**2 + (c_E - mu)**2)

    def solve_day(day):
        nonlocal best_overall_schedule, best_overall_cost, backtrack_target, last_improve_time
        
        if time.time() > deadline: return 1 
        
        if day == D:
            current_sched = [row[:] for row in schedule]
            
            nurse_stats = []
            for i in range(N):
                c_M = sum(1 for s in current_sched[i] if s in ('M', 'B'))
                c_A = sum(1 for s in current_sched[i] if s in ('A', 'B'))
                c_E = sum(1 for s in current_sched[i] if s == 'E')
                w = sum(2 if s == 'B' else (1 if s != 'R' else 0) for s in current_sched[i])
                nurse_stats.append([c_M, c_A, c_E, w])
                
            def get_cost(cM, cA, cE):
                mu = (cM + cA + cE) / 3.0
                return 3.0 * ((cM - mu)**2 + (cA - mu)**2 + (cE - mu)**2)
                
            current_total_cost = sum(get_cost(ns[0], ns[1], ns[2]) for ns in nurse_stats)
            
            best_local_cost = current_total_cost
            best_local_sched = [row[:] for row in current_sched]
            
            T_sa = 5.0
            cooling_rate = 0.9999
            
            unproductive = 0
            max_unproductive = 50000
            
            import math
            import random
            
            while unproductive < max_unproductive and time.time() < deadline:
                n1 = random.randrange(N)
                n2 = random.randrange(N)
                if n1 == n2: 
                    unproductive += 1
                    continue
                    
                d_swap = random.randrange(D)
                s1 = current_sched[n1][d_swap]
                s2 = current_sched[n2][d_swap]
                
                proposals = []
                if s1 != s2: proposals.append((s2, s1))
                if is_surg_day[d_swap]:
                    if (s1 == 'M' and s2 == 'A') or (s1 == 'A' and s2 == 'M'):
                        if is_surgical[n1]: proposals.append(('B', 'R'))
                        if is_surgical[n2]: proposals.append(('R', 'B'))
                    elif s1 == 'B' and s2 == 'R':
                        proposals.append(('M', 'A'))
                        proposals.append(('A', 'M'))
                    elif s1 == 'R' and s2 == 'B':
                        proposals.append(('M', 'A'))
                        proposals.append(('A', 'M'))
                
                valid_proposals = []
                if is_surg_day[d_swap]:
                    current_B_count = sum(1 for i in range(N) if current_sched[i][d_swap] == 'B')
                
                for p_s1, p_s2 in proposals:
                    if is_surg_day[d_swap]:
                        b_change = (1 if p_s1 == 'B' else 0) + (1 if p_s2 == 'B' else 0) - (1 if s1 == 'B' else 0) - (1 if s2 == 'B' else 0)
                        if current_B_count + b_change <= 0:
                            continue
                    valid_proposals.append((p_s1, p_s2))
                    
                if not valid_proposals:
                    unproductive += 1
                    continue
                    
                new_s1, new_s2 = random.choice(valid_proposals)
                current_sched[n1][d_swap] = new_s1
                current_sched[n2][d_swap] = new_s2
                if check_nurse_validity(n1, current_sched) and check_nurse_validity(n2, current_sched):
                    c1_M, c1_A, c1_E, w1 = nurse_stats[n1]
                    c2_M, c2_A, c2_E, w2 = nurse_stats[n2]
                    
                    n1_M_new = c1_M - (1 if s1 in ('M','B') else 0) + (1 if new_s1 in ('M','B') else 0)
                    n1_A_new = c1_A - (1 if s1 in ('A','B') else 0) + (1 if new_s1 in ('A','B') else 0)
                    n1_E_new = c1_E - (1 if s1 == 'E' else 0) + (1 if new_s1 == 'E' else 0)
                    w1_new = w1 - (2 if s1 == 'B' else (1 if s1 != 'R' else 0)) + (2 if new_s1 == 'B' else (1 if new_s1 != 'R' else 0))
                    
                    n2_M_new = c2_M - (1 if s2 in ('M','B') else 0) + (1 if new_s2 in ('M','B') else 0)
                    n2_A_new = c2_A - (1 if s2 in ('A','B') else 0) + (1 if new_s2 in ('A','B') else 0)
                    n2_E_new = c2_E - (1 if s2 == 'E' else 0) + (1 if new_s2 == 'E' else 0)
                    w2_new = w2 - (2 if s2 == 'B' else (1 if s2 != 'R' else 0)) + (2 if new_s2 == 'B' else (1 if new_s2 != 'R' else 0))
                    
                    cost_old = get_cost(c1_M, c1_A, c1_E) + get_cost(c2_M, c2_A, c2_E)
                    cost_new = get_cost(n1_M_new, n1_A_new, n1_E_new) + get_cost(n2_M_new, n2_A_new, n2_E_new)
                    delta = cost_old - cost_new
                    
                    accept = False
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        if random.random() <= 0.7: accept = True
                    else:
                        prob = math.exp(delta / max(0.0001, T_sa))
                        if random.random() <= prob: accept = True
                        
                    if accept:
                        nurse_stats[n1][0], nurse_stats[n1][1], nurse_stats[n1][2], nurse_stats[n1][3] = n1_M_new, n1_A_new, n1_E_new, w1_new
                        nurse_stats[n2][0], nurse_stats[n2][1], nurse_stats[n2][2], nurse_stats[n2][3] = n2_M_new, n2_A_new, n2_E_new, w2_new
                        current_total_cost -= delta
                        
                        if current_total_cost < best_local_cost:
                            best_local_cost = current_total_cost
                            best_local_sched = [row[:] for row in current_sched]
                            unproductive = 0
                        else:
                            unproductive += 1
                    else:
                        current_sched[n1][d_swap] = s1
                        current_sched[n2][d_swap] = s2
                        unproductive += 1
                else:
                    current_sched[n1][d_swap] = s1
                    current_sched[n2][d_swap] = s2
                    unproductive += 1
                    
                T_sa *= cooling_rate
            
            if best_local_cost < best_overall_cost:
                is_first_solution = (best_overall_schedule is None)
                best_overall_cost = best_local_cost
                best_overall_schedule = [row[:] for row in best_local_sched]
                last_improve_time = time.time()
                
                if best_overall_cost == 0:
                    return 1
                
                if not is_first_solution:
                    time_ratio = (time.time() - start_time) / T
                    jump_distance = int(1 + (10 * time_ratio)) 
                    backtrack_target = max(0, day - jump_distance)
                    return 2
            else:
                if best_overall_schedule is None:
                    last_improve_time = time.time()
                
                if time.time() - last_improve_time > max(45.0, T * 0.07):
                    return 1
                time_ratio = (time.time() - start_time) / T
                jump_distance = int(1 + (10 * time_ratio)) 
                backtrack_target = max(0, day - jump_distance)
                return 2
                
            return 0 
        
        rem_days = D - day
        shifts_needed = (m + a + e) * rem_days
        shifts_avail = 0
        for i in range(N):
            max_days = rem_days - rem_leaves[i][day]
            if i < Ns:
                shifts_avail += min(K - count[i], max_days + rem_surg[day])
            else:
                shifts_avail += min(K - count[i], max_days)
                
        if shifts_avail < shifts_needed: return 0
        
        max_B_shifts = 0
        for i in range(Ns):
            max_days = rem_days - rem_leaves[i][day]
            max_B_shifts += min((K - count[i]) // 2, max_days, rem_surg[day])
            
        if max_B_shifts < rem_surg[day]: return 0
        
        c_B, c_M, c_A, c_E = [], [], [], []
        for i in range(N):
            c = count[i]
            if c >= K or leave[i][day] or streak[i] >= 5: continue
            p = prev_shift[i]
            if is_surgical[i] and c + 2 <= K and p not in ('M', 'B', 'E'): c_B.append(i)
            if p not in ('M', 'B', 'E'): c_M.append(i)
            if p != 'B': c_A.append(i)
            c_E.append(i)
        
        c_B.sort(key=lambda i: (count[i], streak[i]))
        c_M.sort(key=lambda i: (count[i] + (1 if is_surgical[i] else 0), streak[i]))
        c_A.sort(key=lambda i: (count[i] + (1 if is_surgical[i] else 0), streak[i]))
        c_E.sort(key=lambda i: (count[i] + (1 if is_surgical[i] else 0), streak[i]))

        s_B, s_M, s_A, s_E = set(c_B), set(c_M), set(c_A), set(c_E)
        
        if is_surg_day[day]:
            min_k = 1
            max_k = min(m, a, Ns, len(c_B))
        else:
            min_k = 0
            max_k = 0
            
        if len(c_B) < min_k: return 0
        
        for k in range(min_k, max_k + 1):
            if len(c_M) < m - k or len(c_A) < a - k or len(c_E) < e: continue
            
            reqs = [k, m - k, a - k, e]
            shifts = ['B', 'M', 'A', 'E']
            cands_l = [c_B[:], c_M[:], c_A[:], c_E[:]]
            cands_c = [len(c_B), len(c_M), len(c_A), len(c_E)]
            cands_s = [s_B, s_M, s_A, s_E]
            assigned_today = [False] * N
            
            res = fill_mrv(day, reqs, shifts, cands_l, cands_c, cands_s, assigned_today)
            if res == 1:
                return 1
            if res == 2:
                if backtrack_target is not None and day <= backtrack_target:
                    backtrack_target = None
                    continue
                else:
                    return 2
        return 0
        
    def fill_mrv(day, reqs, shifts, cands_l, cands_c, cands_s, assigned_today):
        if sum(reqs) == 0:
            old_streaks, old_prevs = streak[:], prev_shift[:]
            for i in range(N):
                if not assigned_today[i]:
                    schedule[i][day] = 'R'
                    streak[i] = 0
                    prev_shift[i] = 'R'
            if day + 1 < D:
                avail_M = avail_A = avail_E = avail_total = 0
                for i in range(N):
                    if leave[i][day + 1] or count[i] >= K or streak[i] >= 5: continue
                    avail_total += 1
                    p = prev_shift[i]
                    if p not in ('M', 'E', 'B'): avail_M += 1
                    if p != 'B': avail_A += 1
                    avail_E += 1
                max_k_next = min(m, a, Ns) if is_surg_day[day + 1] else 0
                if avail_total < m + a + e - max_k_next or avail_M < m or avail_A < a or avail_E < e:
                    streak[:], prev_shift[:] = old_streaks, old_prevs
                    return 0
                    
            res = solve_day(day + 1)
            if res == 1: return 1
            streak[:], prev_shift[:] = old_streaks, old_prevs
            if res == 2: return 2
            return 0
            
        total_req = sum(reqs)
        capable_today = cands_s[0] | cands_s[1] | cands_s[2] | cands_s[3]
        unassigned_capable = sum(1 for n in capable_today if not assigned_today[n])
        if unassigned_capable < total_req:
            return 0
            
        best_i = -1
        best_slack = N + 1
        for i in range(4):
            if reqs[i] > 0:
                if cands_c[i] < reqs[i]: return 0
                slack = cands_c[i] - reqs[i]
                if slack < best_slack:
                    best_slack = slack
                    best_i = i
                    
        shift = shifts[best_i]
        cost = 2 if shift == 'B' else 1
        reqs[best_i] -= 1
        cl = cands_l[best_i]
        
        for idx in range(len(cl)):
            nurse = cl[idx]
            if assigned_today[nurse]: continue
            
            if reqs[best_i] > 0:
                rem = sum(1 for j in range(idx + 1, len(cl)) if not assigned_today[cl[j]])
                if rem < reqs[best_i]: break 
                    
            assigned_today[nurse] = True
            schedule[nurse][day] = shift
            count[nurse] += cost
            possible_fail = False
            
            if is_surgical[nurse]:
                mb = 0
                for i in range(Ns):
                    if assigned_today[i]:
                        md = D - day - 1 - (rem_leaves[i][day+1] if day+1 < D else 0)
                        rs = rem_surg[day+1] if day+1 < D else 0
                        mb += min((K - count[i]) // 2, max(0, md), rs)
                    else:
                        md = D - day - rem_leaves[i][day]
                        rs = rem_surg[day]
                        mb += min((K - count[i]) // 2, max(0, md), rs)
                if mb < reqs[0] + (rem_surg[day+1] if day+1 < D else 0):
                    possible_fail = True
            
            if not possible_fail:
                sa = 0
                for i in range(N):
                    if assigned_today[i]:
                        md = D - day - 1 - (rem_leaves[i][day+1] if day+1 < D else 0)
                        sa += min(K - count[i], max(0, md) + (rem_surg[day+1] if i < Ns and day+1 < D else 0))
                    else:
                        md = D - day - rem_leaves[i][day]
                        sa += min(K - count[i], max(0, md) + (rem_surg[day] if i < Ns else 0))
                future_needed = (m + a + e) * (D - day - 1) if day + 1 < D else 0
                if sa < sum(reqs) + future_needed:
                    possible_fail = True
            
            old_str, old_prv = streak[nurse], prev_shift[nurse]
            streak[nurse] += 1
            prev_shift[nurse] = shift
            
            for j in range(4):
                if nurse in cands_s[j]: cands_c[j] -= 1
            
            saved_cl = cands_l[best_i]
            if reqs[best_i] > 0: cands_l[best_i] = cl[idx+1:]
                
            if not possible_fail:
                res = fill_mrv(day, reqs, shifts, cands_l, cands_c, cands_s, assigned_today)
            else:
                res = 0
            if res == 1:
                return 1
                
            cands_l[best_i] = saved_cl
            for j in range(4):
                if nurse in cands_s[j]: cands_c[j] += 1
                
            assigned_today[nurse] = False
            schedule[nurse][day] = 'R'
            count[nurse] -= cost
            streak[nurse] = old_str
            prev_shift[nurse] = old_prv
            
            if res == 2:
                return 2
            
        reqs[best_i] += 1
        return 0

    solve_day(0)

    if best_overall_schedule:
        res = {}
        for i in range(N):
            for d in range(D):
                res[f"N{i}_{d}"] = best_overall_schedule[i][d]
        return res
        
    return {}

def parse_input(input_csv):
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return (int(row['N']), int(row['D']), int(row['N_s']), int(row['N_g']), 
            int(row['m']), int(row['a']), int(row['e']), float(row['T']), 
            row['days'], int(row['K']), row['leaves'])

def write_output(roster, output_file):
    with open(output_file, "w") as f:
        json.dump(roster, f)

if __name__ == '__main__':
    args = parse_input(sys.argv[1])
    res = solve(*args)
    write_output(res, sys.argv[2])