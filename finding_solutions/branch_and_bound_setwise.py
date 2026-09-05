"""Group branching with MRV over quotas and aggressive global capacity bounds.

Refines the group-beam idea in two ways.  First, the quota to fill next is
chosen by slack - whichever of B/M/A/E has the fewest eligible nurses relative
to what it needs - which is MRV applied to *quotas* rather than to nurses.
Second, before descending it checks several global bounds: total remaining
capacity against total remaining demand, surgical B capacity against the
remaining surgical days, and per-shift availability for tomorrow.

What it teaches: pairing group branching with bounds that look at the whole
remaining horizon, rather than just the next day, is what turns a good
heuristic into one that clears the hard suites in milliseconds.
"""

import sys 
import csv
import json
import time

schedule = []
isSurgical = []
count = []
K = 0
leave = []
N = 0
D = 0
Ns = 0
m = 0
a = 0
e = 0
isSurgDay = []
surgLeft = []
leavesLeft = []
streak = []
prevShift = []

def solveDay(day):
    global schedule, count, streak, prevShift, K, N, D, Ns, m, a, e
    global isSurgical, leave, isSurgDay, surgLeft, leavesLeft
    
    if day == D: return True
    
    daysLeft = D-day
    shiftsNeeded = (m+a+e)*daysLeft
    shiftsAvail = 0
    for i in range(N):
        maxDays = daysLeft-leavesLeft[i][day]
        if i < Ns:
            shiftsAvail += min(K-count[i], maxDays+surgLeft[day])
        else:
            shiftsAvail += min(K-count[i], maxDays)
            
    if shiftsAvail < shiftsNeeded:
        return False
    
    maxB = 0
    for i in range(Ns):
        maxDays = daysLeft-leavesLeft[i][day]
        maxB += min((K-count[i])//2, maxDays, surgLeft[day])
        
    if maxB < surgLeft[day]:
        return False
    
    poolB, poolM, poolA, poolE = [], [], [], []
    for i in range(N):
        currC = count[i]
        if currC >= K or leave[i][day] or streak[i] >= 5: continue
        prevS = prevShift[i]
        if isSurgical[i] and currC+2 <= K and prevS not in ('M', 'B', 'E'): poolB.append(i)
        if prevS not in ('M', 'B', 'E'): poolM.append(i)
        if prevS != 'B': poolA.append(i)
        poolE.append(i)
    
    def sortKeyB(i):
        return (count[i], streak[i])
    
    def sortKeyGeneral(i):
        val = count[i]
        if isSurgical[i]:
            val += 1
        return (val, streak[i])
    
    poolB.sort(key=sortKeyB)
    poolM.sort(key=sortKeyGeneral)
    poolA.sort(key=sortKeyGeneral)
    poolE.sort(key=sortKeyGeneral)

    setB, setM, setA, setE = set(poolB), set(poolM), set(poolA), set(poolE)
    
    if isSurgDay[day]:
        minK = 1
        maxK = min(m, a, Ns, len(poolB))
    else:
        minK = 0
        maxK = 0
        
    if len(poolB) < minK:
        return False
    
    for k in range(minK, maxK+1):
        if len(poolM) < m-k or len(poolA) < a-k or len(poolE) < e:
            continue
        
        reqs = [k, m-k, a-k, e]
        shifts = ['B', 'M', 'A', 'E']
        pools = [poolB[:], poolM[:], poolA[:], poolE[:]]
        poolSizes = [len(poolB), len(poolM), len(poolA), len(poolE)]
        poolSets = [setB, setM, setA, setE]
        assignedToday = [False]*N
        
        if fillMrv(day, reqs, shifts, pools, poolSizes, poolSets, assignedToday):
            return True
    return False
    
def fillMrv(day, reqs, shifts, pools, poolSizes, poolSets, assignedToday):
    global schedule, count, streak, prevShift, K, N, D, Ns, m, a, e
    global isSurgical, leave, isSurgDay, surgLeft, leavesLeft
    
    if sum(reqs) == 0:
        oldStreaks, oldPrevs = streak[:], prevShift[:]
        for i in range(N):
            if not assignedToday[i]:
                schedule[i][day] = 'R'
                streak[i] = 0
                prevShift[i] = 'R'
        if day+1 < D:
            availM = availA = availE = availTot = 0
            for i in range(N):
                if leave[i][day+1] or count[i] >= K or streak[i] >= 5: continue
                availTot += 1
                prevS = prevShift[i]
                if prevS not in ('M', 'E', 'B'): availM += 1
                if prevS != 'B': availA += 1
                availE += 1
            maxKNext = min(m, a, Ns) if isSurgDay[day+1] else 0
            if availTot < m+a+e-maxKNext or availM < m or availA < a or availE < e:
                streak[:], prevShift[:] = oldStreaks, oldPrevs
                return False
                
        if solveDay(day+1): return True
        streak[:], prevShift[:] = oldStreaks, oldPrevs
        return False
        
    totReq = sum(reqs)
    capable = poolSets[0] | poolSets[1] | poolSets[2] | poolSets[3]
    unassigned = sum(1 for n in capable if not assignedToday[n])
    if unassigned < totReq:
        return False
        
    bestIdx = -1
    bestSlack = N+1
    for i in range(4):
        if reqs[i] > 0:
            if poolSizes[i] < reqs[i]: return False
            slack = poolSizes[i]-reqs[i]
            if slack < bestSlack:
                bestSlack = slack
                bestIdx = i
                
    shift = shifts[bestIdx]
    cost = 2 if shift == 'B' else 1
    reqs[bestIdx] -= 1
    currPool = pools[bestIdx]
    
    for idx in range(len(currPool)):
        nurse = currPool[idx]
        if assignedToday[nurse]: continue
        
        if reqs[bestIdx] > 0:
            rem = sum(1 for j in range(idx+1, len(currPool)) if not assignedToday[currPool[j]])
            if rem < reqs[bestIdx]: break 
                
        assignedToday[nurse] = True
        schedule[nurse][day] = shift
        count[nurse] += cost
        
        possibleFail = False
        
        if isSurgical[nurse]:
            futMaxB = 0
            for i in range(Ns):
                if assignedToday[i]:
                    futDays = D-day-1-(leavesLeft[i][day+1] if day+1 < D else 0)
                    futSurg = surgLeft[day+1] if day+1 < D else 0
                    futMaxB += min((K-count[i])//2, max(0, futDays), futSurg)
                else:
                    futDays = D-day-leavesLeft[i][day]
                    futSurg = surgLeft[day]
                    futMaxB += min((K-count[i])//2, max(0, futDays), futSurg)
            if futMaxB < reqs[0]+(surgLeft[day+1] if day+1 < D else 0):
                possibleFail = True
        
        if not possibleFail:
            futAvail = 0
            for i in range(N):
                if assignedToday[i]:
                    futDays = D-day-1-(leavesLeft[i][day+1] if day+1 < D else 0)
                    futAvail += min(K-count[i], max(0, futDays)+(surgLeft[day+1] if i < Ns and day+1 < D else 0))
                else:
                    futDays = D-day-leavesLeft[i][day]
                    futAvail += min(K-count[i], max(0, futDays)+(surgLeft[day] if i < Ns else 0))
            futureNeeded = (m+a+e)*(D-day-1) if day+1 < D else 0
            if futAvail < sum(reqs)+futureNeeded:
                possibleFail = True
        
        oldStrk, oldPrev = streak[nurse], prevShift[nurse]
        streak[nurse] += 1
        prevShift[nurse] = shift
        
        for j in range(4):
            if nurse in poolSets[j]: poolSizes[j] -= 1
        
        savedPool = pools[bestIdx]
        if reqs[bestIdx] > 0: pools[bestIdx] = currPool[idx+1:]
            
        if not possibleFail and fillMrv(day, reqs, shifts, pools, poolSizes, poolSets, assignedToday):
            return True
            
        pools[bestIdx] = savedPool
        for j in range(4):
            if nurse in poolSets[j]: poolSizes[j] += 1
            
        assignedToday[nurse] = False
        schedule[nurse][day] = 'R'
        count[nurse] -= cost
        streak[nurse] = oldStrk
        prevShift[nurse] = oldPrev
        
    reqs[bestIdx] += 1
    return False

def solve(NIn, DIn, NsIn, NgIn, mIn, aIn, eIn, T, days, maxShifts, leaves):
    global schedule, isSurgical, count, K, leave
    global N, D, Ns, m, a, e, isSurgDay, surgLeft, leavesLeft, streak, prevShift
    
    N = NIn
    D = DIn
    Ns = NsIn
    m = mIn
    a = aIn
    e = eIn
    K = maxShifts
    
    schedule = [['R']*D for _ in range(N)]
    count = [0]*N
    leave = [[leaves[i*D+d] == 'L' for d in range(D)] for i in range(N)]
    isSurgical = [i < Ns for i in range(N)]
    isSurgDay = [days[d] == 'S' for d in range(D)]
    streak = [0]*N
    prevShift = ['R']*N
    
    surgLeft = [0]*D
    curr = 0
    for d in range(D-1, -1, -1):
        if isSurgDay[d]: curr += 1
        surgLeft[d] = curr

    leavesLeft = [[0]*(D+1) for _ in range(N)]
    for i in range(N):
        for d in range(D-1, -1, -1):
            leavesLeft[i][d] = leavesLeft[i][d+1]+(1 if leave[i][d] else 0)

    if solveDay(0):
        roster = {}
        for i in range(N):
            for d in range(D):
                roster[f"N{i}_{d}"] = schedule[i][d]
        return roster
    return {}

def parse_input(input_csv):
    """Reads the CSV and initializes problem variables."""
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        print(row)
    
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

def write_output(roster, output_file):
    with open(output_file, "w") as f:
        json.dump(roster, f)

if __name__ == '__main__': 

    N, D, Ns, Ng, m, a, e, T, days, max_shifts, leaves = parse_input(sys.argv[1])
    output_file = sys.argv[2]

    roster = solve(N, D, Ns, Ng, m, a, e, T, days, max_shifts, leaves)

    write_output(roster, output_file)