"""Declarative encoding: state the rules in SMT-LIB and hand them to z3.

Every nurse-day-code triple becomes a Boolean, every hard rule becomes an
assertion, and an external z3 process decides the lot.  No search strategy of
our own is involved - the point is to see what a general-purpose solver makes
of the problem when it is simply told the rules.

What it teaches: the encoding is the bottleneck, not the reasoning.  An
N=50, D=30 instance expands to roughly 40,000 assertions and 1.8 MB of SMT
text that must be generated, serialised, parsed and only then solved, all
inside the same wall clock the hand-written solvers get.  The pairwise
at-most-one constraints and the sliding 6-day rest windows dominate that bulk.
It is competitive on small instances and uncompetitive exactly where the
problem gets interesting.

Requires a `z3` binary on PATH; without one it reports no solution.
"""

import sys
import csv
import json
import subprocess
import os
import tempfile
import re

SHIFTS = ['M', 'A', 'E', 'R', 'B']

def parse_input(input_csv):
    with open(input_csv, 'r') as f:
        row = next(csv.DictReader(f))
        N = int(row['N']); D = int(row['D'])
        N_s = int(row['N_s']); N_g = int(row['N_g'])
        m = int(row['m']); a = int(row['a']); e = int(row['e'])
        T = float(row['T']); days = row['days']
        K = int(row['K']); leaves = row['leaves']
    return N, D, N_s, N_g, m, a, e, T, days, K, leaves

def v(n, d, s):
    return f"x_{n}_{d}_{s}"

def build_smt(N, D, Ns, m, a, e, days, K, leaves):
    L = []
    add = L.append
    add("(set-logic ALL)")

    for n in range(N):
        for d in range(D):
            for s in SHIFTS:
                add(f"(declare-const {v(n,d,s)} Bool)")

    for n in range(N):
        for d in range(D):
            lits = " ".join(v(n, d, s) for s in SHIFTS)
            add(f"(assert (or {lits}))")
            for i in range(len(SHIFTS)):
                for j in range(i + 1, len(SHIFTS)):
                    add(f"(assert (not (and {v(n,d,SHIFTS[i])} {v(n,d,SHIFTS[j])})))")
            if n >= Ns:
                add(f"(assert (not {v(n,d,'B')}))")
            if days[d] != 'S':
                add(f"(assert (not {v(n,d,'B')}))")

    for n in range(N):
        for d in range(D):
            if leaves[n * D + d] == 'L':
                add(f"(assert {v(n,d,'R')})")

    for n in range(N):
        for d in range(D - 1):
            for p in ('M', 'B'):
                for q in ('M', 'B'):
                    add(f"(assert (not (and {v(n,d,p)} {v(n,d+1,q)})))")
            for q in ('M', 'B'):
                add(f"(assert (not (and {v(n,d,'E')} {v(n,d+1,q)})))")
            for q in ('M', 'A', 'B'):
                add(f"(assert (not (and {v(n,d,'B')} {v(n,d+1,q)})))")

    for d in range(D):
        mt = " ".join(f"(ite (or {v(n,d,'M')} {v(n,d,'B')}) 1 0)" for n in range(N))
        at = " ".join(f"(ite (or {v(n,d,'A')} {v(n,d,'B')}) 1 0)" for n in range(N))
        et = " ".join(f"(ite {v(n,d,'E')} 1 0)" for n in range(N))
        add(f"(assert (= (+ {mt}) {m}))")
        add(f"(assert (= (+ {at}) {a}))")
        add(f"(assert (= (+ {et}) {e}))")

    for n in range(N):
        for d in range(D - 5):
            lits = " ".join(v(n, dd, 'R') for dd in range(d, d + 6))
            add(f"(assert (or {lits}))")

    for d in range(D):
        if days[d] == 'S':
            lits = " ".join(v(n, d, 'B') for n in range(Ns))
            add(f"(assert (or {lits}))" if Ns > 0 else "(assert false)")

    for n in range(N):
        terms = []
        for d in range(D):
            terms.append(f"(ite {v(n,d,'M')} 1 0)")
            terms.append(f"(ite {v(n,d,'A')} 1 0)")
            terms.append(f"(ite {v(n,d,'E')} 1 0)")
            terms.append(f"(ite {v(n,d,'B')} 2 0)")
        add(f"(assert (<= (+ {' '.join(terms)}) {K}))")

    add("(check-sat)")
    add("(get-model)")
    return "\n".join(L)

def parse_model(out, N, D):
    vals = {}
    for mo in re.finditer(r"\(define-fun\s+(x_\d+_\d+_[MAERB])\s*\(\)\s*Bool\s+(true|false)\s*\)", out):
        vals[mo.group(1)] = (mo.group(2) == 'true')
    roster = {}
    for n in range(N):
        for d in range(D):
            got = 'R'
            for s in SHIFTS:
                if vals.get(v(n, d, s), False):
                    got = s
                    break
            roster[f"N{n}_{d}"] = got
    return roster

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: smt_encoding_z3.py <input_csv_path> <output_json_path>")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    N, D, Ns, Ng, m, a, e, T, days, K, leaves = parse_input(inp)

    smt = build_smt(N, D, Ns, m, a, e, days, K, leaves)
    fd, path = tempfile.mkstemp(suffix='.smt2')
    with os.fdopen(fd, 'w') as f:
        f.write(smt)

    budget = max(1.0, T - 2.0)
    try:
        r = subprocess.run(['z3', f'-T:{int(budget)}', path],
                           capture_output=True, text=True, timeout=budget + 5)
        out = r.stdout
    except subprocess.TimeoutExpired:
        out = ''
    finally:
        os.unlink(path)

    if out.lstrip().startswith('sat'):
        roster = parse_model(out, N, D)
        with open(outp, 'w') as f:
            json.dump(roster, f, indent=4)
    else:
        with open(outp, 'w') as f:
            json.dump({}, f)
