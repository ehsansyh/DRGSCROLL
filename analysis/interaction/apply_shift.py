#!/usr/bin/env python3
"""Correct residue-numbering offsets in an interaction table, but only where the
evidence is strong (identity-agreement margin >= --min-margin over the next best
shift). Residue numbers are moved into the pdb_residues frame and TP/FP/FN and the
metrics are recomputed for those rows; all other rows are left untouched.

Example:
    python analysis/interaction/apply_shift.py         -i interaction_keep_results_with_residues.csv         -o interaction_keep_results_shift_corrected.csv
"""
import argparse
import re
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('-i', '--input', required=True, help='interaction table (CSV)')
ap.add_argument('-o', '--output', required=True, help='corrected table (CSV)')
ap.add_argument('--min-margin', type=int, default=4)
args = ap.parse_args()
SRC, OUT, MIN_MARGIN = args.input, args.output, args.min_margin

def parse(s):
    if pd.isna(s) or not str(s).strip():
        return set()
    out = set()
    for t in str(s).split(';'):
        m = re.match(r'^([A-Z]{2,3})(-?\d+)$', t.strip())
        if m:
            out.add((int(m.group(2)), m.group(1)))
    return out

def lab(s):
    return ';'.join(sorted('%s%d' % (r[1], r[0]) for r in s))

d = pd.read_csv(SRC)

# --- 1) detect the shift of each (structure, engine) pair
shifts = {}
for (n, me), g in d.groupby(['name', 'method']):
    P, M = set(), set()
    for _, r in g.iterrows():
        P |= parse(r.pdb_residues)
        M |= parse(r.method_residues)
    if not P or not M:
        continue
    sc = {}
    for a, _ in M:
        for b, _ in P:
            k = a - b
            if k in sc:
                continue
            sc[k] = sum(1 for num, nm in M if (num - k, nm) in P)
    o = sorted(sc.items(), key=lambda x: -x[1])
    bk, bn = o[0]
    sn = o[1][1] if len(o) > 1 else 0
    if bk != 0 and bn > sc.get(0, 0) and (bn - sn) >= MIN_MARGIN:
        shifts[(n, me)] = (bk, bn, sc.get(0, 0), bn - sn)

# --- 2) recompute only these rows
rows = []
for _, r in d.iterrows():
    key = (r['name'], r['method'])
    k = shifts.get(key, (0,))[0]
    new = dict(r)
    new['offset_applied'] = k
    new['method_residues_raw'] = '' if pd.isna(r.method_residues) else r.method_residues
    if k:
        P = parse(r.pdb_residues)
        M = {(num - k, nm) for num, nm in parse(r.method_residues)}
        tp = P & M
        TP, FP, FN = len(tp), len(M - P), len(P - M)
        pr = TP / (TP + FP) if (TP + FP) else 0.0
        rc = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        new.update({
            'method_n': len(M), 'TP': TP, 'FP': FP, 'FN': FN,
            'precision': round(pr, 3), 'recall': round(rc, 3),
            'F1': round(f1, 3),
            'kept_pct': round(100.0 * TP / len(P), 2) if P else None,
            'method_residues': lab(M), 'kept_residues': lab(tp),
        })
    rows.append(new)

out = pd.DataFrame(rows)[list(d.columns) + ['offset_applied', 'method_residues_raw']]
out.to_csv(OUT, index=False)

print('shifted (structure, engine) pairs: %d' % len(shifts))
for (n, me), (k, bn, b0, mg) in sorted(shifts.items()):
    print('  %-6s %-6s k=%+5d  (matches %d -> %d, margin %d)' % (n.upper(), me, k, b0, bn, mg))
print()
for tag, df in [('BEFORE', d), ('AFTER', out)]:
    a = df.groupby('method')[['TP', 'FP', 'FN']].sum()
    a['P'] = a.TP / (a.TP + a.FP); a['R'] = a.TP / (a.TP + a.FN)
    a['F1'] = 2 * a.P * a.R / (a.P + a.R)
    print('=== %s ===' % tag)
    print(a[['TP', 'FP', 'FN', 'P', 'R', 'F1']].round(3))
    print()
