#!/usr/bin/env python3
"""Take an existing interaction table and add ONLY the (structure, engine) rows
that are missing, parsed from a directory of PLIP XML reports. Existing rows
are left untouched.

Example:
    python analysis/interaction/merge_csv.py -b interaction_keep_results_with_residues.csv         -x plip_xml/ -o interaction_keep_results_with_residues_merged.csv
"""
import argparse
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

import pandas as pd

BASE = XDIR = OUT = None  # set from the command line in main()

BLOCKS = {'hydrophobic_interactions': 'hydrophobic',
          'hydrogen_bonds': 'hbond',
          'salt_bridges': 'salt_bridge',
          'pi_stacks': 'pi_stack'}
ORDER = ['hbond', 'hydrophobic', 'pi_stack', 'salt_bridge']
NAME = {'drg': 'drg', 'ADFR': 'ADFR', 'vina': 'vina', 'IFD': 'IFD',
        'Glide': 'Glide'}
ADDITIVES = {'SO4', 'PO4', 'GOL', 'EDO', 'ACT', 'NAG', 'MES', 'TRS', 'PEG',
             'DMS', 'CL', 'NA', 'MG', 'ZN', 'CA', 'K', 'MN', 'FE', 'CU', 'NI',
             'CD', 'HOH', 'IOD', 'BR', 'FMT', 'ACY', 'CIT', 'TLA', 'MPD',
             'BME', 'EPE', 'IMD', 'NO3', 'SCN', 'AZI', 'FLC', 'PG4', 'P6G',
             '1PE', '2PE', 'SIN', 'MLI', 'BU3', 'PGE', 'OLC', 'LMT', 'BOG',
             'UNL'}
# fix a dotless-i typo in some file names (4aı5 -> 4ai5)
FIXNAME = {'4aı5': '4ai5'}


def method_of(f):
    low = os.path.basename(f).lower()
    if '_protein' in low:
        return 'pdb'
    if 'ifd' in low:
        return 'IFD'
    if 'glide' in low:
        return 'Glide'
    if 'adfr' in low:
        return 'ADFR'
    if 'vina' in low:
        return 'vina'
    if 'drg' in low or 'comp' in low:
        return 'drg'
    return None


def pdb_of(f):
    b = os.path.basename(f).split('_')[0].lower()
    return FIXNAME.get(b, b)


def best_site(root):
    best, key = None, None
    for s in root.iter('bindingsite'):
        ident = s.find('identifiers')
        if (ident.findtext('hetid') or '').strip().upper() in ADDITIVES:
            continue
        ltype = (ident.findtext('ligtype') or '').strip().upper()
        if 'SMALLMOLECULE' not in ltype and 'PEPTIDE' not in ltype:
            continue
        lp = s.find('lig_properties')
        nh = int(float(lp.findtext('num_heavy_atoms') or 0)) if lp is not None else 0
        inter = s.find('interactions')
        ni = 0 if inter is None else sum(len(inter.find(k)) for k in BLOCKS
                                        if inter.find(k) is not None)
        if key is None or (nh, ni) > key:
            best, key = s, (nh, ni)
    return best


def sets_of(site):
    out = {v: set() for v in BLOCKS.values()}
    inter = site.find('interactions')
    if inter is None:
        return out
    for blk, lbl in BLOCKS.items():
        node = inter.find(blk)
        if node is None:
            continue
        for it in node:
            num = (it.findtext('resnr') or it.findtext('resid') or '').strip()
            nm = (it.findtext('restype') or it.findtext('resname') or '').strip().upper()
            if num and nm:
                out[lbl].add((int(num), nm))
    return out


def lab(s):
    return ';'.join(sorted('%s%d' % (r[1], r[0]) for r in s))


def main():
    global BASE, XDIR, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-b', '--base', required=True, help='existing interaction table (CSV)')
    ap.add_argument('-x', '--xml-dir', required=True, help='directory of PLIP XML reports')
    ap.add_argument('-o', '--output', required=True, help='merged table (CSV)')
    a = ap.parse_args()
    BASE, XDIR, OUT = a.base, a.xml_dir, a.output
    base = pd.read_csv(BASE)
    have = {(r['name'], r['method']) for _, r in base.iterrows()}

    files = {}
    for f in sorted(glob.glob(os.path.join(XDIR, '*.xml'))):
        m = method_of(f)
        if m is None:
            continue
        pdb = pdb_of(f)
        site = None
        try:
            site = best_site(ET.parse(f).getroot())
        except Exception:
            pass
        if site is None:
            continue
        cur = files.get((pdb, m))
        if cur is not None:
            if sum(len(v) for v in sets_of(site).values()) <= \
               sum(len(v) for v in sets_of(cur).values()):
                continue
        files[(pdb, m)] = site

    rows, added = [], []
    for (pdb, m), site in sorted(files.items()):
        if m == 'pdb' or (pdb, m) in have:
            continue
        if (pdb, 'pdb') not in files:
            print('  SKIPPED (no reference): %s %s' % (pdb, m))
            continue
        ps = sets_of(files[(pdb, 'pdb')])
        ms = sets_of(site)
        for lblname in ORDER:
            pset, mset = ps[lblname], ms[lblname]
            tp = pset & mset
            TP, FP, FN = len(tp), len(mset - pset), len(pset - mset)
            pr = TP / (TP + FP) if (TP + FP) else 0.0
            rc = TP / (TP + FN) if (TP + FN) else 0.0
            f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
            kept = 100.0 * TP / len(pset) if pset else None
            rows.append({
                'name': pdb, 'method': NAME[m], 'interaction': lblname,
                'pdb_n': len(pset), 'method_n': len(mset),
                'TP': TP, 'FP': FP, 'FN': FN,
                'precision': round(pr, 3), 'recall': round(rc, 3),
                'F1': round(f1, 3),
                'kept_pct': round(kept, 2) if kept is not None else None,
                'pdb_residues': lab(pset), 'method_residues': lab(mset),
                'kept_residues': lab(tp),
            })
        added.append((pdb, NAME[m]))

    out = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values(['name', 'method', 'interaction'],
                          kind='mergesort').reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print('added (structure, engine): %d  -> %s' % (len(added), added))
    print('total: %d structures, %d rows  (before: %d structures, %d rows)'
          % (out.name.nunique(), len(out), base.name.nunique(), len(base)))
    print('same columns:', list(out.columns) == list(base.columns))


if __name__ == '__main__':
    main()
