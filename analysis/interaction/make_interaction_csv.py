#!/usr/bin/env python3
"""
make_interaction_csv.py -- builds an interaction table in the SAME format as
interaction_keep_results_with_residues.csv from a directory of PLIP XML reports:

  1. RESIDUE NUMBERING OFFSET is corrected. Glide/IFD (Maestro pipeline)
     renumber residues, so without correction their contacts would never
     match the reference. The offset is detected from the amino-acid labels
     of the pocket residues, and engine residue numbers are mapped into the
     reference frame.
  2. LIGAND SELECTION: instead of pooling all binding sites in a file, the
     largest non-additive small molecule is selected (crystallisation
     additives such as SO4, GOL, EDO in reference PDBs would otherwise
     inflate the reference set).

Columns are identical to the original table; three diagnostic columns are
appended at the end (offset_applied, ligand_hetid, method_residues_raw).

    python3 make_interaction_csv.py -x xml_dir -o interaction_keep_v2.csv
"""
import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

INTERACTION_BLOCKS = {
    'hydrophobic_interactions': 'hydrophobic',
    'hydrogen_bonds': 'hbond',
    'salt_bridges': 'salt_bridge',
    'pi_stacks': 'pi_stack',
}
INTERACTION_ORDER = ['hbond', 'hydrophobic', 'pi_stack', 'salt_bridge']
ALL_METHODS = ['drg', 'ADFR', 'vina', 'IFD', 'Glide']
RENUMBERED = {'IFD', 'Glide'}          # Maestro pipeline

ADDITIVES = {
    'SO4', 'PO4', 'GOL', 'EDO', 'ACT', 'NAG', 'MES', 'TRS', 'PEG', 'DMS',
    'CL', 'NA', 'MG', 'ZN', 'CA', 'K', 'MN', 'FE', 'CU', 'NI', 'CD', 'HOH',
    'IOD', 'BR', 'FMT', 'ACY', 'CIT', 'TLA', 'MPD', 'BME', 'EPE', 'IMD',
    'NO3', 'SCN', 'AZI', 'FLC', 'PG4', 'P6G', '1PE', '2PE', 'SIN', 'MLI',
    'BU3', 'PGE', 'OLC', 'LMT', 'BOG', 'UNL',
}


def method_from_filename(fname):
    low = os.path.basename(fname).lower()
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
    return 'unknown'


def base_name(fname):
    return os.path.basename(fname).split('_')[0].lower()


def best_site(root):
    best, key = None, None
    for s in root.iter('bindingsite'):
        ident = s.find('identifiers')
        het = (ident.findtext('hetid') or '').strip().upper()
        ltype = (ident.findtext('ligtype') or '').strip().upper()
        if het in ADDITIVES:
            continue
        if 'SMALLMOLECULE' not in ltype and 'PEPTIDE' not in ltype:
            continue
        lp = s.find('lig_properties')
        nheavy = int(float(lp.findtext('num_heavy_atoms') or 0)) if lp is not None else 0
        inter = s.find('interactions')
        ninter = 0 if inter is None else sum(
            len(inter.find(k)) for k in INTERACTION_BLOCKS
            if inter.find(k) is not None)
        if key is None or (nheavy, ninter) > key:
            best, key = s, (nheavy, ninter)
    return best


def hetid(site):
    return (site.find('identifiers').findtext('hetid') or '').strip().upper()


def pocket(site):
    """{resnum: aa} -- all pocket residues (for offset detection)."""
    out, node = {}, site.find('bs_residues')
    if node is None:
        return out
    for e in node:
        m = re.match(r'^(-?\d+)([A-Za-z]?)$', (e.text or '').strip())
        if m:
            out[int(m.group(1))] = (e.get('aa') or '').upper()
    return out


def extract_sets(site, offset=0):
    """{type: {(resnum, resname)}} -- mapped into the reference frame."""
    sets = {lbl: set() for lbl in INTERACTION_BLOCKS.values()}
    inter = site.find('interactions')
    if inter is None:
        return sets
    for block, label in INTERACTION_BLOCKS.items():
        node = inter.find(block)
        if node is None:
            continue
        for it in node:
            num = (it.findtext('resnr') or it.findtext('resid') or '').strip()
            name = (it.findtext('restype') or it.findtext('resname') or '').strip().upper()
            if not num or not name:
                continue
            try:
                sets[label].add((int(num) - offset, name))
            except ValueError:
                pass
    return sets


def detect_offset(ref_pocket, meth_pocket, strict=True):
    """Find the constant offset that maximises amino-acid label agreement.
    Returns 0 if ambiguous (small margin over the second-best k)."""
    if not ref_pocket or not meth_pocket:
        return 0
    scores = {}
    for rm in list(meth_pocket)[:60]:
        for rr in list(ref_pocket)[:60]:
            k = rm - rr
            if k in scores:
                continue
            scores[k] = sum(1 for q, aa in meth_pocket.items()
                            if ref_pocket.get(q - k) == aa)
    order = sorted(scores.items(), key=lambda x: -x[1])
    best_k, best_n = order[0]
    second_n = order[1][1] if len(order) > 1 else 0
    n0 = scores.get(0, 0)
    shared = sum(1 for q in meth_pocket if q in ref_pocket)
    if best_k == 0:
        return 0
    if strict:
        ok = (best_n - second_n >= 10 and best_n >= 15
              and best_n >= n0 + 10
              and n0 / max(shared, 1) < 0.25)
    else:
        ok = best_n >= max(n0 + 10, 12)
    return best_k if ok else 0


def label(res):
    return '%s%d' % (res[1], res[0])


def joinlab(s):
    return ';'.join(sorted(label(r) for r in s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-x', '--xmldir', required=True)
    ap.add_argument('-o', '--out', default='interaction_keep_v2.csv')
    ap.add_argument('--loose', action='store_true',
                    help='looser offset-detection threshold (default: strict)')
    ap.add_argument('--no-offset', action='store_true',
                    help='disable offset correction entirely')
    args = ap.parse_args()

    groups = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(args.xmldir, '*.xml'))):
        m = method_from_filename(f)
        if m == 'unknown':
            continue
        try:
            site = best_site(ET.parse(f).getroot())
        except Exception:
            continue
        if site is None:
            continue
        b = base_name(f)
        if m in groups[b]:
            old = sum(len(v) for v in extract_sets(groups[b][m]).values())
            new = sum(len(v) for v in extract_sets(site).values())
            if new <= old:
                continue
        groups[b][m] = site

    rows = []
    for base, files in sorted(groups.items()):
        if 'pdb' not in files:
            continue
        ref_site = files['pdb']
        ref_pocket = pocket(ref_site)
        pdb_sets = extract_sets(ref_site, 0)
        for meth in ALL_METHODS:
            if meth not in files:
                continue
            site = files[meth]
            k = 0 if args.no_offset else detect_offset(
                ref_pocket, pocket(site), strict=not args.loose)
            met_sets = extract_sets(site, k)
            raw_sets = extract_sets(site, 0)
            for lbl in INTERACTION_ORDER:
                pset, mset = pdb_sets[lbl], met_sets[lbl]
                tp_set = pset & mset
                TP, FP, FN = len(tp_set), len(mset - pset), len(pset - mset)
                prec = TP / (TP + FP) if (TP + FP) else 0.0
                rec = TP / (TP + FN) if (TP + FN) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                kept = 100.0 * TP / len(pset) if pset else None
                rows.append({
                    'name': base, 'method': meth, 'interaction': lbl,
                    'pdb_n': len(pset), 'method_n': len(mset),
                    'TP': TP, 'FP': FP, 'FN': FN,
                    'precision': round(prec, 3), 'recall': round(rec, 3),
                    'F1': round(f1, 3),
                    'kept_pct': round(kept, 2) if kept is not None else None,
                    'pdb_residues': joinlab(pset),
                    'method_residues': joinlab(mset),
                    'kept_residues': joinlab(tp_set),
                    'offset_applied': k,
                    'ligand_hetid': hetid(site),
                    'method_residues_raw': joinlab(raw_sets[lbl]),
                })

    df = pd.DataFrame(rows).sort_values(['name', 'method', 'interaction'])
    df.to_csv(args.out, index=False)

    agg = df.groupby('method')[['TP', 'FP', 'FN']].sum()
    agg['precision'] = agg.TP / (agg.TP + agg.FP)
    agg['recall'] = agg.TP / (agg.TP + agg.FN)
    agg['F1'] = 2 * agg.precision * agg.recall / (agg.precision + agg.recall)
    print('%d structures, %d rows -> %s' % (df.name.nunique(), len(df), args.out))
    print(agg.round(3))
    nz = df[df.offset_applied != 0][['name', 'method', 'offset_applied']].drop_duplicates()
    print('\n(structure, engine) pairs with offset correction applied: %d' % len(nz))
    print(nz.groupby('method').size().to_dict())


if __name__ == '__main__':
    main()
