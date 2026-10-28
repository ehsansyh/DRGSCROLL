#!/usr/bin/env python3
"""PLIP XML -> interaction fidelity (precision/recall/F1) for each engine."""
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

ITYPES = {
    'hydrophobic_interactions': 'hydrophobic',
    'hydrogen_bonds': 'hbond',
    'water_bridges': 'water_bridge',
    'salt_bridges': 'salt_bridge',
    'pi_stacks': 'pi_stack',
    'pi_cation_interactions': 'pi_cation',
    'halogen_bonds': 'halogen',
    'metal_complexes': 'metal',
}

ADDITIVES = {
    'SO4', 'PO4', 'GOL', 'EDO', 'ACT', 'NAG', 'MES', 'TRS', 'PEG', 'DMS',
    'CL', 'NA', 'MG', 'ZN', 'CA', 'K', 'MN', 'FE', 'CU', 'NI', 'CD', 'HOH',
    'IOD', 'BR', 'FMT', 'ACY', 'CIT', 'TLA', 'MPD', 'BME', 'EPE', 'IMD',
    'NO3', 'SCN', 'AZI', 'FLC', 'PG4', 'P6G', '1PE', '2PE', 'SIN', 'MLI',
    'BU3', 'PGE', 'OLC', 'LMT', 'BOG', 'UNL',
}


def classify(fname):
    n = os.path.basename(fname).lower()
    if '_protein' in n:
        return 'REF'
    if 'ifd' in n:
        return 'IFD'
    if 'glide' in n:
        return 'Glide'
    if 'adfr' in n:
        return 'ADFR'
    if 'vina' in n:
        return 'Vina-GPU'
    if 'drg' in n or 'comp' in n:
        return 'DRGSCROLL'
    return None


def pdb_of(fname):
    return os.path.basename(fname).split('_')[0].lower()


def best_site(root):
    """Select the binding site of the largest non-additive small molecule."""
    best, best_key = None, None
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
        ninter = sum(len(s.find('interactions').find(k) or [])
                     for k in ITYPES if s.find('interactions') is not None
                     and s.find('interactions').find(k) is not None)
        key = (nheavy, ninter)
        if best_key is None or key > best_key:
            best, best_key = s, key
    return best


def bs_residues(site):
    """{resnum: (chain, aa)} -- all pocket residues."""
    out = {}
    node = site.find('bs_residues')
    if node is None:
        return out
    for e in node:
        txt = (e.text or '').strip()
        m = re.match(r'^(-?\d+)([A-Za-z]?)$', txt)
        if not m:
            continue
        out[int(m.group(1))] = (m.group(2), (e.get('aa') or '').upper())
    return out


def detect_offset(ref_bs, meth_bs):
    """An engine's residue numbering may be offset from the reference by a
    constant k (Maestro/Glide/IFD renumbering). Finds the k that maximises
    amino-acid label agreement."""
    if not ref_bs or not meth_bs:
        return 0, 0, 0
    cand = set()
    for rm in list(meth_bs)[:40]:
        for rr in list(ref_bs)[:40]:
            cand.add(rm - rr)
    best_k, best_n = 0, -1
    for k in sorted(cand, key=lambda x: (abs(x) != 0, abs(x))):
        n = sum(1 for rm, (_, aa) in meth_bs.items()
                if rm - k in ref_bs and ref_bs[rm - k][1] == aa)
        if n > best_n:
            best_k, best_n = k, n
    n0 = sum(1 for rm, (_, aa) in meth_bs.items()
             if rm in ref_bs and ref_bs[rm][1] == aa)
    return best_k, best_n, n0


def contacts(site, offset=0):
    """Set of {(type, 'RESNUM:CHAIN:AA')}."""
    out = set()
    inter = site.find('interactions')
    if inter is None:
        return out
    for tag, label in ITYPES.items():
        node = inter.find(tag)
        if node is None:
            continue
        for e in node:
            rn = (e.findtext('resnr') or '').strip()
            ch = (e.findtext('reschain') or '').strip()
            aa = (e.findtext('restype') or '').strip().upper()
            if not rn:
                continue
            try:
                num = int(rn) - offset
            except ValueError:
                continue
            out.add((label, '%d:%s' % (num, aa)))
    return out


def main(xdir, out_csv):
    data = defaultdict(dict)
    bad = []
    for f in sorted(glob.glob(os.path.join(xdir, '*.xml'))):
        m = classify(f)
        if m is None:
            bad.append(f)
            continue
        try:
            root = ET.parse(f).getroot()
        except Exception as e:
            bad.append('%s (%s)' % (f, e))
            continue
        s = best_site(root)
        if s is None:
            bad.append('%s (no bindingsite)' % f)
            continue
        pdb = pdb_of(f)
        if m in data[pdb]:                       # 2nd file for the same engine
            if len(contacts(s)) <= len(contacts(data[pdb][m])):
                continue
        data[pdb][m] = s

    # --- correct numbering offsets, then build contact sets
    offsets = []
    sets = defaultdict(dict)
    for pdb, md in sorted(data.items()):
        if 'REF' not in md:
            continue
        ref_bs = bs_residues(md['REF'])
        sets[pdb]['REF'] = contacts(md['REF'], 0)
        for meth, site in md.items():
            if meth == 'REF':
                continue
            k, nk, n0 = detect_offset(ref_bs, bs_residues(site))
            if nk < max(n0 + 10, 12):            # keep 0 unless there is strong evidence for an offset
                k, nk = 0, n0
            sets[pdb][meth] = contacts(site, k)
            offsets.append({'pdb': pdb.upper(), 'method': meth, 'offset': k,
                            'n_matched_aa': nk, 'n_matched_at_0': n0})
    pd.DataFrame(offsets).to_csv(out_csv.replace('.csv', '_offsets.csv'),
                                 index=False)
    nz = [o for o in offsets if o['offset'] != 0]
    print('files with a numbering offset detected: %d / %d' % (len(nz), len(offsets)))
    for o in nz[:12]:
        print('   %s %-10s offset=%+d (aa matches %d -> %d)'
              % (o['pdb'], o['method'], o['offset'], o['n_matched_at_0'],
                 o['n_matched_aa']))
    data = sets

    rows = []
    for pdb, md in sorted(data.items()):
        ref = md.get('REF')
        if not ref:
            continue
        for meth, cs in md.items():
            if meth == 'REF':
                continue
            tp = len(cs & ref)
            prec = tp / float(len(cs)) if cs else 0.0
            rec = tp / float(len(ref)) if ref else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            jac = tp / float(len(cs | ref)) if (cs | ref) else 0.0
            rows.append({
                'pdb': pdb.upper(), 'method': meth,
                'n_ref': len(ref), 'n_pred': len(cs), 'tp': tp,
                'precision': prec, 'recall': rec, 'f1': f1, 'jaccard': jac,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # recall per interaction type
    trows = []
    for pdb, md in sorted(data.items()):
        ref = md.get('REF')
        if not ref:
            continue
        for meth, cs in md.items():
            if meth == 'REF':
                continue
            for lab in set(ITYPES.values()):
                rr = {c for c in ref if c[0] == lab}
                pp = {c for c in cs if c[0] == lab}
                if not rr:
                    continue
                trows.append({'pdb': pdb.upper(), 'method': meth, 'itype': lab,
                              'n_ref': len(rr), 'tp': len(pp & rr),
                              'recall': len(pp & rr) / float(len(rr))})
    pd.DataFrame(trows).to_csv(out_csv.replace('.csv', '_bytype.csv'),
                               index=False)

    print('number of PDBs:', len(data))
    print(df.groupby('method')[['precision', 'recall', 'f1', 'jaccard']]
          .agg(['count', 'mean', 'median']).round(3))
    if bad:
        print('\n%d problematic files:' % len(bad))
        for b in bad[:10]:
            print('  ', b)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
