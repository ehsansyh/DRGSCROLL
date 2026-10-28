#!/usr/bin/env python3
"""
tanimoto_diversity.py -- chemical diversity of the screening libraries.

Fetches SMILES from ChEMBL for each compound list, computes pairwise
Tanimoto similarity on ECFP4 (Morgan radius 2, 2048 bits), and writes a
publication figure plus a summary table.

    pip install rdkit chembl_webresource_client pandas matplotlib
    python3 tanimoto_diversity.py -i chembl_ids_RET.csv chembl_ids_PARP1.csv \
        -l "RET TK" "PARP1" -o figs

Notes
  - Requires internet access to www.ebi.ac.uk (ChEMBL web services).
  - If you already have SMILES, supply a CSV with a 'smiles' column and the
    fetch step is skipped.
  - Use the same fingerprint settings you used for the DUD-E actives so the
    three libraries are directly comparable.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    sys.exit('rdkit is required: pip install rdkit')

SERIF = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
INK, INK2 = '#111111', '#4e4e4e'
COLORS = ['#d9541f', '#1a4f87', '#7aafe3', '#1f9b72']

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': SERIF, 'mathtext.fontset': 'stix',
    'font.size': 12, 'axes.edgecolor': '#8a8a8a', 'axes.linewidth': 0.9,
    'axes.titlesize': 14, 'axes.labelsize': 12.5,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK2,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'legend.frameon': False, 'legend.fontsize': 11.5,
})


def fetch_smiles(ids):
    """ChEMBL ID -> canonical SMILES."""
    from chembl_webresource_client.new_client import new_client
    mol = new_client.molecule
    out = {}
    chunk = 50
    for i in range(0, len(ids), chunk):
        batch = ids[i:i + chunk]
        try:
            res = mol.filter(molecule_chembl_id__in=batch).only(
                ['molecule_chembl_id', 'molecule_structures'])
            for r in res:
                st = r.get('molecule_structures') or {}
                smi = st.get('canonical_smiles')
                if smi:
                    out[r['molecule_chembl_id']] = smi
        except Exception as e:
            sys.stderr.write('fetch failed for batch %d: %s\n' % (i, e))
        sys.stderr.write('  fetched %d / %d\r' % (len(out), len(ids)))
    sys.stderr.write('\n')
    return out


def fingerprints(smiles_list):
    fps, bad = [], 0
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad += 1
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    return fps, bad


def pairwise(fps):
    vals = []
    for i in range(1, len(fps)):
        vals.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inputs', nargs='+', required=True)
    ap.add_argument('-l', '--labels', nargs='+', required=True)
    ap.add_argument('-o', '--outdir', default='figs')
    ap.add_argument('--cache', default='smiles_cache.csv')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cache = {}
    if os.path.isfile(args.cache):
        c = pd.read_csv(args.cache)
        cache = dict(zip(c.chembl_id, c.smiles))

    data, rows = {}, []
    for path, label in zip(args.inputs, args.labels):
        df = pd.read_csv(path)
        if 'smiles' in df.columns:
            smis = df.smiles.dropna().tolist()
        else:
            idcol = 'chembl_id' if 'chembl_id' in df.columns else df.columns[0]
            ids = [str(x).strip() for x in df[idcol] if str(x).upper().startswith('CHEMBL')]
            missing = [i for i in ids if i not in cache]
            if missing:
                sys.stderr.write('%s: fetching %d SMILES from ChEMBL\n'
                                 % (label, len(missing)))
                cache.update(fetch_smiles(missing))
            smis = [cache[i] for i in ids if i in cache]
            if len(smis) < len(ids):
                sys.stderr.write('%s: %d of %d IDs resolved\n'
                                 % (label, len(smis), len(ids)))
        fps, bad = fingerprints(smis)
        sim = pairwise(fps)
        data[label] = sim
        rows.append({'library': label, 'n_compounds': len(fps),
                     'n_pairs': len(sim), 'median_Tanimoto': np.median(sim),
                     'Q1': np.percentile(sim, 25),
                     'Q3': np.percentile(sim, 75),
                     'frac_pairs_above_0.4': float((sim >= 0.4).mean()),
                     'frac_pairs_above_0.7': float((sim >= 0.7).mean()),
                     'unparsed_smiles': bad})

    if cache:
        pd.DataFrame({'chembl_id': list(cache), 'smiles': list(cache.values())}
                     ).to_csv(args.cache, index=False)

    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.outdir, 'tanimoto_summary.csv'), index=False)
    print(summ.round(3).to_string(index=False))

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), squeeze=False)
    bins = np.arange(0, 1.02, 0.02)
    for ax, (label, sim), col in zip(axes[0], data.items(), COLORS):
        ax.hist(sim, bins=bins, color=col, edgecolor='white', lw=0.3,
                density=True)
        med = np.median(sim)
        ax.axvline(med, color=INK, lw=1.4, ls=(0, (3, 2)))
        ax.text(med + 0.02, ax.get_ylim()[1] * 0.92,
                'median %.2f' % med, fontsize=11.5, color=INK)
        ax.axvline(0.4, color='#8a8a8a', lw=1.0, ls=':')
        ax.set_xlim(0, 1)
        ax.set_xlabel('pairwise Tanimoto similarity (ECFP4)')
        ax.set_title(label, loc='left', color=INK)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel('density of compound pairs')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(args.outdir, 'figS_tanimoto_diversity.%s' % ext),
                    dpi=300)
    print('-> %s/figS_tanimoto_diversity.png' % args.outdir)


if __name__ == '__main__':
    main()
