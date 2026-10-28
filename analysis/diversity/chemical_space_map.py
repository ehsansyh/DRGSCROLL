#!/usr/bin/env python3
"""
chemical_space_map.py -- chemical space of the screening libraries
(companion to the Tanimoto diversity histograms).

Embeds ECFP4 fingerprints with UMAP (or t-SNE) using Tanimoto distance and
colours compounds by target and by measured potency.

    python3 chemical_space_map.py -c smiles_cache.csv \
        -i chembl_ids_RET.csv chembl_ids_PARP1.csv -l "RET TK" "PARP1" \
        -o figs [--method tsne] [--threshold 6.5]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--cache', required=True)
ap.add_argument('-i', '--inputs', nargs='+', required=True)
ap.add_argument('-l', '--labels', nargs='+', required=True)
ap.add_argument('-o', '--outdir', default='figs')
ap.add_argument('--method', choices=('umap', 'tsne'), default='umap')
ap.add_argument('--threshold', type=float, default=6.5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--limit', type=int, default=None,
                help='keep at most this many compounds per library, '
                     'in the order they are listed in the input file')
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

SERIF = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
INK, INK2 = '#111111', '#4e4e4e'
COLORS = ['#d9541f', '#1a4f87', '#1f9b72', '#7aafe3']

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': SERIF, 'mathtext.fontset': 'stix',
    'font.size': 12, 'axes.edgecolor': '#8a8a8a', 'axes.linewidth': 0.9,
    'axes.titlesize': 14, 'axes.labelsize': 12.5,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK2,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'legend.frameon': False, 'legend.fontsize': 11.5,
})

cache = pd.read_csv(args.cache)
smiles = dict(zip(cache.chembl_id, cache.smiles))

rows = []
for path, label in zip(args.inputs, args.labels):
    df = pd.read_csv(path)
    idcol = 'chembl_id' if 'chembl_id' in df.columns else df.columns[0]
    for _, r in df.iterrows():
        cid = str(r[idcol]).strip()
        if cid not in smiles:
            continue
        pic = pd.to_numeric(r.get('pIC50'), errors='coerce')
        rows.append({'chembl_id': cid, 'library': label,
                     'smiles': smiles[cid], 'pIC50': pic})
d = pd.DataFrame(rows).drop_duplicates(subset=['chembl_id', 'library'])
if args.limit:
    d = d.groupby('library', sort=False).head(args.limit).reset_index(drop=True)

fps, keep = [], []
for i, r in d.iterrows():
    m = Chem.MolFromSmiles(r.smiles)
    if m is None:
        continue
    fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    keep.append(i)
d = d.loc[keep].reset_index(drop=True)
sys.stderr.write('%d compounds embedded\n' % len(d))

n = len(fps)
D = np.zeros((n, n))
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    D[i] = 1.0 - np.asarray(sims)
np.fill_diagonal(D, 0.0)

if args.method == 'umap':
    import umap
    emb = umap.UMAP(metric='precomputed', n_neighbors=15, min_dist=0.25,
                    random_state=args.seed).fit_transform(D)
    mname = 'UMAP'
else:
    from sklearn.manifold import TSNE
    emb = TSNE(metric='precomputed', init='random', perplexity=20,
               random_state=args.seed).fit_transform(D)
    mname = 't-SNE'
d['x'], d['y'] = emb[:, 0], emb[:, 1]
d.to_csv(os.path.join(args.outdir, 'chemical_space_coords.csv'), index=False)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))

ax = axes[0]
for lab, col in zip(args.labels, COLORS):
    s = d[d.library == lab]
    ax.scatter(s.x, s.y, s=42, facecolor=col, edgecolor='white', lw=0.6,
               alpha=0.9, label='%s (n=%d)' % (lab, len(s)))
ax.legend(loc='best')
ax.set_title('A   Libraries occupy distinct chemical space', loc='left',
             color=INK)

ax = axes[1]
sub = d.dropna(subset=['pIC50'])
sc = ax.scatter(sub.x, sub.y, s=42, c=sub.pIC50, cmap='viridis',
                edgecolor='white', lw=0.6, alpha=0.95)
miss = d[d.pIC50.isna()]
if len(miss):
    ax.scatter(miss.x, miss.y, s=36, facecolor='#d7d4cf', edgecolor='white',
               lw=0.6, label='no pIC$_{50}$')
    ax.legend(loc='best')
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
cb.set_label('pIC$_{50}$', fontsize=12.5, color=INK2)
cb.ax.tick_params(labelsize=11)
cb.outline.set_visible(False)
ax.set_title('B   Potency is not confined to one region', loc='left',
             color=INK)

for ax in axes:
    ax.set_xlabel('%s 1' % mname)
    ax.set_ylabel('%s 2' % mname)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

fig.suptitle('Chemical space of the screening libraries '
             '(ECFP4, Tanimoto distance, %s)' % mname,
             fontsize=15.5, color=INK, x=0.012, ha='left', y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(args.outdir, 'figS_chemical_space.%s' % ext),
                dpi=300)
plt.close(fig)

act = d.dropna(subset=['pIC50'])
print(d.groupby('library').size().to_dict())
for lab, s in act.groupby('library'):
    print('%s: %d with pIC50, %d above %.1f'
          % (lab, len(s), (s.pIC50 > args.threshold).sum(), args.threshold))
print('-> %s/figS_chemical_space.png' % args.outdir)
