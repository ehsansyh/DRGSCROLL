#!/usr/bin/env python3
"""
md_occupancy_figs.py -- MD occupancy of DRGSCROLL side-chain conformations.

Regenerates two figures with serif (Times) typography and enlarged labels:
  drgscroll_md_category_heatmap_en.*   residue x complex occupancy heatmap
  drgscroll_md_scorecard_en.*          per-complex stacked composition

    python3 md_occupancy_figs.py -i drgscroll_vs_md.csv -o figs
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', default='drgscroll_vs_md.csv')
ap.add_argument('-o', '--outdir', default='figs')
ap.add_argument('--near', type=float, default=30.0)
ap.add_argument('--min-n', type=int, default=30,
                help='minimum complexes per residue x chi row')
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

SERIF = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
INK = '#1a1a1a'
# Not sampled, Rare, Frequent, MD mode, Not rotated
CATCOL = ['#a8cbee', '#4a90d9', '#1f5fa8', '#10305c', '#e6e4e0']
CATLAB = ['Not sampled (10 ns)', 'Rare (<10%)', 'Frequent (10\u201350%)',
          'MD mode (\u226550%)', 'Not rotated by DRGSCROLL']

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': SERIF,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Serif',
    'mathtext.it': 'Liberation Serif:italic',
    'mathtext.bf': 'Liberation Serif:bold',
    'font.size': 15,
    'axes.edgecolor': '#8a8a8a', 'axes.linewidth': 0.9,
    'axes.titlesize': 20, 'axes.labelsize': 17,
    'xtick.labelsize': 11, 'ytick.labelsize': 14,
    'xtick.color': INK, 'ytick.color': INK, 'axes.labelcolor': INK,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'legend.frameon': False, 'legend.fontsize': 15,
})

d = pd.read_csv(args.input)
NEAR = int(args.near)
key = 'frac_md_within_%d' % NEAR
if key not in d.columns:
    key = [c for c in d.columns if c.startswith('frac_md_within_')][0]
    NEAR = int(key.rsplit('_', 1)[1])

d['cat'] = np.select(
    [d.drg_rotated == 0, d[key] <= 0, d[key] < 0.10, d[key] < 0.50],
    [4, 0, 1, 2], 3)
d['row'] = (d.resname.str.title() + d.resnum.astype(str)
            + ' \u03c7' + d.chi.astype(str))
d['col'] = d.complex.str.replace('CHEMBL', '', regex=False)

counts = d[d.drg_rotated == 1].groupby('row').size()
rows = [r for r in counts.index if counts[r] >= args.min_n]
order = (d[d.row.isin(rows)].drop_duplicates('row')
         .sort_values(['resnum', 'chi']).row.tolist())
nrot = d[d.drg_rotated == 1].groupby('row').size()

# rank complexes by fraction of rotated angles that MD supports (>=10%)
rot = d[d.drg_rotated == 1]
supp = rot.groupby('col').apply(
    lambda g: float((g.cat >= 2).mean()), include_groups=False)
cols = supp.sort_values(ascending=False).index.tolist()

M = (d.pivot_table(index='row', columns='col', values='cat', aggfunc='first')
     .reindex(index=order, columns=cols))
M = M.fillna(4).values

# ------------------------------------------------------------- heatmap
cmap = ListedColormap(CATCOL)
norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
fig, ax = plt.subplots(figsize=(0.34 * len(cols) + 7.5,
                                0.46 * len(order) + 3.8))
ax.imshow(M, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(cols, rotation=90, fontsize=15, color=INK)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels(['%s  (%d)' % (r, nrot.get(r, 0)) for r in order],
                   fontsize=20, color=INK)
ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
ax.grid(which='minor', color='white', lw=0.7)
ax.tick_params(which='both', length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xlabel('Complex (ChEMBL ID), ranked by fraction of MD-supported angles',
              labelpad=16, fontsize=24)
ax.set_title('MD occupancy of DRGSCROLL side-chain conformations', pad=24,
             color=INK, fontsize=30)
leg = ax.legend(handles=[Patch(facecolor=c, label=l)
                         for c, l in zip(CATCOL, CATLAB)],
                loc='upper left', bbox_to_anchor=(1.005, 1.0),
                fontsize=23, handlelength=2.0, handleheight=1.6,
                labelspacing=1.1, borderaxespad=1.0)
for txt in leg.get_texts():
    txt.set_fontweight('bold')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(args.outdir,
                             'drgscroll_md_category_heatmap_en.%s' % ext),
                dpi=170)
plt.close(fig)

# ------------------------------------------------------------- scorecard
comp = (rot.groupby(['col', 'cat']).size().unstack(fill_value=0)
        .reindex(columns=[0, 1, 2, 3], fill_value=0))
comp = comp.loc[[c for c in cols if c in comp.index]]
tot = comp.sum(axis=1)
pct = comp.div(tot, axis=0) * 100

fig, ax = plt.subplots(figsize=(14.5, 0.36 * len(pct) + 3.6))
y = np.arange(len(pct))[::-1]
left = np.zeros(len(pct))
for k in range(4):
    ax.barh(y, pct[k].values, 0.82, left=left, color=CATCOL[k],
            edgecolor='white', lw=0.4, zorder=3)
    left += pct[k].values
ax.set_yticks(y)
ax.set_yticklabels(['%s (n=%d)' % (c, t) for c, t in zip(pct.index, tot)],
                   fontsize=12.5, color=INK)
ax.set_xlim(0, 100)
ax.set_ylim(-0.7, len(pct) - 0.3)
ax.set_xlabel('DRGSCROLL-rotated side-chain angles (%)', labelpad=10)
ax.tick_params(axis='y', length=0)
ax.grid(axis='x', alpha=0.22, lw=0.6, zorder=0)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.set_title('Per-complex DRGSCROLL\u2013MD concordance', pad=58, color=INK)
leg = ax.legend(handles=[Patch(facecolor=c, label=l)
                         for c, l in zip(CATCOL[:4], CATLAB[:4])],
                loc='lower center', bbox_to_anchor=(0.5, 1.004), ncol=4,
                fontsize=18, handlelength=1.8, handleheight=1.4,
                columnspacing=2.6, borderaxespad=0.6)
for txt in leg.get_texts():
    txt.set_fontweight('bold')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(args.outdir, 'drgscroll_md_scorecard_en.%s' % ext),
                dpi=170)
plt.close(fig)

print('rows %d, complexes %d' % (len(order), len(cols)))
print('-> %s/drgscroll_md_category_heatmap_en.png' % args.outdir)
print('-> %s/drgscroll_md_scorecard_en.png' % args.outdir)
