#!/usr/bin/env python3
"""
fig_rmsd_en.py -- Pose accuracy figure, two panels, English, Times.

  A  Ligand heavy-atom RMSD distribution per engine (50 PDBbind co-crystals)
  B  Spearman correlation between docking score and RMSD per engine

    python3 fig_rmsd_en.py -r rmsd.tsv -s scores.tsv -o figs
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument('-r', '--rmsd', required=True)
ap.add_argument('-s', '--scores', required=True)
ap.add_argument('-o', '--outdir', default='figs')
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

ORDER = ['DRGSCROLL', 'IFD', 'Glide', 'ADFR', 'Vina-GPU']
# muted, print-safe palette: warm accent for the method, cool tones for others
COL = {'DRGSCROLL': '#c2552b', 'IFD': '#2d4f6b', 'Glide': '#4e7f9e',
       'ADFR': '#8fb3c4', 'Vina-GPU': '#a9a79f'}
FLEX = {'DRGSCROLL', 'IFD'}
INK, INK2, RULE = '#1a1a1a', '#55534e', '#2d4f6b'
SERIF = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': SERIF,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Serif',
    'mathtext.it': 'Liberation Serif:italic',
    'mathtext.bf': 'Liberation Serif:bold',
    'font.size': 14,
    'axes.edgecolor': '#8a8a8a', 'axes.linewidth': 1.0,
    'axes.titlesize': 17, 'axes.labelsize': 15,
    'xtick.labelsize': 13.5, 'ytick.labelsize': 13.5,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK2,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'legend.frameon': False, 'legend.fontsize': 13.5,
})

d = pd.read_csv(args.rmsd, sep=None, engine='python')
d.columns = [c.strip() for c in d.columns]
d = d.set_index(d.columns[0])
d.index = d.index.str.lower()
for c in d.columns:
    d[c] = pd.to_numeric(d[c], errors='coerce')
engines = [e for e in ORDER if e in d.columns]

s = pd.read_csv(args.scores, sep=None, engine='python')
s.columns = [c.strip() for c in s.columns]
s = s.set_index(s.columns[0])
s.index = s.index.str.lower()
for c in s.columns:
    s[c] = pd.to_numeric(s[c], errors='coerce')

fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.6),
                         gridspec_kw={'width_ratios': [1.25, 1.0],
                                      'wspace': 0.26})
fig.subplots_adjust(left=0.075, right=0.975, top=0.755, bottom=0.145)
rng = np.random.RandomState(0)

# ------------------------------------------------------------------ A
ax = axes[0]
for i, e in enumerate(engines):
    v = d[e].dropna().values
    ax.boxplot([v], positions=[i], widths=0.56, patch_artist=True,
               showfliers=False, zorder=3,
               medianprops=dict(color='white' if e != 'ADFR' and e != 'Vina-GPU'
                                else INK, lw=2.0),
               whiskerprops=dict(color=COL[e], lw=1.1),
               capprops=dict(color=COL[e], lw=1.1),
               boxprops=dict(facecolor=COL[e], edgecolor='none'))
    ax.plot(i + rng.uniform(-0.18, 0.18, len(v)), v, ls='none', marker='o',
            ms=3.6, mfc=INK, mec='none', alpha=0.30, zorder=4)
    ax.text(i, 15.9, '%.1f' % np.median(v), ha='center', fontsize=14,
            color=INK, fontweight='bold')
ax.axhline(2.0, color=RULE, lw=1.4, ls=(0, (4, 3)), zorder=2)
ax.text(len(engines) - 0.48, 2.45, u'2 \u00c5', fontsize=13, color=RULE,
        ha='right')
ax.set_xticks(range(len(engines)))
ax.set_xticklabels(engines, rotation=22, ha='right')
for lbl, e in zip(ax.get_xticklabels(), engines):
    lbl.set_color(INK)
    if e in FLEX:
        lbl.set_fontweight('bold')
ax.set_ylim(0, 16.8)
ax.set_ylabel(u'ligand heavy-atom RMSD (\u00c5)')
ax.grid(axis='y', alpha=0.25, lw=0.7, zorder=0)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.set_title(u'A   Pose accuracy', loc='left', color=INK, pad=14)

# ------------------------------------------------------------------ B
ax = axes[1]
rows = []
for e in engines:
    if e not in s.columns:
        continue
    z = pd.concat([s[e].rename('sc'), d[e].rename('rm')], axis=1).dropna()
    rho, pv = spearmanr(z.sc, z.rm)
    rows.append((e, rho, pv, len(z)))
rows.sort(key=lambda t: t[1])
y = np.arange(len(rows))
ax.barh(y, [r[1] for r in rows], 0.60, color=[COL[r[0]] for r in rows],
        zorder=3)
for i, (e, rho, pv, n) in enumerate(rows):
    star = ('***' if pv < 1e-3 else '**' if pv < 1e-2
            else '*' if pv < 0.05 else 'n.s.')
    ax.text(rho + 0.022, i, u'%.2f %s' % (rho, star), va='center',
            fontsize=13.5, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows])
for lbl, r in zip(ax.get_yticklabels(), rows):
    lbl.set_color(INK)
    if r[0] in FLEX:
        lbl.set_fontweight('bold')
ax.set_xlim(0, 0.92)
ax.set_xlabel(r'Spearman $\rho$ between docking score and RMSD')
ax.grid(axis='x', alpha=0.25, lw=0.7, zorder=0)
ax.tick_params(axis='y', length=0)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.set_title(u'B   Does the score identify the correct pose?', loc='left',
             color=INK, pad=14)

fig.suptitle(u'Pose accuracy relative to the crystallographic reference',
             fontsize=19, color=INK, x=0.075, ha='left', y=0.975)
fig.text(0.075, 0.905,
         u'50 PDBbind co-crystal complexes. Bold labels mark engines that '
         r'model side-chain flexibility. In panel B a higher $\rho$ means '
         u'better scores correspond to more accurate poses',
         fontsize=13, color=INK2)
fig.text(0.075, 0.862, u'(*p < 0.05, ***p < 0.001, n.s. not significant).',
         fontsize=13, color=INK2)

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(args.outdir, 'fig_rmsd_2panel.%s' % ext), dpi=300)
plt.close(fig)

summ = pd.DataFrame({
    'n': d[engines].notna().sum(),
    'median': d[engines].median().round(2),
    'pct_le_2A': (100 * (d[engines] <= 2).sum() / d[engines].notna().sum()).round(1),
    'pct_le_3A': (100 * (d[engines] <= 3).sum() / d[engines].notna().sum()).round(1),
    'pct_gt_5A': (100 * (d[engines] > 5).sum() / d[engines].notna().sum()).round(1),
}).sort_values('median')
summ.to_csv(os.path.join(args.outdir, 'rmsd_summary.csv'))
print(summ)
print('-> %s/fig_rmsd_2panel.png' % args.outdir)
