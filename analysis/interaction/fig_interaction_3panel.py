#!/usr/bin/env python3
"""
fig_interaction_3panel.py -- Interaction fidelity figure (Figure S3).

  A  Per-structure accuracy for the engines chosen with --engines-a
     (default: the three rigid engines)
  B  Per-structure F1 for the same engines
  C  Reference contacts recovered by one engine only (all engines)

    python3 fig_interaction_3panel.py -i interaction_keep_FINAL.csv -o figs
    python3 fig_interaction_3panel.py -i ... --engines-a all
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True)
ap.add_argument('-o', '--outdir', default='figs')
ap.add_argument('--engines-a', default='rigid',
                choices=('rigid', 'all'),
                help='engines shown in panel A')
ap.add_argument('--exclude-unresolved', action='store_true')
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

NAMES = {'drg': 'DRGSCROLL', 'IFD': 'IFD', 'Glide': 'Glide',
         'ADFR': 'ADFR', 'vina': 'Vina-GPU'}
ALL = ['DRGSCROLL', 'IFD', 'Glide', 'ADFR', 'Vina-GPU']
RIGID = ['Glide', 'ADFR', 'Vina-GPU']
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
    'font.size': 16,
    'axes.edgecolor': '#8a8a8a', 'axes.linewidth': 1.0,
    'axes.titlesize': 19, 'axes.labelsize': 17,
    'xtick.labelsize': 15.5, 'ytick.labelsize': 15.5,
    'xtick.color': INK, 'ytick.color': INK, 'axes.labelcolor': INK,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'legend.frameon': False, 'legend.fontsize': 13.5,
})

d = pd.read_csv(args.input)
if args.exclude_unresolved and 'unresolved' in d.columns:
    d = d[d.unresolved == 0]
d['engine'] = d.method.map(NAMES)

per = d.groupby(['name', 'engine'])[['pdb_n', 'TP', 'FP', 'FN']].sum().reset_index()
per = per[per.pdb_n > 0]
per['recovery'] = 100.0 * per.TP / per.pdb_n
den = per.TP + per.FP + per.FN
per['accuracy'] = np.where(den > 0, per.TP / den, 0.0)
f1den = 2 * per.TP + per.FP + per.FN
per['f1'] = np.where(f1den > 0, 2 * per.TP / f1den, 0.0)

eng_a = RIGID if args.engines_a == 'rigid' else ALL
eng_a = [e for e in eng_a if e in set(per.engine)]


def pset(s):
    if pd.isna(s) or not str(s).strip():
        return set()
    return {t.strip() for t in str(s).split(';') if t.strip()}


uniq = {e: 0 for e in ALL}
nobody = 0
for (n, i), g in d.groupby(['name', 'interaction']):
    ref = pset(g.pdb_residues.iloc[0])
    if not ref:
        continue
    kept = {r.engine: pset(r.kept_residues) for _, r in g.iterrows()}
    allk = set().union(*kept.values()) if kept else set()
    nobody += len(ref - allk)
    for c in ref:
        who = [e for e, s in kept.items() if c in s]
        if len(who) == 1 and who[0] in uniq:
            uniq[who[0]] += 1
eng_b = [e for e in ALL if e in set(d.engine)]

fig, axes = plt.subplots(1, 3, figsize=(18.5, 7.0),
                         gridspec_kw={'width_ratios': [1.0, 1.0, 1.15],
                                      'wspace': 0.28})
fig.subplots_adjust(left=0.075, right=0.975, top=0.760, bottom=0.155)
rng = np.random.RandomState(0)

# ------------------------------------------------------------------ A, B
def box_panel(ax, key, title, letter):
    for i, e in enumerate(eng_a):
        v = per.loc[per.engine == e, key].dropna().values
        ax.boxplot([v], positions=[i], widths=0.56, patch_artist=True,
                   showfliers=False, zorder=3,
                   medianprops=dict(
                       color='white' if e in ('Glide', 'DRGSCROLL', 'IFD')
                       else INK, lw=2.0),
                   whiskerprops=dict(color=COL[e], lw=1.1),
                   capprops=dict(color=COL[e], lw=1.1),
                   boxprops=dict(facecolor=COL[e], edgecolor='none'))
        ax.plot(i + rng.uniform(-0.18, 0.18, len(v)), v, ls='none', marker='o',
                ms=3.6, mfc=INK, mec='none', alpha=0.30, zorder=4)
        ax.text(i, 1.07, '%.2f' % np.median(v), ha='center', fontsize=16,
                color=INK, fontweight='bold')
    ax.set_xticks(range(len(eng_a)))
    ax.set_xticklabels(eng_a, rotation=20, ha='right')
    for lbl, e in zip(ax.get_xticklabels(), eng_a):
        lbl.set_color(INK)
        if e in FLEX:
            lbl.set_fontweight('bold')
    ax.set_ylim(-0.03, 1.13)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(title)
    ax.grid(axis='y', alpha=0.25, lw=0.7, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(letter, loc='left', color=INK, pad=14)


box_panel(axes[0], 'accuracy', 'accuracy per structure', 'A   Accuracy')
box_panel(axes[1], 'f1', 'F1 per structure', 'B   F1 score')

# ------------------------------------------------------------------ C
ax = axes[2]
vals = [uniq[e] for e in eng_b]
ax.bar(range(len(eng_b)), vals, 0.60, color=[COL[e] for e in eng_b], zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 0.3, '%d' % v, ha='center', fontsize=16, color=INK)
ax.set_xticks(range(len(eng_b)))
ax.set_xticklabels(eng_b, rotation=20, ha='right')
for lbl, e in zip(ax.get_xticklabels(), eng_b):
    lbl.set_color(INK)
    if e in FLEX:
        lbl.set_fontweight('bold')
ax.set_ylim(0, max(vals) * 1.28)
ax.set_ylabel('number of reference contacts')
ax.grid(axis='y', alpha=0.25, lw=0.7, zorder=0)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.set_title('C   Contacts recovered by one engine only', loc='left',
             color=INK, pad=14)

fig.suptitle('Interaction fidelity relative to the co-crystal reference',
             fontsize=21, color=INK, x=0.075, ha='left', y=0.975)
fig.text(0.075, 0.895,
         'PLIP contacts from %d PDBbind complexes, matched by residue '
         'identity and interaction type. Bold labels mark engines that '
         'model side-chain flexibility.' % per.name.nunique(),
         fontsize=14.5, color=INK)
fig.text(0.075, 0.855,
         'In panel C, %d further reference contacts were recovered by no '
         'engine.' % nobody, fontsize=14.5, color=INK)

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(args.outdir, 'fig_interaction_3panel.%s' % ext),
                dpi=300)
plt.close(fig)

print(per.groupby('engine')[['accuracy', 'f1', 'recovery']].median().round(3))
print('unique:', uniq, '| recovered by none:', nobody)
print('-> %s/fig_interaction_3panel.png' % args.outdir)
