#!/usr/bin/env python3
"""
chi_figures_en.py -- DRGSCROLL vs MD publication figures (English, Times).

    python3 chi_figures_en.py -i drgscroll_vs_md.csv -o figs

Outputs
  fig1_md_sampling.*       4 panels: MD sampling adequacy, agreement by
                           mobility stratum, agreement classes, displacement
  fig2_per_residue.*       per residue x chi agreement + MD mobility
  fig3_angle_ranges.*      MD angle range and DRGSCROLL angles per residue
  fig4_overview.*          combined overview (recommended main figure)

180 deg-periodic torsions (PHE/TYR chi2, ASP chi2, GLU chi3) are folded to
[-90, 90) and marked with an asterisk.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', default='drgscroll_vs_md.csv')
ap.add_argument('-o', '--outdir', default='figs')
ap.add_argument('--near', type=float, default=30.0)
ap.add_argument('--best-only', action='store_true')
ap.add_argument('--min-n', type=int, default=30)
ap.add_argument('--label', default='')
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
OUT = args.outdir

SYM180 = {('PHE', 2), ('TYR', 2), ('ASP', 2), ('GLU', 3)}

d = pd.read_csv(args.input)
NEAR = int(args.near)
nearkey = 'frac_md_within_%d' % NEAR
if nearkey not in d.columns:
    nearkey = [c for c in d.columns if c.startswith('frac_md_within_')][0]
    NEAR = int(nearkey.rsplit('_', 1)[1])
c1key = 'frac_md_c1_within_%d' % NEAR
CHANCE = 2.0 * NEAR / 360.0
HAS_V2 = 'sym180' in d.columns
if args.best_only and 'drg_rank' in d.columns:
    d = d[d.drg_rank == 1]
if 'sym180' not in d.columns:
    d['sym180'] = [int((rn, c) in SYM180) for rn, c in zip(d.resname, d.chi)]
d['frac'] = d[nearkey]
d['frac_c1'] = d[c1key]
r = d[d.drg_rotated == 1].copy()

INK, INK2 = '#111111', '#4e4e4e'
BLUE = ['#bcd7f2', '#7aafe3', '#3a7fc4', '#1a4f87']
DRG, C1, GREY = '#d9541f', '#1f9b72', '#b9b7b2'
MDBAND, MDLINE = '#c7d4de', '#4a6676'
SERIF = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': SERIF, 'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.edgecolor': '#8a8a8a', 'axes.linewidth': 0.9,
    'axes.titlesize': 14, 'axes.labelsize': 12.5,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK2,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'legend.frameon': False, 'legend.fontsize': 11.5,
})

STRATA = [(0, 20, u'< 20\u00b0\n(immobile)'), (20, 40, u'20\u201340\u00b0'),
          (40, 70, u'40\u201370\u00b0'), (70, 1e9, u'> 70\u00b0\n(mobile)')]


def cats(f):
    return np.select([f <= 0, f < 0.10, f < CHANCE], [0, 1, 2], 3)


r['ucat'] = cats(r['frac'].values)
SUB = (u'%d complexes; %d side-chain torsions actually rotated by DRGSCROLL'
       % (d.complex.nunique(), len(r)))
if args.label:
    SUB += u' \u2014 ' + args.label


def save(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (stem, ext)), dpi=300)
    plt.close(fig)


def fold(a, sym):
    a = np.asarray(a, dtype=float)
    if not sym:
        return (a + 180.0) % 360.0 - 180.0
    return (a + 90.0) % 180.0 - 90.0


# ============================================================ FIGURE 1
fig = plt.figure(figsize=(11.0, 8.8))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.26, left=0.085,
                      right=0.975, top=0.855, bottom=0.155)
uniq = d.drop_duplicates(['complex', 'resnum', 'chi'])

ax = fig.add_subplot(gs[0, 0])
ax.hist(uniq.md_circ_std_deg.clip(0, 150), bins=np.arange(0, 155, 5),
        color=BLUE[1], edgecolor='white', lw=0.4)
ax.axvline(30, color=INK, lw=1.3, ls=(0, (3, 2)))
ax.text(35, ax.get_ylim()[1] * 0.93,
        u'%.0f%% below 30\u00b0\n(essentially fixed\nover 10 ns)'
        % (100 * (uniq.md_circ_std_deg < 30).mean()),
        fontsize=12, color=INK, va='top')
ax.set_xlabel(u'MD circular standard deviation (\u00b0)')
ax.set_ylabel(u'number of torsions')
ax.set_title(u'A   How much did MD sample in 10 ns?', loc='left', color=INK)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

ax = fig.add_subplot(gs[0, 1])
lab, md_, c1_, ns = [], [], [], []
for lo, hi, nm in STRATA:
    sub = r[(r.md_circ_std_deg >= lo) & (r.md_circ_std_deg < hi)]
    lab.append(nm); ns.append(len(sub))
    md_.append(sub['frac'].median()); c1_.append(sub['frac_c1'].median())
x = np.arange(len(lab)); w = 0.38
ax.bar(x - w / 2, md_, w, color=DRG, label=u'DRGSCROLL angle', zorder=3)
ax.bar(x + w / 2, c1_, w, color=C1, label=u'starting angle', zorder=3)
ax.axhline(CHANCE, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=4)
ax.text(-0.42, CHANCE + 0.03, u'chance level (%d%%)' % round(CHANCE * 100),
        fontsize=11, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(['%s\nn=%d' % (l, n) for l, n in zip(lab, ns)], fontsize=11)
ax.set_ylim(0, 1.08)
ax.set_ylabel(u'median fraction of MD time\nwithin \u00b1%d\u00b0' % NEAR)
ax.set_xlabel(u'MD mobility of that torsion', labelpad=6)
ax.set_title(u'B   Time spent at that angle', loc='left', color=INK)
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.03))
ax.grid(axis='y', alpha=0.25, lw=0.6, zorder=0)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

ax = fig.add_subplot(gs[1, 0])
names = [u'never visited', u'rare (<10%)', u'below chance',
         u'sampled by MD']
M = np.array([[(r[(r.md_circ_std_deg >= lo) & (r.md_circ_std_deg < hi)].ucat
                == i).mean() * 100 for i in range(4)] for lo, hi, _ in STRATA])
bottom = np.zeros(len(STRATA))
for i in range(4):
    ax.bar(np.arange(len(STRATA)), M[:, i], 0.66, bottom=bottom, color=BLUE[i],
           edgecolor='white', lw=0.6, zorder=3)
    bottom += M[:, i]
handles_c = [Patch(facecolor=BLUE[i], edgecolor='white', lw=0.6, label=names[i])
             for i in range(4)]
ax.set_xticks(np.arange(len(STRATA)))
ax.set_xticklabels([t[2] for t in STRATA], fontsize=11)
ax.set_ylim(0, 100)
ax.set_ylabel(u'percentage of DRGSCROLL angles')
ax.set_xlabel(u'MD mobility of that torsion')
ax.set_title(u'C   Outcome is bimodal', loc='left', color=INK)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

ax = fig.add_subplot(gs[1, 1])
bins = [(0.3, 2), (2, 3.5), (3.5, 5), (5, 1e9)]
labs = [u'0.3\u20132', u'2\u20133.5', u'3.5\u20135', u'> 5']
med, never, nn = [], [], []
for lo, hi in bins:
    sub = r[(r.drg_moved_A >= lo) & (r.drg_moved_A < hi)]
    med.append(sub['frac'].median())
    never.append((sub['frac'] <= 0).mean() * 100)
    nn.append(len(sub))
x = np.arange(len(bins))
ax.bar(x, med, 0.6, color=BLUE[2], zorder=3, label=u'median MD fraction')
ax.axhline(CHANCE, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=4)
ax.set_ylim(0, max(0.52, max(med) * 1.25))
ax.set_ylabel(u'fraction of MD time within \u00b1%d\u00b0' % NEAR)
ax.set_xticks(x)
ax.set_xticklabels(['%s\nn=%d' % (l, n) for l, n in zip(labs, nn)], fontsize=11)
ax.set_xlabel(u'side-chain displacement by DRGSCROLL (\u00c5)', labelpad=6)
ax2 = ax.twinx()
ax2.plot(x, never, color=DRG, marker='o', ms=7, lw=2.0, zorder=5,
         label=u'never-visited fraction')
ax2.set_ylim(0, max(60, max(never) * 1.2))
ax2.set_ylabel(u'percentage never visited', color=DRG)
ax2.tick_params(colors=DRG, labelsize=11)
ax2.spines['right'].set_color(DRG)
ax.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=11)
ax.set_title(u'D   Large rotations lack MD support', loc='left', color=INK)
ax.grid(axis='y', alpha=0.25, lw=0.6, zorder=0)

fig.legend(handles=handles_c, loc='lower center', ncol=4, fontsize=12,
           bbox_to_anchor=(0.5, 0.008), handlelength=1.4,
           title=u'agreement class in panel C \u2014 fraction of MD time '
                 u'within \u00b1%d\u00b0 of the DRGSCROLL angle' % NEAR,
           title_fontsize=11.5)
fig.suptitle(u'Are DRGSCROLL side-chain angles sampled by MD?',
             fontsize=17, color=INK, x=0.085, ha='left', y=0.975)
fig.text(0.085, 0.915, SUB, fontsize=12.5, color=INK2, ha='left')
save(fig, 'fig1_md_sampling')

# ============================================================ FIGURE 2
t = r.groupby(['resname', 'resnum', 'chi']).agg(
    n=('frac', 'size'), med=('frac', 'median'),
    mob=('md_circ_std_deg', 'median'), sym=('sym180', 'max')).reset_index()
t = t[t.n >= args.min_n].sort_values(['resnum', 'chi']).iloc[::-1]

fig, axes = plt.subplots(1, 2, figsize=(11.0, 0.38 * len(t) + 2.6),
                         gridspec_kw={'width_ratios': [2.5, 1], 'wspace': 0.05})
y = np.arange(len(t))
ax = axes[0]
cols = [BLUE[3] if m >= CHANCE else (BLUE[1] if m > 0.02 else GREY)
        for m in t['med']]
ax.barh(y, t['med'], 0.70, color=cols, zorder=3)
ax.axvline(CHANCE, color=INK, lw=1.3, ls=(0, (3, 2)), zorder=4)
ax.text(CHANCE + 0.015, len(t) - 0.4, u'chance (%d%%)' % round(CHANCE * 100),
        fontsize=11.5, color=INK, va='center')
ticks = [u'%s%d  \u03c7%d%s' % (a.title(), b, c, u' *' if s else u'')
         for a, b, c, s in zip(t.resname, t.resnum, t.chi, t.sym)]
ax.set_yticks(y); ax.set_yticklabels(ticks, fontsize=12, color=INK)
for i, (n, m) in enumerate(zip(t.n, t['med'])):
    ax.text(m + 0.012, i, 'n=%d' % n, va='center', fontsize=10, color=INK2)
ax.set_ylim(-0.7, len(t) - 0.3); ax.set_xlim(0, 1.14)
ax.set_xlabel(u'median fraction of MD time within \u00b1%d\u00b0 of the '
              u'DRGSCROLL angle' % NEAR)
ax.grid(axis='x', alpha=0.25, lw=0.6, zorder=0)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.set_title(u'A   Is the DRGSCROLL angle seen in MD?', loc='left', color=INK)

ax = axes[1]
ax.barh(y, t['mob'], 0.70, color='#8fa9bd', zorder=3)
ax.axvline(30, color=INK, lw=1.3, ls=(0, (3, 2)), zorder=4)
ax.text(34, len(t) - 0.4, u'30\u00b0', fontsize=11.5, color=INK, va='center')
ax.set_yticks(y); ax.set_yticklabels([])
ax.set_ylim(-0.7, len(t) - 0.3)
ax.set_xlabel(u'MD circular SD (\u00b0)')
ax.grid(axis='x', alpha=0.25, lw=0.6, zorder=0)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.set_title(u'B   Is it mobile in MD?', loc='left', color=INK)

h = [Patch(facecolor=BLUE[3], label=u'sampled above chance'),
     Patch(facecolor=BLUE[1], label=u'rarely visited'),
     Patch(facecolor=GREY, label=u'essentially never seen')]
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=12,
           bbox_to_anchor=(0.5, 0.008), handlelength=1.3)
fig.text(0.012, 0.035,
         u'*  180\u00b0-periodic torsion (\u03c7 and \u03c7+180 are the same '
         u'structure)' if HAS_V2 else
         u'*  180\u00b0-periodic torsion \u2014 not symmetry-corrected in this '
         u'input; rerun with the corrected pipeline',
         fontsize=11, color=INK2)
fig.suptitle(u'Per-residue agreement between DRGSCROLL and MD',
             fontsize=17, color=INK, x=0.012, ha='left', y=0.988)
fig.text(0.012, 0.962, SUB, fontsize=12, color=INK2)
fig.tight_layout(rect=[0, 0.075, 1, 0.950])
save(fig, 'fig2_per_residue')

# ============================================================ FIGURE 3
keys = list(zip(t.resname, t.resnum, t.chi))[::-1]
fig, ax = plt.subplots(figsize=(max(11.0, 0.42 * len(keys) + 2.5), 6.6))
rng = np.random.RandomState(0)
for i, (rn, num, chi) in enumerate(keys):
    sub = r[(r.resname == rn) & (r.resnum == num) & (r.chi == chi)]
    sym = bool(sub.sym180.max())
    if sym:
        ax.axvspan(i - 0.5, i + 0.5, color='#f3f0ea', zorder=0)
    mu = float(np.median(fold(sub.md_circ_mean_deg.values, sym)))
    sd = float(np.median(sub.md_circ_std_deg.values)) / (2.0 if sym else 1.0)
    lim = 90.0 if sym else 180.0
    lo, hi = max(mu - sd, -lim), min(mu + sd, lim)
    ax.add_patch(plt.Rectangle((i - 0.30, lo), 0.60, hi - lo,
                               facecolor=MDBAND, edgecolor='none', zorder=2))
    ax.plot([i - 0.34, i + 0.34], [mu, mu], color=MDLINE, lw=2.0, zorder=3)
    a = fold(sub.angle_drgscroll_deg.values, sym)
    ax.plot(i + rng.uniform(-0.22, 0.22, len(a)), a, ls='none', marker='o',
            ms=3.4, mfc=DRG, mec='none', alpha=0.55, zorder=4)
    c1 = fold(sub.angle_complex1_deg.values, sym)
    ax.plot([i], [np.median(c1)], marker='_', ms=14, mew=2.0, color=C1, zorder=5)
ax.set_xlim(-0.6, len(keys) - 0.4)
ax.set_ylim(-185, 185)
ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
ax.set_ylabel(u'side-chain torsion angle (\u00b0)')
ax.set_xticks(np.arange(len(keys)))
ax.set_xticklabels([u'%s%d \u03c7%d%s' % (a.title(), b, c,
                                          u' *' if (a, c) in SYM180 else u'')
                    for a, b, c in keys], rotation=90, fontsize=11)
ax.grid(axis='y', alpha=0.25, lw=0.6)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
leg = [Patch(facecolor=MDBAND, label=u'MD range (circular mean \u00b1 1 SD)'),
       Line2D([], [], color=MDLINE, lw=2.0, label=u'MD circular mean'),
       Line2D([], [], ls='none', marker='o', ms=6, mfc=DRG, mec='none',
              label=u'DRGSCROLL angle (one point per complex)'),
       Line2D([], [], ls='none', marker='_', ms=12, mew=2.0, color=C1,
              label=u'starting angle')]
ax.legend(handles=leg, fontsize=12, loc='upper center', ncol=4,
          bbox_to_anchor=(0.5, 1.10), columnspacing=1.6)
fig.suptitle(u'MD-sampled angular range versus the angles chosen by DRGSCROLL',
             fontsize=17, color=INK, x=0.012, ha='left', y=0.995)
fig.text(0.012, 0.015,
         u'*  180\u00b0-periodic torsion \u2014 angles folded to '
         u'[\u221290\u00b0, 90\u00b0)', fontsize=11, color=INK2)
fig.tight_layout(rect=[0, 0.045, 1, 0.945])
save(fig, 'fig3_angle_ranges')

# ============================================================ FIGURE 4
CMAP = LinearSegmentedColormap.from_list(
    'agree', ['#e4e2dd', '#cfe0ef', '#7aafe3', '#2f6fb0', '#12375e'])
t4 = r.groupby(['resname', 'resnum', 'chi']).agg(
    n=('frac', 'size'), med=('frac', 'median'),
    mob=('md_circ_std_deg', 'median'), sym=('sym180', 'max')).reset_index()
t4 = t4[t4.n >= args.min_n].sort_values('med', ascending=False)
keys4 = list(zip(t4.resname, t4.resnum, t4.chi, t4['n'], t4['med'], t4['sym']))

ncol = 2
per = int(np.ceil(len(keys4) / float(ncol)))
fig = plt.figure(figsize=(15.5, 0.40 * per + 3.0))
gs = fig.add_gridspec(1, ncol, wspace=0.50, left=0.135, right=0.955,
                      top=0.835, bottom=0.165)
rng = np.random.RandomState(1)
sc = None
for c in range(ncol):
    chunk = keys4[c * per:(c + 1) * per]
    ax = fig.add_subplot(gs[0, c])
    for i, (rn, num, chi, n, med, sym) in enumerate(chunk):
        yy = len(chunk) - 1 - i
        if i % 2 == 0:
            ax.axhspan(yy - 0.5, yy + 0.5, color='#faf9f7', zorder=0)
        sub = r[(r.resname == rn) & (r.resnum == num) & (r.chi == chi)]
        sym = bool(sym)
        mu = float(np.median(fold(sub.md_circ_mean_deg.values, sym)))
        sd = float(np.median(sub.md_circ_std_deg.values)) / (2.0 if sym else 1.0)
        lim = 90.0 if sym else 180.0
        ax.add_patch(plt.Rectangle((max(mu - sd, -lim), yy - 0.34), 0.0 +
                                   min(mu + sd, lim) - max(mu - sd, -lim), 0.68,
                                   facecolor=MDBAND, edgecolor='none',
                                   alpha=0.9, zorder=1))
        ax.plot([mu, mu], [yy - 0.38, yy + 0.38], color=MDLINE, lw=1.7, zorder=2)
        a = fold(sub.angle_drgscroll_deg.values, sym)
        sc = ax.scatter(a, yy + rng.uniform(-0.26, 0.26, len(a)),
                        c=sub['frac'].values, cmap=CMAP, vmin=0, vmax=1,
                        s=20, linewidths=0.3, edgecolors='white', zorder=3)
        ax.text(1.03, yy, '%2.0f%%' % (100 * med), fontsize=12.5, va='center',
                color=INK if med >= CHANCE else INK2,
                transform=ax.get_yaxis_transform(),
                fontweight='bold' if med >= CHANCE else 'normal')
    ax.set_xlim(-185, 185); ax.set_ylim(-0.6, len(chunk) - 0.4)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks(range(len(chunk)))
    ax.set_yticklabels([u'%s%d  \u03c7%d%s  (n=%d)'
                        % (k[0].title(), k[1], k[2], u' *' if k[5] else u'',
                           k[3]) for k in chunk][::-1], fontsize=12, color=INK)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(u'side-chain torsion angle (\u00b0)')
    ax.grid(axis='x', alpha=0.22, lw=0.6, zorder=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.text(1.03, len(chunk) - 0.1, u'in MD', fontsize=11.5, color=INK2,
            transform=ax.get_yaxis_transform())

cax = fig.add_axes([0.135, 0.070, 0.22, 0.018])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
cb.set_label(u'fraction of MD time within \u00b1%d\u00b0 of that angle' % NEAR,
             fontsize=11.5, color=INK2, labelpad=4)
cb.ax.tick_params(labelsize=11)
cb.outline.set_visible(False)
leg4 = [Patch(facecolor=MDBAND, label=u'MD range (circular mean \u00b1 1 SD)'),
        Line2D([], [], color=MDLINE, lw=1.7, label=u'MD circular mean'),
        Line2D([], [], ls='none', marker='o', ms=7, mfc='#2f6fb0',
               mec='white', mew=0.5,
               label=u'DRGSCROLL angle \u2014 one point per complex')]
fig.legend(handles=leg4, loc='lower center', ncol=3, fontsize=12.5,
           bbox_to_anchor=(0.70, 0.050), handlelength=1.5, columnspacing=2.0)
fig.suptitle(u'Side-chain angles chosen by DRGSCROLL relative to the range '
             u'sampled by MD', fontsize=18, color=INK, x=0.012, ha='left',
             y=0.980)
fig.text(0.012, 0.935,
         SUB + u'. Rows are ordered by agreement; dark points mark angles '
               u'where MD spends substantial time.',
         fontsize=12.5, color=INK2)
fig.text(0.012, 0.014,
         u'*  180\u00b0-periodic torsion \u2014 angles folded to '
         u'[\u221290\u00b0, 90\u00b0) (\u03c7 and \u03c7+180 are the same '
         u'structure)', fontsize=11, color=INK2)
save(fig, 'fig4_overview')

print('fig1-fig4 -> %s  (%d rows, %d residue x chi)' % (OUT, len(r), len(t)))
