import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = f"{ROOT}/analysis/figures"; os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(f"{ROOT}/analysis/merged.csv"); c = pd.read_csv(f"{ROOT}/analysis/chi.csv")
plt.rcParams.update({"font.size":8,"axes.titlesize":9,"figure.dpi":300,
                     "axes.spines.top":False,"axes.spines.right":False})
BL, OR, GY = "#2166ac", "#d6604d", "#777777"

# ---------- Figure 1: score agreement ----------
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
a, h = d.apo_best, d.holo_best
lim = [min(a.min(), h.min())-.4, max(a.max(), h.max())+.4]
ax[0].fill_between(lim, [x-1 for x in lim], [x+1 for x in lim], color=BL, alpha=.10, lw=0)
ax[0].plot(lim, lim, "-", c=GY, lw=.8)
ax[0].scatter(h, a, s=14, c=BL, alpha=.75, ec="none")
ax[0].set(xlim=lim, ylim=lim, xlabel="holo-started score (kcal/mol)",
          ylabel="apo-started score (kcal/mol)", title="a  Final DRGSCROLL score")
ax[0].text(.04,.94, f"$\\rho$ = {stats.spearmanr(a,h).statistic:.2f}\n"
           f"r = {stats.pearsonr(a,h)[0]:.2f}\n"
           f"Wilcoxon p = {stats.wilcoxon(a,h).pvalue:.2f}\n"
           f"84% within 1 kcal/mol", transform=ax[0].transAxes, va="top", fontsize=7)

mean, diff = (a+h)/2, a-h
ax[1].axhline(0, c=GY, lw=.8)
for s, st in ((diff.mean(),"-"), (diff.mean()+1.96*diff.std(),"--"), (diff.mean()-1.96*diff.std(),"--")):
    ax[1].axhline(s, ls=st, c=OR, lw=.9)
ax[1].scatter(mean, diff, s=14, c=BL, alpha=.75, ec="none")
ax[1].set(xlabel="mean score (kcal/mol)", ylabel="apo - holo (kcal/mol)",
          title="b  Bland-Altman")
ax[1].text(.04,.04, f"bias {diff.mean():+.2f}\n95% LoA [{diff.mean()-1.96*diff.std():.2f}, "
           f"{diff.mean()+1.96*diff.std():.2f}]", transform=ax[1].transAxes, fontsize=7)

y = (d.pIC50 >= 7).astype(int)
for col, nm, col_ in [("apo_best","apo-started",OR), ("holo_best","holo-started",BL),
                      ("vina_rigid","rigid Vina-GPU",GY)]:
    fpr, tpr, _ = roc_curve(y, -d[col])
    ax[2].plot(fpr, tpr, c=col_, lw=1.4, label=f"{nm} (AUC {roc_auc_score(y,-d[col]):.2f})")
ax[2].plot([0,1],[0,1],":",c="k",lw=.7)
ax[2].set(xlabel="false positive rate", ylabel="true positive rate",
          title="c  Actives (pIC$_{50}\\geq$7) retrieval")
ax[2].legend(frameon=False, fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_score_agreement.png"); plt.close(fig)

# ---------- Figure 2: is the difference driven by the starting point? ----------
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for i, (v, lab) in enumerate([("rmsd_start","ligand RMSD of the two starting poses ($\\AA$)"),
                              ("pocket_jaccard","Jaccard overlap of 4-$\\AA$ residue sets")]):
    s = stats.spearmanr(d[v], d.delta.abs(), nan_policy="omit")
    ax[i].scatter(d[v], d.delta.abs(), s=14, c=BL, alpha=.75, ec="none")
    ax[i].set(xlabel=lab, ylabel="|apo - holo| score (kcal/mol)",
              title=f"{'ab'[i]}  $\\rho$ = {s.statistic:+.2f}, p = {s.pvalue:.2f}")
bins = np.linspace(0, 3, 40)
ax[2].hist(d.delta.abs(), bins=bins, color=OR, alpha=.75, label="observed |apo - holo|")
ax[2].hist(d.expected_abs_delta, bins=bins, color=BL, alpha=.65,
           label="within-run fluctuation\n(generations 40-49)")
ax[2].set(xlabel="|score difference| (kcal/mol)", ylabel="ligands",
          title="c  Difference vs. protocol noise")
ax[2].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_start_dependence.png"); plt.close(fig)

# ---------- Figure 3: poses and side chains ----------
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
m = min(d.rmsd_start.max(), d.rmsd_final.max())+1
ax[0].plot([0,m],[0,m],"-",c=GY,lw=.8)
ax[0].scatter(d.rmsd_start, d.rmsd_final, s=14, c=BL, alpha=.75, ec="none")
ax[0].set(xlim=(0,m), ylim=(0,m), xlabel="start: apo vs holo ligand RMSD ($\\AA$)",
          ylabel="final: apo vs holo ligand RMSD ($\\AA$)", title="a  Pose convergence")
ax[0].text(.04,.94,f"below the line: {(d.rmsd_final<d.rmsd_start).mean()*100:.0f}%\n"
           f"final <2.5 $\\AA$: {(d.ens_min<2.5).mean()*100:.0f}%",
           transform=ax[0].transAxes, va="top", fontsize=7)

bins = np.linspace(0, 180, 37)
ax[1].hist(c.d_chi1_start.dropna(), bins=bins, color=BL, alpha=.7, label="starting structures")
ax[1].hist(c.d_chi1_final.dropna(), bins=bins, color=OR, alpha=.7, label="final structures")
ax[1].set(xlabel="|$\\Delta\\chi_1$| apo vs holo (deg)", ylabel="pocket residues",
          title="b  Side-chain states", xticks=[0,60,120,180])
ax[1].legend(frameon=False, fontsize=6.5)

key = [861,889,896,897,907,862,863,903,904,890,988,866]
g = c[c.parp.isin(key)].dropna(subset=["rot_apo","rot_holo"]).groupby("parp").apply(
    lambda x: pd.Series({"agree":(x.rot_apo==x.rot_holo).mean()*100,
                         "res":x.res_apo.iloc[0],"n":len(x)}), include_groups=False)
g = g[g.n>=20].sort_values("agree")
ax[2].barh(range(len(g)), g.agree.astype(float), color=BL, alpha=.8)
ax[2].set_yticks(range(len(g)), [f"{r.res.title()}{i}" for i, r in g.iterrows()], fontsize=6.5)
ax[2].axvline(100/3, ls=":", c=GY, lw=.8)
ax[2].set(xlabel="$\\chi_1$ rotamer agreement (%)", xlim=(0,100),
          title="c  Catalytic-site residues")
ax[2].text(34, -.8, "chance", fontsize=6, color=GY)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_pose_sidechain.png"); plt.close(fig)
print("figures written to analysis/figures/")
for f in sorted(os.listdir(OUT)): print("  ", f)
