"""Step 3: paired statistics + virtual-screening performance, apo-start vs holo-start."""
import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(f"{ROOT}/analysis/scores.csv").merge(
    pd.read_csv(f"{ROOT}/analysis/rmsd.csv"), on="ligand")
d["delta"] = d.apo_best - d.holo_best
out = []
def rep(k, v): out.append((k, v)); print(f"{k:<46s} {v}")

print("="*70); print("B1. FINAL SCORE AGREEMENT (n=100, paired)"); print("="*70)
rep("apo  best, mean +/- sd", f"{d.apo_best.mean():.2f} +/- {d.apo_best.std():.2f}")
rep("holo best, mean +/- sd", f"{d.holo_best.mean():.2f} +/- {d.holo_best.std():.2f}")
rep("delta (apo-holo), median [IQR]",
    f"{d.delta.median():+.2f} [{d.delta.quantile(.25):+.2f}, {d.delta.quantile(.75):+.2f}]")
w = stats.wilcoxon(d.apo_best, d.holo_best)
rep("Wilcoxon signed-rank", f"W={w.statistic:.0f}, p={w.pvalue:.3f}")
rep("Pearson r",  f"{stats.pearsonr(d.apo_best, d.holo_best)[0]:.3f}")
sp = stats.spearmanr(d.apo_best, d.holo_best)
rep("Spearman rho (ranking agreement)", f"{sp.statistic:.3f}  (p={sp.pvalue:.2e})")
rep("|delta| <= 0.5 kcal/mol", f"{(d.delta.abs()<=0.5).mean()*100:.0f}%")
rep("|delta| <= 1.0 kcal/mol", f"{(d.delta.abs()<=1.0).mean()*100:.0f}%")
rep("Bland-Altman LoA (95%)",
    f"{d.delta.mean()-1.96*d.delta.std():+.2f} .. {d.delta.mean()+1.96*d.delta.std():+.2f}")

print(); print("="*70); print("B2. TOP-N OVERLAP"); print("="*70)
for n in (10, 20, 30):
    a = set(d.nsmallest(n, "apo_best").ligand); h = set(d.nsmallest(n, "holo_best").ligand)
    rep(f"top-{n} overlap", f"{len(a&h)}/{n}  (Jaccard {len(a&h)/len(a|h):.2f})")

print(); print("="*70); print("B3. ACTIVITY CORRELATION / SCREENING PERFORMANCE"); print("="*70)
for col, nm in [("apo_best","DRGSCROLL apo-start"), ("holo_best","DRGSCROLL holo-start"),
                ("vina_rigid","Vina-GPU rigid (holo)")]:
    sp = stats.spearmanr(d[col], d.pIC50)
    rep(f"{nm}: Spearman vs pIC50", f"{sp.statistic:+.3f} (p={sp.pvalue:.1e})")
lab = d.dropna(subset=["label"]).copy()
lab = lab[lab.label.isin(["active","inactive"])]
y = (lab.label == "active").astype(int)
rep("labelled subset (active/inactive)", f"n={len(lab)} ({y.sum()} act / {len(y)-y.sum()} inact)")
for col, nm in [("apo_best","apo-start"), ("holo_best","holo-start"), ("vina_rigid","Vina rigid")]:
    rep(f"ROC-AUC {nm}", f"{roc_auc_score(y, -lab[col]):.3f}")
# pIC50>=7 as actives over the full set
y2 = (d.pIC50 >= 7).astype(int)
rep(f"full set, pIC50>=7 as active", f"n={len(d)} ({y2.sum()} act)")
for col, nm in [("apo_best","apo-start"), ("holo_best","holo-start"), ("vina_rigid","Vina rigid")]:
    rep(f"ROC-AUC {nm}", f"{roc_auc_score(y2, -d[col]):.3f}")

print(); print("="*70); print("C. POSE / POCKET"); print("="*70)
rep("backbone CA RMSD apo vs holo (global)", f"{d.ca_rmsd_global.iloc[0]:.2f} A")
rep("backbone CA RMSD apo vs holo (pocket)", f"{d.ca_rmsd_pocket.mean():.2f} A")
rep("4-A residue-set Jaccard, median [IQR]",
    f"{d.pocket_jaccard.median():.2f} [{d.pocket_jaccard.quantile(.25):.2f}, {d.pocket_jaccard.quantile(.75):.2f}]")
rep("ligand RMSD apo-start vs holo-start, median", f"{d.rmsd_start.median():.2f} A")
rep("ligand RMSD apo-final vs holo-final, median", f"{d.rmsd_final.median():.2f} A")
rep("fraction where final RMSD < start RMSD", f"{(d.rmsd_final<d.rmsd_start).mean()*100:.0f}%")
rep("redocking displacement, apo  (median)", f"{d.move_apo.median():.2f} A")
rep("redocking displacement, holo (median)", f"{d.move_holo.median():.2f} A")
rep("corr(start RMSD, |delta score|)",
    f"{stats.spearmanr(d.rmsd_start, d.delta.abs()).statistic:+.3f} "
    f"(p={stats.spearmanr(d.rmsd_start, d.delta.abs()).pvalue:.2f})")
rep("corr(pocket Jaccard, |delta score|)",
    f"{stats.spearmanr(d.pocket_jaccard, d.delta.abs()).statistic:+.3f} "
    f"(p={stats.spearmanr(d.pocket_jaccard, d.delta.abs()).pvalue:.2f})")

print(); print("="*70); print("D. SEARCH BEHAVIOUR"); print("="*70)
rep("n cluster representatives, apo  median [IQR]",
    f"{d.apo_n_reps.median():.0f} [{d.apo_n_reps.quantile(.25):.0f}, {d.apo_n_reps.quantile(.75):.0f}]")
rep("n cluster representatives, holo median [IQR]",
    f"{d.holo_n_reps.median():.0f} [{d.holo_n_reps.quantile(.25):.0f}, {d.holo_n_reps.quantile(.75):.0f}]")
rep("ADFR initial affinity apo  (median)", f"{d.apo_adfr_best.median():.2f}")
rep("ADFR initial affinity holo (median)", f"{d.holo_adfr_best.median():.2f}")
w2 = stats.wilcoxon(d.apo_adfr_best, d.holo_adfr_best)
rep("Wilcoxon, ADFR initial apo vs holo", f"W={w2.statistic:.0f}, p={w2.pvalue:.2e}")
rep("gain apo  (ADFR start -> DRGSCROLL final)", f"{(d.apo_best-d.apo_adfr_best).median():.2f} kcal/mol")
rep("gain holo (ADFR start -> DRGSCROLL final)", f"{(d.holo_best-d.holo_adfr_best).median():.2f} kcal/mol")

pd.DataFrame(out, columns=["metric","value"]).to_csv(f"{ROOT}/analysis/summary_stats.csv", index=False)
d.to_csv(f"{ROOT}/analysis/merged.csv", index=False)
