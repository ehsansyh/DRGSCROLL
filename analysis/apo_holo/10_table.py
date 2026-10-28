import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d=pd.read_csv(f"{R}/analysis/merged.csv"); c=pd.read_csv(f"{R}/analysis/chi.csv")
y=(d.pIC50>=7).astype(int); lab=d[d.label.isin(["active","inactive"])]; yl=(lab.label=="active").astype(int)
ok=c.dropna(subset=["rot_apo","rot_holo"]); ok0=c.dropna(subset=["rot_apo0","rot_holo0"])
def iqr(x): x=pd.Series(x).dropna(); return f"{x.median():.2f} [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}]"
T=[("Starting point","Backbone Ca RMSD, apo vs holo (whole domain)","1.33 A","-"),
 ("","Backbone Ca RMSD, apo vs holo (binding pocket)",f"{d.ca_rmsd_pocket.median():.2f} A","-"),
 ("","Per-residue all-atom RMSD > 1 A","139 / 347 residues","-"),
 ("","Ligand RMSD between the two ADFR starting poses",f"{iqr(d.rmsd_start)} A","-"),
 ("","Overlap of the 4-A flexible-residue sets (Jaccard)",iqr(d.pocket_jaccard),"-"),
 ("","|dchi1| of pocket side chains, starting structures",f"{c.d_chi1_start.median():.0f} deg","-"),
 ("","chi1 rotamer agreement, starting structures",f"{(ok0.rot_apo0==ok0.rot_holo0).mean()*100:.1f}%","-"),
 ("Final score","Mean final score, apo-started",f"{d.apo_best.mean():.2f} +/- {d.apo_best.std():.2f}","kcal/mol"),
 ("","Mean final score, holo-started",f"{d.holo_best.mean():.2f} +/- {d.holo_best.std():.2f}","kcal/mol"),
 ("","Paired difference (apo - holo), median [IQR]",iqr(d.delta),"kcal/mol"),
 ("","Wilcoxon signed-rank",f"W={stats.wilcoxon(d.apo_best,d.holo_best).statistic:.0f}, p={stats.wilcoxon(d.apo_best,d.holo_best).pvalue:.2f}","n.s."),
 ("","Pearson r / Spearman rho",f"{stats.pearsonr(d.apo_best,d.holo_best)[0]:.3f} / {stats.spearmanr(d.apo_best,d.holo_best).statistic:.3f}","-"),
 ("","Ligands within 1.0 kcal/mol",f"{(d.delta.abs()<=1).mean()*100:.0f}%","-"),
 ("","Bland-Altman 95% limits of agreement",f"[{d.delta.mean()-1.96*d.delta.std():.2f}, {d.delta.mean()+1.96*d.delta.std():.2f}]","kcal/mol"),
 ("Ranking","Top-10 overlap","9 / 10","-"),
 ("","Top-30 overlap","22 / 30","-"),
 ("","Spearman vs pIC50, apo-started",f"{stats.spearmanr(d.apo_best,d.pIC50).statistic:+.3f} (p={stats.spearmanr(d.apo_best,d.pIC50).pvalue:.3f})","-"),
 ("","Spearman vs pIC50, holo-started",f"{stats.spearmanr(d.holo_best,d.pIC50).statistic:+.3f} (p={stats.spearmanr(d.holo_best,d.pIC50).pvalue:.3f})","-"),
 ("","Spearman vs pIC50, rigid Vina-GPU",f"{stats.spearmanr(d.vina_rigid,d.pIC50).statistic:+.3f} (p={stats.spearmanr(d.vina_rigid,d.pIC50).pvalue:.2f})","n.s."),
 ("","ROC-AUC (pIC50>=7), apo / holo / rigid",f"{roc_auc_score(y,-d.apo_best):.3f} / {roc_auc_score(y,-d.holo_best):.3f} / {roc_auc_score(y,-d.vina_rigid):.3f}","-"),
 ("","ROC-AUC (annotated set, n=44), apo / holo / rigid",f"{roc_auc_score(yl,-lab.apo_best):.3f} / {roc_auc_score(yl,-lab.holo_best):.3f} / {roc_auc_score(yl,-lab.vina_rigid):.3f}","-"),
 ("Start dependence","Spearman(|dscore|, starting-pose RMSD)",f"{stats.spearmanr(d.rmsd_start,d.delta.abs()).statistic:+.3f} (p={stats.spearmanr(d.rmsd_start,d.delta.abs()).pvalue:.2f})","n.s."),
 ("","Spearman(|dscore|, flexible-residue-set overlap)",f"{stats.spearmanr(d.pocket_jaccard,d.delta.abs()).statistic:+.3f} (p={stats.spearmanr(d.pocket_jaccard,d.delta.abs()).pvalue:.2f})","n.s."),
 ("","Spearman(|dscore|, ADFR starting-affinity difference)",f"{stats.spearmanr(d.apo_adfr_best.sub(d.holo_adfr_best).abs(),d.delta.abs()).statistic:+.3f} (p={stats.spearmanr(d.apo_adfr_best.sub(d.holo_adfr_best).abs(),d.delta.abs()).pvalue:.2f})","n.s."),
 ("","Within-run score SD (generations 40-49)",f"{d.holo_gen_sd.median():.2f}","kcal/mol"),
 ("","|apo-holo| within the holo run's own gen 40-49 range",f"{(d.delta.abs()<=d.holo_gen_range).mean()*100:.0f}% of ligands","-"),
 ("","Ligand displacement from its own ADFR start, holo run",f"{iqr(d.move_holo)} A","-"),
 ("Pose / structure","Ligand RMSD apo-final vs holo-final",f"{iqr(d.ens_min)} A","-"),
 ("","Poses reproduced within 2.5 A",f"{(d.ens_min<2.5).mean()*100:.0f}%","-"),
 ("","Pocket contact fingerprint Tanimoto (final)",iqr(d.fpT_final),"-"),
 ("","Catalytic-contact fingerprint Tanimoto (final)",iqr(d.fp_tanimoto),"-"),
 ("","Tyr907 / Tyr896 / Gly863 contact retained (apo | holo)","100|100 / 100|100 / 95|95 %","-"),
 ("","chi1 rotamer agreement, final structures",f"{(ok.rot_apo==ok.rot_holo).mean()*100:.1f}%","-"),
 ("","Pose-vs-score: |dscore| in reproduced vs diverged poses",f"p={stats.mannwhitneyu(d[d.ens_min<2.5].delta.abs(),d[d.ens_min>=2.5].delta.abs()).pvalue:.2f}","n.s.")]
t=pd.DataFrame(T,columns=["Block","Metric","Value","Unit"])
t.to_csv(f"{R}/analysis/TABLE_R1.csv",index=False)
w=max(t.Metric.str.len())
print(f"{'':2s}{'Metric':<{w}}  Value")
blk=None
for _,r in t.iterrows():
    if r.Block!=blk: blk=r.Block; print(f"\n--- {blk} " + "-"*(w+20-len(blk)))
    print(f"  {r.Metric:<{w}}  {r.Value}")
