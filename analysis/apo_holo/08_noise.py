"""Step 8: noise floor. Is the apo-vs-holo score difference larger than the
within-run generation-to-generation fluctuation of the same protocol?"""
import os, numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(f"{ROOT}/analysis/merged.csv")
G = [f"{t}_g{g}" for t in ("apo","holo") for g in range(40,50)]

for t in ("apo","holo"):
    cols = [f"{t}_g{g}" for g in range(40,50)]
    d[f"{t}_gen_sd"]    = d[cols].std(axis=1)
    d[f"{t}_gen_range"] = d[cols].max(axis=1) - d[cols].min(axis=1)
    d[f"{t}_gen_n"]     = d[cols].notna().sum(axis=1)
d["abs_delta"] = d.delta.abs()
# pooled within-run sd -> expected |difference| of two independent runs
pooled = np.sqrt((d.apo_gen_sd**2 + d.holo_gen_sd**2) / 2)
d["expected_abs_delta"] = pooled * np.sqrt(2) * np.sqrt(2/np.pi)   # E|N(0, 2s^2)|

print("="*72); print("NOISE FLOOR OF THE PROTOCOL (generations 40-49 within one run)"); print("="*72)
def q(x): x=x.dropna(); return f"{x.median():.2f} [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}]"
print(f"within-run SD of best score, apo        {q(d.apo_gen_sd)} kcal/mol")
print(f"within-run SD of best score, holo       {q(d.holo_gen_sd)} kcal/mol")
print(f"within-run range (gen40-49), holo       {q(d.holo_gen_range)} kcal/mol")
print(f"\nobserved |apo - holo|                   {q(d.abs_delta)} kcal/mol")
print(f"expected |difference| from noise alone  {q(d.expected_abs_delta)} kcal/mol")
sub = d.dropna(subset=["abs_delta","expected_abs_delta"])
w = stats.wilcoxon(sub.abs_delta, sub.expected_abs_delta)
print(f"\nWilcoxon observed vs noise-expected: W={w.statistic:.0f}, p={w.pvalue:.2e} (n={len(sub)})")
print(f"ratio observed/expected (median): {(sub.abs_delta/sub.expected_abs_delta.replace(0,np.nan)).median():.2f}x")
print(f"|apo-holo| <= within-run range of the holo run: "
      f"{(d.abs_delta <= d.holo_gen_range).mean()*100:.0f}% of ligands")

print("\n" + "="*72); print("IS THE DIFFERENCE EXPLAINED BY HOW DIFFERENT THE START WAS?"); print("="*72)
for v, nm in [("rmsd_start","ligand RMSD of the two starting poses"),
              ("pocket_jaccard","overlap of the 4-A flexible residue sets"),
              ("ca_rmsd_pocket","backbone difference in the pocket"),
              ("d_adfr", "difference in ADFR starting affinity")]:
    if v == "d_adfr": d[v] = (d.apo_adfr_best - d.holo_adfr_best).abs()
    if d[v].nunique() < 3: print(f"{nm:<44s} constant, skipped"); continue
    s = stats.spearmanr(d[v], d.abs_delta, nan_policy="omit")
    print(f"{nm:<44s} rho={s.statistic:+.3f}  p={s.pvalue:.2f}")
d.to_csv(f"{ROOT}/analysis/merged.csv", index=False)
