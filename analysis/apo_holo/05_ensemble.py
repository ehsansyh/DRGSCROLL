"""Step 5: ensemble-level pose comparison + the key control.

Question: is the apo-vs-holo pose difference larger than the pose scatter the
method produces anyway within a single run? If not, the starting structure adds
nothing beyond the method's intrinsic redocking noise.
"""
import os, sys, glob, re, itertools, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_struct as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(f"{ROOT}/analysis/merged.csv")
MAXREP = 12

def start_pdb(dd):
    p = os.path.join(dd, "complex1.pdb")
    return p if os.path.exists(p) else [x for x in glob.glob(os.path.join(dd,"*_complex.pdb"))
                                        if "rotated" not in x][0]

def reps(dd, limit=MAXREP):
    """Cluster representatives of one run, newest-generation first."""
    fs = glob.glob(os.path.join(dd, "complex_rotated_*_complex.pdb"))
    key = lambda f: tuple(int(x) for x in re.search(r"rotated_(\d+)_(\d+)_", f).groups())
    return sorted(fs, key=key, reverse=True)[:limit]

rows = []
for _, r in d.iterrows():
    a0, la0 = L.read_pdb(start_pdb(r.apo_dir))
    h0, lh0 = L.read_pdb(start_pdb(r.holo_dir))
    R, t, _, _ = L.super_apo_on_holo(a0, h0)
    A = [L.lig_mol(L.read_pdb(f)[1], R, t) for f in reps(r.apo_dir)]
    H = [L.lig_mol(L.read_pdb(f)[1])       for f in reps(r.holo_dir)]
    A = [m for m in A if m]; H = [m for m in H if m]
    o = dict(ligand=r.ligand, n_apo=len(A), n_holo=len(H))
    # cross-ensemble: does the apo ensemble contain a holo-like pose?
    cross = [L.sym_rmsd(a, h) for a in A for h in H]
    o["ens_min"] = np.nanmin(cross); o["ens_median"] = np.nanmedian(cross)
    # within-ensemble spread (the method's own pose scatter) - the null model
    wa = [L.sym_rmsd(x, y) for x, y in itertools.combinations(A, 2)]
    wh = [L.sym_rmsd(x, y) for x, y in itertools.combinations(H, 2)]
    o["within_apo"]  = np.nanmedian(wa) if wa else np.nan
    o["within_holo"] = np.nanmedian(wh) if wh else np.nan
    rows.append(o)
    print(f"{r.ligand:16s} n={len(A)}/{len(H)}  ens_min={o['ens_min']:.2f} "
          f"within a/h={o['within_apo'] if wa else float('nan'):.2f}/"
          f"{o['within_holo'] if wh else float('nan'):.2f}", flush=True)

e = pd.DataFrame(rows); e.to_csv(f"{ROOT}/analysis/ensemble.csv", index=False)
m = d.merge(e, on="ligand")

print("\n" + "="*70); print("KEY CONTROL: between-structure vs within-run pose scatter"); print("="*70)
def q(x): return f"{np.nanmedian(x):.2f} [{np.nanpercentile(x,25):.2f}, {np.nanpercentile(x,75):.2f}]"
print(f"apo-final vs holo-final (best pose)     {q(m.rmsd_final)}")
print(f"apo-final vs holo-final (best matching) {q(m.ens_min)}")
print(f"within-run scatter, holo (null)         {q(m.within_holo)}")
print(f"within-run scatter, apo  (null)         {q(m.within_apo)}")
print(f"redock displacement vs own start, holo  {q(m.move_holo)}")
sub = m.dropna(subset=["rmsd_final","within_holo"])
w = stats.wilcoxon(sub.rmsd_final, sub.within_holo)
print(f"\nWilcoxon  between-structure vs within-run(holo):  "
      f"W={w.statistic:.0f}, p={w.pvalue:.3f}  (n={len(sub)})")
sub2 = m.dropna(subset=["ens_min","within_holo"])
w2 = stats.wilcoxon(sub2.ens_min, sub2.within_holo)
print(f"Wilcoxon  best-matching  vs within-run(holo):  "
      f"W={w2.statistic:.0f}, p={w2.pvalue:.2e}  (n={len(sub2)})")
print(f"\nens_min < 2 A: {(m.ens_min<2).mean()*100:.0f}%    ens_min < 2.5 A: {(m.ens_min<2.5).mean()*100:.0f}%")
m.to_csv(f"{ROOT}/analysis/merged.csv", index=False)
