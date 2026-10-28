"""Step 6: full-pocket contact fingerprint + pose-agreement stratification."""
import os, sys, glob, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_struct as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(f"{ROOT}/analysis/merged.csv")

def start_pdb(dd):
    p = os.path.join(dd, "complex1.pdb")
    return p if os.path.exists(p) else [x for x in glob.glob(os.path.join(dd,"*_complex.pdb"))
                                        if "rotated" not in x][0]

def fp(prot, lig_block, to_apo, cutoff=4.5):
    """Binary residue-contact fingerprint, keyed on apo (= common) indexing."""
    lig = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lig_block.splitlines() if l.startswith(("ATOM","HETATM"))])
    byres = {}
    for (ri, an), v in prot.items(): byres.setdefault(ri, []).append(v)
    s = set()
    for ri, pts in byres.items():
        if np.linalg.norm(np.array(pts)[:,None,:]-lig[None,:,:],axis=2).min() <= cutoff:
            k = to_apo(ri)
            if k: s.add(k)
    return s

rows = []
for _, r in d.iterrows():
    a0, la0 = L.read_pdb(start_pdb(r.apo_dir)); h0, lh0 = L.read_pdb(start_pdb(r.holo_dir))
    af, laf = L.read_pdb(os.path.join(r.apo_dir,  f"complex_rotated_{r.apo_best_tag}_complex.pdb"))
    hf, lhf = L.read_pdb(os.path.join(r.holo_dir, f"complex_rotated_{r.holo_best_tag}_complex.pdb"))
    id_ = lambda i: i
    sA0, sH0 = fp(a0, la0, id_), fp(h0, lh0, L.holo2apo)
    sAf, sHf = fp(af, laf, id_), fp(hf, lhf, L.holo2apo)
    tan = lambda x, y: len(x & y) / max(1, len(x | y))
    rows.append(dict(ligand=r.ligand, fpT_start=tan(sA0, sH0), fpT_final=tan(sAf, sHf),
                     fpT_apo_move=tan(sA0, sAf), fpT_holo_move=tan(sH0, sHf),
                     n_contact_apo=len(sAf), n_contact_holo=len(sHf)))
f = pd.DataFrame(rows); f.to_csv(f"{ROOT}/analysis/fingerprint.csv", index=False)
m = d.merge(f, on="ligand").merge(pd.read_csv(f"{ROOT}/analysis/core_contacts.csv"), on="ligand")
m.to_csv(f"{ROOT}/analysis/merged.csv", index=False)

def q(x): return f"{np.nanmedian(x):.2f} [{np.nanpercentile(x,25):.2f}, {np.nanpercentile(x,75):.2f}]"
print("="*72); print("FULL-POCKET CONTACT FINGERPRINT (Tanimoto, all residues <=4.5 A)"); print("="*72)
print(f"apo-start  vs holo-start   {q(m.fpT_start)}")
print(f"apo-final  vs holo-final   {q(m.fpT_final)}")
print(f"apo-start  vs apo-final    {q(m.fpT_apo_move)}")
print(f"holo-start vs holo-final   {q(m.fpT_holo_move)}")
w = stats.wilcoxon(m.fpT_final, m.fpT_start)
print(f"\nfinal vs start fingerprint agreement: Wilcoxon p={w.pvalue:.2e} "
      f"(final higher in {(m.fpT_final>m.fpT_start).mean()*100:.0f}% of ligands)")

print("\n" + "="*72); print("STRATIFICATION BY POSE AGREEMENT"); print("="*72)
m["grp"] = np.where(m.ens_min < 2.5, "pose reproduced (<2.5 A)", "pose differs (>=2.5 A)")
g = m.groupby("grp").agg(n=("ligand","size"),
        d_score_med=("delta", lambda x: np.median(np.abs(x))),
        within1=("delta", lambda x: (np.abs(x)<=1).mean()*100),
        fpT=("fpT_final","median"), key_fp=("fp_tanimoto","median"),
        rmsd_start=("rmsd_start","median"), pocketJ=("pocket_jaccard","median"))
print(g.round(2).to_string())
a = m[m.grp.str.startswith("pose reproduced")].delta.abs()
b = m[m.grp.str.startswith("pose differs")].delta.abs()
print(f"\n|delta score| differs between groups? Mann-Whitney p="
      f"{stats.mannwhitneyu(a,b).pvalue:.3f}")
print(f"\nSpearman(apo,holo) within 'pose reproduced': "
      f"{stats.spearmanr(m[m.grp.str.startswith('pose rep')].apo_best, m[m.grp.str.startswith('pose rep')].holo_best).statistic:.3f}")
print(f"Spearman(apo,holo) within 'pose differs'   : "
      f"{stats.spearmanr(m[m.grp.str.startswith('pose dif')].apo_best, m[m.grp.str.startswith('pose dif')].holo_best).statistic:.3f}")
