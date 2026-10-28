"""Step 2: pose-level comparison. Backbone fit, initial/final ligand RMSD, displacement."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_struct as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "analysis", "scores.csv"))

def pocket_resis(prot, lig_block, cutoff=4.0):
    lig = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lig_block.splitlines() if l.startswith(("ATOM", "HETATM"))])
    keys = list(prot); xyz = np.array([prot[k] for k in keys])
    d = np.linalg.norm(xyz[:, None, :] - lig[None, :, :], axis=2).min(1)
    return sorted({keys[i][0] for i in np.where(d <= cutoff)[0]})

def start_pdb(d):
    p = os.path.join(d, "complex1.pdb")
    if os.path.exists(p): return p
    import glob as g
    c = [x for x in g.glob(os.path.join(d, "*_complex.pdb")) if "rotated" not in x]
    return c[0]

rows = []
for _, r in df.iterrows():
    out = dict(ligand=r.ligand)
    ap, hp = r.apo_dir, r.holo_dir
    a0, la0 = L.read_pdb(start_pdb(ap))
    h0, lh0 = L.read_pdb(start_pdb(hp))
    af, laf = L.read_pdb(os.path.join(ap, f"complex_rotated_{r.apo_best_tag}_complex.pdb"))
    hf, lhf = L.read_pdb(os.path.join(hp, f"complex_rotated_{r.holo_best_tag}_complex.pdb"))

    # pocket (apo indexing) from each starting complex
    pa = set(pocket_resis(a0, la0))
    ph = {L.holo2apo(j) for j in pocket_resis(h0, lh0)}
    out["n_pocket_apo"], out["n_pocket_holo"] = len(pa), len(ph)
    out["pocket_jaccard"] = len(pa & ph) / len(pa | ph)
    pocket = pa | ph

    # global and pocket-restricted backbone fit of apo onto holo
    R, t, out["ca_rmsd_global"], n = L.super_apo_on_holo(a0, h0)
    out["n_ca"] = n
    Rp, tp, out["ca_rmsd_pocket"], _ = L.super_apo_on_holo(a0, h0, pocket=pocket)

    # ligand poses, all expressed in the holo frame (global CA fit)
    mA0, mH0 = L.lig_mol(la0, R, t), L.lig_mol(lh0)
    mAf, mHf = L.lig_mol(laf, R, t), L.lig_mol(lhf)
    out["rmsd_start"]  = L.sym_rmsd(mA0, mH0)   # apo start  vs holo start
    out["rmsd_final"]  = L.sym_rmsd(mAf, mHf)   # apo final  vs holo final
    out["move_apo"]    = L.sym_rmsd(mAf, mA0)   # apo  final vs apo  start
    out["move_holo"]   = L.sym_rmsd(mHf, mH0)   # holo final vs holo start
    out["rmsd_cross"]  = L.sym_rmsd(mAf, mH0)   # apo final  vs holo start
    rows.append(out)
    print(f"{r.ligand:16s} CA {out['ca_rmsd_global']:.2f} | start {out['rmsd_start']:.2f} "
          f"final {out['rmsd_final']:.2f} | move a/h {out['move_apo']:.2f}/{out['move_holo']:.2f}",
          flush=True)

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "analysis", "rmsd.csv"), index=False)
print("\nwritten analysis/rmsd.csv")
