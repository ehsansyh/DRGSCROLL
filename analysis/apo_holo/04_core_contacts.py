"""Step 4: anchor-core RMSD + key-residue contact fingerprints."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_struct as L
from rdkit import Chem
from rdkit.Chem import rdFMCS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(f"{ROOT}/analysis/merged.csv")

# Rigid scaffold = all ring atoms. Bond orders are not reliably perceived from PDB,
# but ring perception only needs connectivity, so this is robust.
def ring_atoms(m):
    return sorted({i for r in m.GetRingInfo().AtomRings() for i in r})

def start_pdb(dd):
    p = os.path.join(dd, "complex1.pdb")
    if os.path.exists(p): return p
    return [x for x in glob.glob(os.path.join(dd, "*_complex.pdb")) if "rotated" not in x][0]

def core_rmsd(mA, mH):
    """Scaffold (ring-atom) RMSD, minimised over all graph automorphisms."""
    core = ring_atoms(mH)
    if not core: return np.nan, 0
    maps = mA.GetSubstructMatches(mH, uniquify=False, useChirality=False, maxMatches=5000)
    if not maps: return np.nan, 0
    ca, ch = mA.GetConformer(), mH.GetConformer()
    XH = np.array([list(ch.GetAtomPosition(i)) for i in core])
    best = np.inf
    for mp in maps:
        XA = np.array([list(ca.GetAtomPosition(mp[i])) for i in core])
        best = min(best, np.sqrt(((XA - XH)**2).sum(1).mean()))
    return best, len(core)

# PARP1 catalytic-site residues (true numbering) -> apo file index
KEY = {"Gly863":863, "Ser904":904, "Tyr907":907, "Tyr896":896, "His862":862,
       "Ala898":898, "Lys903":903, "Glu988":988, "Met890":890, "Tyr889":889}
KEY_APO = {k: v - 661 for k, v in KEY.items()}

def contacts(prot, lig_block, idxmap, cutoff=4.0):
    lig = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lig_block.splitlines() if l.startswith(("ATOM","HETATM"))])
    fp = {}
    for nm, apo_i in KEY_APO.items():
        i = idxmap(apo_i)
        pts = np.array([v for (ri, an), v in prot.items() if ri == i]) if i else np.empty((0,3))
        fp[nm] = 0 if len(pts) == 0 else int(
            (np.linalg.norm(pts[:,None,:]-lig[None,:,:],axis=2).min() <= cutoff))
    return fp

rows = []
for _, r in d.iterrows():
    a0, la0 = L.read_pdb(start_pdb(r.apo_dir))
    h0, lh0 = L.read_pdb(start_pdb(r.holo_dir))
    af, laf = L.read_pdb(os.path.join(r.apo_dir,  f"complex_rotated_{r.apo_best_tag}_complex.pdb"))
    hf, lhf = L.read_pdb(os.path.join(r.holo_dir, f"complex_rotated_{r.holo_best_tag}_complex.pdb"))
    R, t, _, _ = L.super_apo_on_holo(a0, h0)
    mA0, mH0 = L.lig_mol(la0, R, t), L.lig_mol(lh0)
    mAf, mHf = L.lig_mol(laf, R, t), L.lig_mol(lhf)
    o = dict(ligand=r.ligand)
    o["core_start"], o["core_n"] = core_rmsd(mA0, mH0)
    o["core_final"], _ = core_rmsd(mAf, mHf)
    o["core_move_apo"],  _ = core_rmsd(mAf, mA0)
    o["core_move_holo"], _ = core_rmsd(mHf, mH0)
    fa = contacts(af, laf, lambda i: i)                      # apo final, apo indexing
    fh = contacts(hf, lhf, L.apo2holo)                       # holo final, holo indexing
    o["fp_tanimoto"] = (sum(fa[k] and fh[k] for k in KEY) /
                        max(1, sum(fa[k] or fh[k] for k in KEY)))
    o["fp_agree"] = np.mean([fa[k] == fh[k] for k in KEY])
    for k in KEY: o[f"apo_{k}"], o[f"holo_{k}"] = fa[k], fh[k]
    rows.append(o)

c = pd.DataFrame(rows)
c.to_csv(f"{ROOT}/analysis/core_contacts.csv", index=False)
print("scaffold (ring) atoms found in %d/100 ligands (median %d atoms)\n" %
      (c.core_start.notna().sum(), np.nanmedian(c.core_n)))
print(c[["core_start","core_final","core_move_apo","core_move_holo",
         "fp_tanimoto","fp_agree"]].describe().round(2).to_string())
print("\ncore RMSD apo-final vs holo-final  <2 A: %.0f%%" % ((c.core_final<2).mean()*100))
print("core RMSD apo-start vs holo-start  <2 A: %.0f%%" % ((c.core_start<2).mean()*100))
print("\nkey-contact frequency (apo-final / holo-final):")
for k in KEY:
    print(f"  {k:8s} {c['apo_'+k].mean()*100:5.0f}% / {c['holo_'+k].mean()*100:5.0f}%")
