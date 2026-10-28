"""Step 7: side-chain chi1/chi2 comparison between apo-started and holo-started runs."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_struct as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(f"{ROOT}/analysis/merged.csv")

CHI1 = {"ARG":"CG","ASN":"CG","ASP":"CG","CYS":"SG","GLN":"CG","GLU":"CG","HIS":"CG",
        "ILE":"CG1","LEU":"CG","LYS":"CG","MET":"CG","PHE":"CG","PRO":"CG","SER":"OG",
        "THR":"OG1","TRP":"CG","TYR":"CG","VAL":"CG1"}
CHI2 = {"ARG":("CB","CG","CD"),"ASN":("CB","CG","OD1"),"ASP":("CB","CG","OD1"),
        "GLN":("CB","CG","CD"),"GLU":("CB","CG","CD"),"HIS":("CB","CG","ND1"),
        "ILE":("CB","CG1","CD1"),"LEU":("CB","CG","CD1"),"LYS":("CB","CG","CD"),
        "MET":("CB","CG","SD"),"PHE":("CB","CG","CD1"),"TRP":("CB","CG","CD1"),
        "TYR":("CB","CG","CD1")}

def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0-p1, p2-p1, p3-p2
    b1 = b1/np.linalg.norm(b1)
    v = b0 - b0.dot(b1)*b1; w = b2 - b2.dot(b1)*b1
    return np.degrees(np.arctan2(np.cross(b1, v).dot(w), v.dot(w)))

def read_full(path):
    """-> dict[resi] = (resname, {atomname: xyz})"""
    res = {}
    for l in open(path, errors="ignore"):
        if l[:6] not in ("ATOM  ", "HETATM"): continue
        rn = l[17:20].strip()
        if rn == "UNK": continue
        ri = int(l[22:26])
        res.setdefault(ri, [rn, {}])[1][l[12:16].strip()] = np.array(
            [float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return res

def chis(res, ri):
    if ri not in res: return (np.nan, np.nan, None)
    rn, at = res[ri]
    c1 = c2 = np.nan
    g = CHI1.get(rn)
    if g and all(a in at for a in ("N","CA","CB",g)):
        c1 = dihedral(at["N"], at["CA"], at["CB"], at[g])
    if rn in CHI2 and all(a in at for a in CHI2[rn]) and "CA" in at:
        b, c, dd = CHI2[rn]
        c2 = dihedral(at["CA"], at[b], at[c], at[dd])
    return (c1, c2, rn)

def angdiff(a, b):
    if np.isnan(a) or np.isnan(b): return np.nan
    return abs((a - b + 180) % 360 - 180)

def rotamer(x):
    if np.isnan(x): return None
    x = (x + 360) % 360
    return "g+" if 0 <= x < 120 else ("t" if 120 <= x < 240 else "g-")

def start_pdb(dd):
    p = os.path.join(dd, "complex1.pdb")
    return p if os.path.exists(p) else [x for x in glob.glob(os.path.join(dd,"*_complex.pdb"))
                                        if "rotated" not in x][0]

def pocket(dd, to_apo, cutoff=4.0):
    prot, lig = L.read_pdb(start_pdb(dd))
    lg = np.array([[float(l[30:38]),float(l[38:46]),float(l[46:54])]
                   for l in lig.splitlines() if l.startswith(("ATOM","HETATM"))])
    byres = {}
    for (ri,an),v in prot.items(): byres.setdefault(ri,[]).append(v)
    out = set()
    for ri,pts in byres.items():
        if np.linalg.norm(np.array(pts)[:,None,:]-lg[None,:,:],axis=2).min() <= cutoff:
            k = to_apo(ri)
            if k: out.add(k)
    return out

rows = []
for _, r in d.iterrows():
    A0, Af = read_full(start_pdb(r.apo_dir)), read_full(os.path.join(
        r.apo_dir, f"complex_rotated_{r.apo_best_tag}_complex.pdb"))
    H0, Hf = read_full(start_pdb(r.holo_dir)), read_full(os.path.join(
        r.holo_dir, f"complex_rotated_{r.holo_best_tag}_complex.pdb"))
    pk = pocket(r.apo_dir, lambda i: i) | pocket(r.holo_dir, L.holo2apo)
    for ai in sorted(pk):
        hj = L.apo2holo(ai)
        if hj is None: continue
        a0c, a0c2, rnA = chis(A0, ai); afc, afc2, _ = chis(Af, ai)
        h0c, h0c2, rnH = chis(H0, hj); hfc, hfc2, _ = chis(Hf, hj)
        if rnA is None or rnH is None: continue
        rows.append(dict(ligand=r.ligand, parp=L.apo2parp(ai), res_apo=rnA, res_holo=rnH,
                         same_res=rnA == rnH,
                         d_chi1_final=angdiff(afc, hfc), d_chi1_start=angdiff(a0c, h0c),
                         d_chi2_final=angdiff(afc2, hfc2),
                         move_chi1_apo=angdiff(afc, a0c), move_chi1_holo=angdiff(hfc, h0c),
                         rot_apo=rotamer(afc), rot_holo=rotamer(hfc),
                         rot_apo0=rotamer(a0c), rot_holo0=rotamer(h0c)))
c = pd.DataFrame(rows)
c = c[c.same_res]                      # drop the A762V position
c.to_csv(f"{ROOT}/analysis/chi.csv", index=False)

def q(x): x = x.dropna(); return f"{x.median():.0f} [{x.quantile(.25):.0f}, {x.quantile(.75):.0f}]"
print("="*72); print(f"SIDE-CHAIN CHI ANALYSIS  ({len(c)} residue instances, "
                     f"{c.parp.nunique()} distinct PARP1 residues)"); print("="*72)
print(f"|dchi1| apo-start  vs holo-start  (deg)  {q(c.d_chi1_start)}")
print(f"|dchi1| apo-final  vs holo-final  (deg)  {q(c.d_chi1_final)}")
print(f"|dchi1| moved by GA, apo  run     (deg)  {q(c.move_chi1_apo)}")
print(f"|dchi1| moved by GA, holo run     (deg)  {q(c.move_chi1_holo)}")
print(f"|dchi2| apo-final  vs holo-final  (deg)  {q(c.d_chi2_final)}")
ok = c.dropna(subset=["rot_apo","rot_holo"])
print(f"\nchi1 rotamer-state agreement, final structures: "
      f"{(ok.rot_apo==ok.rot_holo).mean()*100:.1f}%  (n={len(ok)})")
ok0 = c.dropna(subset=["rot_apo0","rot_holo0"])
print(f"chi1 rotamer-state agreement, start structures: "
      f"{(ok0.rot_apo0==ok0.rot_holo0).mean()*100:.1f}%  (n={len(ok0)})")
print(f"|dchi1_final| <= 30 deg: {(c.d_chi1_final<=30).mean()*100:.0f}%   "
      f"<= 60 deg: {(c.d_chi1_final<=60).mean()*100:.0f}%")

print("\nper-residue, catalytic site (n>=20 observations):")
g = c.groupby("parp").agg(n=("ligand","size"), res=("res_apo","first"),
    dchi1_start=("d_chi1_start","median"), dchi1_final=("d_chi1_final","median"),
    move_apo=("move_chi1_apo","median"), move_holo=("move_chi1_holo","median"))
g["rot_agree"] = c.dropna(subset=["rot_apo","rot_holo"]).groupby("parp").apply(
    lambda x: (x.rot_apo == x.rot_holo).mean()*100, include_groups=False)
g = g[g.n >= 20].sort_values("n", ascending=False)
print(g.round(1).to_string())
