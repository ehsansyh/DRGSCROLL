"""Shared structure utilities: apo<->holo residue mapping, superposition, ligand RMSD."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign
RDLogger.DisableLog("rdApp.*")

# --- residue-numbering map (user-supplied, verified against sequence) ---
# file indices are 1..350 in BOTH apo and holo complex PDBs.
# holo idx <=122  -> apo idx = holo idx
# holo idx >=123  -> apo idx = holo idx + 3   (apo 784-786 absent in holo)
# true PARP1 number = apo idx + 661
def holo2apo(j):  return j if j <= 122 else j + 3
def apo2holo(i):
    if i <= 122: return i
    if i in (123, 124, 125): return None          # not present in holo
    return i - 3
def apo2parp(i):  return i + 661
def holo2parp(j): return apo2parp(holo2apo(j))

PAIRS = [(holo2apo(j), j) for j in range(1, 351)]   # (apo_idx, holo_idx)

def read_pdb(path):
    """-> (prot: dict[(resi,atomname)] = xyz, lig_block: str)"""
    prot, lig = {}, []
    for l in open(path, errors="ignore"):
        rec = l[:6]
        if rec not in ("ATOM  ", "HETATM"): continue
        resi = int(l[22:26]); name = l[12:16].strip()
        xyz = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
        if l[17:20].strip() == "UNK" or resi == 900: lig.append(l)
        else: prot[(resi, name)] = xyz
    return prot, "".join(lig) + "END\n"

def kabsch(P, Q):
    """Rotation+translation mapping P onto Q."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cq - R @ cp

def super_apo_on_holo(apo_prot, holo_prot, atoms=("CA",), pocket=None):
    """Least-squares fit of apo protein onto holo protein over mapped residues.
    pocket: optional set of apo residue indices to restrict the fit to."""
    P, Q = [], []
    for ai, hj in PAIRS:
        if pocket is not None and ai not in pocket: continue
        for an in atoms:
            if (ai, an) in apo_prot and (hj, an) in holo_prot:
                P.append(apo_prot[(ai, an)]); Q.append(holo_prot[(hj, an)])
    P, Q = np.array(P), np.array(Q)
    R, t = kabsch(P, Q)
    rmsd = np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean())
    return R, t, rmsd, len(P)

def lig_mol(block, R=None, t=None):
    """Ligand PDB block -> heavy-atom RDKit mol, optionally transformed."""
    m = Chem.MolFromPDBBlock(block, removeHs=True, sanitize=False)
    if m is None: return None
    try: Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^
                             Chem.SanitizeFlags.SANITIZE_PROPERTIES ^
                             Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception: pass
    m = Chem.RemoveHs(m, sanitize=False)
    Chem.FastFindRings(m)
    if R is not None:
        conf = m.GetConformer()
        for i in range(m.GetNumAtoms()):
            p = np.array(conf.GetAtomPosition(i))
            q = R @ p + t
            conf.SetAtomPosition(i, q.tolist())
    return m

def sym_rmsd(probe, ref):
    """Symmetry-corrected RMSD in place (no further alignment)."""
    try: return rdMolAlign.CalcRMS(probe, ref)
    except Exception: return np.nan
