"""Step 1: per-ligand score inventory (apo vs holo)."""
import os, re, glob, pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_COLS = [f"Gen {g}" for g in range(40, 50)]

def ligdirs(sub, pat):
    d = {}
    for p in glob.glob(os.path.join(ROOT, sub, pat)):
        m = re.match(r"(CHEMBL\d+)", os.path.basename(p))
        if m: d[m.group(1)] = p
    return d

def parse_selected(path):
    """Return DataFrame: file, gen, pop, score (Vina kcal/mol)."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        sc = [r[c] for c in GEN_COLS if c in df.columns and pd.notna(r[c])]
        sc = [float(x) for x in sc if str(x).strip() != ""]
        if not sc: continue
        rows.append(dict(file=r["Ligand"], cluster=r.get("Cluster"),
                         gen=int(r["Generation"]), pop=int(r["Population_x"]),
                         score=min(sc)))
    return pd.DataFrame(rows)

def parse_dlg(path):
    """ADFR initial docking: best affinity + mode table + cluster sizes."""
    txt = open(path, errors="ignore").read()
    modes = re.findall(r"^\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)",
                       txt, re.M)
    if not modes: return dict(adfr_best=np.nan, adfr_nmodes=np.nan, adfr_top_clustsize=np.nan)
    aff = [float(m[1]) for m in modes]
    return dict(adfr_best=min(aff), adfr_nmodes=len(modes),
                adfr_top_clustsize=int(modes[0][4]))

apo = ligdirs("analyze_apo", "CHEMBL*_analyze_apo")
holo = ligdirs("holo_analyze", "CHEMBL*_analyze")
common = sorted(set(apo) & set(holo))
print(f"paired ligands: {len(common)}")

recs = []
for lig in common:
    row = dict(ligand=lig)
    for tag, dd in (("apo", apo), ("holo", holo)):
        d = dd[lig]
        row[f"{tag}_dir"] = d
        sel = parse_selected(os.path.join(d, "selected_ligands.csv"))
        row[f"{tag}_n_reps"] = len(glob.glob(os.path.join(d, "complex_rotated_*_complex.pdb")))
        row[f"{tag}_n_sel"] = len(sel)
        row[f"{tag}_best"] = sel.score.min()
        row[f"{tag}_mean"] = sel.score.mean()
        row[f"{tag}_sd"] = sel.score.std()
        row[f"{tag}_nclust"] = sel.cluster.nunique() if "cluster" in sel else np.nan
        best = sel.loc[sel.score.idxmin()]
        row[f"{tag}_best_tag"] = f"{int(best['gen'])}_{int(best['pop'])}"
        row[f"{tag}_best_gen"] = int(best["gen"])
        # per-generation best, for convergence curves
        for g in range(40, 50):
            sub = sel[sel.gen == g]
            row[f"{tag}_g{g}"] = sub.score.min() if len(sub) else np.nan
        row.update({f"{tag}_{k}": v for k, v in
                    parse_dlg(os.path.join(d, "ligand_rigid_summary.dlg")).items()})
    recs.append(row)

df = pd.DataFrame(recs)
act = pd.read_csv(os.path.join(ROOT, "PARP1_aktivity.csv")).rename(
    columns={"Name": "ligand", "Vina GPU 2.1": "vina_rigid", "DRGSCROLL": "drgscroll_paper"})
df = df.merge(act, on="ligand", how="left")
df["label"] = df.holo_dir.str.extract(r"CHEMBL\d+_(active|inactive|moderate)_")[0]
df.to_csv(os.path.join(ROOT, "analysis", "scores.csv"), index=False)

print(df[["apo_best","holo_best","drgscroll_paper","vina_rigid","pIC50"]].describe().round(2))
print("\nholo_best vs paper DRGSCROLL: r =",
      round(df.holo_best.corr(df.drgscroll_paper), 4),
      "| max |diff| =", round((df.holo_best - df.drgscroll_paper).abs().max(), 2))
print("labels:", df.label.value_counts(dropna=False).to_dict())
