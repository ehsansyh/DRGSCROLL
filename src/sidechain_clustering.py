#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 of the DRGSCROLL pipeline: select the final protein-ligand complex.

For each run directory (after drgscroll.py and score_collection.py):
  1. take the clash-free population members of the last N generations (default 10)
     and keep the 10 best-scoring of each generation;
  2. compute the pairwise ligand heavy-atom RMSD of these poses (RDKit GetBestRMS);
  3. cluster them by average-linkage hierarchical clustering into at most 10 clusters;
  4. copy the best-scoring complex of the most populated cluster to <run>/Best_complex/.

Inputs  <run>/clash_map_<run>.csv, <run>/docking_scores_<run>.csv,
        complex_rotated_{g}_{p}_complex.pdb, ligand_complex_rotated_{g}_{p}_out.pdbqt
Outputs <run>/top_docking_scores.csv, <run>/clusters/, <run>/clustered_files/,
        <run>/Best_complex/

Examples:
    python src/sidechain_clustering.py run1
    python src/sidechain_clustering.py --root all_runs/      # every sub-directory
"""
import argparse
import glob
import os
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from scipy.cluster.hierarchy import fcluster


def top10_gen(cluster_folder, folder, last_n=10, top_k=10):
    clash_map_file = cluster_folder + f"/clash_map_{folder}.csv"
    docking_scores_file = cluster_folder + f"/docking_scores_{folder}.csv"
    output_csv_file = cluster_folder + "/top_docking_scores.csv"
    clusters_folder = cluster_folder + "/clusters/"
    os.makedirs(clusters_folder, exist_ok=True)

    clash_map_df = pd.read_csv(clash_map_file)
    # first data row holds the generation numbers (header written by DataFrame.unstack)
    clash_map_df.columns = ["Population"] + list(clash_map_df.iloc[0, 1:].astype(int))
    clash_map_df = clash_map_df.iloc[2:].copy()
    clash_map_df["Population"] = clash_map_df["Population"].astype(int)
    clash_map_df.columns = clash_map_df.columns.astype(str).str.strip()

    all_gens = sorted(int(c) for c in clash_map_df.columns if c != "Population")
    gens = all_gens[-last_n:]
    gen_columns = [str(i) for i in gens]
    clash_map_df[gen_columns] = clash_map_df[gen_columns].apply(pd.to_numeric, errors="coerce")

    # keep only clash-free entries
    filtered_df = clash_map_df.copy()
    for col in gen_columns:
        filtered_df[col] = filtered_df[col].where(filtered_df[col] == 0.0, other=float("nan"))
    filtered_df = filtered_df[["Population"] + gen_columns]
    filtered_df.to_csv(cluster_folder + f"/filtered_clash_map_{folder}.csv", index=False)

    filtered_rows = filtered_df[filtered_df[gen_columns].eq(0).any(axis=1)]
    populations = filtered_rows["Population"].tolist()
    docking_scores_df = pd.read_csv(docking_scores_file)

    top_scores_list = []
    for gen in gens:
        gen_col = f"Gen {gen}"
        gen_data = docking_scores_df[docking_scores_df["Pop"].isin(populations)][["Pop", gen_col]]
        gen_clash = filtered_rows[filtered_rows["Population"].isin(populations)][["Population", str(gen)]]
        merged_data = gen_data.merge(gen_clash, left_on="Pop", right_on="Population")
        filtered_data = merged_data[merged_data[str(gen)] == 0.0]
        selected_rows = filtered_data.nsmallest(top_k, gen_col)
        top_scores_list.append(selected_rows.assign(Generation=gen))

        for _, row in selected_rows.iterrows():
            pop = int(row["Pop"])
            pdb_filename = f"complex_rotated_{gen}_{pop}_complex.pdb"
            ligand_filename = f"ligand_complex_rotated_{gen}_{pop}_out.pdbqt"
            pdb_path = os.path.join(cluster_folder, pdb_filename)
            if os.path.exists(pdb_path):
                shutil.copy(pdb_path, os.path.join(clusters_folder, pdb_filename))
                shutil.copy(os.path.join(cluster_folder, ligand_filename),
                            os.path.join(clusters_folder, ligand_filename))

    top_scores_df = pd.concat(top_scores_list, ignore_index=True)
    top_scores_df.to_csv(output_csv_file, index=False)


def extract_first_ligand_conformation(ligand_out_file):
    ligand_data = []
    capture = False
    with open(ligand_out_file, "r") as file:
        for line in file:
            if "MODEL 1" in line:
                capture = True
                ligand_data.append(line)
            elif "ENDMDL" in line and capture:
                break
            elif capture:
                ligand_data.append(line)
    return ligand_data


def load_ligands_from_sdf(source_folder):
    ligands = {}
    for sdf_file in glob.glob(os.path.join(source_folder, "*.sdf")):
        suppl = Chem.SDMolSupplier(sdf_file, sanitize=False)
        if len(suppl) > 0 and suppl[0] is not None:
            ligands[os.path.basename(sdf_file)] = suppl[0]
    return ligands


def compute_rmsd_matrix(ligands):
    """Pairwise heavy-atom RMSD (RDKit GetBestRMS, symmetry-aware, after optimal alignment)."""
    ligand_names = sorted(ligands.keys())
    n = len(ligand_names)
    rmsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mol1, mol2 = ligands[ligand_names[i]], ligands[ligand_names[j]]
            if mol1.GetNumHeavyAtoms() != mol2.GetNumHeavyAtoms():
                rmsd_matrix[i, j] = rmsd_matrix[j, i] = np.nan
                continue
            try:
                rmsd_matrix[i, j] = rmsd_matrix[j, i] = rdMolAlign.GetBestRMS(mol1, mol2)
            except RuntimeError:
                rmsd_matrix[i, j] = rmsd_matrix[j, i] = np.nan
    rmsd_matrix[np.isnan(rmsd_matrix)] = 10  # failed pairs are treated as far apart
    return rmsd_matrix, ligand_names


def hierarchical_clustering(rmsd_matrix, ligand_names, output_csv, n_clusters=10, plot_path=None):
    condensed_matrix = rmsd_matrix[np.triu_indices(len(ligand_names), k=1)]
    linkage_matrix = sch.linkage(condensed_matrix, method="average")
    clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    if plot_path:
        plt.figure(figsize=(12, 6))
        sch.dendrogram(linkage_matrix, labels=np.array(ligand_names), leaf_rotation=90, color_threshold=10)
        plt.ylabel("RMSD")
        plt.title("Hierarchical clustering of ligand poses")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    cluster_data = np.column_stack((ligand_names, clusters.astype(str)))
    np.savetxt(output_csv, cluster_data, fmt="%s", delimiter=",", header="Ligand,Cluster", comments="")
    return clusters


def copy_clustered_files(ligand_names, cluster_labels, source_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for ligand, cluster_id in zip(ligand_names, cluster_labels):
        gener, popul = ligand.split("_")[3], ligand.split("_")[4]
        comp_name = f"complex_rotated_{gener}_{popul}_complex.pdb"
        cluster_dir = os.path.join(output_folder, f"Cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        shutil.copy(os.path.join(source_folder, "clusters", ligand), os.path.join(cluster_dir, ligand))
        shutil.copy(os.path.join(source_folder, "clusters", comp_name), os.path.join(cluster_dir, comp_name))


def select_best_pose(cluster_folder):
    """Copy the best-scoring complex of the most populated cluster to Best_complex/."""
    cluster_file = os.path.join(cluster_folder, "clusters", "ligand_clusters.csv")
    if not os.path.exists(cluster_file):
        print(f"Cluster file not found: {cluster_file}")
        return

    cluster_df = pd.read_csv(cluster_file, header=None, names=["Ligand", "Cluster"], skiprows=1)
    top_cluster = cluster_df["Cluster"].value_counts().idxmax()
    top_cluster_ligands = cluster_df[cluster_df["Cluster"] == top_cluster].copy()
    top_cluster_ligands["Generation"] = top_cluster_ligands["Ligand"].str.extract(r"_(\d+)_")[0].astype(int)
    top_cluster_ligands["Population"] = top_cluster_ligands["Ligand"].str.extract(r"_(\d+)_out")[0].astype(int)

    docking_scores_file = os.path.join(cluster_folder, "top_docking_scores.csv")
    if not os.path.exists(docking_scores_file):
        print(f"Docking scores file not found: {docking_scores_file}")
        return
    docking_scores_df = pd.read_csv(docking_scores_file, delimiter=",")

    merged_df = top_cluster_ligands.merge(
        docking_scores_df, how="left", left_on=["Generation", "Population"], right_on=["Generation", "Pop"])
    score_cols = [c for c in merged_df.columns if c.startswith("Gen ")]
    row_scores = merged_df[score_cols].min(axis=1)
    min_score = row_scores.min()
    best_ligands = merged_df.loc[row_scores == min_score, "Ligand"]

    output_folder = os.path.join(cluster_folder, "Best_complex")
    os.makedirs(output_folder, exist_ok=True)
    for ligand in best_ligands:
        bgen, bpop = ligand.split("_")[3], ligand.split("_")[4]
        best_complex = f"complex_rotated_{bgen}_{bpop}_complex.pdb"
        source_path = os.path.join(cluster_folder, best_complex)
        if os.path.exists(source_path):
            shutil.copy(source_path, os.path.join(output_folder, best_complex))
        else:
            print(f"File not found: {source_path}")
    merged_df.to_csv(os.path.join(output_folder, "selected_ligands.csv"), index=False)


def process_run(cluster_folder, last_n=10, top_k=10, n_clusters=10):
    cluster_folder = os.path.abspath(cluster_folder)
    folder = os.path.basename(os.path.normpath(cluster_folder))
    print(f"Processing: {cluster_folder}")
    top10_gen(cluster_folder, folder, last_n, top_k)

    clusters_dir = os.path.join(cluster_folder, "clusters")
    for lig in [f for f in os.listdir(clusters_dir) if f.startswith("ligand_") and f.endswith(".pdbqt")]:
        path = os.path.join(clusters_dir, lig)
        top_lig = extract_first_ligand_conformation(path)
        with open(path, "w") as top:
            top.writelines(top_lig)
        sdf = os.path.join(clusters_dir, lig.split(".")[0] + ".sdf")
        subprocess.run(["obabel", path, "-O", sdf], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ligands = load_ligands_from_sdf(clusters_dir)
    rmsd_matrix, ligand_names = compute_rmsd_matrix(ligands)
    np.savetxt(os.path.join(clusters_dir, "rmsd_values.csv"), rmsd_matrix, delimiter=",", fmt="%.3f")
    cluster_labels = hierarchical_clustering(
        rmsd_matrix, ligand_names, os.path.join(clusters_dir, "ligand_clusters.csv"),
        n_clusters=n_clusters, plot_path=os.path.join(clusters_dir, "dendrogram.png"))
    copy_clustered_files(ligand_names, cluster_labels, cluster_folder,
                         os.path.join(cluster_folder, "clustered_files"))
    select_best_pose(cluster_folder)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="DRGSCROLL run directories")
    ap.add_argument("--root", help="process every sub-directory of this folder")
    ap.add_argument("--last-generations", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=10, help="best clash-free poses kept per generation")
    ap.add_argument("--clusters", type=int, default=10, help="maximum number of clusters")
    args = ap.parse_args(argv)

    runs = list(args.runs)
    if args.root:
        runs += [os.path.join(args.root, f) for f in sorted(os.listdir(args.root))
                 if os.path.isdir(os.path.join(args.root, f))]
    if not runs:
        ap.error("give at least one run directory or --root")
    for r in runs:
        process_run(r, args.last_generations, args.top_k, args.clusters)
    print("Processing complete.")


if __name__ == "__main__":
    main()
