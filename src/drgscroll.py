#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRGSCROLL - genetic-algorithm optimisation of receptor side-chain chi angles
coupled to Vina-GPU 2.1 docking.

Stage 1 of the pipeline. Run it inside a working directory that contains

    protein.pdb        receptor (any protonation; hydrogens are added at pH 7.4)
    ligand.pdb         ligand to dock (residue name UNK)
    <box ligand>       ligand PDBQT used by AGFR to define the pocket
                       (default: boxligand.pdbqt)

It writes, for every generation g and population member p,
complex_rotated_{g}_{p}.pdb / .pdbqt, the Vina-GPU logs
ligand_complex_rotated_{g}_{p}.log and the docked complexes
complex_rotated_{g}_{p}_complex.pdb, plus clash_map_<workdir>.csv.

Then run score_collection.py (stage 2) and sidechain_clustering.py (stage 3)
on the same directory to select the final complex.

Example:
    python src/drgscroll.py --workdir run1 \
        --adfr-bin /opt/ADFRsuite/bin \
        --vina-gpu /opt/Vina-GPU-2.1/AutoDock-Vina-GPU-2-1 \
        --opencl /opt/Vina-GPU-2.1
"""

import argparse
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from tqdm import tqdm

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
CLASH_CUTOFF = 1.75        # heavy-atom distance (A) below which a pair clashes
NEIGHBOUR_RADIUS = 4.0     # neighbour search radius (A) for the clash test
DISULFIDE_CUTOFF = 2.05    # S-S distance (A) below which a Cys is treated as bonded

# file names used inside the working directory
protein_file = "protein.pdbqt"
ligand_file = "ligand.pdbqt"
binding_pocket = "binding_pocket"
ligand_out_file = "ligand_rigid_out.pdbqt"
complex_file_pdbqt = "complex1.pdbqt"
complex_file_pdb = "complex1.pdb"


# ----------------------------------------------------------------------------
# preparation and initial pose
# ----------------------------------------------------------------------------
def run(cmd, quiet=False):
    """Run a shell command, report failures without aborting (as in the original)."""
    try:
        subprocess.run(" ".join(cmd), shell=True, check=True,
                       stdout=subprocess.PIPE if quiet else None,
                       stderr=subprocess.PIPE if quiet else None)
    except subprocess.CalledProcessError as e:
        print("command failed:", " ".join(cmd), "\n", e)


def preparation(input_pdb):
    fixer = PDBFixer(input_pdb)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(input_pdb, "w"))


def prepare_and_dock_initial(adfr_bin, box_ligand_file):
    """PDBFixer -> prepare_receptor/prepare_ligand -> AGFR pocket -> ADFR initial pose."""
    preparation("protein.pdb")
    run([f'"{adfr_bin / "prepare_receptor"}"', "-r", '"protein.pdb"', "-o", f'"{protein_file}"',
         "-A", "checkhydrogens"])
    run([f'"{adfr_bin / "prepare_ligand"}"', "-l", "ligand.pdb", "-o", ligand_file])
    run([f'"{adfr_bin / "agfr"}"', "-r", f'"{protein_file}"', "-l", f'"{box_ligand_file}"',
         "-o", f'"{binding_pocket}"'])
    run([f'"{adfr_bin / "adfr"}"', "-l", f'"{ligand_file}"', "-t", f'"{binding_pocket}.trg"',
         "--jobName", "rigid", "--nbRuns", "8", "--maxEvals", "20000", "-O", "--seed", "1"])
    best_ligand_data = extract_first_ligand_conformation(ligand_out_file)
    create_complex_file(protein_file, best_ligand_data, complex_file_pdbqt)
    convert_pdbqt_to_pdb(complex_file_pdbqt, complex_file_pdb)


def extract_first_ligand_conformation(ligand_out_file):
    """Return the lines of MODEL 1 of a multi-model PDBQT file."""
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


def create_complex_file(protein_file, ligand_data, output_file):
    """Concatenate receptor and ligand PDBQT records into one file."""
    try:
        with open(protein_file, "r") as protein, open(output_file, "w") as complex_file:
            for line in protein:
                complex_file.write(line)
            for line in ligand_data:
                complex_file.write(line)
    except Exception as e:
        print(f"could not write {output_file}: {e}")


def convert_pdbqt_to_pdb(input_file, output_file):
    subprocess.run(["obabel", input_file, "-O", output_file],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def generate_config(opencl, box_size, threads):
    """Write the Vina-GPU config centred on the AGFR pocket centre."""
    with open("binding_pocket.log", "r") as grid:
        center = grid.read().split("\n")[4].split()
        x, y, z = center[2], center[3], center[4]
    with open("config.txt", "w") as conf:
        conf.write(f"receptor = ./protein.pdbqt\nligand = ./ligand.pdbqt\n"
                   f"opencl_binary_path = {opencl}\n"
                   f"center_x = {x}\ncenter_y = {y}\ncenter_z = {z}\n"
                   f"size_x = {box_size}\nsize_y = {box_size}\nsize_z = {box_size}\n"
                   f"thread = {threads}")


# ----------------------------------------------------------------------------
# structure helpers
# ----------------------------------------------------------------------------
def ext_lig(chain):
    ligid, ligid_mol2 = [], []
    for res in list(chain.get_residues()):
        if is_aa(res) == False and res.get_id()[0] != "W":
            ligid.append(f"{res.get_resname()}")
            ligid_mol2.append(f"{res.get_resname()} {res.get_id()[1]}")
    return ligid, ligid_mol2


def find_residues_around_ligand(chain, radius=4.0):
    """Residues (excluding Pro and Gly) with any atom within `radius` of the ligand (UNK)."""
    ligand_atoms = [atom for atom in chain.get_atoms() if atom.get_parent().resname == "UNK"]
    if not ligand_atoms:
        return []
    ns = NeighborSearch(list(chain.get_atoms()))
    close_residues = set()
    for atom in ligand_atoms:
        for neighbor in ns.search(atom.coord, radius):
            residue = neighbor.get_parent()
            if residue.get_resname() not in ("PRO", "GLY") and is_aa(residue):
                close_residues.add(residue)
    return close_residues


def is_disulfide_cys(residue, ns):
    """True if this Cys SG has another residue's sulfur within DISULFIDE_CUTOFF."""
    sulfur = [a for a in residue.get_atoms() if a.element == "S"]
    if not sulfur:
        return False
    return any(n.element == "S" and n.get_parent() is not residue
               for n in ns.search(sulfur[0].coord, DISULFIDE_CUTOFF))


class ProteinSelect(Select):
    """Save only protein residues."""
    def __init__(self, ligid):
        self.ligid = ligid

    def accept_residue(self, residue):
        return residue.get_resname() not in [self.ligid[0], "HET", "WAT"]


def get_updated_residue(residue, chain):
    for updated_residue in chain.get_residues():
        if (updated_residue.get_resname() == residue.get_resname()
                and updated_residue.get_id() == residue.get_id()):
            return updated_residue
    return None


def is_backbone_atom(atom):
    return atom.get_name().strip() in BACKBONE_ATOMS


def check_clash(residue, ns, distance_threshold=NEIGHBOUR_RADIUS):
    """True if any heavy atom of `residue` is closer than CLASH_CUTOFF to another residue."""
    for atom in residue.get_atoms():
        for neighbor in ns.search(atom.coord, distance_threshold):
            if neighbor.get_parent() == atom.get_parent():
                continue
            if is_backbone_atom(atom) and is_backbone_atom(neighbor):
                continue
            if atom.element == "H" or neighbor.element == "H":
                continue
            if np.linalg.norm(atom.coord - neighbor.coord) < CLASH_CUTOFF:
                return True
    return False


def count_clashes_in_structure(structure, distance_threshold=NEIGHBOUR_RADIUS):
    """Number of clashing heavy-atom pairs (each pair counted twice)."""
    all_atoms = list(structure.get_atoms())
    ns = NeighborSearch(all_atoms)
    total_clashes = 0
    for atom in all_atoms:
        for neighbor in ns.search(atom.coord, distance_threshold):
            if neighbor.get_parent() == atom.get_parent():
                continue
            if is_backbone_atom(atom) and is_backbone_atom(neighbor):
                continue
            if atom.element == "H" or neighbor.element == "H":
                continue
            if np.linalg.norm(atom.coord - neighbor.coord) < CLASH_CUTOFF:
                total_clashes += 1
    return total_clashes


def chi_keys(residue):
    keys, chi_idx = [], 1
    while True:
        angle_key = f"chi{chi_idx}"
        if residue.internal_coord.get_angle(angle_key) is None:
            return keys
        keys.append(angle_key)
        chi_idx += 1


def precompute_chi_angles(residues, population_size):
    """Random initial chi angles, U(-180, 180), for every individual."""
    population_chi_angles = []
    for _ in range(population_size):
        chi_angle_dict = {}
        for residue in residues:
            chi_angles = {k: random.uniform(-180, 180) for k in chi_keys(residue)}
            if chi_angles:
                chi_angle_dict[residue] = chi_angles
        population_chi_angles.append(chi_angle_dict)
    return population_chi_angles


def parse_gen_pop_from_filename(file_name):
    base = file_name.replace(".pdb", "").split("_")
    return int(base[2]), int(base[3])


# ----------------------------------------------------------------------------
# docking
# ----------------------------------------------------------------------------
def pro_prep(file_name, prepare_receptor):
    file_name = file_name.split(".pdb")[0]
    run([f'"{prepare_receptor}"', "-r", f'"{file_name}.pdb"', "-o", f'"{file_name}.pdbqt"',
         "-A", "checkhydrogens"], quiet=True)


def vina_GPU(file_name, vina_gpu):
    """Dock the ligand into one rotated receptor; return (score, pose file)."""
    file_name = file_name.split(".pdbqt")[0]
    dock_out_name = f"ligand_{file_name}"
    with open("config.txt", "r") as conf:
        totallines = conf.readlines()
    totallines[0] = f"receptor = ./{file_name}.pdbqt\n"
    with open(f"{file_name}_conf.txt", "w") as nconf:
        nconf.write("".join(totallines))
    os.system(f'"{vina_gpu}" --config {file_name}_conf.txt --out {dock_out_name}_out.pdbqt '
              f'--log {dock_out_name}.log > /dev/null 2>&1')
    with open(f"{dock_out_name}.log", "r") as file:
        data = file.read().split("\n")
    lines = [i for i, line in enumerate(data) if "kcal/mol" in line]
    score = data[lines[0] + 2].split()[1]
    best_pose = f"{dock_out_name}_out.pdbqt"
    best_ligand = extract_first_ligand_conformation(best_pose)
    create_complex_file(f"{file_name}.pdbqt", best_ligand, f"{file_name}_complex.pdbqt")
    convert_pdbqt_to_pdb(f"{file_name}_complex.pdbqt", f"{file_name}_complex.pdb")
    return score, best_pose


# ----------------------------------------------------------------------------
# genetic algorithm
# ----------------------------------------------------------------------------
def genetic_algorithm(pdb_file, workdir_name, prepare_receptor, vina_gpu,
                      generations=50, population_size=100, shell=4.0):
    population_map = {}
    clash_map = {}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex_structure", pdb_file)
    chain = structure[0]
    ns = NeighborSearch(list(chain.get_atoms()))
    chain.atom_to_internal_coordinates(verbose=True)

    residues = find_residues_around_ligand(chain, shell)
    # Cys residues whose sulfur lies within 2.05 A of another sulfur (disulfide) are
    # kept fixed; free Cys residues are rotated like any other residue.
    residues = [r for r in residues
                if not (r.get_resname() == "CYS" and is_disulfide_cys(r, ns))]
    population = precompute_chi_angles(residues, population_size)

    for generation in range(generations):
        pop = 0
        with tqdm(total=3, desc=f"Generation {generation + 1}", unit="step") as generation_bar:
            # 1) apply the chi angles of every individual and save the receptor
            with tqdm(total=population_size, desc="Rotating individuals", unit="individual",
                      position=1) as rotation_bar:
                for p in range(population_size):
                    pop += 1
                    individuals = []
                    for residue in residues:
                        keys = [(k, population[p][residue][k]) for k in chi_keys(residue)
                                if residue in population[p] and k in population[p][residue]]
                        residue_angles = {}
                        if generation == 0:
                            # first generation: resample any angle that creates a clash
                            for angle_key, new_angle in keys:
                                residue.internal_coord.set_angle(angle_key, new_angle)
                                while True:
                                    chain.internal_to_atom_coordinates()
                                    ns_1 = NeighborSearch(list(chain.get_atoms()))
                                    updated_residue = get_updated_residue(residue, chain)
                                    if check_clash(updated_residue, ns_1) == False:
                                        residue_angles[angle_key] = new_angle
                                        break
                                    new_angle = random.uniform(-180, 180)
                                    residue.internal_coord.set_angle(angle_key, new_angle)
                        else:
                            for angle_key, new_angle in keys:
                                residue.internal_coord.set_angle(angle_key, new_angle)
                                chain.internal_to_atom_coordinates()
                                residue_angles[angle_key] = new_angle
                        if residue_angles:
                            individuals.append({residue: residue_angles})

                    out_file_name = f"complex_rotated_{generation}_{pop}.pdb"
                    ligid, _ = ext_lig(chain)
                    io = PDBIO()
                    io.set_structure(chain)
                    io.save(out_file_name, select=ProteinSelect(ligid))
                    no_clash = count_clashes_in_structure(chain)
                    population_map[(generation, pop)] = individuals
                    clash_map[(generation, pop)] = no_clash
                    rotation_bar.update(1)
            generation_bar.update(1)

            # 2) receptor preparation
            protein_files = [f for f in os.listdir()
                             if f.startswith(f"complex_rotated_{generation}_") and f.endswith(".pdb")
                             and not f.endswith("_complex.pdb")]
            with ThreadPoolExecutor(max_workers=25) as executor:
                list(executor.map(lambda fn: pro_prep(fn, prepare_receptor), protein_files))
            generation_bar.update(1)

            # 3) docking
            docking_scores_dict = {}
            docking_files = [f for f in os.listdir()
                             if f.startswith(f"complex_rotated_{generation}_") and f.endswith(".pdbqt")
                             and not f.endswith("_complex.pdbqt")]
            for file_name in tqdm(docking_files, desc="Docking", unit="file", position=1):
                g, p = parse_gen_pop_from_filename(file_name.replace("pdbqt", "pdb"))
                docking_scores_dict[(g, p)] = vina_GPU(file_name, vina_gpu)[0]
            generation_bar.update(1)

        # fitness: clash count first, docking score as tie-breaker
        fitness_scores = []
        for p in range(1, population_size + 1):
            combined_individual = {}
            for d in population_map.get((generation, p), []):
                for residue, angles in d.items():
                    combined_individual.setdefault(residue, {}).update(angles)
            clash_number = clash_map.get((generation, p))
            docking_score = docking_scores_dict.get((generation, p))
            score = float(docking_score) if docking_score is not None else float("inf")
            fitness_scores.append((combined_individual, score, clash_number))

        fitness_scores.sort(key=lambda x: (x[2], x[1]))
        next_population = [ind for ind, _, _ in fitness_scores[:population_size // 4]]

        while len(next_population) < population_size:
            # per-residue crossover between two random parents of the retained pool
            parent1, parent2 = random.sample(next_population, 2)
            child = {}
            for residue in set(parent1) | set(parent2):
                if residue in parent1 and residue in parent2:
                    chosen_parent = random.choice([parent1, parent2])
                elif residue in parent1:
                    chosen_parent = parent1
                else:
                    chosen_parent = parent2
                child[residue] = dict(chosen_parent[residue])
            next_population.append(child)

            # mutation: U(-1, 1) degree increment on every chi angle of the next population.
            # Note: this is applied each time a child is added (as in the runs reported in
            # the paper), so earlier members accumulate more perturbation steps.
            for individual in next_population:
                for residue, angles in individual.items():
                    for angle_key in angles:
                        angles[angle_key] += random.uniform(-1, 1)

            population = next_population

    clash_df = pd.DataFrame.from_dict(clash_map, orient="index", columns=["Clash_Count"])
    clash_df.index = pd.MultiIndex.from_tuples(clash_df.index, names=["Generation", "Population"])
    clash_df = clash_df.unstack(level=0)
    clash_df.to_csv(f"clash_map_{workdir_name}.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, type=Path,
                    help="directory containing protein.pdb, ligand.pdb and the box ligand")
    ap.add_argument("--box-ligand", default="boxligand.pdbqt",
                    help="ligand PDBQT used by AGFR to place the pocket (default: boxligand.pdbqt)")
    ap.add_argument("--adfr-bin", type=Path, default=os.environ.get("ADFR_BIN"),
                    help="ADFR suite bin directory (or set $ADFR_BIN)")
    ap.add_argument("--vina-gpu", default=os.environ.get("VINA_GPU"),
                    help="Vina-GPU 2.1 executable (or set $VINA_GPU)")
    ap.add_argument("--opencl", default=os.environ.get("VINA_GPU_OPENCL"),
                    help="Vina-GPU opencl_binary_path (or set $VINA_GPU_OPENCL)")
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--population", type=int, default=100)
    ap.add_argument("--shell", type=float, default=4.0, help="flexible-residue shell (A)")
    ap.add_argument("--box-size", type=float, default=10, help="Vina-GPU box edge (A)")
    ap.add_argument("--threads", type=int, default=8000, help="Vina-GPU thread setting")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default: unseeded)")
    args = ap.parse_args(argv)

    for name in ("adfr_bin", "vina_gpu", "opencl"):
        if getattr(args, name) is None:
            ap.error(f"--{name.replace('_', '-')} is required (or set the environment variable)")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    workdir = args.workdir.resolve()
    os.chdir(workdir)
    start_time = time.time()
    prepare_and_dock_initial(Path(args.adfr_bin), args.box_ligand)
    generate_config(args.opencl, args.box_size, args.threads)
    genetic_algorithm(complex_file_pdb, workdir.name, Path(args.adfr_bin) / "prepare_receptor",
                      args.vina_gpu, generations=args.generations,
                      population_size=args.population, shell=args.shell)
    print(f"Duration: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    sys.exit(main())
