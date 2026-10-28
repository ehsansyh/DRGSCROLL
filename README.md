# DRGSCROLL

**D**urdağı **R**esearch **G**roup's **S**ide-**C**hain **RO**tations for **L**igand **L**anding — a docking
engine that optimises ligand pose and continuous receptor side-chain χ angles together inside a
genetic algorithm, built on Vina-GPU 2.1.

- Web server: https://drgscroll.bau.edu.tr
- Project page: https://www.drgscroll.com
- Archived data: https://doi.org/10.5281/zenodo.17733136
- Paper: *(citation — add on acceptance)*

---

## What this repository contains

```
src/                     DRGSCROLL engine (GA, score collection, final clustering)
analysis/
  interaction/           PLIP contact-fidelity analysis across docking engines
  rmsd/                  Ligand RMSD to the crystallographic reference
  md_chi/                Figures comparing GA-selected χ angles with MD occupancy
  apo_holo/              Apo- vs holo-started screening comparison
  diversity/             Chemical diversity of the screening libraries
data/                    Compound lists and per-complex result tables
docs/                    Method notes
```

The analyses added during revision (interaction fidelity, ligand RMSD, χ angles versus MD,
apo versus holo, library diversity) can be regenerated from `analysis/` given the
corresponding input files from Zenodo. Each script is standalone and documents its inputs in
its docstring.

---

## Installation

DRGSCROLL requires a CUDA-capable GPU.

```bash
git clone https://github.com/ehsansyh/DRGSCROLL.git
cd DRGSCROLL
conda env create -f environment.yml
conda activate drgscroll
```

External dependencies that must be installed separately:

| Tool | Used for | Notes |
| --- | --- | --- |
| Vina-GPU 2.1 | docking inside the GA loop | https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1 (its Boost libraries must be on `LD_LIBRARY_PATH`) |
| ADFR suite | receptor and ligand preparation, initial pose | `prepare_receptor`, `prepare_ligand`, `agfr` |
| Open Babel | ligand format conversion | |
| PDBFixer / OpenMM | adding hydrogens at pH 7.4 | |

The analysis scripts additionally need `pandas`, `numpy`, `scipy`, `matplotlib`, and for
specific modules `rdkit` and `umap-learn` (diversity).

---

## Running a docking job

Prepare a working directory containing `protein.pdb`, `ligand.pdb` (residue name `UNK`) and a
ligand PDBQT that defines the pocket for AGFR (default name `boxligand.pdbqt`). Paths to the
external tools are given on the command line or through environment variables.

```bash
export ADFR_BIN=/opt/ADFRsuite_x86_64Linux_1.0/bin
export VINA_GPU=/opt/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1/AutoDock-Vina-GPU-2-1
export VINA_GPU_OPENCL=/opt/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1

# 1) preparation, initial ADFR pose and 50 generations x 100 individuals of GA refinement
python src/drgscroll.py --workdir run1 --box-ligand boxligand.pdbqt

# 2) collect the docking score of every generation and individual
python src/score_collection.py run1

# 3) cluster the clash-free top-10 poses of the last 10 generations and pick the final complex
python src/sidechain_clustering.py run1          # result in run1/Best_complex/
```

Stage 1 prepares the receptor (PDBFixer, pH 7.4) and ligand, places an initial pose with ADFR,
and then runs the genetic algorithm: χ angles of the side chains within 4 Å of that pose are
initialised at random (resampling any angle that creates a steric clash), each individual is
docked with Vina-GPU 2.1, individuals are ranked by clash count and then docking score, and the
top 25 % seed the next generation through per-residue crossover and small random mutations.
Cys residues in a disulfide bond (S–S < 2.05 Å) are kept fixed. Stage 3 clusters the selected
poses by ligand heavy-atom RMSD and returns the best-scoring complex of the largest cluster.

Run `python src/drgscroll.py -h` for all options (generations, population size, box size,
random seed). See `docs/method_notes.md` for parameters and their defaults.

---

## Reproducing the analyses

### Interaction fidelity

```bash
# parse PLIP XML reports, correcting residue-numbering offsets between pipelines
python analysis/interaction/plip_parse.py xml_dir/ fidelity.csv

# or rebuild the contact table from an existing results file
python analysis/interaction/make_interaction_csv.py -x xml_dir/ -o interaction_keep.csv

# add missing (structure, engine) rows to an existing table / apply offsets to an existing table
python analysis/interaction/merge_csv.py -b interaction_keep.csv -x xml_dir/ -o interaction_merged.csv
python analysis/interaction/apply_shift.py -i interaction_merged.csv -o interaction_shift_corrected.csv

python analysis/interaction/fig_interaction_3panel.py -i interaction_keep.csv -o figs/
```

Glide and IFD renumber binding-site residues relative to the reference structure. Contacts
matched by residue number therefore fail for these engines unless the offset is corrected;
`plip_parse.py` and `apply_shift.py` detect the offset from the amino-acid identities of the
binding-site residue list and apply it only where the evidence is unambiguous. See
`docs/numbering_offsets.md`.

### Pose accuracy

```bash
python analysis/rmsd/fig_rmsd_en.py -r data/rmsd.tsv -s data/scores.tsv -o figs/
```

### χ angles versus MD

The figure scripts take the per-torsion comparison table `drgscroll_vs_md.csv` (fraction of
MD frames within ±30° of each DRGSCROLL-selected χ angle) as input.

```bash
python analysis/md_chi/chi_figures_en.py -i chi_results/drgscroll_vs_md.csv -o figs/
python analysis/md_chi/md_occupancy_figs.py -i chi_results/drgscroll_vs_md.csv -o figs/
```

180°-periodic torsions (Phe/Tyr χ2, Asp χ2, Glu χ3) must be folded when building this table;
omitting this produces systematic false disagreements.

### Apo versus holo

```bash
cd analysis/apo_holo && bash run_all.sh      # or run 01_*.py … 10_*.py in order
```

### Library diversity

```bash
python analysis/diversity/tanimoto_diversity.py \
    -i data/chembl_ids_RET.csv data/chembl_ids_PARP1.csv -l "RET TK" PARP1 -o figs/
python analysis/diversity/chemical_space_map.py \
    -c smiles_cache.csv -i data/chembl_ids_*.csv -l "RET TK" PARP1 -o figs/
```

---

## Data

`data/` holds the compound lists and the per-complex tables needed to regenerate the figures.
Structures, trajectories and docked poses are too large for version control and are deposited on
Zenodo: https://doi.org/10.5281/zenodo.17733136.

---

## Citation

```bibtex
@article{drgscroll,
  title   = {DRGSCROLL: Achieving Full Side-Chain Flexibility in Docking Simulations through Genetic Algorithm Framework},
  author  = {Sayyah, Ehsan and Ulug, Muhammet Eren and Tunc, Huseyin and Durdağı, Serdar},
  journal = {},
  year    = {},
  doi     = {}
}
```

## License

Released under the MIT License; see `LICENSE`.

## Contributing

Issues and pull requests are welcome. For bug reports please include the receptor and ligand
input, the command line used, and the DRGSCROLL version or commit hash.
