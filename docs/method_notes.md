# Method notes

## Pipeline

| Stage | Script | Output |
| --- | --- | --- |
| 1. Preparation, initial pose, GA | `src/drgscroll.py` | rotated receptors, Vina-GPU logs and complexes per generation/member, `clash_map_<run>.csv` |
| 2. Score table | `src/score_collection.py` | `docking_scores_<run>.csv` |
| 3. Final selection | `src/sidechain_clustering.py` | `top_docking_scores.csv`, `clusters/`, `clustered_files/`, `Best_complex/` |

## Default parameters

| Parameter | Value | Where set |
| --- | --- | --- |
| Flexible-residue shell | 4.0 Å around the initial ligand pose (Pro and Gly excluded) | `--shell`, `find_residues_around_ligand` |
| Disulfide Cys | a Cys whose sulfur has another sulfur within 2.05 Å is kept fixed; free Cys are rotated | `is_disulfide_cys` |
| Search box | 10 × 10 × 10 Å, centred on the AGFR pocket centre | `--box-size`, `generate_config` |
| Population size | 100 | `--population` |
| Generations | 50 | `--generations` |
| Initial χ angles | U(−180°, 180°); in generation 0 an angle that creates a clash is resampled until the residue is clash-free | `precompute_chi_angles`, GA loop |
| Fitness | lexicographic: clash count first, Vina-GPU score as tie-breaker | GA loop |
| Selection | top 25 % retained | GA loop |
| Crossover | per residue: the complete χ set of a residue is copied from one of two random parents | GA loop |
| Mutation | U(−1°, 1°) added to every χ angle of the next population (see note below) | GA loop |
| Clash cutoff | 1.75 Å between heavy atoms (hydrogens and backbone–backbone pairs ignored) | `check_clash`, `count_clashes_in_structure` |
| Neighbour search radius | 4.0 Å | same |
| Final selection | clash-free members of the last 10 generations; best 10 scores per generation; pairwise ligand heavy-atom RMSD (RDKit `GetBestRMS`); average-linkage hierarchical clustering into at most 10 clusters; best-scoring member of the most populated cluster | `sidechain_clustering.py` |

**Mutation note.** The mutation step is applied each time a child is added to the next
population, so members that enter the population early (including the retained top 25 %)
accumulate more ±1° steps than those added late. This is the behaviour used for the runs
reported in the paper and is kept for reproducibility.

**RMSD note.** `GetBestRMS` superimposes the two poses before computing the RMSD, so the
clustering groups poses by conformation and, within the small search box, by position.

**Seeds.** `drgscroll.py --seed N` makes the GA reproducible; without it the run is unseeded,
as in the paper. ADFR always uses `--seed 1` for the initial pose.

## Preparation

The receptor is prepared with PDBFixer, hydrogens added at pH 7.4, then converted with ADFR
`prepare_receptor`. Ligands are converted with Open Babel and prepared with ADFR
`prepare_ligand` in a single protonation and tautomeric state; protonation and tautomer states
are not enumerated, and ligand conformers are not pre-generated since Vina-GPU samples ligand
torsions internally. The ligand must carry the residue name `UNK`.

## Known limitations

- The flexible-residue shell is defined once from the initial pose and the search box is fixed,
  so results depend on that starting pose.
- The backbone is held fixed.
- χ angles are optimised against the docking score alone, with no rotamer prior or internal
  strain term.
- Scores obtained under different receptor conformations are not directly comparable.
