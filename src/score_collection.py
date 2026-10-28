#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 of the DRGSCROLL pipeline: collect the Vina-GPU docking scores of every
generation and population member of a run into one table.

Reads  <workdir>/ligand_complex_rotated_{gen}_{pop}.log
Writes <workdir>/docking_scores_<workdir name>.csv
       (rows: population member; columns: "Gen 0" ... "Gen N-1")

Example:
    python src/score_collection.py run1
"""
import argparse
import csv
import os
from collections import defaultdict


def read_score_from_log(log_path):
    """Return the best docking score of a Vina-GPU log (2 lines below the 'kcal/mol' header)."""
    with open(log_path, "r") as file:
        lines = file.read().split("\n")
    indices = [i for i, line in enumerate(lines) if "kcal/mol" in line]
    if indices:
        return lines[indices[0] + 2].split()[1]
    return ""


def collect(log_directory):
    # scores[generation][population] = docking_score
    scores = defaultdict(dict)
    for file_name in os.listdir(log_directory):
        if file_name.endswith(".log") and file_name.startswith("ligand_complex_rotated_"):
            # ligand_complex_rotated_{gen}_{pop}.log
            parts = file_name.replace(".log", "").split("_")
            try:
                generation = int(parts[3])
                population = int(parts[4])
            except (ValueError, IndexError):
                continue
            scores[generation][population] = read_score_from_log(os.path.join(log_directory, file_name))

    max_generation = max(scores) if scores else 0
    max_population = max((p for g in scores for p in scores[g]), default=0)

    name = os.path.basename(os.path.normpath(log_directory))
    output_csv = os.path.join(log_directory, f"docking_scores_{name}.csv")
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Pop"] + [f"Gen {g}" for g in range(0, max_generation + 1)])
        for p in range(0, max_population + 1):
            writer.writerow([p] + [scores[g].get(p, "") for g in range(0, max_generation + 1)])
    print(f"Docking scores written to: {output_csv}")
    return output_csv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workdir", nargs="+", help="one or more DRGSCROLL run directories")
    args = ap.parse_args(argv)
    for w in args.workdir:
        collect(w)


if __name__ == "__main__":
    main()
