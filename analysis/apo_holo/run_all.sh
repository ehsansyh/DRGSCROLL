#!/usr/bin/env bash
set -euo pipefail
for s in 01_scores 02_rmsd 03_stats 04_core_contacts 05_ensemble \
         06_fingerprint 07_chi 08_noise 09_figures 10_table; do
    echo "== $s"
    python "${s}.py"
done
