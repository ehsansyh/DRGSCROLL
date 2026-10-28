# Residue-numbering offsets between docking pipelines

Engines that pass through different preparation pipelines may renumber binding-site residues
relative to the reference structure. Contacts matched by residue number then fail for those
engines even when every contacting residue is identical.

Observed offsets in the PDBbind benchmark include +1 (1C83), +24 (2IHJ), +95 (4AGL–4AGQ),
+227 (4PP3), +414 (3FUZ, 3FVK) and +529 (1NHZ).

## Detection

`plip_parse.py` and `apply_shift.py` take the full binding-site residue list of each structure,
which carries both residue numbers and amino-acid identities, and find the integer shift that
maximises identity agreement with the reference. A shift is applied only when the agreement at
that shift exceeds the agreement at zero by a clear margin and the best shift is well separated
from the second best. Ambiguous cases are left uncorrected, which is the conservative choice.

Pass `--no-offset` to disable the correction and reproduce the uncorrected values.
