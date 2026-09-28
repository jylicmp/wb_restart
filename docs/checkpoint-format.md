# Checkpoint format 1

`restart.json` is the atomic commit record. It contains a scientific configuration
fingerprint and entries indexed by global iteration. Each entry references an
immutable, checksummed pickle containing the K-point metadata and weight vector
as one object. A generation is published before dispatch and marked complete only
after output is written. Incomplete output is regenerated on restart.

Each new point result is an atomic pickle envelope containing its index, geometric
identity, scientific configuration digest, and a checksummed result payload. New
filenames include a geometry digest so historical branches cannot reuse another
point's result accidentally. Weights are excluded from point identity: refinement
changes weights without changing that point's calculated tensor.

The driver holds an advisory filesystem lock for the whole run. Never delete a
lock file to unlock a running process. Locks are released by the OS on exit.
The old unmodified 1.8 process does not participate in this locking protocol:
never run recovery against its live output directory.

Recovery scans result files and rebuilds flags, maxima, and the weighted sum.
Only missing or explicitly quarantined corrupt files are recalculated. A config,
energy, shape, or identity mismatch always fails, including with `recompute`.
Historical restart selects numeric iteration order and discards later iteration
entries from the new manifest; their immutable files remain on disk.

Legacy complete K-list/weights checkpoints are readable. Missing-list recovery is
explicit and limited to regular-grid iteration 0; its regenerated per-index
weights must be exactly equal. Raw legacy results have no identity proof. Their
provenance stays explicitly marked when imported. Validate input provenance and
sample recomputations before trusting them. Only load trusted pickle files.

Persistent configuration hashing currently supports System_R and its constructors
(including System_tb). Non-checkpoint runs retain the original system support.
The configuration includes all loaded real-space matrices, lattice, centers,
R vectors, symmetry, grid, calculators, parameters_K and refinement settings, plus
the scientific Python source hashes. Source changes affecting physics require a
new checkpoint. Output prefixes and final iteration targets may change.

Atomic writes use fsync, same-filesystem rename, and directory fsync. Unreferenced
metadata generations or temporary files from a crash are harmless and are not
automatically deleted. Metadata corruption fails closed; keep the backup.
