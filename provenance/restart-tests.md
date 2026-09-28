# Restart implementation milestone

Executed in the cloned Python 3.12 / WannierBerri 1.8.0 environment:
`python -B tests/test_restart.py` — 8 tests passed (1.5 s).

Covers continuous versus split adaptive integration, interruption in the first
and a later round, explicit missing-list legacy recovery, read-only inspection,
missing/truncated point repair, relocation, configuration/point identity rejection,
historical iteration selection, interruption before manifest/result rename, and
exclusive writer locking. Tensor comparison uses rtol=1e-10, atol=1e-12; refined
point coordinates and weights are compared exactly.

## Expanded acceptance regression

The final suite contains 16 passing unittest cases (6.916 s in the cloned
environment). Additional cases cover the public read-only inspection CLI,
continuation to a final global iteration, complete-checkpoint zero-recompute,
interrupted completion/refinement, changed model/legacy weights, old complete
K-list files, deployment isolation and managed-file rollback, and a 1e-10 weight
whose old contribution must be removed during refinement.

Real compute-node validation also passed: untouched 1.8 versus new serial
integration agrees exactly on the small model; real Ray parallel integration
agrees exactly, and missing-point recovery differs by at most
4.440892098500626e-16. Main process and workers used the same committed package.

## Iteration 0 focus

After the requested expansion, **21 tests passed in 11.656 s** in the deployed
cloned environment. Five additional test methods use 84 initial K points and
exercise these subcases:

- Legacy checkpoints without a K-list, with 0, 1, 42, 57, 83 or all 84 results.
  Randomized write order and sparse holes are included.
- Native interruptions after 0, 1, 41 or 83 completed points.
- Sparse legacy recovery interrupted twice more (after 7 and 11 new results),
  then completed and continued through two adaptive refinements.
- Four missing files and three truncated files in the same legacy checkpoint.
- All first-round results saved, but interruption before the completion marker.

The tests record actual computed coordinates, require every needed point exactly
once across attempts, and require saved legacy files to remain byte-identical.
A repeated complete restart performs no computation. Raw and smoothed totals
agree within rtol=1e-10, atol=1e-12. Final adaptive coordinates and weights agree
exactly. All fixtures reside in disposable independent directories.

A further boundary test exposed and fixed recovery before publication of the
first K-list. A durable initializing manifest now permits restart after either
the initial metadata rename or its manifest publication is interrupted. Both
boundaries are tested, with all 84 points computed exactly once after recovery.
The updated development-source suite passed **22 tests in 12.083 s**.

Final compute-node acceptance of committed implementation
`dc2784eac88acac5723f4155e83206faa3dd5197`: **22 tests passed in 12.850 s**.
The untouched-original comparison and real-Ray regression also passed. Scheduler
state was COMPLETED with exit code 0 and the expected success marker. All 94
original package file hashes remained unchanged; the three custom physics files
also matched the baseline in the new environment.
