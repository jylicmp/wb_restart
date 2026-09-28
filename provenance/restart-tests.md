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
