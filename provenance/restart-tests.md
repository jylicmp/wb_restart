# Restart implementation milestone

Executed in the cloned Python 3.12 / WannierBerri 1.8.0 environment:
`python -B tests/test_restart.py` — 8 tests passed (1.5 s).

Covers continuous versus split adaptive integration, interruption in the first
and a later round, explicit missing-list legacy recovery, read-only inspection,
missing/truncated point repair, relocation, configuration/point identity rejection,
historical iteration selection, interruption before manifest/result rename, and
exclusive writer locking. Tensor comparison uses rtol=1e-10, atol=1e-12; refined
point coordinates and weights are compared exactly.
