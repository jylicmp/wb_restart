# Live-copy integration test

This harness copies a running task into a new sibling directory, then resumes the
copy through global iterations 0 and 1. The original task remains running.
Private source/destination paths, Slurm partitions and job IDs stay outside Git.

1. Supply `WB_COPY_SOURCE`, `WB_TEST_ROOT`, `WB_REPO` and `WB_ENV` to
   `examples/copy_live.sbatch`. Use a small single-node staging allocation.
   The target must not exist. Each file is read with before/after stability checks,
   independently copied, SHA-256 verified and recorded in `_restart_test`.
   The copy is explicitly a dynamic snapshot; later-created source files are not
   included. Any copy failure blocks continuation. Directory symlinks are rejected;
   ordinary file symlinks are copied as independent regular files.
2. Submit `examples/resume_multinode.sbatch` after successful staging, with four
   exclusive nodes, 64 allocated CPUs per node and the user-approved time limit.
   Set `WB_SOURCE` to the original checkpoint, `WB_TB_FILE` to the original model,
   and preserve the other variables above. The test uses 30 Ray CPUs on the head
   and 32 on each other node, with 32 GiB object stores. This bounds initial
   concurrency while leaving memory for expensive individual K points.
3. Each Ray node verifies its Python/package location, source checksums and Git
   deployment against the driver. The cluster uses a job-specific port and local
   temporary directory. Cleanup targets only this job's launch steps.
4. Recovery reuses valid copied results, repairs missing/corrupt points and emits
   iteration-zero snapshots every 900 seconds. The driver finishes iteration 1.
   Acceptance checks both completed manifests, finite refined tensors and
   agreement between the final partial and official iteration-zero tensors.

The copied original launch script is preserved for provenance and is **not**
executed. Submit only the new integration harness. Do not run the copied original
`restart=False` script against its populated checkpoint.

Watch the staging log first, then the Slurm log and `_restart_test/restart-JOB.log`.
Snapshots are under `partial`; completed round outputs are under `restart-results`.
A successful run writes `_restart_test/acceptance.json` and prints
`INTEGRATION_VALIDATION_PASSED`. Scheduler completion alone is insufficient.
A timeout/failure preserves the copied checkpoint and snapshots for diagnosis.
