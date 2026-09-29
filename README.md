# WannierBerri 1.8 Restart

**English** | [简体中文](README.zh-CN.md)

This project adds interruption-safe restart, partial result export, and bounded Ray result collection to a customized WannierBerri **1.8.0** codebase. The existing MQM calculators, formulas, and matrix requirements are preserved. The complete source baseline is tagged `baseline-v1.8-mqm`.

The upstream code is distributed under the [GPL](LICENSE).

## Features

- Commits the complete K-point list and weights before dispatching each iteration.
- Resumes interrupted iteration 0 calculations and later adaptive refinement iterations.
- Writes per-point results atomically and validates their index, configuration, shape, and checksum.
- Quarantines corrupt files for recomputation, supports checkpoint relocation, and enforces a single writer.
- Explicitly reconstructs a missing legacy iteration 0 K-point list, reuses valid results, and computes only missing points.
- Publishes integrity-checked partial snapshots while iteration 0 is still running.
- Collects Ray results through an `ObjectRef -> K-point` mapping with bounded pending and deserialization batches.
- Restricts incremental symmetry deduplication to newly added K points, avoiding quadratic scans over large existing grids.

## Python API

```python
result = wb.run(
    system, grid, calculators,
    restart=True,
    restart_recover=True,        # Required only for legacy iteration 0 without a K-point list
    restart_on_corrupt="error",  # Or "recompute" to quarantine and recompute corrupt files
    file_Klist_path="/path/to/independent/checkpoint",
    allow_restart=True,
    dump_results=True,
    partial_save_interval=900,
    partial_output_dir="/path/to/partial-results",
    adpt_num_iter=20,
)
```

`adpt_num_iter` keeps its original meaning: it is the number of additional refinement iterations after the selected global iteration. For example, continuing from iteration 5 through iteration 20 requires 15 additional iterations. The command-line entry point performs this conversion automatically:

```bash
export WB_TB_FILE=/path/to/CrSe_SOC_tb.dat
python /path/to/repo/scripts/restart_run.py \
  --config /path/to/repo/examples/crse_config.py \
  --checkpoint /path/to/independent/checkpoint --recover --inspect

python /path/to/repo/scripts/restart_run.py \
  --config /path/to/repo/examples/crse_config.py \
  --checkpoint /path/to/independent/checkpoint --recover \
  --on-corrupt recompute --until-iteration 20 --output /path/to/results/CrSe \
  --progress-interval 300 \
  --partial-interval 900 --partial-output /path/to/partial-results
```

Inspection is read-only, but it still loads the full model and reconstructs the grid, so run it through an appropriate compute job. `--inspect` reports valid, missing, and corrupt indices. Configuration or result-schema mismatches fail immediately and are never treated as files that may be recomputed.

Progress lines are emitted at most once every 300 seconds by default. Later adaptive rounds normally add only a few K points, so `examples/resume.sbatch` requires a single-node allocation. Use the multi-node harness only for an initial recovery with enough missing points to occupy multiple nodes.

## Deployment, validation, and operations

Clone the original environment as `wberri_v1.8_restart`, verify that its source files are independent from the original environment, and deploy from a clean Git commit:

```bash
python scripts/deploy.py --environment /path/to/wberri_v1.8_restart
```

The deployment script copies and verifies the package files and records the deployed Git commit. It does not use an editable installation and refuses to deploy into an environment with any other name. Run production calculations outside the repository root so the checkout cannot shadow the installed package.

- [Iteration 0 partial output and read-only export](docs/partial-output.md)
- [Validation report: iteration 0, real CrSe, and Ray](docs/validation.md)
- [Checkpoint format and compatibility boundaries](docs/checkpoint-format.md)
- [Production cutover, testing, and rollback](docs/operations.md)
- Small-model tests: `python -m unittest discover -v -s /path/to/repo/tests`
- Original/new implementation and real-Ray validation template: `examples/validate_small.sbatch`
- Real-grid and sampled recomputation template: `examples/validate_crse.sbatch`

Persistent checkpoints currently support `System_R`, including `System_tb`. Calculations without checkpoints retain the original system-type support.

Legacy pickle files do not contain coordinates or a complete input fingerprint. Matching reconstructed weights alone does not independently prove their provenance. Verify the original inputs and recompute a small sample before recovery. Load pickle files only from trusted calculations.
