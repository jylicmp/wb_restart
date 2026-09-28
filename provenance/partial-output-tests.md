# Iteration-zero partial output acceptance

Implementation commit: `4d9027ada5f9e7284f7d14be1ad44e50c67a24dd`.

The committed installed package passed **31 unittest cases in 24.030 seconds**
on a compute node. This includes the previous 22 cases and nine new snapshot
cases: first/final outputs, exact per-snapshot weighted contributions, immediate
resume output, interval/no-change behavior, publication retry, four interrupted
write boundaries, nonfatal output errors, read-only sparse/corrupt/empty export,
writer locking/path/configuration checks, pre-upgrade checkpoint compatibility,
and exclusion of files changed during reading. Some methods cover several cases.

The real Ray regression passed as well. The driver was interrupted after two
point files were saved; restart computed exactly the remaining two points.
The three retained snapshots contain 1, 2 and 4 points. The final iteration-zero
snapshot and result agree with continuous serial calculation within
rtol=1e-10, atol=1e-12; raw DOS and CumDOS maximum absolute differences are zero.
The existing two-refinement comparison passed; its missing-file DOS recovery
error was 4.440892098500626e-16. Worker and driver used the same committed package
in the isolated environment. The small acceptance job completed with exit code
0 in 116 seconds, including original-version comparison and actual Ray workers.

All tests use independent directories. Snapshots contain raw weighted sums,
smoothed arrays and completion metadata; they are not restart authorities.
Detailed scientific sample files are kept outside the public repository.

## Real CrSe sample export

A separate compute-node job loaded the original model and regenerated all 620,775
iteration-zero points. It read the eight existing independent legacy sample
copies and exported their partial IMD/QMD contributions. All input file names and
SHA-256 values remained unchanged. Exported raw and smoothed tensors matched the
independent weighted sum with **maximum absolute error 0** for all four fields.
The metadata correctly indicated eight valid contributions, incomplete coverage,
and no renormalization. This did not recompute MQM or export the entire live run.
The job completed with exit code 0 in 313 seconds, using a 128 GiB allocation in
the user-selected large-memory partition. Both new acceptance jobs together used
429 node-seconds; the accumulated campaign total at this point is 3,033 seconds
(0.8425 node-hours), including the earlier failures and retries.

A follow-up read-only optimization skips absent files after the initial stat,
avoiding redundant failed opens and stats. A dedicated regression asserts that
missing files are never passed to the point loader. It does not change valid-file
loading, numerical accumulation or Ray scheduling.

The final release regression passed **33 tests in 30.270 seconds**, including the
missing-file optimization and an additional test requiring explicit identity or
configuration conflicts to fail even if the file changed during reading.
All original package hashes and the three custom physics files were rechecked
unchanged. The release is tagged `partial-output-v1`.
