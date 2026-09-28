# Bounded collection milestone

Nine unittest cases passed in the cloned environment (1.684 s).
The 49,999-task collection simulation verified one get/write per point,
no submission of an already evaluated point, and a maximum of 256 pending refs.
Measured collection time: 0.263 s; traced Python allocation peak: 2,894,140 bytes.
These timings measure dispatch bookkeeping with synthetic tasks, not MQM speedup.
Real Ray workers and CrSe numerical checks are separate acceptance jobs.
