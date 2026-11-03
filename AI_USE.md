# AI use disclosure - HW4.a / HW4.5

**Tool:** Claude Code (Anthropic), model Claude Opus 5.5, 2026-10-21.

## What the AI did

| Area | AI contribution |
|---|---|
| Running experiments | Helped execute and monitor experiment commands while I selected the parameters, used the reserved GPU, and reviewed the outputs. |
| Debugging and checking | Helped troubleshoot package and runtime issues, the Windows system-memory fallback, and the PyTorch memory cap. |
| Analysis and formatting | Helped organize result tables, format the report, and check the architecture, configuration, and reported measurements. |
| Documentation | Helped organize the README, run log, written answers, GPU-hour record, and report structure. |

## What I did

- I wrote the experiment scripts in this repository, including `common.py`, `bench_one.py`, `hw4a_train.py`, `hw4a_serve.py`, `hw45_partA_batch_scaling.py`, `hw45_partB_memory.py`, `hw45_partC_ddp.py`, `hw45_partD_kv_cache.py`, `make_report.py`, `runlog.py`, and `capture_uuid.py`.
- I selected the experiment parameters and provided my personal parameters: SID4 = 1384, SLICE = 384, and SEED = 1384.
- I ran the reported experiments on the reserved RTX 4090 GPU. The commands and timestamps are recorded in `RUN_LOG.txt`.
- I reviewed the experiment settings, result tables, plots, diagnoses, and written answers.
- I checked that the reported values matched the saved JSON results and run logs.
- I assembled the final submission files and report.

## Verification

Every number in `REPORT.md` and `METRICS.md` comes from a JSON result file in `results/` or from the recorded run logs. I checked the GPU identity and environment details against the UUID capture and run log, reviewed the batch-scaling and memory-technique results, and verified that the HW4.5 summary was not labeled as an HW5 baseline.
