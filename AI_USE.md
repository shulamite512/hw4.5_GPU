# AI use disclosure - HW4.a / HW4.5

**Tool:** Claude Code (Anthropic), model Claude Opus 5.5, used inside VS Code on the lab machine, 2026-10-21.

## What the AI did

| Area | AI contribution |
|---|---|
| Code | Wrote all scripts in this repository from the assignment text: `common.py`, `bench_one.py`, `hw4a_train.py`, `hw4a_serve.py`, `hw45_partA_batch_scaling.py`, `hw45_partB_memory.py`, `hw45_partC_ddp.py`, `hw45_partD_kv_cache.py`, `make_report.py`, `runlog.py`, `capture_uuid.py`. |
| Environment | Checked the GPU/driver and installed missing Python packages (numpy, transformers, datasets, scikit-learn, pandas, bitsandbytes, accelerate, matplotlib); guided the WSL2 Ubuntu + vLLM setup, including a failed Docker attempt that filled the disk and was cleaned up. Wrote the environment notes in `README.md`. |
| Smoke tests | Ran short test runs with placeholder settings (slice 0, seed 42, 1 epoch) to catch errors before the real runs; these are listed in `RUN_LOG.txt` and the GPU-hour record and their outputs were overwritten. |
| Debugging | Found and fixed: (1) `batch_encode_plus` removed in transformers 5; (2) Windows "sysmem fallback" letting batch 1024 spill into system RAM instead of raising OOM; (3) `set_per_process_memory_fraction(1.0)` not enforcing a cap in PyTorch (fixed with 0.99; verified by test). |
| Analysis | Interpreted results and drafted the five misclassification diagnoses (`diagnoses.json`) and explanatory text in `REPORT.md`. |
| Documentation | Drafted `README.md`, `RUN_LOG.txt` backfill entries, this file, and the reservation/GPU-hour templates. |

## What I did

- Ran every reported experiment myself on the reserved GPU (commands and timestamps in `RUN_LOG.txt`).
- Provided my personal parameters (SID4 = 1384 -> SLICE = 384, SEED = 1384) and the assignment text.
- Asked the instructor which Twitter dataset to use; `cardiffnlp/tweet_eval` (sentiment) was used pending the reply.
- Reviewed the results, the diagnoses and the written answers, and edited them into my own words: _[describe what you changed]_.

## Verification

- Every number in `REPORT.md` / `METRICS.md` comes from a JSON file in `results/` written by the scripts during my runs; none were typed in by hand.
- _[Add anything you checked independently, e.g. re-ran a configuration, compared against nvidia-smi.]_
