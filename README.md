# HW4.a + HW4.5: BERT fine-tuning, vLLM serving, training throughput & memory

SLICE = 384 and SEED = 1384 (from SID4 = 1384) are the scripts' defaults. Every script writes JSON to `results/`,
tagged with the GPU name, and appends its run to `RUN_LOG.txt`.

## Environment (lab machine ADS-R15-836-04, RTX 4090 24 GB, Windows 11)

- **BERT training (4.a.1, Parts A, B): Windows Python 3.13**:
  `pip install torch --index-url https://download.pytorch.org/whl/cu128` then
  `pip install transformers datasets scikit-learn pandas bitsandbytes accelerate matplotlib`
  (torch 2.11.0, transformers 5.20.2, bitsandbytes 0.50.2).
- **vLLM (4.a.2, Part D) and DDP (Part C): WSL2 Ubuntu 24.04** (vLLM and NCCL are Linux-only):
  `pip install vllm==0.11.0` then `pip install "transformers>=4.56,<5" "datasets<5" scikit-learn pandas matplotlib`
  (vLLM 0.11 fails on transformers 5).
- **Windows/WSL caveat:** past VRAM the NVIDIA driver spills into system RAM instead of raising OOM (bs=1024 ran
  17x slower with a 39.9 GiB "peak" on a 24 GiB card). The scripts cap PyTorch at 99% of VRAM to get a real OOM;
  a cap of exactly 1.0 is treated by PyTorch as no cap.

**Dataset:** by default, `cardiffnlp/tweet_eval` (sentiment: negative/neutral/positive, ~60k tweets; downloads
automatically; rows numbered train → validation → test). To use a course-provided CSV instead, add
`--csv file.csv --text-col <column> --label-col <column>` to every training command.

## Run order (on each card)

| Step | Command (4.a.2, C, D in WSL; the rest in Windows) | Approx. time on a 4090 |
|---|---|---|
| HW4.a.1 BERT | `python hw4a_train.py --slice SLICE --seed SEED` | 2-3 min |
| HW4.a.2 serving | `python hw4a_serve.py --backend hf` then `python hw4a_serve.py --backend vllm` | 5 min |
| 4.5 A | `python hw45_partA_batch_scaling.py --slice SLICE --seed SEED` | 10 min |
| 4.5 B | `python hw45_partB_memory.py --slice SLICE --seed SEED` | 30-40 min |
| 4.5 C baseline | `torchrun --nproc_per_node=1 hw45_partC_ddp.py --slice SLICE --seed SEED` | 3 min |
| 4.5 C pair | on the instructor's 2-GPU pairing: `torchrun --nproc_per_node=2 hw45_partC_ddp.py --slice SLICE --seed SEED --tag <pair>` | 3 min |
| 4.5 D serving | `python hw4a_serve.py --backend hf --tag hw45` and `--backend vllm --tag hw45` | 5 min |
| 4.5 D KV cache | `python hw45_partD_kv_cache.py --backend hf` then `--backend vllm` | 5 min |
| Report | `python make_report.py` → `REPORT.md`, `METRICS.md`, `plots/` | seconds |

To get the "both cards" comparisons (Parts A, D), copy `results/` from the second machine (e.g. a lab RTX 5090)
into this folder and re-run `make_report.py`; the tables and the throughput plot include every card found.

## What each measurement means

- **Peak memory** = `torch.cuda.max_memory_allocated()` (tensors). Reserved memory (allocator cache) is reported alongside.
- **Throughput** (Parts A/B) = samples/s over 20 timed optimiser steps after 3 warm-up steps, batches sampled with
  replacement from the 70% training split so batch size isn't capped by the split's size. Time/epoch = training
  split size ÷ throughput.
- **AMP** = bf16 autocast (no GradScaler needed on Ada/Blackwell). **8-bit optimiser** = bitsandbytes `AdamW8bit`.
- **Serving**: every request generates exactly 128 tokens (greedy, EOS ignored); a request's latency is its batch's
  wall time; tokens/s counts generated tokens only. Model: `Qwen/Qwen2.5-1.5B-Instruct`, bf16.
- **DDP comm overhead** = step time with gradient all-reduce − slowest rank's step time under `no_sync()`
  (the communication not hidden behind backward).

## Deliverables checklist

| Deliverable | Where | Produced by |
|---|---|---|
| Table HW4.5.1, once per card | `REPORT.md` → Part B | `make_report.py` (one table per `results/partB_<GPU>.json`) |
| Both UUID captures | `captures/uuid_*.txt` | `python capture_uuid.py` on each card |
| All figures | `plots/*.png` | `make_report.py` |
| METRICS.md (HW4.5 summary) | `METRICS.md` | `make_report.py` |
| RUN_LOG.txt | `RUN_LOG.txt` | automatic (`runlog.py`), plus backfilled early entries |
| Saved terminal output of every run | `outputs/<date>_<time>_<script>.log` | automatic (`runlog.py`); runs before 2026-10-22 saved from the terminal text copied at the time (marked in each file) |
| AI_USE.md | `AI_USE.md` | draft; fill in the bracketed parts |
| Reservation record | `RESERVATION.md` + `captures/reservation_*` | fill in by hand |
| GPU-hour record | `GPU_HOURS.md` | `make_report.py`, from the run log |
| Repository tagged `hw4-5` | git | `git tag hw4-5 && git push origin hw4-5` once results are final |

## Written answers

`make_report.py` computes every number; the written answers live in `ANSWERS.md` (and the misclassification
diagnoses in `diagnoses.json`), which `make_report.py` merges into `REPORT.md`. Edit those two files, never
`REPORT.md` itself: it is rebuilt from scratch on every run.
