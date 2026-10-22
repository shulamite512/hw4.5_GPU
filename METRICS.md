# METRICS - HW4.5 summary

> HW4.5 summary of training-throughput and memory engineering on the HW4.a BERT fine-tune. This is **not** an HW5 baseline (HW5 uses a different model and task).

**HW4.a reference run (NVIDIA GeForce RTX 4090)**: peak 2.472 GiB, wall clock 21.5 s, test acc 0.6907, rows 384-5383, SEED 1384.

## RTX4090

| Metric | Configuration | Value | Eval accuracy |
|---|---|---|---|
| Best peak memory | Combo: AMP + 8-bit AdamW + checkpointing | 1.082 GiB @ bs32 | 0.6867 |
| Best throughput | Mixed precision (bf16) | 767.0 samp/s @ bs32 | 0.6893 |
| Best wall-clock | Best combo: AMP + 8-bit AdamW | 16.98 s full fine-tune | 0.6867 |


Baseline (HW4 settings) on this card: 699.6 samp/s, 2.472 GiB @ bs32, 21.22 s, acc 0.688. Candidates restricted to configurations within 1 point of baseline accuracy.
