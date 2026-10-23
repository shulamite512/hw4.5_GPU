## Written answers

> DRAFT written with AI assistance from the measured results; reword in your own voice, then delete this line.
> Edit this file, not REPORT.md: make_report.py rebuilds REPORT.md and appends this file to it.

### HW4.a - Serving: latency vs throughput (3-4 sentences)

On the RTX 4090, per-request latency never became worse than the HF pipeline baseline at the batch sizes tested:
vLLM's p50 rose from 619 ms at batch 1 to 734 ms at batch 8 and 761 ms at batch 32, still well below the HF
pipeline's 2,021 ms, while total throughput grew 26x (207 to 5,434 tok/s). Relative to vLLM's own batch 1, latency
got 19-23% worse from batch 8 upward while throughput kept climbing, so batch 8 is where the trade-off begins on this
card. For an interactive application the number that matters is per-request latency (p50 and especially p95),
because each user waits for their own response; total throughput matters for serving cost and offline batch jobs.

### Part A - Where throughput stops improving, and the batch/VRAM ratio

Throughput rises almost linearly up to batch 16 (the GPU is launch-bound and mostly idle), gains shrink from batch
32 to 128, and beyond batch 256 each doubling adds only ~5% while memory doubles; by the 95%-of-peak rule throughput
stops improving at batch 256. Only the RTX 4090 was measured, so the same-on-both-cards comparison and the measured
max-batch ratio are not available.

Why the max-batch ratio would not equal the VRAM ratio: peak memory is a fixed part (weights, gradients and AdamW
state, ~2 GiB, plus CUDA context) and a part that grows with batch (activations, ~38 MiB per sample at seq len 64),
so extra VRAM only buys extra batch after the fixed part is paid. Doubling also skips everything between powers of
two: batch 1024 needs ~40 GiB, so a 32 GB card would very likely also stop at 512, giving a max-batch ratio of 1.0
against a VRAM ratio of 32/24 = 1.33.

Windows caveat: the driver's system-memory fallback let batch 1024 and 2048 "run" past the 24 GiB card by spilling
into system RAM over PCIe, at 70 and 43 samp/s instead of 1,146 (16-27x slower). The scripts cap PyTorch at 99% of
VRAM so the memory wall shows up as a real OOM.

### Part B - Which technique first, and which to skip on a 32 GB card

Apply mixed precision (bf16) first: it is the only technique that saves memory and adds throughput at the same time
(+10% at batch 32, about 2x from batch 128 up: 2,391 vs 1,147 samp/s at batch 512) with unchanged accuracy (0.689
vs 0.688). On a 32 GB card I would not bother with gradient checkpointing: this fine-tune peaks at 2.5 GiB at the
HW4 batch size and 20.6 GiB even at batch 512, so checkpointing would cost 20-27% throughput for memory that is not
needed. The 8-bit optimiser is also marginal here, saving a constant ~0.6 GiB of optimiser state on a 110M-parameter
model.

The ranking depends on batch size: at batch 32 activations are small, so checkpointing saves only 0.4 GiB and ranks
last; at batch 512 it saves 16.2 GiB (20.6 to 4.4 GiB) for 26.7% throughput, which would put it above 8-bit AdamW.
Gradient accumulation lost 1.3 points of accuracy because 8 x 16 = 128 gives 4x fewer optimiser steps per epoch at
the same learning rate, not because of the technique itself.

### Part C - Two cards at once

The TA confirmed the two-GPU pairing is optional (2026-10-22), and the lab machine has a single RTX 4090, so only the
single-GPU DDP baseline was run. Interconnect path, effective all-reduce bandwidth, scaling efficiency and the batch
size at which two GPUs pay off therefore could not be measured. At batch 64 and above the single-GPU run reproduces
Part A within ~2%; small batches are launch-bound (the GPU waits on the CPU) and vary between runs. With no
all-reduce to do, the measured "comm overhead" (up to ~7.5 ms per step) is the noise floor of the method.

### Part D - Serving on real hardware vs HW4

The Part D re-run matches the HW4.a serving numbers within about 2% at every configuration (for example 5,434 vs
5,547 tok/s at vLLM batch 32), because both runs used the same card (RTX 4090), the same precision (bf16) and the same
software. So nothing changed in hardware or precision, and the close agreement is evidence the measurement is not
flawed.

KV-cache memory grows exactly linearly with context: 28 KiB per token per sequence (2 x 28 layers x 2 KV heads x 128
dims x 2 bytes), i.e. 224 KiB per token of context at batch 8, and the measured values match this within 8 MiB. At
batch 8 the 4090 runs out of memory at 58,368 tokens with HF transformers (prefill activations and the attention mask
add ~3.6 GiB on top of 12.3 GiB of KV) and at 84,680 tokens with vLLM (PagedAttention, one 18.1 GiB pool), but
Qwen2.5-1.5B supports only 32,768 positions, so the longest servable context on this card is 32,768 with either
backend: the model, not memory, is the limit. Only the RTX 4090 was measured.

### Part E - Configuration for a full fine-tune

I would use bf16 mixed precision at the HW4 batch size of 32. It was the fastest configuration whose accuracy was
measured (767 samp/s, 0.689 test accuracy vs 0.688 for the fp32 baseline), it lowers peak memory (2.26 vs 2.47 GiB),
and it needs no other change to the recipe; a full fine-tune of all 59,899 tweets then takes about 0.06 GPU-hours on
one 4090. Batch 128 with bf16 would be about 2.8x faster (2,165 samp/s) and still fit easily (4.5 GiB), but it changes
the optimisation (4x fewer steps per epoch), so the learning rate would need retuning and its accuracy was not
measured; it is the next experiment rather than the recommendation.
