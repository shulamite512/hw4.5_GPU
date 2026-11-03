## Written answers

These answers are based on the measured results saved in `results/` and the run records in `RUN_LOG.txt`.


### HW4.a - Serving: latency vs throughput (3-4 sentences)

On the RTX 4090, per-request latency never became worse than the HF pipeline baseline at the batch sizes tested: vLLM's p50 rose from 619 ms at batch 1 to 734 ms at batch 8 and 761 ms at batch 32, still well below the HF pipeline's 2,021 ms, while total throughput grew 26x from 207 to 5,434 tokens/s. Relative to vLLM's own batch 1, latency became 19-23% worse from batch 8 upward while throughput continued increasing, so batch 8 is where the latency-throughput trade-off begins on this card. For an interactive application, per-request latency, especially p95 latency, matters more because it determines how long each user waits. Total throughput matters more for serving many offline requests efficiently.

### Part A - Where throughput stops improving, and the batch/VRAM ratio

Throughput rises almost linearly up to batch 16 because the GPU is launch-bound and mostly idle. The gains shrink from batch 32 to batch 128, and beyond batch 256 each doubling adds only about 5% while memory continues to increase substantially. Using the 95%-of-peak rule, throughput stops improving meaningfully at batch 256.

Only the RTX 4090 was measured because the second-GPU experiment was optional and was not performed. Therefore, a cross-card comparison and maximum-batch ratio are not reported.

The maximum-batch ratio would not necessarily equal the VRAM ratio because GPU memory includes both a fixed component and a batch-dependent component. The fixed component includes model weights, gradients, optimizer state, and CUDA overhead, while activations grow with batch size. Therefore, additional VRAM does not translate directly into the same proportional increase in maximum batch size.

During early invalid runs, the Windows driver allowed batches larger than the available VRAM to spill into system memory over PCIe. Batch sizes 1024 and 2048 appeared to run, but throughput fell to approximately 70 and 43 samples/s instead of about 1,146 samples/s. The scripts were updated to cap PyTorch at 99% of VRAM so that the memory wall produced a real out-of-memory result.

### Part B - Which technique should be applied first, and which should be skipped on a 32 GB card

I would apply mixed precision first because it saved memory, improved throughput, and preserved accuracy. At batch 32, mixed precision improved throughput from 699.6 to 767.0 samples/s while reducing peak allocated memory from 2.472 to 2.256 GiB. Its evaluation accuracy was 0.6893 compared with 0.6880 for the baseline.

On a 32 GB card, I would not apply gradient checkpointing for this particular workload because the baseline fine-tune uses only about 2.5 GiB at batch 32 and about 20.6 GiB at batch 512. Gradient checkpointing reduced throughput from 699.6 to 559.5 samples/s, a 20.0% slowdown, while saving only about 0.4 GiB at batch 32. It becomes more useful only when the batch size or model size is large enough that memory is the limiting factor.

The ranking depends on batch size. At batch 32, checkpointing saves relatively little memory and has the largest throughput penalty. At batch 512, however, checkpointing saves much more activation memory and becomes more useful. Gradient accumulation also reduced accuracy because the unchanged learning rate produced fewer optimizer steps per epoch when the effective batch size was increased.

### Part C - Two cards at once

The TA confirmed that the two-GPU pairing was optional on 2026-10-22. The available lab machine had one RTX 4090, so only the single-GPU DDP baseline was run.

The single-GPU run was homogeneous, single-node, and had world size 1. Because there was only one GPU, no gradient all-reduce communication occurred. Therefore, interconnect path, effective all-reduce bandwidth, two-GPU scaling efficiency, and the batch size at which two GPUs become worthwhile could not be measured.

At batch sizes 64 and above, the single-GPU DDP run reproduced the Part A throughput results within approximately 2%. At small batch sizes, timing varied because the workload was launch-bound and the GPU sometimes waited for the CPU. The reported communication overhead, up to approximately 7.5 ms per step, should therefore be interpreted as the timing noise floor of the single-GPU measurement rather than real multi-GPU communication overhead.

### Part D - Serving on real hardware versus HW4.a

The HW4.5 serving rerun matched the HW4.a serving results within approximately 2% at every tested configuration. For example, vLLM batch 32 throughput changed from 5,434 tokens/s in HW4.a to 5,547 tokens/s in HW4.5.

Both runs used the same RTX 4090, the same bf16 precision, the same Qwen/Qwen2.5-1.5B-Instruct model, and the same 128-token generation length. The close agreement indicates that there was no meaningful hardware or precision change and provides evidence that the earlier measurement was not flawed.

KV-cache memory grew linearly with context length. The measured cache size was approximately 28 KiB per token per sequence, which becomes approximately 224 KiB per token of context at batch size 8. The measured values agreed with the analytic estimates within approximately 8 MiB.

At batch size 8, the HF Transformers implementation reached 58,368 tokens before running out of memory. vLLM's paged KV-cache pool allowed approximately 84,680 tokens per sequence by memory capacity. However, Qwen2.5-1.5B supports a maximum context length of 32,768 tokens, so the longest model-supported context on this RTX 4090 was 32,768 tokens.

### Part E - Configuration for a full fine-tune

I would use bf16 mixed precision with the original HW4 batch size of 32. It was the fastest measured configuration for which evaluation accuracy was recorded, reaching 767.0 samples/s with 0.6893 accuracy compared with 0.6880 for the fp32 baseline. It also reduced peak memory from 2.472 to 2.256 GiB without changing the overall training recipe.

For the full dataset, the calculation is:

- Training data: 59,899 samples x 70% = 41,929 samples
- Four epochs: 41,929 x 4 = 167,716 training samples
- Measured throughput: 767.0 samples/s
- Training time: 167,716 / 767.0 = approximately 219 seconds
- GPU time: 219 / 3,600 = approximately 0.06 GPU-hours

A larger batch size with bf16 could provide higher throughput, but it would change the number of optimizer steps per epoch. Since those larger-batch accuracy results were not measured with a retuned learning rate, batch size 32 with bf16 is the safest configuration for the full fine-tune.
