"""HW4.5 Part D - KV-cache memory vs context length at a fixed batch size (default 8).

Two complementary measurements, each run in its own process:

  --backend hf    Measured. Chunked prefill of `batch` random sequences into a growing KV cache; records
                  torch.cuda.memory_allocated() minus the weights at each context length until OOM.
  --backend vllm  Capacity. Starts vLLM, reads the KV-cache pool size it allocates (from its startup log),
                  and derives the longest context that `batch` concurrent sequences fit in.

The analytic size per token, 2 (K,V) x layers x kv_heads x head_dim x bytes, is reported for both.

    python hw45_partD_kv_cache.py --backend hf
    python hw45_partD_kv_cache.py --backend vllm
"""
import runlog  # noqa: F401  (appends this run to RUN_LOG.txt)
import argparse
import re
import subprocess
import sys

import torch
from transformers import AutoConfig

from common import GIB, env_info, gpu_tag, is_oom, save_json

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["hf", "vllm"], required=True)
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--batch", type=int, default=8)
p.add_argument("--chunk", type=int, default=1024)
p.add_argument("--max-ctx", type=int, default=262144)
p.add_argument("--gpu-util", type=float, default=0.90)
p.add_argument("--vllm-child", action="store_true", help=argparse.SUPPRESS)
a = p.parse_args()

cfg = AutoConfig.from_pretrained(a.model)
head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
bytes_per_token = 2 * cfg.num_hidden_layers * kv_heads * head_dim * 2  # bf16
max_pos = cfg.max_position_embeddings
arch = dict(layers=cfg.num_hidden_layers, kv_heads=kv_heads, head_dim=head_dim,
            kv_bytes_per_token=bytes_per_token, max_position_embeddings=max_pos)

if a.vllm_child:  # runs inside the subprocess so the parent can read vLLM's log output
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=max_pos, max_num_seqs=a.batch,
              gpu_memory_utilization=a.gpu_util)
    llm.generate(["hello"] * a.batch, SamplingParams(max_tokens=8), use_tqdm=False)
    sys.exit(0)

if a.backend == "vllm":
    out = subprocess.run([sys.executable, __file__, "--backend", "vllm", "--vllm-child", "--model", a.model,
                          "--batch", str(a.batch), "--gpu-util", str(a.gpu_util)],
                         capture_output=True, text=True)
    log = out.stdout + out.stderr
    m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log)          # vLLM V1 engine
    if m:
        capacity = int(m.group(1).replace(",", ""))
    else:
        m = re.search(r"# GPU blocks:\s*(\d+)", log)                        # V0 engine, 16-token blocks
        capacity = int(m.group(1)) * 16 if m else None
    if capacity is None:
        print(log[-4000:])
        raise SystemExit("could not find the KV-cache size in vLLM's log (see output above)")
    mem_limited = capacity // a.batch
    res = dict(env=env_info(), model=a.model, arch=arch, batch=a.batch, gpu_memory_utilization=a.gpu_util,
               kv_pool_tokens=capacity, kv_pool_gib=round(capacity * bytes_per_token / GIB, 2),
               longest_ctx_memory_limited=mem_limited, longest_ctx_model_limited=max_pos,
               longest_ctx_servable=min(mem_limited, max_pos),
               table=[dict(ctx=L, kv_gib=round(a.batch * L * bytes_per_token / GIB, 3),
                           fits=a.batch * L <= capacity) for L in [2 ** k for k in range(9, 19)]])
    print(f"vLLM KV pool: {capacity:,} tokens ({res['kv_pool_gib']} GiB). At batch {a.batch}: memory allows "
          f"{mem_limited:,} tokens/seq, model allows {max_pos:,} -> servable {res['longest_ctx_servable']:,}")
    save_json(res, f"partD_kv_vllm_{gpu_tag()}.json")
    sys.exit(0)

# ---- HF measured curve
from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

# WSL uses the Windows driver, whose sysmem fallback spills past VRAM instead of raising OOM (the first run
# crawled at 100% GPU with 24.0/24.6 GB used); same 0.99 cap as common.cap_to_physical_vram (1.0 means no cap).
torch.cuda.set_per_process_memory_fraction(0.99)
lm = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
base = lm.model  # skip the LM head: full-sequence logits (batch x ctx x vocab) would dwarf the KV cache
torch.cuda.synchronize()
weights = torch.cuda.memory_allocated()
cache, ctx, rows, oom_at, last_ok, last_row = DynamicCache(), 0, [], None, 0, None
gen = torch.Generator(device="cuda").manual_seed(0)
next_mark = 512
with torch.no_grad():
    while ctx < a.max_ctx:
        ids = torch.randint(0, cfg.vocab_size, (a.batch, a.chunk), device="cuda", generator=gen)
        try:
            torch.cuda.reset_peak_memory_stats()
            out = base(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            del out
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            oom_at = ctx + a.chunk
            break
        ctx += a.chunk
        row = dict(ctx=ctx, kv_measured_gib=round((torch.cuda.memory_allocated() - weights) / GIB, 3),
                   kv_analytic_gib=round(a.batch * ctx * bytes_per_token / GIB, 3),
                   peak_total_gib=round(torch.cuda.max_memory_allocated() / GIB, 3))
        if ctx >= next_mark:
            rows.append(row)
            next_mark *= 2
        if ctx % (8 * a.chunk) == 0 or (rows and rows[-1] is row):
            print(f"ctx {ctx:>7}: KV {row['kv_measured_gib']} GiB measured, "
                  f"{row['kv_analytic_gib']} analytic; peak {row['peak_total_gib']} GiB", flush=True)
        last_ok, last_row = ctx, row

if last_row is not None and (not rows or last_row is not rows[-1]):
    rows.append(last_row)  # the largest context that fit, so the table ends at the memory wall

print(f"HF longest context at batch {a.batch}: {last_ok:,} tokens (OOM at {oom_at})")
save_json(dict(env=env_info(), model=a.model, arch=arch, batch=a.batch, weights_gib=round(weights / GIB, 3),
               rows=rows, longest_ctx=last_ok, oom_at=oom_at,
               note="beyond max_position_embeddings the memory number is valid but output quality is not"),
          f"partD_kv_hf_{gpu_tag()}.json")
