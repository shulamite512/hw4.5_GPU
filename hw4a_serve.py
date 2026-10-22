"""HW4.a part 2 (and HW4.5 Part D) - vLLM inference and serving throughput vs a plain HF pipeline.

Run each backend in its own process (vLLM pre-allocates most of the GPU):
    python hw4a_serve.py --backend hf      # HF pipeline, batch 1 (baseline)
    python hw4a_serve.py --backend vllm    # vLLM, batch 1 / 8 / 32

Every request generates exactly --new-tokens tokens (greedy, EOS ignored) so all configurations do the
same work. A "batch" is a group of requests submitted together; each request's latency is the wall time
of its batch. Throughput = generated tokens / total wall time.
"""
import runlog  # noqa: F401  (appends this run to RUN_LOG.txt)
import argparse
import time

import numpy as np
import torch

from common import env_info, gpu_tag, save_json

MY_PROMPTS = [
    "Explain in two sentences why a GPU is faster than a CPU for training neural networks.",
    "Write a tweet complaining about a delayed flight, then rewrite it to sound polite.",
    "What is the difference between precision and recall? Give a spam-filter example.",
    "Summarise the plot of Romeo and Juliet for a ten-year-old.",
    "List three ways to reduce GPU memory use when fine-tuning BERT.",
    "Is the sentence 'great, my phone died again' positive or negative? Explain.",
    "Write a haiku about batch sizes.",
    "Give me a one-paragraph pitch for a coffee shop run by robots.",
]

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["hf", "vllm"], required=True)
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--requests", type=int, default=96, help="requests per configuration (divisible by 32)")
p.add_argument("--new-tokens", type=int, default=128)
p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
p.add_argument("--dtype", default="bfloat16")
p.add_argument("--tag", default="", help="suffix for the results file, e.g. 'hw45'")
p.add_argument("--seed", type=int, default=1384)
a = p.parse_args()

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model)
chat = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        for q in MY_PROMPTS]
prompts = [chat[i % len(chat)] for i in range(a.requests)]


def summarise(name, batch, latencies, tokens, wall):
    lat_ms = np.array(latencies) * 1000
    r = dict(config=name, batch_size=batch, requests=len(latencies), generated_tokens=int(tokens),
             wall_s=round(wall, 3), tokens_per_s=round(tokens / wall, 1),
             p50_ms=round(float(np.percentile(lat_ms, 50)), 1), p95_ms=round(float(np.percentile(lat_ms, 95)), 1))
    print(f"{name:>22}: {r['tokens_per_s']:>8} tok/s   p50 {r['p50_ms']:>8} ms   p95 {r['p95_ms']:>8} ms")
    return r


rows, samples = [], []
if a.backend == "hf":
    from transformers import AutoModelForCausalLM, pipeline

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).cuda()
    model = model.to(getattr(torch, a.dtype))  # in case the loader ignored the dtype kwarg
    pipe = pipeline("text-generation", model=model, tokenizer=tok, device=0)
    gen = dict(max_new_tokens=a.new_tokens, min_new_tokens=a.new_tokens, do_sample=False, return_full_text=False)
    for pr in prompts[:2]:  # warm-up (CUDA context, kernels)
        pipe(pr, **gen)
    lat, toks = [], 0
    t_all = time.perf_counter()
    for pr in prompts:
        t = time.perf_counter()
        out = pipe(pr, **gen)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t)
        toks += a.new_tokens  # min_new_tokens == max_new_tokens forces exactly this many
    rows.append(summarise("HF pipeline, batch 1", 1, lat, toks, time.perf_counter() - t_all))
    samples = [dict(prompt=MY_PROMPTS[i], output=pipe(chat[i], max_new_tokens=200, do_sample=False,
                                                      return_full_text=False)[0]["generated_text"]) for i in range(3)]
    precision = str(model.dtype)
else:
    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype=a.dtype, seed=a.seed, gpu_memory_utilization=0.85, max_model_len=4096)
    sp = SamplingParams(temperature=0.0, max_tokens=a.new_tokens, ignore_eos=True)
    llm.generate(prompts[:4], sp, use_tqdm=False)  # warm-up / graph capture
    for b in a.batch_sizes:
        lat, toks = [], 0
        t_all = time.perf_counter()
        for i in range(0, len(prompts), b):
            chunk = prompts[i:i + b]
            t = time.perf_counter()
            outs = llm.generate(chunk, sp, use_tqdm=False)
            dt = time.perf_counter() - t
            lat += [dt] * len(chunk)
            toks += sum(len(o.outputs[0].token_ids) for o in outs)
        rows.append(summarise(f"vLLM, batch {b}", b, lat, toks, time.perf_counter() - t_all))
    outs = llm.generate(chat, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200, seed=a.seed), use_tqdm=False)
    samples = [dict(prompt=q, output=o.outputs[0].text) for q, o in zip(MY_PROMPTS, outs)]
    precision = a.dtype

print("\nSample generations:")
for s in samples[:3]:
    print(f"> {s['prompt']}\n{s['output'].strip()}\n")

suffix = f"_{a.tag}" if a.tag else ""
save_json(dict(env=env_info(), backend=a.backend, model=a.model, precision=precision, new_tokens=a.new_tokens,
               rows=rows, samples=samples), f"serve_{a.backend}_{gpu_tag()}{suffix}.json")
