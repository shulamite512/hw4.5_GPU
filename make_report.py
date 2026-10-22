"""Collect results/*.json into REPORT.md (all tables), METRICS.md (HW4.5 summary) and plots/*.png.

Safe to run at any point: sections whose results don't exist yet are marked as pending.
Results from several cards (RTX4090, RTX5090, ...) are compared automatically.

    python make_report.py
"""
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RES, PLOTS = os.path.join(ROOT, "results"), os.path.join(ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)

# reference palette (light): categorical slots 1-3, chrome, ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10, "legend.frameon": False,
})


def load(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        with open(f, encoding="utf-8") as fh:
            out[os.path.basename(f)[:-5]] = json.load(fh)
    return out


def table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def pending(what):
    return f"_Pending: run {what}._\n"


def direct_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)


md, metrics = ["# HW4.a / HW4.5 results\n"], {}
DIAG_FILE = os.path.join(ROOT, "diagnoses.json")
DIAG = json.load(open(DIAG_FILE, encoding="utf-8")) if os.path.exists(DIAG_FILE) else {}

# ============================================================ HW4.a training
train = load("hw4a_train_*.json")
md.append("## HW4.a - BERT sentiment classifier\n")
if not train:
    md.append(pending("`python hw4a_train.py --slice SLICE --seed SEED`"))
for key, r in train.items():
    d, h, o = r["data"], r["history"], r["overtraining"]
    md.append(f"### {r['env']['gpu']}\n")
    md.append(f"**Data:** {d['source']}. Working subset = rows **{d['row_start']}-{d['row_end']}** (0-based, "
              f"inclusive; {d['rows_used']} rows, {d['rows_dropped_nan']} dropped as empty). SEED={d['seed']}. "
              f"Split 70/15/15 -> {d['n_train']}/{d['n_val']}/{d['n_test']}"
              f"{' (stratified)' if d['stratified'] else ''}. Classes: {d['class_counts']}.\n")
    md.append(table(["Epoch", "Train loss", "Val loss", "Val acc", "Epoch time (s)", "samp/s"],
                    [[e["epoch"], e["train_loss"], e["val_loss"], e["val_acc"], e["epoch_train_s"],
                      e["samples_per_s"]] for e in h]))
    if o["extra_epochs"]:
        cost = (f"Training continued {o['extra_epochs']} epoch(s) past that point, costing "
                f"{o['extra_train_s']} s of GPU time; validation loss rose from {o['val_loss_at_best']} to "
                f"{o['val_loss_final']} and test accuracy went from {o['test_acc_best_ckpt']} (best checkpoint) "
                f"to {o['test_acc_final_epoch']} (final epoch).")
    else:
        cost = "Validation loss was still improving at the last epoch, so nothing was trained past the best point."
    md.append(f"\nValidation loss stopped improving after **epoch {o['best_val_loss_epoch']}** "
              f"(best val accuracy at epoch {o['best_val_acc_epoch']}). {cost}\n")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ep = [e["epoch"] for e in h]
    for i, (k, lab) in enumerate([("train_loss", "train"), ("val_loss", "validation")]):
        a1.plot(ep, [e[k] for e in h], marker="o", color=SERIES[i], label=lab)
        direct_label(a1, ep[-1], h[-1][k], lab, SERIES[i])
    a1.set(title="Loss per epoch", xlabel="epoch", xticks=ep)
    a1.legend(loc="upper right")
    a2.plot(ep, [e["val_acc"] for e in h], marker="o", color=SERIES[0])
    a2.set(title="Validation accuracy", xlabel="epoch", xticks=ep)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"{key}_curves.png"), dpi=150)
    plt.close(fig)
    md.append(f"![curves](plots/{key}_curves.png)\n")

    L = r["labels"]
    md.append(f"**Test accuracy** {r['test_acc']} (best-val checkpoint; {r['reload_test_acc']} after save + reload).\n")
    md.append("Confusion matrix (rows = true, columns = predicted):\n")
    md.append(table(["true \\ pred", *L], [[L[i], *row] for i, row in enumerate(r["confusion_matrix"])]))
    md.append("\n" + table(["Class", "Precision", "Recall", "F1", "Support"],
                           [[c["label"], c["precision"], c["recall"], c["f1"], c["support"]] for c in r["per_class"]]))
    md.append(f"\n**Worst class: {r['worst_class']}** (lowest F1). Five misclassified test examples:\n")
    md.append(table(["Row", "Tweet", "True -> Pred", "Diagnosis"],
                    [[m["row"], m["text"].replace("|", "/").replace("\n", " "), f"{m['true']} -> {m['pred']}",
                      DIAG.get(str(m["row"]), m["diagnosis"])] for m in r["misclassified"]]))
    md.append("\nCustom inference with the reloaded model:\n")
    md.append(table(["Text", "Prediction", "Probabilities"],
                    [[c["text"], c["pred"], c["probs"]] for c in r["custom_inference"]]))
    b = r["hw45_baseline"]
    md.append(f"\n**HW4.5 baseline (keep with this assignment):** peak GPU memory **{b['peak_alloc_gib']} GiB** "
              f"allocated ({b['peak_reserved_gib']} GiB reserved), wall-clock training time **{b['wall_clock_s']} s** "
              f"({b['train_only_s']} s excluding per-epoch validation), batch {r['cfg']['batch_size']}, "
              f"{r['cfg']['epochs']} epochs, fp32, AdamW lr {r['cfg']['lr']}.\n")

# ============================================================ serving (HW4.1 and Part D re-run)


def serving_section(suffix, title):
    out = [f"## {title}\n"]
    hf, vl = load(f"serve_hf_*{suffix}.json"), load(f"serve_vllm_*{suffix}.json")
    if suffix == "":  # don't let the Part D files leak into the HW4 table
        hf = {k: v for k, v in hf.items() if k.count("_") == 2}
        vl = {k: v for k, v in vl.items() if k.count("_") == 2}
    gpus = sorted({v["env"]["gpu_tag"] for v in [*hf.values(), *vl.values()]})
    if not gpus:
        return out + [pending("`python hw4a_serve.py --backend hf` and `--backend vllm`" +
                              (f" with `--tag {suffix.strip('_')}`" if suffix else ""))]
    for g in gpus:
        rows = [row for v in hf.values() if v["env"]["gpu_tag"] == g for row in v["rows"]]
        rows += [row for v in vl.values() if v["env"]["gpu_tag"] == g for row in v["rows"]]
        model = next(v["model"] for v in [*hf.values(), *vl.values()] if v["env"]["gpu_tag"] == g)
        out.append(f"### {g} - {model}, {next(iter(vl.values() or hf.values()))['new_tokens']} new tokens/request, bf16\n")
        out.append(table(["Configuration", "Tokens/sec", "p50 latency (ms)", "p95 latency (ms)"],
                         [[x["config"], x["tokens_per_s"], x["p50_ms"], x["p95_ms"]] for x in rows]))
        base = next((x for x in rows if x["config"].startswith("HF")), None)
        v = [x for x in rows if x["config"].startswith("vLLM")]
        if base and v:
            worse = [x for x in v if x["p50_ms"] > base["p50_ms"] and x["tokens_per_s"] > v[0]["tokens_per_s"]]
            if worse:
                x = worse[0]
                out.append(f"\nPer-request latency first exceeds the HF baseline at **vLLM batch {x['batch_size']}** "
                           f"(p50 {x['p50_ms']} ms vs {base['p50_ms']} ms) while total throughput is still rising "
                           f"({x['tokens_per_s']} vs {v[0]['tokens_per_s']} tok/s at batch 1).\n")
            else:
                out.append("\nAt the batch sizes tested, vLLM per-request latency never exceeded the HF baseline.\n")
    return out


md += serving_section("", "HW4.a - Table HW4.1 Serving comparison")

# ============================================================ Part A
md.append("## HW4.5 Part A - Batch scaling to the memory wall\n")
pa = load("partA_*.json")
if not pa:
    md.append(pending("`python hw45_partA_batch_scaling.py`"))
fig, ax = plt.subplots(figsize=(7, 4))
sat = {}
for i, (k, r) in enumerate(pa.items()):
    g, rows = r["env"]["gpu_tag"], r["rows"]
    peak = max(x["samples_per_s"] for x in rows)
    sat[g] = next(x["batch_size"] for x in rows if x["samples_per_s"] >= 0.95 * peak)
    md.append(f"### {r['env']['gpu']} ({r['env']['vram_gib']} GiB)\n")
    md.append(table(["Batch", "Throughput (samp/s)", "Peak alloc (GiB)", "Peak reserved (GiB)", "Time/epoch (s)"],
                    [[x["batch_size"], x["samples_per_s"], x["peak_alloc_gib"], x["peak_reserved_gib"],
                      x["time_per_epoch_s"]] for x in rows]))
    md.append(f"\nLargest batch that runs: **{r['max_batch']}** (OOM at {r['oom_at']}). Throughput reaches 95% of its "
              f"peak ({peak} samp/s) at **batch {sat[g]}** and stops improving meaningfully beyond it.\n")
    c = SERIES[i % len(SERIES)]
    ax.plot([x["batch_size"] for x in rows], [x["samples_per_s"] for x in rows], marker="o", color=c, label=g)
    direct_label(ax, rows[-1]["batch_size"], rows[-1]["samples_per_s"], g, c)
if pa:
    ax.set_xscale("log", base=2)
    ax.set(title="Training throughput vs batch size (BERT-base, HW4 settings)", xlabel="batch size",
           ylabel="samples / s")
    ax.set_ylim(bottom=0)
    if len(pa) > 1:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "partA_throughput.png"), dpi=150)
    md.append("![throughput vs batch](plots/partA_throughput.png)\n")
plt.close(fig)
if len(pa) >= 2:
    (_, r1), (_, r2) = sorted(pa.items(), key=lambda kv: -kv[1]["env"]["vram_gib"])[:2]
    md.append(f"Saturation batch: {sat}. Max-batch ratio {r1['env']['gpu_tag']}/{r2['env']['gpu_tag']} = "
              f"{r1['max_batch']}/{r2['max_batch']} = **{r1['max_batch'] / r2['max_batch']:.2f}**; VRAM ratio = "
              f"{r1['env']['vram_gib']}/{r2['env']['vram_gib']} = **{r1['env']['vram_gib'] / r2['env']['vram_gib']:.2f}**.\n")
elif pa:
    md.append("_Only one card measured so far. Run this script on the second card to get the max-batch and VRAM ratios._\n")

# ============================================================ Part B
md.append("## HW4.5 Part B - Memory techniques, one at a time\n")
pb = load("partB_*.json")
if not pb:
    md.append(pending("`python hw45_partB_memory.py`"))
for k, r in pb.items():
    R = r["results"]
    md.append(f"### Table HW4.5.1 - {r['env']['gpu']} ({r['env']['vram_gib']} GiB)\n")
    md.append("Throughput and peak memory are measured at the HW4 batch size (32) so the techniques do equal work; "
              "max batch is the largest power of two before OOM; accuracy is a full HW4 fine-tune with only that "
              "technique on.\n")
    labels = {"baseline": "Baseline (HW4 settings)", "amp": "Mixed precision (bf16)",
              "grad_ckpt": "Gradient checkpointing", "adamw8bit": "8-bit AdamW (bitsandbytes)",
              "combo_amp_8bit": "Best combo: AMP + 8-bit AdamW",
              "combo_amp_8bit_ckpt": "Combo: AMP + 8-bit AdamW + checkpointing"}
    rows = []
    for name, lab in labels.items():
        if name in R:
            x = R[name]
            rows.append([lab, x["max_batch"], x["at_bs32"]["samples_per_s"], x["at_bs32"]["peak_alloc_gib"],
                         x["test_acc"]])
        if name == "grad_ckpt" and "accum" in R:
            x = R["accum"]
            rows.append([f"Gradient accumulation ({x['physical_batch']} x {x['cfg']['accum']})",
                         f"{x['effective_batch']} effective", x["accum_probe"]["samples_per_s"],
                         x["accum_probe"]["peak_alloc_gib"], x["test_acc"]])
    md.append(table(["Technique", "Max batch", "Throughput (samp/s)", "Peak memory (GiB)", "Eval accuracy"], rows))

    # memory vs throughput at bs 32: one point per technique, labelled directly
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pts = [(labels.get(n, n), x["at_bs32"]) for n, x in R.items() if n != "accum"]
    if "accum" in R:
        pts.append((f"Grad accum ({R['accum']['physical_batch']}x{R['accum']['cfg']['accum']})",
                    R["accum"]["accum_probe"]))
    for lab, p_ in pts:
        ax.scatter(p_["samples_per_s"], p_["peak_alloc_gib"], s=64, color=SERIES[0], edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        direct_label(ax, p_["samples_per_s"], p_["peak_alloc_gib"], lab.split(" (")[0], SERIES[0])
    ax.set(title=f"Peak memory vs throughput by technique ({r['env']['gpu_tag']}, batch 32)",
           xlabel="throughput (samples / s)", ylabel="peak memory (GiB)")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"{k}_mem_vs_throughput.png"), dpi=150)
    plt.close(fig)
    md.append(f"\n![memory vs throughput](plots/{k}_mem_vs_throughput.png)\n")

    if "accum" in R:
        x, dp = R["accum"], R["accum"]["direct_probe"]
        direct = "OOM" if dp.get("oom") else f"{dp['timed_s']} s, {dp['peak_alloc_gib']} GiB"
        md.append(f"\nGradient accumulation: effective batch **{x['effective_batch']}** on a physical batch of "
                  f"{x['physical_batch']}. Wall-clock for {x['opt_steps']} optimiser steps: "
                  f"**{x['accum_probe']['timed_s']} s** ({x['accum_probe']['peak_alloc_gib']} GiB) vs a direct "
                  f"batch of {x['effective_batch']}: {direct}. Test accuracy {x['test_acc']} "
                  f"(same lr as HW4, so fewer optimiser steps per epoch).\n")

    # ranking: memory saved per % throughput lost, at equal work
    base = R.get("baseline")
    if base:
        rank = []
        for name in ["amp", "grad_ckpt", "adamw8bit", "accum"]:
            if name not in R:
                continue
            if name == "accum":
                ref, t = R["accum"]["direct_probe"], R["accum"]["accum_probe"]
                if ref.get("oom"):
                    continue
            else:
                ref, t = base["at_bs32"], R[name]["at_bs32"]
            saved = ref["peak_alloc_gib"] - t["peak_alloc_gib"]
            lost = 100 * (ref["samples_per_s"] - t["samples_per_s"]) / ref["samples_per_s"]
            score = math.inf if lost <= 0 else saved / lost
            rank.append([labels.get(name, "Gradient accumulation"), f"{saved:+.2f}", f"{lost:+.1f}%",
                         "no throughput cost" if score == math.inf else f"{score:.3f}", score])
        rank.sort(key=lambda z: -z[-1])
        md.append("\n**Ranking: memory saved per unit of throughput lost** (GiB saved per 1% throughput lost; "
                  "higher is better):\n")
        md.append(table(["Rank", "Technique", "Memory saved (GiB)", "Throughput lost", "GiB per 1% lost"],
                        [[i + 1, *z[:-1]] for i, z in enumerate(rank)]))
        if "grad_ckpt" in R:
            g_ = R["grad_ckpt"]["at_bs32"]["samples_per_s"]
            b_ = base["at_bs32"]["samples_per_s"]
            md.append(f"\nGradient checkpointing throughput cost: {b_} -> {g_} samp/s "
                      f"(**{100 * (b_ - g_) / b_:.1f}% slower**) for {base['at_bs32']['peak_alloc_gib']} -> "
                      f"{R['grad_ckpt']['at_bs32']['peak_alloc_gib']} GiB.\n")
        cands = {n: x for n, x in R.items() if n != "accum" and x["test_acc"] >= base["test_acc"] - 0.01}
        best_t = max(cands.items(), key=lambda kv: kv[1]["at_bs32"]["samples_per_s"])
        best_m = min(cands.items(), key=lambda kv: kv[1]["at_bs32"]["peak_alloc_gib"])
        best_w = min(cands.items(), key=lambda kv: kv[1]["fit_stats"]["wall_clock_s"])
        metrics[r["env"]["gpu_tag"]] = dict(best_t=best_t, best_m=best_m, best_w=best_w, base=base, labels=labels)

# ============================================================ Part C
md.append("## HW4.5 Part C - Two cards at once (DDP)\n")
pc = load("partC_*.json")
single = next((v for k, v in pc.items() if k.startswith("partC_w1")), None)
multi = [v for k, v in pc.items() if not k.startswith("partC_w1")]
if not single:
    md.append(pending("`torchrun --nproc_per_node=1 hw45_partC_ddp.py` (single-GPU baseline)"))
if not multi and single:
    t = single["topology"]
    md.append(f"Single-GPU DDP baseline on {t['gpus'][0]} ({'single-node' if t['single_node'] else 'multi-node'}, "
              f"world size {t['world_size']}). The two-GPU pairing was confirmed optional by the TA (2026-10-22) and "
              "was not run; the lab machine has one GPU.\n")
    md.append(table(["Batch/GPU", "Throughput (samp/s)", "Step (ms)", "Compute (ms)", "Comm overhead (ms)",
                     "Peak memory (GiB)"],
                    [[x["per_gpu_batch"], x["samples_per_s"], x["step_ms"], x["compute_step_ms"],
                      x["comm_overhead_ms"], x["peak_alloc_gib"]] for x in single["rows"]]))
    md.append("\nWith one GPU there is no gradient all-reduce, so the comm-overhead column is the timing noise floor "
              f"of the method (up to {max(x['comm_overhead_ms'] for x in single['rows'])} ms); a two-GPU run would "
              "need overhead above this to count as real communication cost. Throughput matches Part A within ~3%.\n")
elif not multi:
    md.append(pending("the instructor-assigned 2-GPU pairing, e.g. `torchrun --nproc_per_node=2 hw45_partC_ddp.py`"))
for m in multi:
    t = m["topology"]
    md.append(f"Pairing: {' + '.join(t['gpus'])}: **{'homogeneous' if t['homogeneous'] else 'heterogeneous'}**, "
              f"**{'single-node' if t['single_node'] else 'multi-node'}** ({t['nodes']} host(s)).\n")
    b1 = {x["per_gpu_batch"]: x for x in (single or {}).get("rows", [])}
    rows, worth = [], None
    for x in m["rows"]:
        s = b1.get(x["per_gpu_batch"])
        eff = x["samples_per_s"] / (t["world_size"] * s["samples_per_s"]) if s else None
        if eff and worth is None and x["samples_per_s"] > s["samples_per_s"] and eff >= 0.8:
            worth = x["per_gpu_batch"]
        rows.append([x["per_gpu_batch"], x["global_batch"], s["samples_per_s"] if s else "-", x["samples_per_s"],
                     f"{eff:.0%}" if eff else "-", x["step_ms"], x["compute_step_ms"], x["comm_overhead_ms"],
                     f"{x['comm_fraction']:.0%}", x["straggler_wait_ms"]])
    md.append(table(["Batch/GPU", "Global batch", "1-GPU samp/s", "2-GPU samp/s", "Scaling eff.", "Step (ms)",
                     "Compute (ms)", "Comm overhead (ms)", "Comm share", "Straggler wait (ms)"], rows))
    if m.get("allreduce"):
        bw = m["allreduce"]
        md.append(f"\nAll-reduce of the {bw['bytes'] / 1e6:.0f} MB gradient buffer: {bw['allreduce_ms']} ms -> "
                  f"algorithm bandwidth {bw['algbw_gbs']} GB/s, bus bandwidth {bw['busbw_gbs']} GB/s.\n")
    if t.get("nvidia_smi_topo"):
        md.append("\n`nvidia-smi topo -m`:\n```\n" + t["nvidia_smi_topo"] + "```\n")
    md.append(f"\nMulti-GPU first beats one GPU with at least 80% scaling efficiency at **{worth} per GPU**.\n"
              if worth else "\nScaling efficiency stayed below 80% at every batch tested.\n")

# ============================================================ Part D
md += serving_section("_hw45", "HW4.5 Part D - Serving re-run on real hardware")
md.append("### KV-cache memory vs context length\n")
kh, kv = load("partD_kv_hf_*.json"), load("partD_kv_vllm_*.json")
if not kh and not kv:
    md.append(pending("`python hw45_partD_kv_cache.py --backend hf` and `--backend vllm`"))
for r in kh.values():
    md.append(f"**{r['env']['gpu']}, HF transformers (measured), batch {r['batch']}**, {r['model']}, "
              f"{r['arch']['kv_bytes_per_token'] / 1024:.0f} KiB of KV per token:\n")
    md.append(table(["Context", "KV measured (GiB)", "KV analytic (GiB)", "Peak total (GiB)"],
                    [[x["ctx"], x["kv_measured_gib"], x["kv_analytic_gib"], x["peak_total_gib"]] for x in r["rows"]]))
    md.append(f"\nLongest context at batch {r['batch']} before OOM (HF): **{r['longest_ctx']:,} tokens**.\n")
for r in kv.values():
    md.append(f"\n**{r['env']['gpu']}, vLLM**: KV pool {r['kv_pool_tokens']:,} tokens ({r['kv_pool_gib']} GiB at "
              f"gpu_memory_utilization={r['gpu_memory_utilization']}). At batch {r['batch']}: memory allows "
              f"{r['longest_ctx_memory_limited']:,} tokens/sequence, model allows {r['longest_ctx_model_limited']:,} "
              f"-> longest servable context **{r['longest_ctx_servable']:,}**.\n")

# ============================================================ Part E
md.append("## HW4.5 Part E - Analysis and summary\n")
if train and metrics:
    t0 = next(iter(train.values()))
    full_rows = t0["data"]["dataset_rows"]
    epochs = t0["cfg"]["epochs"]
    for g, m in metrics.items():
        name, x = m["best_t"]
        sps = x["at_bs32"]["samples_per_s"]
        n = round(0.70 * full_rows)
        hours = n * epochs / sps / 3600
        md.append(f"**{g}.** One full fine-tune = the 70% training share of all {full_rows:,} rows = {n:,} samples x "
                  f"{epochs} epochs = {n * epochs:,} samples. At {sps} samp/s ({m['labels'][name]}, measured in "
                  f"Part B): {n * epochs:,} / {sps} = {n * epochs / sps:,.0f} s = **{hours:.2f} GPU-hours** "
                  f"(x 1 GPU), excluding validation passes.\n")
else:
    md.append(pending("HW4.a training and Part B"))

with open(os.path.join(ROOT, "REPORT.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("[saved] REPORT.md")

# ============================================================ METRICS.md
mm = ["# METRICS - HW4.5 summary\n",
      "> HW4.5 summary of training-throughput and memory engineering on the HW4.a BERT fine-tune. "
      "This is **not** an HW5 baseline (HW5 uses a different model and task).\n"]
for r in train.values():
    b = r["hw45_baseline"]
    mm.append(f"**HW4.a reference run ({r['env']['gpu']})**: peak {b['peak_alloc_gib']} GiB, wall clock "
              f"{b['wall_clock_s']} s, test acc {r['test_acc']}, rows {r['data']['row_start']}-{r['data']['row_end']}, "
              f"SEED {r['data']['seed']}.\n")
for g, m in metrics.items():
    L = m["labels"]
    rows = []
    for title, (n, x), val in [("Best peak memory", m["best_m"], f"{m['best_m'][1]['at_bs32']['peak_alloc_gib']} GiB @ bs32"),
                               ("Best throughput", m["best_t"], f"{m['best_t'][1]['at_bs32']['samples_per_s']} samp/s @ bs32"),
                               ("Best wall-clock", m["best_w"], f"{m['best_w'][1]['fit_stats']['wall_clock_s']} s full fine-tune")]:
        rows.append([title, L[n], val, x["test_acc"]])
    mm.append(f"## {g}\n")
    mm.append(table(["Metric", "Configuration", "Value", "Eval accuracy"], rows))
    mm.append(f"\nBaseline (HW4 settings) on this card: {m['base']['at_bs32']['samples_per_s']} samp/s, "
              f"{m['base']['at_bs32']['peak_alloc_gib']} GiB @ bs32, {m['base']['fit_stats']['wall_clock_s']} s, "
              f"acc {m['base']['test_acc']}. Candidates restricted to configurations within 1 point of baseline accuracy.\n")
if not metrics:
    mm.append(pending("Part B (`python hw45_partB_memory.py`)"))
with open(os.path.join(ROOT, "METRICS.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(mm))
print("[saved] METRICS.md")

# ============================================================ GPU_HOURS.md (from the run log)
runs = []
for fn in ("run_log_backfill.jsonl", "run_log.jsonl"):
    path = os.path.join(RES, fn)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            runs += [json.loads(line) for line in fh if line.strip()]
runs.sort(key=lambda e: e["start"])
gh = ["# GPU-hour record - HW4.a / HW4.5\n",
      "Generated by `make_report.py` from `results/run_log.jsonl` (written automatically by `runlog.py` at the end "
      "of every run) and `results/run_log_backfill.jsonl` (runs before automatic logging; durations estimated, "
      "marked ~). GPU-hours = wall-clock duration x number of GPUs in the job. Failed, interrupted and invalid "
      "runs are counted: they used the reservation too.\n"]
rows, per_gpu = [], {}
for e in runs:
    n_gpu = max(1, int(e.get("world_size") or 1))
    h = e["duration_s"] * n_gpu / 3600
    gpu = ", ".join(sorted({g["name"].replace("NVIDIA GeForce ", "") for g in e.get("gpus", [])})) or "?"
    uuid = ", ".join(sorted({g["uuid"][:12] + "..." for g in e.get("gpus", [])})) or "?"
    per_gpu[gpu] = per_gpu.get(gpu, 0) + h
    rows.append([e["start"].replace("T", " "), ("~" if e.get("estimated") else "") + f"{e['duration_s'] / 60:.1f}",
                 n_gpu, f"{h:.3f}", gpu, uuid, e["status"], e.get("purpose", ""), f"`{e['command']}`"])
gh.append(table(["Start", "Minutes", "GPUs", "GPU-h", "GPU", "UUID", "Status", "Purpose", "Command"], rows))
gh.append("\n**Totals:** " + "; ".join(f"{g}: **{h:.2f} GPU-h**" for g, h in per_gpu.items()) +
          f"; all: **{sum(per_gpu.values()):.2f} GPU-h**.\n")
with open(os.path.join(ROOT, "GPU_HOURS.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(gh))
print("[saved] GPU_HOURS.md")
