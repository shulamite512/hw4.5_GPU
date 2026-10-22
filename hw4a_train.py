"""HW4.a part 1 - BERT sentiment classifier on Twitter data.

BertTokenizer preprocessing -> TensorDatasets -> BertForSequenceClassification fine-tune -> per-epoch
train/val loss + val accuracy -> test confusion matrix and per-class precision/recall -> worst class and
five misclassified examples -> save, reload, custom inference. Peak GPU memory and wall-clock time are
recorded as the HW4.5 baseline.

    python hw4a_train.py --slice <SLICE> --seed <SEED> [--epochs 4] [--csv tweets.csv]
"""
import runlog  # noqa: F401  (appends this run to RUN_LOG.txt)
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import BertForSequenceClassification, BertTokenizer

from common import (MODELS, add_data_args, add_train_args, cfg_from_args, env_info, evaluate, fit, gpu_tag,
                    prepare_data, save_json)

p = add_train_args(add_data_args(argparse.ArgumentParser()))
p.add_argument("--predict", nargs="*", default=[
    "just landed and my bag is already at the carousel, best flight ever",
    "three hours on the tarmac and nobody tells us anything",
    "the meeting got moved to thursday",
    "oh great, another monday. exactly what i needed",
])
a = p.parse_args()
cfg = cfg_from_args(a)

data = prepare_data(a)
info = data["info"]
print(f"Data: {info['source']}")
print(f"Rows {info['row_start']}-{info['row_end']} (0-based, inclusive) = {info['rows_used']} rows; "
      f"train/val/test = {info['n_train']}/{info['n_val']}/{info['n_test']}; classes {info['class_counts']}")

# ---- train
model, history, stats, best_state = fit(data, cfg, keep_best=True)
final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

best_ep = min(history, key=lambda h: h["val_loss"])["epoch"]
best_acc_ep = max(history, key=lambda h: h["val_acc"])["epoch"]
_, final_test_acc, _, _ = evaluate(model, data["test_ds"])

# ---- evaluate the best-validation checkpoint on test (this is the model we save)
model.load_state_dict(best_state)
test_loss, test_acc, preds, labels = evaluate(model, data["test_ds"])
names = data["labels"]
cm = confusion_matrix(labels, preds, labels=list(range(len(names))))
prec, rec, f1, sup = precision_recall_fscore_support(labels, preds, labels=list(range(len(names))), zero_division=0)
per_class = [dict(label=n, precision=round(float(p_), 4), recall=round(float(r_), 4), f1=round(float(f_), 4),
                  support=int(s_)) for n, p_, r_, f_, s_ in zip(names, prec, rec, f1, sup)]
worst = int(np.argmin(f1))

test_df = data["test_df"].reset_index(drop=True)
miss_idx = [i for i in range(len(labels)) if labels[i] == worst and preds[i] != worst][:5]
misclassified = [dict(row=int(test_df.loc[i, "row"]), text=test_df.loc[i, "text"], true=names[labels[i]],
                      pred=names[preds[i]], diagnosis="TODO") for i in miss_idx]

after = history[best_ep:]  # epochs trained past the best validation loss
overtrain = dict(
    best_val_loss_epoch=best_ep, best_val_acc_epoch=best_acc_ep, epochs_trained=len(history),
    extra_epochs=len(after),
    extra_train_s=round(sum(h["epoch_train_s"] for h in after), 2),
    val_loss_at_best=history[best_ep - 1]["val_loss"], val_loss_final=history[-1]["val_loss"],
    test_acc_best_ckpt=round(test_acc, 4), test_acc_final_epoch=round(final_test_acc, 4),
)

# ---- save, reload, custom inference
out_dir = os.path.join(MODELS, f"bert_twitter_s{a.slice}_seed{a.seed}")
model.save_pretrained(out_dir)
data["tokenizer"].save_pretrained(out_dir)
del model
torch.cuda.empty_cache()

reloaded = BertForSequenceClassification.from_pretrained(out_dir).cuda().eval()
tok = BertTokenizer.from_pretrained(out_dir)
_, reload_acc, _, _ = evaluate(reloaded, data["test_ds"])
enc = tok(a.predict, padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to("cuda")
with torch.no_grad():
    probs = reloaded(**enc).logits.softmax(-1).cpu().numpy()
custom = [dict(text=t, pred=names[int(pr.argmax())], probs={n: round(float(x), 3) for n, x in zip(names, pr)})
          for t, pr in zip(a.predict, probs)]

# ---- report
print("\nConfusion matrix (rows = true, cols = predicted):", names)
print(cm)
for c in per_class:
    print(f"  {c['label']:>10}: P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f} n={c['support']}")
print(f"Worst class: {names[worst]}")
for m in misclassified:
    print(f"  row {m['row']}: [{m['true']} -> {m['pred']}] {m['text']}")
print(f"Validation loss stopped improving after epoch {best_ep}; {overtrain}")
print(f"Test acc best ckpt {test_acc:.4f}; after reload {reload_acc:.4f}")
print(f"HW4.5 BASELINE: peak alloc {stats['peak_alloc_gib']} GiB, wall clock {stats['wall_clock_s']} s")
for c in custom:
    print(f"  {c['pred']:>9} {c['probs']}  <- {c['text']}")

save_json(dict(env=env_info(), data=info, cfg=vars(cfg),
               history=history, overtraining=overtrain, test_loss=round(test_loss, 4), test_acc=round(test_acc, 4),
               reload_test_acc=round(reload_acc, 4), confusion_matrix=cm.tolist(), labels=names,
               per_class=per_class, worst_class=names[worst], misclassified=misclassified,
               custom_inference=custom, hw45_baseline=stats, saved_model=out_dir),
          f"hw4a_train_{gpu_tag()}.json")
