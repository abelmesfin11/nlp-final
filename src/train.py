"""Fine-tune a Qwen model for SST-2 sentiment with full FT, LoRA, or DoRA.

One run = one (method, noise rate, seed) combination. During training we
periodically measure:
  * sst2_acc   - accuracy on the clean SST-2 validation set (our test set)
  * mem_rate   - on a fixed sample of CORRUPTED training examples, the fraction
                 where the model predicts the wrong label we gave it
                 (= it has memorized the noise)
  * fix_rate   - on those same corrupted examples, the fraction where the model
                 predicts the TRUE label (it "ignored" the noise)
  * clean_fit  - accuracy on a sample of correctly-labeled training examples
and at the end of every epoch the IMDb accuracy.

At the end we save per-example predictions for SST-2, IMDb and the whole
training set so you can do error analysis without re-running the model.

Example:
  python src/train.py --method lora --noise 0.3 --seed 1
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from data import get_datasets

DEFAULT_LR = {"full": 1e-5, "lora": 2e-4, "dora": 2e-4}
QWEN_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["full", "lora", "dora"], required=True)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1, help="training seed (init + data order)")
    p.add_argument("--data_seed", type=int, default=0, help="which examples get corrupted")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_imdb", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None, help="default depends on method")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--imdb_max_len", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--evals_per_epoch", type=int, default=4)
    p.add_argument("--track_n", type=int, default=1000,
                   help="how many corrupted / clean train examples to track during training")
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--num_threads", type=int, default=4,
                   help="CPU threads for torch (gpmoo asks for 1-4 on GPU jobs)")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--cache_dir", default="data")
    p.add_argument("--out_root", default="runs")
    # smoke-test helpers
    p.add_argument("--max_steps", type=int, default=None, help="stop early (debugging)")
    p.add_argument("--limit_eval", type=int, default=None, help="cap eval set sizes (debugging)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device_and_precision():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return dev, dtype
    if torch.backends.mps.is_available():
        return torch.device("mps"), None
    return torch.device("cpu"), None


def build_model(args, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, dtype=torch.float32
    )
    # The classification head reads the hidden state of the last non-pad token,
    # so the model must know which id is padding.
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.method in ("lora", "dora"):
        from peft import LoraConfig, TaskType, get_peft_model

        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # also makes the new "score" head trainable
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=QWEN_LINEAR,
            use_dora=(args.method == "dora"),
        )
        model = get_peft_model(model, cfg)

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def make_loader(ds, tokenizer, max_len, batch_size, shuffle, seed=0):
    texts, labels = list(ds["text"]), list(ds["label"])

    def collate(batch_idx):
        enc = tokenizer([texts[i] for i in batch_idx], truncation=True,
                        max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in batch_idx])
        return enc

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(list(range(len(texts))), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=g)


@torch.no_grad()
def predict(model, ds, tokenizer, max_len, batch_size, device, amp_dtype):
    """Returns (pred labels, P(positive)) for every example in ds, in order."""
    model.eval()
    # sort by length so batches have little padding (big speedup for IMDb)
    order = np.argsort([len(t) for t in ds["text"]])
    sorted_ds = ds.select(order.tolist())
    loader = make_loader(sorted_ds, tokenizer, max_len, batch_size, shuffle=False)
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    probs_sorted = np.concatenate(probs)
    p_pos = np.empty_like(probs_sorted)
    p_pos[order] = probs_sorted
    model.train()
    return (p_pos > 0.5).astype(int), p_pos


def main():
    args = parse_args()
    args.lr = args.lr or DEFAULT_LR[args.method]
    set_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device, amp_dtype = get_device_and_precision()

    run_name = f"{args.method}_noise{args.noise:.2f}_seed{args.seed}"
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"== {run_name} on {device} (amp={amp_dtype}) -> {out_dir}")

    # ---------------- data ----------------
    train, sst2, imdb = get_datasets(args.cache_dir, args.n_train, args.noise,
                                     args.data_seed, args.n_imdb)
    if args.limit_eval:
        sst2 = sst2.select(range(min(args.limit_eval, len(sst2))))
        imdb = imdb.shuffle(seed=0).select(range(min(args.limit_eval, len(imdb))))

    rng = np.random.default_rng(123)
    corrupted_idx = np.where(np.array(train["is_corrupted"]))[0]
    clean_idx = np.where(~np.array(train["is_corrupted"]))[0]
    track_corr = train.select(rng.choice(corrupted_idx, min(args.track_n, len(corrupted_idx)),
                                         replace=False).tolist()) if len(corrupted_idx) else None
    track_clean = train.select(rng.choice(clean_idx, min(args.track_n, len(clean_idx)),
                                          replace=False).tolist())

    # ---------------- model ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = build_model(args, tokenizer).to(device)
    n_trainable, n_total = count_params(model)
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.3f}%)")

    loader = make_loader(train, tokenizer, args.max_len, args.batch_size, shuffle=True,
                         seed=args.seed)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    history = []

    def evaluate(step, epoch_float, with_imdb):
        rec = {"step": step, "epoch": round(epoch_float, 3)}
        pred, _ = predict(model, sst2, tokenizer, args.max_len, args.eval_batch_size, device, amp_dtype)
        rec["sst2_acc"] = float(np.mean(pred == np.array(sst2["label"])))
        if track_corr is not None:
            pred, _ = predict(model, track_corr, tokenizer, args.max_len, args.eval_batch_size, device, amp_dtype)
            rec["mem_rate"] = float(np.mean(pred == np.array(track_corr["label"])))
            rec["fix_rate"] = float(np.mean(pred == np.array(track_corr["true_label"])))
        pred, _ = predict(model, track_clean, tokenizer, args.max_len, args.eval_batch_size, device, amp_dtype)
        rec["clean_fit"] = float(np.mean(pred == np.array(track_clean["label"])))
        if with_imdb:
            pred, _ = predict(model, imdb, tokenizer, args.imdb_max_len, args.eval_batch_size // 4 or 1,
                              device, amp_dtype)
            rec["imdb_acc"] = float(np.mean(pred == np.array(imdb["label"])))
        history.append(rec)
        print("   eval", json.dumps(rec))
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    # ---------------- train ----------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    evaluate(0, 0.0, with_imdb=True)  # zero-shot head = random baseline sanity check
    model.train()
    step, running, n_run = 0, 0.0, 0
    done = False
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum
            n_run += 1
            if (i + 1) % args.grad_accum != 0:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % 50 == 0:
                print(f"step {step}/{total_steps}  loss {running / n_run:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s")
                running, n_run = 0.0, 0
            end_of_epoch = (i + 1) >= len(loader) - (len(loader) % args.grad_accum)
            if step % eval_every == 0 and not end_of_epoch:
                evaluate(step, step / steps_per_epoch, with_imdb=False)
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        evaluate(step, step / steps_per_epoch, with_imdb=True)
        if done:
            break
    train_time = time.time() - t0

    # ---------------- final predictions for error analysis ----------------
    print("saving per-example predictions ...")
    pred, prob = predict(model, sst2, tokenizer, args.max_len, args.eval_batch_size, device, amp_dtype)
    pd.DataFrame({"uid": sst2["uid"], "text": sst2["text"], "label": sst2["label"],
                  "pred": pred, "p_pos": prob}).to_csv(os.path.join(out_dir, "preds_sst2.csv"), index=False)
    pred, prob = predict(model, imdb, tokenizer, args.imdb_max_len, args.eval_batch_size // 4 or 1,
                         device, amp_dtype)
    pd.DataFrame({"uid": imdb["uid"], "text": imdb["text"], "label": imdb["label"],
                  "pred": pred, "p_pos": prob}).to_csv(os.path.join(out_dir, "preds_imdb.csv"), index=False)
    train_eval = train if not args.limit_eval else train.select(range(min(args.limit_eval * 4, len(train))))
    pred, prob = predict(model, train_eval, tokenizer, args.max_len, args.eval_batch_size, device, amp_dtype)
    pd.DataFrame({"uid": train_eval["uid"], "text": train_eval["text"], "label": train_eval["label"],
                  "true_label": train_eval["true_label"], "is_corrupted": train_eval["is_corrupted"],
                  "pred": pred, "p_pos": prob}).to_csv(os.path.join(out_dir, "preds_train.csv"), index=False)

    final = history[-1]
    results = {
        "run": run_name,
        "config": vars(args),
        "trainable_params": n_trainable,
        "total_params": n_total,
        "train_time_sec": round(train_time, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "final": final,
        "best_sst2_during_training": max(h["sst2_acc"] for h in history),
        "history": history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    if args.save_model:
        model.save_pretrained(os.path.join(out_dir, "model"))
        tokenizer.save_pretrained(os.path.join(out_dir, "model"))
    print("FINAL", json.dumps(final))


if __name__ == "__main__":
    main()
