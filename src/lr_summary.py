"""Summarize the learning-rate sanity check (seed 1).

Combines runs_lr/lr*/<run>/results.json with the default-LR seed-1 runs in runs/
and writes analysis/lr_sweep.md.

Usage:  python src/lr_summary.py
"""

import glob
import json
import os

NAMES = {"full": "Full FT", "lora": "LoRA", "dora": "DoRA"}

rows = []
paths = glob.glob("runs_lr/lr*/*_seed1/results.json") + glob.glob("runs/*_seed1/results.json")
for p in paths:
    r = json.load(open(p))
    c, f = r["config"], r["final"]
    rows.append((c["method"], c["lr"], c["noise"], f["sst2_acc"], f.get("imdb_acc"), f.get("mem_rate"),
                 r["best_sst2_during_training"]))

lines = ["# Learning-rate sanity check (seed 1)", "",
         "Default LRs used in the main grid: Full FT 1e-5, LoRA/DoRA 2e-4 (marked *).", "",
         "| Method | LR | Clean SST-2 | Noisy SST-2 | Noisy IMDb | Memorization (noisy) | Best noisy SST-2 (oracle) |",
         "|---|---|---|---|---|---|---|"]
default = {"full": 1e-5, "lora": 2e-4, "dora": 2e-4}
for m in ["full", "lora", "dora"]:
    for lr in sorted({r[1] for r in rows if r[0] == m}):
        clean = [r for r in rows if r[0] == m and r[1] == lr and r[2] == 0.0]
        noisy = [r for r in rows if r[0] == m and r[1] == lr and r[2] > 0]
        pct = lambda x: f"{100 * x:.1f}" if x is not None else "–"
        lines.append(f"| {NAMES[m]} | {lr:g}{'*' if abs(lr - default[m]) < 1e-12 else ''} | "
                     f"{pct(clean[0][3]) if clean else '–'} | {pct(noisy[0][3]) if noisy else '–'} | "
                     f"{pct(noisy[0][4]) if noisy else '–'} | {pct(noisy[0][5]) if noisy else '–'} | "
                     f"{pct(noisy[0][6]) if noisy else '–'} |")
md = "\n".join(lines) + "\n"
os.makedirs("analysis", exist_ok=True)
open("analysis/lr_sweep.md", "w").write(md)
print(md)
