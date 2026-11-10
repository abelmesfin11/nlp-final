"""Aggregate all runs into tables and figures for the report.

Reads runs/*/results.json and writes to analysis/:
  * summary.csv / summary.md  - mean ± std over seeds per (method, noise)
  * fig_memorization.png      - memorization rate over training (noisy runs)
  * fig_sst2_curve.png        - SST-2 accuracy over training, clean vs noisy
  * fig_final_bars.png        - final SST-2 / IMDb accuracy per method & condition

Usage:  python src/analyze.py --runs runs --out analysis
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical slots 1-3 (blue, orange, aqua). Noise condition is shown
# by line style / hatching too, so color never carries meaning alone.
COLORS = {"full": "#2a78d6", "lora": "#eb6834", "dora": "#1baf7a"}
NAMES = {"full": "Full FT", "lora": "LoRA", "dora": "DoRA"}
METHODS = ["full", "lora", "dora"]
INK, MUTED, GRID = "#1f1f1d", "#6b6a63", "#e1e0d9"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_runs(run_dir):
    rows, curves = [], []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "results.json"))):
        r = json.load(open(path))
        c = r["config"]
        f = r["final"]
        rows.append({
            "method": c["method"], "noise": c["noise"], "seed": c["seed"],
            "sst2_acc": f["sst2_acc"], "imdb_acc": f.get("imdb_acc"),
            "mem_rate": f.get("mem_rate"), "fix_rate": f.get("fix_rate"),
            "clean_fit": f["clean_fit"], "best_sst2": r["best_sst2_during_training"],
            "trainable_params": r["trainable_params"], "train_min": r["train_time_sec"] / 60,
            "peak_gb": r["peak_gpu_mem_gb"],
        })
        for h in r["history"]:
            curves.append({"method": c["method"], "noise": c["noise"], "seed": c["seed"], **h})
    curves = pd.DataFrame(curves)
    # train.py evaluates a couple of steps before AND at each epoch end; snap epochs to the
    # 1/4-epoch grid and keep the later point so each run has one value per grid point.
    curves["epoch"] = (curves.epoch * 4).round() / 4
    curves = curves.groupby(["method", "noise", "seed", "epoch"], as_index=False).last()
    return pd.DataFrame(rows), curves


def summarize(df, out):
    metrics = ["sst2_acc", "imdb_acc", "mem_rate", "fix_rate", "clean_fit", "best_sst2", "train_min"]
    g = df.groupby(["method", "noise"])
    summ = g[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ["n_seeds"] = g.size()
    summ["trainable_params"] = g["trainable_params"].first()
    summ.to_csv(os.path.join(out, "summary.csv"))

    lines = ["| Method | Noise | Seeds | Trainable params | SST-2 acc | IMDb acc | Memorization | Corrected | Best SST-2 (oracle) |",
             "|---|---|---|---|---|---|---|---|---|"]

    def fmt(m, s):
        if m is None or pd.isna(m):
            return "–"
        return f"{100 * m:.1f} ± {100 * (0 if pd.isna(s) else s):.1f}"

    for (method, noise), row in summ.iterrows():
        lines.append(
            f"| {NAMES[method]} | {int(noise * 100)}% | {int(row['n_seeds'])} | {int(row['trainable_params']):,} | "
            f"{fmt(row['sst2_acc_mean'], row['sst2_acc_std'])} | "
            f"{fmt(row['imdb_acc_mean'], row['imdb_acc_std'])} | "
            f"{fmt(row['mem_rate_mean'], row['mem_rate_std']) if noise > 0 else '–'} | "
            f"{fmt(row['fix_rate_mean'], row['fix_rate_std']) if noise > 0 else '–'} | "
            f"{fmt(row['best_sst2_mean'], row['best_sst2_std'])} |")

    # robustness gap: how much accuracy each method loses going from clean to noisy
    lines += ["", "**Accuracy drop from clean → noisy (percentage points, mean over seeds)**", "",
              "| Method | SST-2 drop | IMDb drop |", "|---|---|---|"]
    means = df.groupby(["method", "noise"])[["sst2_acc", "imdb_acc"]].mean()
    for m in METHODS:
        try:
            clean, noisy = means.loc[(m, 0.0)], means.loc[(m, max(df.noise))]
            lines.append(f"| {NAMES[m]} | {100 * (clean.sst2_acc - noisy.sst2_acc):.1f} | "
                         f"{100 * (clean.imdb_acc - noisy.imdb_acc):.1f} |")
        except KeyError:
            pass
    md = "\n".join(lines)
    open(os.path.join(out, "summary.md"), "w").write(md + "\n")
    print(md)


def plot_curve(curves, metric, noise_filter, ylabel, title, path, show_both_conditions=False):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    style(ax)
    for m in METHODS:
        conds = sorted(curves.noise.unique()) if show_both_conditions else [noise_filter]
        for noise in conds:
            d = curves[(curves.method == m) & (curves.noise == noise)].dropna(subset=[metric])
            if d.empty:
                continue
            agg = d.groupby("epoch")[metric].agg(["mean", "std"]).reset_index().fillna(0)
            ls = "-" if noise > 0 or not show_both_conditions else "--"
            label = NAMES[m] + (f" ({int(noise * 100)}% noise)" if show_both_conditions else "")
            ax.plot(agg.epoch, 100 * agg["mean"], color=COLORS[m], lw=2, ls=ls, label=label,
                    marker="o", ms=4, markeredgecolor="white", markeredgewidth=1)
            ax.fill_between(agg.epoch, 100 * (agg["mean"] - agg["std"]),
                            100 * (agg["mean"] + agg["std"]), color=COLORS[m], alpha=0.12, lw=0)
            if not show_both_conditions:  # few lines -> label them directly
                ax.annotate(label, (agg.epoch.iloc[-1], 100 * agg["mean"].iloc[-1]),
                            xytext=(6, 0), textcoords="offset points", va="center",
                            fontsize=8, color=INK)
    ax.set_xlabel("Epoch", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.margins(x=0.02)
    if show_both_conditions:  # 6 lines converge -> a legend instead of colliding labels
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars(df, path):
    means = df.groupby(["method", "noise"])[["sst2_acc", "imdb_acc"]].agg(["mean", "std"])
    noises = sorted(df.noise.unique())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=150, sharey=False)
    for ax, metric, title in zip(axes, ["sst2_acc", "imdb_acc"], ["SST-2 (in-domain)", "IMDb (out-of-domain)"]):
        style(ax)
        width = 0.8 / len(noises)
        for j, noise in enumerate(noises):
            for i, m in enumerate(METHODS):
                if (m, noise) not in means.index:
                    continue
                mu = 100 * means.loc[(m, noise), (metric, "mean")]
                sd = 100 * np.nan_to_num(means.loc[(m, noise), (metric, "std")])
                x = i + (j - (len(noises) - 1) / 2) * width
                ax.bar(x, mu, width * 0.92, color=COLORS[m], edgecolor="white", linewidth=2,
                       hatch="///" if noise > 0 else None, alpha=1.0 if noise == 0 else 0.75)
                ax.errorbar(x, mu, yerr=sd, color=INK, lw=1, capsize=2)
                ax.text(x, mu + sd + 0.3, f"{mu:.1f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(METHODS)), [NAMES[m] for m in METHODS])
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        lo = 100 * df[metric].min()
        ax.set_ylim(max(0, lo - 8), 100)
    axes[0].set_ylabel("Accuracy (%)", color=MUTED, fontsize=9)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#9b9a92", edgecolor="white", label="clean labels")]
    if len(noises) > 1:
        handles.append(Patch(facecolor="#9b9a92", edgecolor="white", hatch="///",
                             label=f"{int(100 * max(noises))}% noisy labels"))
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="analysis")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df, curves = load_runs(a.runs)
    if df.empty:
        raise SystemExit(f"no results.json found under {a.runs}/")
    df.to_csv(os.path.join(a.out, "all_runs.csv"), index=False)
    summarize(df, a.out)

    noisy = max(df.noise)
    if noisy > 0:
        plot_curve(curves, "mem_rate", noisy, "Corrupted examples fit to wrong label (%)",
                   f"Memorization of noisy labels ({int(noisy * 100)}% noise)",
                   os.path.join(a.out, "fig_memorization.png"))
    plot_curve(curves, "sst2_acc", None, "SST-2 validation accuracy (%)",
               "SST-2 accuracy during training",
               os.path.join(a.out, "fig_sst2_curve.png"), show_both_conditions=True)
    plot_bars(df, os.path.join(a.out, "fig_final_bars.png"))
    print(f"\nwrote tables and figures to {a.out}/")


if __name__ == "__main__":
    main()
