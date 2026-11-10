"""Support for MANUAL error analysis (the professor's requirement).

The script does not decide error types for you. It (1) gathers the errors,
(2) adds cheap automatic flags that help you spot patterns, and (3) writes
spreadsheets with empty `category` / `notes` columns for you to fill in by hand.
After you label them, `tally` counts categories per method.

Step 1 - build the sheets:
    python src/error_analysis.py sample --runs runs --out analysis/errors

  writes, for each noise condition:
    sst2_errors_noise{X}.csv   every SST-2 test example at least one method gets wrong
    imdb_errors_noise{X}.csv   a sample of IMDb errors per method
    memorized_noise{X}.csv     corrupted training examples: which methods memorized them
    auto_breakdown.md          error rates split by automatic flags (negation, contrast, length)

Step 2 - open the CSVs, fill in `category` (see CATEGORIES below / README) and `notes`.

Step 3 - count them:
    python src/error_analysis.py tally analysis/errors/sst2_errors_noise0.30.csv
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

METHODS = ["full", "lora", "dora"]

CATEGORIES = {
    "negation": "sentiment flipped by not / never / no / n't (e.g. 'not bad')",
    "contrast": "mixed sentiment with but / although / however; model follows the wrong clause",
    "sarcasm": "irony or sarcasm; literal words disagree with intended meaning",
    "comparison": "compares to other films ('no Citizen Kane, but ...')",
    "world_knowledge": "needs knowledge of actors, genres, references",
    "rare_words": "unusual vocabulary, idioms, slang the model misreads",
    "neutral_ambiguous": "no clear sentiment; a human could go either way",
    "label_noise": "the GOLD label itself looks wrong",
    "plot_vs_opinion": "(IMDb) describes plot events that sound negative/positive but the review is the opposite",
    "truncation": "(IMDb) the deciding sentence is past the 512-token cutoff",
    "other": "anything else - describe in notes",
}

NEGATION = re.compile(r"\b(not|no|never|nothing|nobody|neither|nor|none|hardly|barely)\b|n't\b", re.I)
CONTRAST = re.compile(r"\b(but|although|though|however|yet|despite|while|even if)\b", re.I)
INTENSIFIER = re.compile(r"\b(very|really|so|too|extremely|incredibly|utterly|totally)\b", re.I)


def add_flags(df):
    df = df.copy()
    df["n_words"] = df.text.str.split().str.len()
    df["has_negation"] = df.text.str.contains(NEGATION)
    df["has_contrast"] = df.text.str.contains(CONTRAST)
    df["has_intensifier"] = df.text.str.contains(INTENSIFIER)
    return df


def load_preds(run_dir, method, noise, split):
    """Per-example predictions for one method/noise, combined over seeds by majority vote."""
    paths = sorted(glob.glob(os.path.join(run_dir, f"{method}_noise{noise:.2f}_seed*", f"preds_{split}.csv")))
    if not paths:
        return None
    frames = [pd.read_csv(p) for p in paths]
    base = frames[0].drop(columns=["pred", "p_pos"])
    preds = np.stack([f.pred.values for f in frames])
    probs = np.stack([f.p_pos.values for f in frames])
    base["pred"] = (preds.mean(0) > 0.5).astype(int)
    base["p_pos"] = probs.mean(0)
    base["seed_agreement"] = (preds == base["pred"].values).mean(0)  # 1.0 = all seeds agree
    base["n_seeds"] = len(frames)
    return base


def noises_in(run_dir):
    found = set()
    for d in glob.glob(os.path.join(run_dir, "*_noise*_seed*")):
        found.add(float(re.search(r"noise([\d.]+)_seed", d).group(1)))
    return sorted(found)


def merged_errors(run_dir, noise, split):
    merged = None
    for m in METHODS:
        d = load_preds(run_dir, m, noise, split)
        if d is None:
            continue
        cols = {"pred": f"pred_{m}", "p_pos": f"p_pos_{m}", "seed_agreement": f"agree_{m}"}
        d = d.rename(columns=cols)[["uid", "text", "label", *cols.values()]]
        merged = d if merged is None else merged.merge(d[["uid", *cols.values()]], on="uid")
    if merged is None:
        return None
    present = [m for m in METHODS if f"pred_{m}" in merged]
    for m in present:
        merged[f"wrong_{m}"] = (merged[f"pred_{m}"] != merged.label).astype(int)
    merged["n_methods_wrong"] = merged[[f"wrong_{m}" for m in present]].sum(1)
    return add_flags(merged), present


def breakdown(df, present, title):
    """Error rate per method within slices defined by the automatic flags."""
    slices = {
        "all": np.ones(len(df), bool),
        "has negation": df.has_negation.values,
        "has contrast word": df.has_contrast.values,
        "short (<=10 words)": (df.n_words <= 10).values,
        "long (>25 words)": (df.n_words > 25).values,
    }
    if df.n_words.median() > 100:  # IMDb-sized texts
        del slices["short (<=10 words)"], slices["long (>25 words)"]
        slices["short (<=150 words)"] = (df.n_words <= 150).values
        slices["long (>400 words, likely truncated)"] = (df.n_words > 400).values
    lines = [f"### {title}", "", "| Slice | n | " + " | ".join(f"{m} err%" for m in present) + " |",
             "|---|---|" + "---|" * len(present)]
    for name, mask in slices.items():
        if mask.sum() == 0:
            continue
        errs = " | ".join(f"{100 * df.loc[mask, f'wrong_{m}'].mean():.1f}" for m in present)
        lines.append(f"| {name} | {mask.sum()} | {errs} |")
    return "\n".join(lines) + "\n"


def cmd_sample(a):
    os.makedirs(a.out, exist_ok=True)
    report = ["# Automatic error breakdown",
              "Error rates (%) within slices flagged by simple regexes. Use these to choose what to",
              "look at by hand - they are hints, not conclusions.", ""]
    for noise in noises_in(a.runs):
        tag = f"noise{noise:.2f}"
        # ---- SST-2: all errors (the test set is small enough to read every one) ----
        res = merged_errors(a.runs, noise, "sst2")
        if res:
            df, present = res
            report.append(breakdown(df, present, f"SST-2, {int(noise * 100)}% noise"))
            err = df[df.n_methods_wrong > 0].sort_values("n_methods_wrong", ascending=False)
            err = err.assign(category="", notes="")
            err.to_csv(os.path.join(a.out, f"sst2_errors_{tag}.csv"), index=False)
            print(f"SST-2 {tag}: {len(err)} examples wrong by >=1 method "
                  f"({(err.n_methods_wrong == len(present)).sum()} wrong by all)")

        # ---- IMDb: sample errors per method (2000 examples -> too many to read all) ----
        res = merged_errors(a.runs, noise, "imdb")
        if res:
            df, present = res
            report.append(breakdown(df, present, f"IMDb, {int(noise * 100)}% noise"))
            picks = []
            for m in present:
                e = df[df[f"wrong_{m}"] == 1]
                picks.append(e.sample(min(a.imdb_per_method, len(e)), random_state=0))
            samp = pd.concat(picks).drop_duplicates("uid")
            samp = samp.assign(category="", notes="")
            samp.to_csv(os.path.join(a.out, f"imdb_errors_{tag}.csv"), index=False)
            print(f"IMDb {tag}: sampled {len(samp)} error examples to read")

        # ---- memorization: which corrupted training examples got learned ----
        if noise > 0:
            merged = None
            for m in METHODS:
                d = load_preds(a.runs, m, noise, "train")
                if d is None:
                    continue
                d = d[d.is_corrupted]
                d = d.assign(**{f"memorized_{m}": (d.pred == d.label).astype(int)})
                keep = ["uid", f"memorized_{m}"]
                merged = d[["uid", "text", "label", "true_label", f"memorized_{m}"]] if merged is None \
                    else merged.merge(d[keep], on="uid")
            if merged is not None:
                merged = add_flags(merged.rename(columns={"label": "noisy_label"}))
                mcols = [c for c in merged if c.startswith("memorized_")]
                merged["n_memorized"] = merged[mcols].sum(1)
                # stratified sample: memorized by everyone / by some / by none
                groups = [merged[merged.n_memorized == k] for k in range(len(mcols) + 1)]
                samp = pd.concat([g.sample(min(a.mem_per_group, len(g)), random_state=0) for g in groups])
                samp.assign(category="", notes="").to_csv(
                    os.path.join(a.out, f"memorized_{tag}.csv"), index=False)
                rates = ", ".join(f"{c.split('_')[1]}={100 * merged[c].mean():.1f}%" for c in mcols)
                report += [f"### Memorization, {int(noise * 100)}% noise", "",
                           f"Share of corrupted training examples fit to the WRONG label: {rates}", "",
                           "Overlap (how many methods memorized each example): " +
                           ", ".join(f"{k} methods: {len(g)}" for k, g in enumerate(groups)), "",
                           "Mean length of memorized vs not (words), per method:", ""]
                for c in mcols:
                    report.append(f"- {c.split('_')[1]}: memorized {merged[merged[c] == 1].n_words.mean():.1f}, "
                                  f"not memorized {merged[merged[c] == 0].n_words.mean():.1f}")
                report.append("")
    open(os.path.join(a.out, "auto_breakdown.md"), "w").write("\n".join(report))
    with open(os.path.join(a.out, "CATEGORIES.md"), "w") as f:
        f.write("# Error categories for manual labeling\n\nPut ONE key in the `category` column "
                "(add a second with `;` only if truly needed).\n\n")
        for k, v in CATEGORIES.items():
            f.write(f"- `{k}` - {v}\n")
    print(f"wrote sheets + auto_breakdown.md + CATEGORIES.md to {a.out}/")


def cmd_tally(a):
    for path in a.files:
        df = pd.read_csv(path, dtype={"category": str, "notes": str}).fillna({"category": ""})
        df = df[df.category.str.strip() != ""]
        if df.empty:
            print(f"{path}: no labeled rows yet")
            continue
        df = df.assign(category=df.category.str.split(";")).explode("category")
        df["category"] = df.category.str.strip()
        flag_cols = [c for c in df if c.startswith("wrong_") or c.startswith("memorized_")]
        out = {c.split("_", 1)[1]: df[df[c] == 1].category.value_counts() for c in flag_cols}
        table = pd.DataFrame(out).fillna(0).astype(int)
        table["total_labeled"] = df.category.value_counts()
        print(f"\n== {path} ({df.uid.nunique()} labeled examples)\n")
        print(table.sort_values("total_labeled", ascending=False).to_markdown())
        unknown = set(table.index) - set(CATEGORIES)
        if unknown:
            print(f"\n(note: categories not in the standard list: {sorted(unknown)})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--runs", default="runs")
    s.add_argument("--out", default="analysis/errors")
    s.add_argument("--imdb_per_method", type=int, default=40)
    s.add_argument("--mem_per_group", type=int, default=20)
    t = sub.add_parser("tally")
    t.add_argument("files", nargs="+")
    a = p.parse_args()
    cmd_sample(a) if a.cmd == "sample" else cmd_tally(a)


if __name__ == "__main__":
    main()
