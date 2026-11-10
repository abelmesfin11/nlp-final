# Does Restricting Fine-Tuning Capacity Resist Noisy Labels?
### Full fine-tuning vs LoRA vs DoRA on Qwen3-0.6B, with 30% corrupted SST-2 labels

This is the step-by-step guide for running the whole project. It already includes the
professor's two changes:

1. **GPT-2 → Qwen3-0.6B-Base.** Same size class as GPT-2 Small/Medium, so full fine-tuning
   still fits on one Colab GPU, but a much stronger, better-curated starting point.
2. **Manual error analysis.** Tooling in `src/error_analysis.py` gathers the mistakes and
   gives you spreadsheets to label by hand; the guide below says how to do it so it counts.

---

## 1. The experiment at a glance

| | |
|---|---|
| Model | `Qwen/Qwen3-0.6B-Base` + a 2-way classification head |
| Methods | Full FT (all ~600M params), LoRA (r=16), DoRA (r=16) — adapters on all attention + MLP projections |
| Training data | 20,000 examples sampled from SST-2 train |
| Conditions | clean labels; **30%** of those 20k flipped (same flipped examples for every method) |
| In-domain test | SST-2 validation (872, clean) |
| Out-of-domain test | IMDb test, 2,000 balanced reviews, truncated at 512 tokens |
| Seeds | 3 per configuration → 3 × 2 × 3 = **18 runs** |
| Training | 3 epochs, AdamW, linear decay with 6% warmup. LR: full 1e-5, LoRA/DoRA 2e-4 |

**Why the Base model and not Instruct?** We want to measure what fine-tuning does to the
model. An instruction-tuned model has already been trained on sentiment-like tasks, so it
would muddy the "how much does it memorize our labels" question. (You could add Instruct as
an extension.)

**Why a classification head instead of generating "positive"/"negative"?** It makes
"what label did the model learn" unambiguous and the three methods directly comparable — the
same setup you'd have used with GPT-2.

### What gets measured

| Metric | Meaning |
|---|---|
| `sst2_acc` | accuracy on clean SST-2 validation |
| `imdb_acc` | accuracy on IMDb (generalization to a new domain and longer text) |
| **`mem_rate`** | of 1,000 tracked **corrupted** training examples, % where the model predicts the *wrong label we gave it* → memorization |
| `fix_rate` | same examples, % where it predicts the *true* label → it resisted the noise (`fix_rate = 1 − mem_rate` for 2 classes) |
| `clean_fit` | accuracy on 1,000 correctly-labeled training examples |

These are logged 4× per epoch (IMDb once per epoch), so you get **curves over training**, not
just end numbers. That matters: the usual finding with label noise is that models first learn
the real pattern (test accuracy rises), then start memorizing the noise (mem_rate climbs,
test accuracy drops). Whether LoRA delays or caps that second phase is the heart of your
hypothesis.

We do **not** early-stop on a dev set, because in a real noisy-label setting your dev set would
be noisy too. We report the final checkpoint, and separately the best test accuracy seen during
training, clearly labeled as an "oracle" upper bound.

---

## 2. Project layout

```
NLP Final/
├── README.md              ← this guide
├── requirements.txt
├── colab_runner.ipynb     ← run everything on a Colab GPU
├── gpmoo/                ← gpmoo cluster: env setup, data prep, pilot + grid Slurm jobs
├── scripts/run_all.sh     ← the 18-run grid for Colab/single machine (skips finished runs)
├── logs/                  ← Slurm job output
└── src/
    ├── data.py            ← loads SST-2 / IMDb, creates the corrupted training set
    ├── train.py           ← one run: --method {full,lora,dora} --noise {0.0,0.3} --seed N
    ├── analyze.py         ← summary table + figures
    └── error_analysis.py  ← builds sheets for manual error analysis, then tallies them
```

Each run writes `runs/<method>_noise<x>_seed<n>/`:
`results.json` (config, param counts, time, memory, full history), `history.json`,
and per-example predictions `preds_sst2.csv`, `preds_imdb.csv`, `preds_train.csv`.

---

## 3. Step-by-step plan

### Step 0 — Compute: gpmoo (do this first)
Your Mac's Python runs under Rosetta (x86), so it has no GPU access; it's fine for reading code
and labeling errors, not for training. We run on **gpmoo** (Williams CS Slurm cluster,
[docs](https://gpmoo-docs.cs.williams.edu/onboarding/)). Colab is a fallback (Step 1).

gpmoo rules that the scripts already follow: every GPU job goes through Slurm (`sbatch`);
**max 1 GPU and 8 CPUs per job**; no `tmux`/`nohup`/`&` inside job scripts; set time and memory.
Partitions: `gpmoo-b` (A6000 48GB, default — what we use) and `gpmoo-a` (A100s, for heavier jobs).

**0a. SSH key (once, on your Mac)** — so you (and Claude) can connect without typing a password
every time. You must be on campus or on the college VPN (or jump through `angus`, see the docs).
```bash
ssh-keygen -t ed25519            # press Enter for all prompts
ssh-copy-id 26agm6@gpmoo.cs.williams.edu   # asks for your gpmoo password once
```
Then add to `~/.ssh/config`:
```
Host gpmoo gpmoo.cs.williams.edu
  HostName gpmoo.cs.williams.edu
  User 26agm6
```

**0b. Copy the project up** (re-run after any local code change):
```bash
rsync -av --exclude runs --exclude analysis --exclude data "/Users/abelmesfin/Desktop/NLP Final/" gpmoo:~/nlp-final/
```

**0c. On gpmoo (`ssh gpmoo`, then `cd ~/nlp-final`):**
```bash
bash gpmoo/setup_env.sh      # creates conda env "noisylora" (once, ~5 min)
bash gpmoo/prepare.sh        # downloads Qwen3-0.6B + datasets, builds noisy/clean train sets
sbatch gpmoo/pilot.sbatch    # 1-epoch LoRA + full FT: checks speed and memory
squeue -u $USER              # watch it; logs land in logs/
```
Look at `logs/pilot_<jobid>.out`: you should see accuracy jump from ~50% to ~90%, and note the
seconds per step. Then the real grid, as a Slurm array (one GPU job per run, at most 3 at a time
so we don't hog the cluster):
```bash
sbatch --array=0-5%3 gpmoo/grid.sbatch     # seed 1: all 6 configurations
# check results look sane, then:
sbatch --array=6-17%3 gpmoo/grid.sbatch    # seeds 2 and 3
```
Finished runs are skipped, so if a job fails just resubmit the same command. If full FT runs out
of memory: `EXTRA_ARGS="--grad_checkpointing" sbatch --array=... gpmoo/grid.sbatch`.

**0d. After the runs:**
```bash
~/.conda/envs/noisylora/bin/python src/analyze.py
~/.conda/envs/noisylora/bin/python src/error_analysis.py sample
```
(both are light — seconds of pandas). Then pull results back to your Mac to read and label:
```bash
rsync -av gpmoo:~/nlp-final/analysis gpmoo:~/nlp-final/runs "/Users/abelmesfin/Desktop/NLP Final/"
```

Rough cost per run (20k examples, 3 epochs, short SST-2 sentences): ~15–30 min on an A6000,
full FT the slowest. With 3 runs in parallel, 18 runs ≈ 2–3 hours of wall-clock time.
**Do seed 1 of all 6 configurations first**, check the results, then add seeds 2–3.

### Step 1 — Set up (Colab fallback only, ≈30 min)
1. Upload this whole folder to Google Drive → `My Drive/NLP Final`.
2. Open `colab_runner.ipynb` in Colab, set the runtime to GPU, run the first cells.
3. Run `python src/data.py` and read the printed corrupted examples. You should see 30.0%
   corrupted and ~70% agreement with gold.

### Step 2 — Pilot run (≈30 min)
*(On gpmoo this is `sbatch gpmoo/pilot.sbatch`, Step 0c.)*

Run one 1-epoch LoRA job (cell in the notebook). Check:
- the step-0 eval shows ~50% accuracy (untrained head = random), and it rises fast after;
- loss goes down; no out-of-memory;
- note seconds per step, so you can budget the full grid.

Then try one 1-epoch **full FT** run — it's the one most likely to run out of memory. If it does:
`--grad_checkpointing`, or `--batch_size 8 --grad_accum 2` (same effective batch).

### Step 3 — Learning-rate sanity check (optional, ≈1–2 h, recommended)
A reviewer's first question will be "maybe LoRA just looked robust because its LR was off."
On **clean** data, seed 1, try full FT at {5e-6, 1e-5, 2e-5} and LoRA/DoRA at {1e-4, 2e-4, 5e-4}
(`--lr`, `--out_root runs_lr`; on gpmoo use `EXTRA_ARGS="--lr 2e-5 --out_root runs_lr"` with a single-task array, or copy `pilot.sbatch`). Pick the LR per method with the best clean SST-2 accuracy, and
use it for every noise condition. Mention this in the report — choosing on clean data keeps the
choice independent of the noise.

### Step 4 — Main grid
*(On gpmoo: the `grid.sbatch` array in Step 0c. Below is the Colab version.)*

```bash
SEEDS="1" bash scripts/run_all.sh
SEEDS="2 3" bash scripts/run_all.sh
```
(or the notebook cells). Finished runs are skipped, so Colab disconnects only cost the run in
progress.

### Step 5 — Numbers and figures
```bash
python src/analyze.py
```
writes `analysis/summary.md` (main table, mean ± std over seeds, plus the clean→noisy accuracy
drop per method) and three figures:
- `fig_memorization.png` — memorization rate vs epoch, one line per method. **Your key figure.**
- `fig_sst2_curve.png` — test accuracy over training, clean (dashed) vs noisy (solid).
- `fig_final_bars.png` — final SST-2 and IMDb accuracy.

### Step 6 — Manual error analysis (the part the professor asked for)
```bash
python src/error_analysis.py sample
```
creates in `analysis/errors/`:

| File | What's in it | How many to label |
|---|---|---|
| `sst2_errors_noise0.00.csv` / `…0.30.csv` | every SST-2 test example at least one method got wrong (majority vote across seeds), with which methods erred and their confidence | all (usually ~50–120) |
| `imdb_errors_noise*.csv` | 40 sampled errors per method | all of them |
| `memorized_noise0.30.csv` | corrupted training examples, grouped by how many methods memorized them (0–3) | all (~80) |
| `auto_breakdown.md` | error rates within automatic slices: negation, contrast words, short vs long | — read it |
| `CATEGORIES.md` | the label set | — |

**How to label.** Open each CSV in Excel/Google Sheets. For each row, read the text, then fill:
- `category` — one of: `negation`, `contrast`, `sarcasm`, `comparison`, `world_knowledge`,
  `rare_words`, `neutral_ambiguous`, `label_noise` (gold label itself looks wrong),
  `plot_vs_opinion` (IMDb), `truncation` (IMDb), `other`.
- `notes` — one sentence in your own words. These become quotes in your report.

Tips so it holds up:
- **Label before looking at which method got it wrong** (hide the `wrong_*` columns), so you
  don't unconsciously label to fit your hypothesis.
- If you're a team, have two people label the same ~40 SST-2 rows independently and report how
  often they agree. That one number makes the analysis much more credible.
- For the **memorization sheet**, ask: are the memorized noisy examples ones that are
  *ambiguous anyway* (short fragments like "adults , other than the parents ..." — easy to
  memorize because the true label is weak) or *clearly* wrong labels (e.g. "an insult to every
  family" marked positive)? If full FT memorizes the clear ones and LoRA only the ambiguous
  ones, that is a real insight about *what* constrained updates protect against.

Then count:
```bash
python src/error_analysis.py tally analysis/errors/sst2_errors_noise0.30.csv
```
→ a table of categories × method you can paste into the report.

**Questions your error analysis should answer:**
1. Which error types does noise *add*? Compare categories at 0% vs 30% noise for the same method.
2. Do the methods fail on the *same* examples? (`n_methods_wrong` column.) Errors shared by all
   three point to hard examples; method-specific errors point to the training method.
3. On IMDb, are errors about long, mixed reviews (`contrast`, `plot_vs_opinion`, `truncation`)?
   Does noise make those worse more for one method?
4. Which corrupted examples get memorized, and how does that differ between methods?
5. Pick 3–5 striking examples to quote in full in the report.

### Step 7 — Write up
Suggested structure:
1. **Intro & hypothesis** — PEFT is usually judged by matching full FT; we ask about robustness to label noise.
2. **Related work** — label-noise memorization (Zhang et al. 2017 "Understanding deep learning requires rethinking generalization"; Arpit et al. 2017 "A closer look at memorization"), LoRA (Hu et al. 2021), DoRA (Liu et al. 2024), "LoRA learns less and forgets less" (Biderman et al. 2024).
3. **Setup** — the tables in section 1; why Qwen3 Base; LR selection.
4. **Results** — summary table; memorization curve; clean vs noisy accuracy drop; IMDb.
5. **Error analysis** — category table + examples + answers to the 5 questions above.
6. **Discussion & limitations** — one model size, one noise type (random flips), binary task, IMDb truncated.

---

## 4. Running things manually

```bash
pip install -r requirements.txt
python src/data.py --n_train 20000 --noise 0.3          # inspect data
python src/train.py --method dora --noise 0.3 --seed 1   # one run
python src/analyze.py
python src/error_analysis.py sample
```

Useful `train.py` flags: `--lr`, `--epochs`, `--lora_r`, `--batch_size`, `--grad_accum`,
`--grad_checkpointing`, `--save_model` (off by default; full-FT checkpoints are ~2.4 GB each),
`--n_train`, `--model`.

---

## 5. Extensions if you have time (pick at most one or two)
- **Rank sweep:** LoRA r ∈ {4, 16, 64, 256}. Does memorization grow with rank? This tests the "capacity" story directly and is cheap.
- **Noise levels:** 10% / 30% / 45% → a robustness curve.
- **More epochs (5–10):** memorization often keeps rising long after accuracy plateaus.
- **Scale:** Qwen3-1.7B with LoRA/DoRA only (full FT gets expensive).

## 6. Pitfalls to avoid
- **Don't compare methods at different LRs chosen on noisy data.** Choose on clean data (Step 3).
- **Report std over seeds.** Differences under ~1 point on SST-2 (872 examples) are likely noise.
- **The pilot run folder** `runs/lora_noise0.30_seed1` must be deleted before the grid, or it
  will be skipped as "done" (the notebook does this for you).
- **IMDb is 2,000 reviews, not 25,000,** and truncated at 512 tokens — state both in the report.
