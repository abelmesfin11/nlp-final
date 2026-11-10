# Does LoRA Resist Noisy Labels?

Full fine-tuning vs. LoRA vs. DoRA on Qwen3-0.6B-Base, trained on SST-2 with 30% of the labels flipped.

The idea: LoRA and DoRA update far fewer parameters than full fine-tuning. Does that limited capacity make them memorize wrong labels less?

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3-0.6B-Base` with a 2-class classification head |
| Methods | Full FT, LoRA (r=16), DoRA (r=16). Adapters on all attention and MLP projections |
| Train data | 20k examples from SST-2 train |
| Noise | 0% and 30% (the same flipped examples for every method) |
| Test | SST-2 validation (872) and IMDb (2,000 balanced reviews, truncated to 512 tokens) |
| Training | 3 epochs, AdamW, linear decay with 6% warmup. LR 1e-5 (full), 2e-4 (LoRA/DoRA) |
| Seeds | 3, so 18 runs in total |

I used the base model rather than Instruct because Instruct has already seen sentiment-style tasks, which would make it harder to tell what the model learned from our labels. A classification head (instead of generating "positive"/"negative") keeps predictions unambiguous and makes the three methods easy to compare.

## Metrics

- **sst2_acc / imdb_acc**: accuracy on clean SST-2 and on IMDb (out of domain)
- **mem_rate**: on 1,000 tracked corrupted training examples, how often the model predicts the wrong label it was given
- **fix_rate**: how often it predicts the true label instead (1 − mem_rate)
- **clean_fit**: accuracy on 1,000 correctly labeled training examples

Everything is logged 4 times per epoch (IMDb once per epoch), so you can see the usual pattern: the model learns the real signal first, then starts memorizing the noise. The question is whether LoRA/DoRA delay or limit that second phase.

There's no early stopping, since a real noisy dataset would have a noisy dev set too. I report the final checkpoint, plus the best test accuracy during training as an "oracle" upper bound.

## Layout

```
├── requirements.txt
├── colab_runner.ipynb     # run on Colab
├── gpmoo/                 # Slurm scripts: env setup, data prep, pilot, grid
├── scripts/run_all.sh     # full grid on a single machine (skips finished runs)
└── src/
    ├── data.py            # loads SST-2/IMDb, builds the noisy train set
    ├── train.py           # one run: --method {full,lora,dora} --noise {0.0,0.3} --seed N
    ├── analyze.py         # summary table and figures
    └── error_analysis.py  # sheets for manual error analysis + tallying
```

Each run saves to `runs/<method>_noise<x>_seed<n>/`: `results.json`, `history.json`, and per-example predictions (`preds_sst2.csv`, `preds_imdb.csv`, `preds_train.csv`).

## Running it

### Locally / Colab

```bash
pip install -r requirements.txt
python src/data.py --n_train 20000 --noise 0.3          # check the noisy data (should be 30% flipped)
python src/train.py --method lora --noise 0.3 --seed 1   # single run
SEEDS="1 2 3" bash scripts/run_all.sh                    # full grid
python src/analyze.py
```

Useful `train.py` flags: `--lr`, `--epochs`, `--lora_r`, `--batch_size`, `--grad_accum`, `--grad_checkpointing`, `--save_model` (off by default, full-FT checkpoints are ~2.4 GB), `--n_train`, `--model`.

If full FT runs out of memory, add `--grad_checkpointing` or use `--batch_size 8 --grad_accum 2`.

### On a Slurm cluster (gpmoo)

```bash
bash gpmoo/setup_env.sh                    # create conda env (once)
bash gpmoo/prepare.sh                      # download model + data
sbatch gpmoo/pilot.sbatch                  # quick 1-epoch sanity check
sbatch --array=0-5%3 gpmoo/grid.sbatch     # seed 1, all 6 configs
sbatch --array=6-17%3 gpmoo/grid.sbatch    # seeds 2 and 3
```

Finished runs are skipped, so you can just resubmit if something fails. For OOM: `EXTRA_ARGS="--grad_checkpointing" sbatch ...`. Each run takes roughly 15–30 min on an A6000.

Delete the pilot's output folder (`runs/lora_noise0.30_seed1`) before the grid, otherwise it gets skipped as already done.

## Results

`python src/analyze.py` writes `analysis/summary.md` (mean ± std over seeds and the clean → noisy drop per method) and three figures:

- `fig_memorization.png`: memorization rate over training, per method
- `fig_sst2_curve.png`: SST-2 accuracy over training, clean vs. noisy
- `fig_final_bars.png`: final SST-2 and IMDb accuracy

## Error analysis

```bash
python src/error_analysis.py sample
```

This writes CSVs to `analysis/errors/`: SST-2 errors, sampled IMDb errors, and the corrupted training examples grouped by how many methods memorized them. It also writes `auto_breakdown.md` with error rates on slices like negation and contrast words.

I labeled each row by hand with a category (`negation`, `contrast`, `sarcasm`, `comparison`, `world_knowledge`, `rare_words`, `neutral_ambiguous`, `label_noise`, `plot_vs_opinion`, `truncation`, `other`) and a short note, hiding the columns that show which method got it wrong so I wouldn't bias the labels. Then:

```bash
python src/error_analysis.py tally analysis/errors/sst2_errors_noise0.30.csv
```

gives a category × method table.

## Notes

- Learning rates were picked on **clean** data (full FT in {5e-6, 1e-5, 2e-5}, LoRA/DoRA in {1e-4, 2e-4, 5e-4}) so the choice doesn't depend on the noise.
- SST-2 validation is only 872 examples, so differences under about 1 point probably aren't meaningful.
- Limitations: one model size, one noise type (random flips), binary task, truncated IMDb subset.

## References

- Zhang et al., 2017. *Understanding deep learning requires rethinking generalization*
- Arpit et al., 2017. *A closer look at memorization in deep networks*
- Hu et al., 2021. *LoRA: Low-rank adaptation of large language models*
- Liu et al., 2024. *DoRA: Weight-decomposed low-rank adaptation*
- Biderman et al., 2024. *LoRA learns less and forgets less*
