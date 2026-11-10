#!/usr/bin/env bash
# The main experiment as a Slurm job array: one GPU job per run.
#   index = 6*(seed-1) + 3*noise_idx + method_idx
#   sbatch --array=0-5%3   gpmoo/grid.sbatch   # seed 1 only (6 runs, max 3 at once)
#   sbatch --array=6-17%3  gpmoo/grid.sbatch   # seeds 2 and 3
# Finished runs (results.json exists) exit immediately, so resubmitting is safe.
#SBATCH -J noisy-grid
#SBATCH -p gpmoo-b
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-04:00
#SBATCH -o logs/grid_%A_%a.out
#SBATCH -e logs/grid_%A_%a.err
set -e
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=~/.conda/envs/noisylora/bin/python
METHODS=(lora dora full)
NOISES=(0.0 0.3)
i=${SLURM_ARRAY_TASK_ID}
seed=$(( i / 6 + 1 ))
noise=${NOISES[$(( (i % 6) / 3 ))]}
method=${METHODS[$(( i % 3 ))]}
run="runs/${method}_noise$(printf '%.2f' $noise)_seed${seed}"
if [ -f "$run/results.json" ]; then echo "already done: $run"; exit 0; fi
echo "task $i -> $run on $(hostname)"
$PY -c "import torch; assert torch.cuda.is_available(), 'no GPU visible - check driver / torch CUDA version'; print('GPU:', torch.cuda.get_device_name(0))"
$PY src/train.py --method $method --noise $noise --seed $seed ${EXTRA_ARGS}
