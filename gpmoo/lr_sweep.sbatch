#!/usr/bin/env bash
# Learning-rate sanity check (seed 1): two extra LRs per method, clean AND noisy labels.
# The default LRs (full 1e-5, LoRA/DoRA 2e-4) are already in runs/.
#   sbatch --array=0-11%3 gpmoo/lr_sweep.sbatch
#   index = 4*method_idx + 2*lr_idx + noise_idx
#SBATCH -J noisy-lr
#SBATCH -p gpmoo-b
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-04:00
#SBATCH -o logs/lr_%A_%a.out
#SBATCH -e logs/lr_%A_%a.err
set -e
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=~/.conda/envs/noisylora/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"
METHODS=(full lora dora)
FULL_LRS=(5e-6 2e-5)
PEFT_LRS=(1e-4 5e-4)
NOISES=(0.0 0.3)
i=${SLURM_ARRAY_TASK_ID}
method=${METHODS[$(( i / 4 ))]}
if [ "$method" = full ]; then lr=${FULL_LRS[$(( (i % 4) / 2 ))]}; else lr=${PEFT_LRS[$(( (i % 4) / 2 ))]}; fi
noise=${NOISES[$(( i % 2 ))]}
out="runs_lr/lr${lr}"
run="$out/${method}_noise$(printf '%.2f' $noise)_seed1"
if [ -f "$run/results.json" ]; then echo "already done: $run"; exit 0; fi
echo "task $i -> $run"
$PY src/train.py --method $method --noise $noise --seed 1 --lr $lr --out_root $out
