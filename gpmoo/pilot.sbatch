#!/usr/bin/env bash
# Pilot: 1 epoch of LoRA and of full FT, to check speed + memory before the grid.
#   sbatch gpmoo/pilot.sbatch
#SBATCH -J noisy-pilot
#SBATCH -p gpmoo-b
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-02:00
#SBATCH -o logs/pilot_%j.out
#SBATCH -e logs/pilot_%j.err
set -e
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=~/.conda/envs/noisylora/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'no GPU visible - check driver / torch CUDA version'; print('GPU:', torch.cuda.get_device_name(0))"
for method in lora full; do
  $PY src/train.py --method $method --noise 0.3 --seed 1 --epochs 1 --out_root runs_pilot
done
