#!/usr/bin/env bash
# Runs the full experiment grid: 3 methods x 2 noise levels x N seeds.
# Already-finished runs (results.json exists) are skipped, so you can re-run
# this after a Colab disconnect and it picks up where it left off.
#
#   bash scripts/run_all.sh            # seeds 1 2 3
#   SEEDS="1" bash scripts/run_all.sh  # just one seed first
set -e
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"1 2 3"}
EXTRA=${EXTRA:-""}   # e.g. EXTRA="--grad_checkpointing" if full FT runs out of memory

for seed in $SEEDS; do
  for noise in 0.0 0.3; do
    for method in lora dora full; do
      run="runs/${method}_noise$(printf '%.2f' $noise)_seed${seed}"
      if [ -f "$run/results.json" ]; then echo "skip $run"; continue; fi
      mkdir -p "$run"
      python src/train.py --method $method --noise $noise --seed $seed $EXTRA 2>&1 | tee "$run/log.txt"
    done
  done
done
python src/analyze.py
python src/error_analysis.py sample
