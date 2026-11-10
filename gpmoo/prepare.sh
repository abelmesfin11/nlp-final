#!/usr/bin/env bash
# Downloads the model + datasets and builds the cached train/test sets ONCE,
# so the GPU jobs never touch the network and never race to build the cache.
# Light work (downloads), fine on the head node.
#   bash gpmoo/prepare.sh
set -e
cd "$(dirname "$0")/.."
PY=~/.conda/envs/noisylora/bin/python
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B-Base')"
for noise in 0.0 0.3; do
  $PY src/data.py --n_train 20000 --noise $noise
done
