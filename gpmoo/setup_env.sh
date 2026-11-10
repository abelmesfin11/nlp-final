#!/usr/bin/env bash
# One-time setup on gpmoo (run on the head node; it only installs packages).
#   bash gpmoo/setup_env.sh
set -e
ENV=noisylora
if ! conda env list | grep -q "^$ENV "; then
  conda create -y --name $ENV python=3.12
fi
PY=~/.conda/envs/$ENV/bin/python
$PY -m pip install --upgrade pip
# gpmoo's driver supports CUDA 12.6; the default torch wheel needs a newer driver
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install -r requirements.txt
$PY -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"
echo "Environment ready: $PY"
