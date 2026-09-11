#!/bin/bash
# Replicate local 'inference' conda env on remote for bspline_policy eval.
set -e

# Use local clash proxy via reverse SSH tunnel for faster downloads
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
NO_PROXY="localhost,127.0.0.1,.local"
export NO_PROXY

# Initialize conda explicitly for non-interactive shells
CONDA_BASE=/home/byq/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_NAME=${1:-bspline_eval}
FORCE=${2:-0}

if conda env list | grep -q "^$ENV_NAME "; then
    if [ "$FORCE" = "1" ]; then
        echo "=== Removing existing env: $ENV_NAME ==="
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Env $ENV_NAME already exists. Use 'bash setup_eval_env.sh $ENV_NAME 1' to force recreate."
        exit 1
    fi
fi

echo "=== Creating conda env: $ENV_NAME (python 3.10) ==="
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

echo "=== Configuring pip mirror ==="
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip config set global.trusted-host mirrors.aliyun.com
pip install --upgrade pip==26.2.1 setuptools==84.0.0 wheel==0.48.0

REQ_FILE=/home/byq/bspline/requirements_inference.txt
FILTERED_REQ=/tmp/requirements_inference_filtered.txt

# Filter out packages that must be installed separately or from source
grep -vE '^(torch==|torchvision==|robomimic==|robosuite==)' "$REQ_FILE" > "$FILTERED_REQ"

echo "=== Installing CUDA torch (this may take several minutes) ==="
pip install --timeout 120 --retries 5 torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130

echo "=== Installing remaining requirements ==="
pip install -r "$FILTERED_REQ"

echo "=== Installing robomimic / robosuite from source (editable) ==="
cd /home/byq/bspline/robosuite && pip install -e .
cd /home/byq/bspline/robomimic && pip install -e .

echo "=== Installing diffusion_policy / bspline_policy (editable) ==="
cd /home/byq/bspline/diffusion_policy && pip install -e .
cd /home/byq/bspline/bspline_policy && pip install -e .

echo "=== Done. Activate with: conda activate $ENV_NAME ==="
