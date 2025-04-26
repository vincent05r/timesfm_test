#!/bin/bash

# Exit immediately on error
set -e

ENV_NAME="ffm_benchmark"

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "⚙️ Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

echo "📦 Installing pip dependencies..."
pip install -r requirement_v2.txt
pip install -e .
pip install chronos-forecasting
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

echo "✅ All packages installed in Conda env '$ENV_NAME'."