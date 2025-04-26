#!/bin/bash

# Exit immediately on error
set -e

ENV_NAME="ffm_benchmark"
PYTHON_VERSION="3.11.11"

echo "🔧 Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION..."
conda create --yes --name "$ENV_NAME" python="$PYTHON_VERSION"

echo "✅ Conda environment '$ENV_NAME' created successfully."