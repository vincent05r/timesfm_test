#!/bin/bash

# === Configuration ===
# Directory containing all datasets
base_dir=datasets/test_v1_nv_flat
datetime_col=Datetime

# Result base directory (all results will go under this)
result_base_dir=./results/tsfm_200m_t1
run_id=1

# Model & Data settings
horizon_len=96
context_len=128
logging=0
logging_name=exp

model_path=google/timesfm-1.0-200m-pytorch
# google/timesfm-2.0-500m-pytorch   

# === Execution ===
for dataset_path in "$base_dir"/*; do
  if [ -d "$dataset_path" ]; then
    dataset_name=$(basename "$dataset_path")

    # Compose result directory
    result_dir="${result_base_dir}/${dataset_name}"

    echo "🚀 Running evaluation for $dataset_name"
    echo "🕒 Datetime column: $datetime_col"
    echo "📁 Saving results to: $result_dir"

    python -u src/run_eval_tsfm_stock.py \
      --dataset_dir "$dataset_path" \
      --run_id "$run_id" \
      --datetime_col "$datetime_col" \
      --horizon_len "$horizon_len" \
      --context_len "$context_len" \
      --logging "$logging" \
      --logging_name "$logging_name" \
      --result_dir "$result_dir" \
      --model_path "$model_path" \
      --normalize \

    echo "✅ Done with $dataset_name"
    echo "--------------------------------------------"
  fi
done
