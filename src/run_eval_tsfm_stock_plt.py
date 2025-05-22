# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Eval pipeline."""

import json
import os
import sys
import time
from absl import flags
import chronos #type: ignore
import numpy as np
import pandas as pd
import timesfm
from timesfm import data_loader
import torch
import tqdm
from typing import Optional, Tuple, List, Dict

import argparse
import logging


# FLAGS = flags.FLAGS

# _BATCH_SIZE = flags.DEFINE_integer("batch_size", 64,
#                                    "Batch size for the randomly sampled batch")
# _DATASET = flags.DEFINE_string("dataset", "etth1", "The name of the dataset.")

# _MODEL_PATH = flags.DEFINE_string("model_path", "google/timesfm-2.0-500m-pytorch",
#                                   "The name of the model.")
# _DATETIME_COL = flags.DEFINE_string("datetime_col", "date",
#                                     "Column having datetime.")
# _NUM_COV_COLS = flags.DEFINE_list("num_cov_cols", None,
#                                   "Column having numerical features.")
# _CAT_COV_COLS = flags.DEFINE_list("cat_cov_cols", None,
#                                   "Column having categorical features.")
# _TS_COLS = flags.DEFINE_list("ts_cols", None, "Columns of time-series features")
# _NORMALIZE = flags.DEFINE_bool("normalize", True,
#                                "normalize data for eval or not")
# _CONTEXT_LEN = flags.DEFINE_integer("context_len", 2048,
#                                     "Length of the context window")
# _PRED_LEN = flags.DEFINE_integer("pred_len", 96, "prediction length.")
# _BACKEND = flags.DEFINE_string("backend", "gpu", "backend to use")
# _RESULTS_DIR = flags.DEFINE_string("results_dir", "./results/long_horizon",
#                                    "results directory")

# DATA_DICT = {
#     "ettm2": {
#         "boundaries": [34560, 46080, 57600],
#         "data_path": "./datasets/ETT-small/ETTm2.csv",
#         "freq": "15min",
#     },
#     "ettm1": {
#         "boundaries": [34560, 46080, 57600],
#         "data_path": "./datasets/ETT-small/ETTm1.csv",
#         "freq": "15min",
#     },
#     "etth2": {
#         "boundaries": [8640, 11520, 14400],
#         "data_path": "./datasets/ETT-small/ETTh2.csv",
#         "freq": "H",
#     },
#     "etth1": {
#         "boundaries": [8640, 11520, 14400],
#         "data_path": "./datasets/ETT-small/ETTh1.csv",
#         "freq": "H",
#     },
#     "elec": {
#         "boundaries": [18413, 21044, 26304],
#         "data_path": "./datasets/electricity/electricity.csv",
#         "freq": "H",
#     },
#     "traffic": {
#         "boundaries": [12280, 14036, 17544],
#         "data_path": "./datasets/traffic/traffic.csv",
#         "freq": "H",
#     },
#     "weather": {
#         "boundaries": [36887, 42157, 52696],
#         "data_path": "./datasets/weather/weather.csv",
#         "freq": "10min",
#     },
# }

QUANTILES = list(np.arange(1, 10) / 10.0)
EPS = 1e-7


def get_forecasts(model_path, model, past, freq, pred_len):
  """Get forecasts."""
  if model_path.startswith("amazon"):
    out = model.predict(
        torch.tensor(past),
        prediction_length=pred_len,
        limit_prediction_length=False,
    )
    out = out.numpy()
    out = np.median(out, axis=1)
  else:
    lfreq = [freq] * past.shape[0]
    _, out = model.forecast(list(past), lfreq)
    out = out[:, :, 5]
  return out


def _mse(y_pred, y_true):
  """mse loss."""
  return np.square(y_pred - y_true)


def _mae(y_pred, y_true):
  """mae loss."""
  return np.abs(y_pred - y_true)


def _smape(y_pred, y_true):
  """_smape loss."""
  abs_diff = np.abs(y_pred - y_true)
  abs_val = (np.abs(y_true) + np.abs(y_pred)) / 2
  abs_val = np.where(abs_val > EPS, abs_val, 1.0)
  abs_diff = np.where(abs_val > EPS, abs_diff, 0.0)
  return abs_diff / abs_val


def find_files_with_suffix(directory: str, suffix: str) -> List[str]:
  """
  Recursively traverse the given directory to find all files ending with `suffix`.

  Args:
      directory (str): Path to the directory where the search should begin.
      suffix (str): File suffix (e.g., ".csv", ".txt") to filter by.

  Returns:
      List[str]: List of full file paths that match the given suffix.
  """
  matched_files = []
  for root, dirs, files in os.walk(directory):
      dirs.sort()   # Ensures deterministic traversal of subdirectories
      files.sort()  # Ensures deterministic file ordering
      for f in files:
          if f.endswith(suffix):
              matched_files.append(os.path.join(root, f))
  return matched_files





def get_model_api(config):
  model_path = config.model_path
  #implement correct path for each models

  if model_path.startswith("amazon"):
    model = chronos.ChronosPipeline.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

  else:
    if model_path == "google/timesfm-2.0-500m-pytorch":
      model = timesfm.TimesFm(
          hparams=timesfm.TimesFmHparams(
              backend="gpu",
              per_core_batch_size=32,
              horizon_len=128,
              num_layers=50,
              context_len=config.context_len,
              use_positional_embedding=False,
          ),
          checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_path),
      )
    elif model_path == "google/timesfm-1.0-200m-pytorch":
      model = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
      )
    
    model._model.eval()

  return model


def eval(config, model):
  """Eval pipeline."""
  dataset = os.path.basename(config.dataset)
  data_path = config.dataset
  freq = data_path.split('_')[-1].split('.')[0]
  if freq == '1wk':
    freq = '1w'
  elif freq == '1m':
    freq = '1min'
  int_freq = timesfm.freq_map(freq)
  boundary = pd.read_csv(open(data_path, "r")).shape[0]

  data_df = pd.read_csv(open(data_path, "r"))

  if config.ts_cols is not None:
    raise NotImplementedError("todo")
    # ts_cols = DATA_DICT[dataset]["ts_cols"]
    # num_cov_cols = DATA_DICT[dataset]["num_cov_cols"]
    # cat_cov_cols = DATA_DICT[dataset]["cat_cov_cols"]
  else:
    ts_cols = [col for col in data_df.columns if col != config.datetime_col]
    num_cov_cols = None
    cat_cov_cols = None
  batch_size = 1
  dtl = data_loader.TimeSeriesdata(
      data_path=data_path,
      datetime_col=config.datetime_col,
      num_cov_cols=num_cov_cols,  
      cat_cov_cols=cat_cov_cols,
      ts_cols=np.array(ts_cols),
      train_range=[0, boundary],
      val_range=[0, boundary],
      test_range=[config.context_len, boundary],
      hist_len=config.context_len,
      pred_len=config.horizon_len,
      batch_size=batch_size,
      freq=freq,
      normalize=config.normalize,
      epoch_len=None,
      holiday=False,
      permute=False,
  )
  eval_itr = dtl.tf_dataset(mode="test",
                            shift=config.horizon_len).as_numpy_iterator()
  
  model_path = config.model_path

  smape_run_losses = []
  mse_run_losses = []
  mae_run_losses = []

  num_elements = 0
  abs_sum = 0
  start_time = time.time()

  for batch in tqdm.tqdm(eval_itr):
    past = batch[0]
    actuals = batch[3]
    forecasts = get_forecasts(model_path, model, past, int_freq,
                              config.horizon_len)
    forecasts = forecasts[:, 0:actuals.shape[1]]
    mae_run_losses.append(_mae(forecasts, actuals).sum())
    mse_run_losses.append(_mse(forecasts, actuals).sum())
    smape_run_losses.append(_smape(forecasts, actuals).sum())
    num_elements += actuals.shape[0] * actuals.shape[1]
    abs_sum += np.abs(actuals).sum()


    import matplotlib.pyplot as plt


    # Convert tensors to numpy (if not already)
    past_np = past.detach().cpu().numpy() if isinstance(past, torch.Tensor) else past
    actuals_np = actuals.detach().cpu().numpy() if isinstance(actuals, torch.Tensor) else actuals
    forecasts_np = forecasts.detach().cpu().numpy() if isinstance(forecasts, torch.Tensor) else forecasts

    # print(past_np)
    # print(len(past_np[0]))
    # print(actuals_np)
    # print(len(actuals_np[0]))
    # print(forecasts_np)
    # print(len(forecasts_np[0]))

    # Pick a sample (assume batch dimension exists)
    idx = 0
    past_seq = past_np[idx]
    actual_seq = actuals_np[idx]
    forecast_seq = forecasts_np[idx]

    # Concatenate past and actuals
    gt_full = np.concatenate([past_seq, actual_seq], axis=0)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(gt_full, label='Ground Truth', color='blue', linewidth=2)
    plt.plot(np.arange(len(past_seq), len(past_seq) + len(forecast_seq)), forecast_seq, label='Forecast', color='red', linewidth=3)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    plt.savefig(r"pics/chrono_s_60_p2.jpg", dpi=400, bbox_inches='tight')
    break


  class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)

  if num_elements != 0:

    mse_val = np.sum(mse_run_losses) / num_elements
    
    result_dict = {
        "mse": mse_val,
        "smape": np.sum(smape_run_losses) / num_elements,
        "mae": np.sum(mae_run_losses) / num_elements,
        "wape": np.sum(mae_run_losses) / abs_sum,
        "nrmse": np.sqrt(mse_val) / (abs_sum / num_elements),
        "num_elements": num_elements,
        "abs_sum": abs_sum,
        "total_time": time.time() - start_time,
        "model_path": model_path,
        "dataset": dataset,
        "freq": freq,
        "pred_len": config.horizon_len,
        "context_len": config.context_len,
    }
    run_id = config.run_id
    result_sv_name = "{}_{}_h{}_id{}".format(os.path.basename(model_path).split('.')[0], dataset, config.horizon_len, run_id)
    save_path = os.path.join(config.result_dir, result_sv_name)
    print(f"Saving results to {save_path}", flush=True)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "results.json"), "w") as f:
      json.dump(result_dict, f, cls=NumpyEncoder)
    print(result_dict, flush=True)
    logging.info("Result dictionary: %s", result_dict)
  
  else:
    print("no data for this set")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='zeroshot_longeval')

  #data
  parser.add_argument('--dataset_dir', required=True, help='directory for dataset name')
  parser.add_argument('--horizon_len', type=int, default=96)
  parser.add_argument('--context_len', type=int, default=128)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--datetime_col', type=str, default="date")
  parser.add_argument('--ts_cols', nargs='*', default=None,
                  help="List of time-series feature column names.")
  parser.add_argument('--normalize', action='store_true',
                  help="Apply normalization if set.")
  parser.add_argument('--num_cov_cols', nargs='*', default=None,
                  help="List of numerical covariate column names.")
  parser.add_argument('--cat_cov_cols', nargs='*', default=None,
                  help="List of categorical covariate column names.")

  #logging
  parser.add_argument('--logging', type=int, default=1, help='1 = logging, 0 = not logging')
  parser.add_argument('--logging_name', type=str, default='exp')
  parser.add_argument('--result_dir', type=str, default="./results/long_horizon")
  parser.add_argument('--run_id', default=None)

  #model
  parser.add_argument('--model_path', type=str, required=True, help='correct model checkpoint path')

  args = parser.parse_args()

  # torch.backends.cuda.matmul.allow_tf32 = False
  # torch.backends.cudnn.allow_tf32 = False
  # torch.backends.cudnn.benchmark = False
  # torch.backends.cudnn.deterministic = True
  print(f"TF32 Allow Matmul : {torch.backends.cuda.matmul.allow_tf32}")
  print(f"TF32 Allow Convolution : {torch.backends.cudnn.allow_tf32}")
  print(f"CUDNN Benchmark Enabled : {torch.backends.cudnn.benchmark}")
  print(f"CUDA Version : {torch.version.cuda}")
  print(f"Torch Compile Active : {torch._dynamo.config.verbose}")
  # torch.cuda.set_per_process_memory_fraction(0.3, device=0)  #memory reserve ratio cap for each process
  
  model = get_model_api(args)

  dataset_list = find_files_with_suffix(args.dataset_dir, suffix='.csv')

  for dataset in dataset_list:
    print(f"\n🧪 Running evaluation for dataset: {dataset}")
    config = argparse.Namespace(**vars(args))  # Shallow copy
    config.dataset = dataset

    if config.run_id is None:
        config.run_id = int(time.time())

    eval(config, model)
