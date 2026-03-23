"""
aggregate_results.py - Script for aggregating multi-seed training runs.

Overview
This script aggregates reinforcement-learning training results produced with
the same experiment configuration but different random seeds. It computes
statistics such as mean and standard deviation, then saves them as CSV files
and TensorBoard logs.

Main features
1. aggregate_progresses function:
     - Reads multiple progress.csv files and aggregates them by the specified
         mode (average, median, max, min, std).
     - Processes eval, leader, and follower logs separately when
         is_separate_logs=True.
     - Saves outputs in CSV and TensorBoard formats.
     - Validates that all input dataframes have the same number of rows.

2. Main routine:
     - Scans subdirectories under the specified directory.
     - Reads experiment name (exp_name) from each directory's config.yaml.
     - Groups directories by identical experiment name.
     - Computes and saves average and standard deviation per experiment name.

Expected directory structure
Input:
    log_dir/
        ├── run_seed_0/  (training directory whose config.yaml contains exp_name)
        │   ├── config.yaml
        │   ├── eval/progress.csv
        │   ├── leader/progress.csv
        │   └── follower/progress.csv
        ├── run_seed_1/
        │   ├── config.yaml
        │   ├── eval/progress.csv
        │   ├── leader/progress.csv
        │   └── follower/progress.csv
        └── run_seed_2/
                └── ...

Output:
    log_dir/
        └── aggregated/
                ├── exp_name_average/
                │   ├── eval/progress_average.csv
                │   ├── leader/progress_average.csv
                │   └── follower/progress_average.csv
                └── exp_name_std/
                        ├── eval/progress_std.csv
                        ├── leader/progress_std.csv
                        └── follower/progress_std.csv

Usage
Basic:
    python aggregate_results.py LOG_DIR

Example:
    python aggregate_results.py data/local/experiment/2025_01_27_20_15_23_test

Arguments
    log_dir: Path to the directory containing training results (required).
                     Assumes multiple training-run subdirectories directly under it.

Notes
- Each run directory must contain config.yaml (to read exp_name).
- progress.csv files from directories with the same exp_name must have the
    same row count.
- A mismatch in row count raises an error.
- Empty CSV files also raise an error.
- Directories without config.yaml are skipped.

Difference from evaluate_aggregate.py
- aggregate_results.py: Aggregates training-time results (simple 1-level structure)
- evaluate_aggregate.py: Aggregates evaluation-time results (recursive search, per iteration)
"""

import os
import yaml
import argparse
import pandas as pd
import warnings
from torch.utils.tensorboard import SummaryWriter

def aggregate_progresses(log_dirs: list, 
                        aggregate_log_dir: str, 
                        mode='average',
                        x_axis='TotalEnvSteps',
                        groupby='TotalEnvSteps',
                        use_tensorboard=True,
                        is_separate_logs=True):
    data_list = []
    save_log_list = []
    tensorboard_dirs = []

    separate_logs = ['eval', 'leader', 'follower'] if is_separate_logs else ['']
    for l in separate_logs:
        data = []
        for log_dir in log_dirs:  
            path = os.path.join(log_dir, l, 'progress.csv')
            try:
                df = pd.read_csv(path)
                print(f"Read from {path}")
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
                raise ValueError(
                    f"No data found in {path}. Using empty DataFrame instead."
                    )
            data.append(df)
        if not all([len(d) == len(data[0]) for d in data]):
            raise ValueError(
                f"DataFrames have different numbers of rows for {aggregate_log_dir}"
                )
        data_list.append(data)
        save_log = os.path.join(aggregate_log_dir, l, f'progress_{mode}.csv')
        save_log_list.append(save_log)
        tensorboard_dirs.append(os.path.join(aggregate_log_dir, l))
 
    for dir in tensorboard_dirs:
        os.makedirs(dir, exist_ok=True)

    for d, sd, td in zip(data_list, save_log_list, tensorboard_dirs):
        data_concat = pd.concat(d)
        if data_concat.empty:
            continue
        elif mode == 'average':
            progress_agregate = data_concat.groupby(groupby).mean().reset_index()
        elif mode == 'median':
            progress_agregate = data_concat.groupby(groupby).median().reset_index()
        elif mode == 'max':
            progress_agregate = data_concat.groupby(groupby).max().reset_index()
        elif mode == 'min':
            progress_agregate = data_concat.groupby(groupby).min().reset_index()
        elif mode == 'std':
            progress_agregate = data_concat.groupby(groupby).std().reset_index()

        print(f"Write to {sd}")
        progress_agregate.to_csv(sd, index=False)

        if use_tensorboard:
            summary_writer = SummaryWriter(td)
            if x_axis in progress_agregate:
                x = progress_agregate[x_axis]
                for key, value in progress_agregate.items():
                    if key != x_axis:
                        for i in range(len(x)):
                            summary_writer.add_scalar(key, value[i], x[i])
            else:
                warnings.warn(f"{x_axis} is not in progress_average")
            summary_writer.close()

    return progress_agregate


if __name__ == '__main__':
    # Take log_dir (e.g. 'data/local/experiment/2025_01_27_20_15_23_test') as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str, help='Path to the log directory.')
    
    args = parser.parse_args()

    # Add directories under log_dir to log_dirs
    log_dirs = dict()
    for dir in os.listdir(args.log_dir):
        config_path = os.path.join(args.log_dir, dir, 'config.yaml')
        if not os.path.isfile(config_path):
            print(f"Skipping {dir} as config.yaml is not found.")
            continue
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file)
        if not cfg['exp_name'] in log_dirs:
            log_dirs[cfg['exp_name']] = []
        log_dirs[cfg['exp_name']].append(os.path.join(args.log_dir, dir))

    # aggregation
    save_dir = os.path.join(args.log_dir, 'aggregated')
    for exp_name, l_dirs in log_dirs.items():
        # Average
        average_log_dir = os.path.join(save_dir, f"{exp_name}_average")
        aggregate_progresses(log_dirs=l_dirs, 
                             aggregate_log_dir=average_log_dir, 
                             mode='average', 
                             use_tensorboard=True,
                             is_separate_logs=True)
        # Std
        std_log_dir = os.path.join(save_dir, f"{exp_name}_std")
        aggregate_progresses(log_dirs=l_dirs, 
                             aggregate_log_dir=std_log_dir, 
                             mode='std', 
                             use_tensorboard=True,
                             is_separate_logs=True)
        print("Succeeded !")


