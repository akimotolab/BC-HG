"""
aggregate_results.py - 学習実験の複数シード実行結果を集約するスクリプト

【概要】
強化学習の学習実験において、同じ実験設定で異なるシード（ランダムシード）により
実行された複数の学習結果を集約し、平均値・標準偏差などの統計量を計算するスクリプト。
結果はCSVファイルとTensorBoardログとして保存される。

【主な機能】
1. aggregate_progresses 関数:
   - 複数のprogress.csvファイルを読み込み、指定されたモード（average, median, max, min, std）で集約
   - eval, leader, follower の3つのログを個別に処理（is_separate_logs=True の場合）
   - 結果をCSVとTensorBoard形式で保存
   - 全ての入力データが同じ行数を持つことを検証

2. メイン処理:
   - 指定ディレクトリ直下のサブディレクトリをスキャン
   - 各ディレクトリのconfig.yamlから実験名（exp_name）を取得
   - 同じ実験名を持つディレクトリをグループ化
   - 各実験名ごとに平均値（average）と標準偏差（std）を計算して保存

【ディレクトリ構造の想定】
入力:
  log_dir/
    ├── run_seed_0/  (config.yaml に exp_name が含まれる学習ディレクトリ)
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

出力:
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

【使い方】
基本:
  python aggregate_results.py LOG_DIR

例:
  python aggregate_results.py data/local/experiment/2025_01_27_20_15_23_test

【引数】
  log_dir: 学習結果が含まれるディレクトリのパス（必須）
           このディレクトリ直下に複数の学習実行ディレクトリが存在する想定

【注意事項】
- 各実行ディレクトリには config.yaml が必要（exp_name を取得するため）
- 同じ実験名を持つディレクトリの progress.csv は同じ行数である必要がある
- 行数が異なる場合はエラーが発生する
- 空のCSVファイルが存在する場合もエラーが発生する
- config.yaml が存在しないディレクトリはスキップされる

【evaluate_aggregate.py との違い】
- aggregate_results.py: 学習時の結果を集約（シンプルな1階層構造）
- evaluate_aggregate.py: 評価時の結果を集約（再帰的探索、イテレーション別）
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
    # log_dir（e.g. 'data/local/experiment/2025_01_27_20_15_23_test'）を引数に取る
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str, help='Path to the log directory.')
    
    args = parser.parse_args()

    # log_dir傘下のディレクトリのをlog_dirsに追加
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


