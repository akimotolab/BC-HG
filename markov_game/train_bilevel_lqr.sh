#!/bin/bash

# 実行確認
echo "このスクリプトを実行しますか？"
read -p "実行する場合は 'y' を入力してください (y/n): " confirm

case $confirm in
    [Yy]* )
        echo "スクリプトを実行します..."
        ;;
    [Nn]* )
        echo "実行をキャンセルしました。"
        exit 0
        ;;
    * )
        echo "無効な入力です。実行をキャンセルしました。"
        exit 1
        ;;
esac

# markov_game/config/config_bilevel_lqr_*.yaml
configs=(
    "config_bilevel_lqr_bchg.yaml"
    "config_bilevel_lqr_baseline_on.yaml"
    "config_bilevel_lqr_baseline_off.yaml"
)

# Experiment settings
exp_name="BilevelLQR"
datetime=$(date +"%Y%m%d_%H%M%S")
log_label="$datetime"
config="${configs[0]}"  # Config file to use
gpu_id=0  # GPU ID to use (e.g., 0, 1, 2, ...)

# Leader grid search parameters
actor_update_steps_ns=(1 2 5 10)

for actor_update_steps_n in "${actor_update_steps_ns[@]}"; do

    log_file="$log_label"_"$exp_name"_"$algo"_an_"$actor_update_steps_n".log

    # Run experiments in the background for all the combinations of parameters
    nohup python scripts/train_bilevel_lqr.py \
        --config "markov_game/config/$config" \
        --no_aggregate \
        datetime="'$datetime'" \
        gpu_id="$gpu_id" \
        leader.actor_update_steps_n="$actor_update_steps_n" \
        > "$log_file" 2>&1 &
done

# Note: if you use --no_aggregate option, execute "scripts/aggregate_results.py" after all the experiments are done