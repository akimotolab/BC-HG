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

method=(
    "baseline" "configurable_mdp" "hpgd_td" "hpgd_oracle" "proposal" "sobirl"
)
exp=(
    "experiment_bilevel_lqr__reg_lambda_0_1_steps_10000"
)

# gpu:0 - baseline, hogd, sobirl
nohup bash -c "

    CUDA_VISIBLE_DEVICES=0 python configurable_mdp/train_bilevel_lqr_${method[0]}.py \
        --experiment_dir configurable_mdp/data/${exp[0]} \
        > configurable_mdp/experiment_${method[0]}_${exp[0]}.log 2>&1 && \
    echo 'Experiment (${method[0]} - ${exp[0]}) completed' && \

    CUDA_VISIBLE_DEVICES=0 python configurable_mdp/train_bilevel_lqr_${method[1]}.py \
        --experiment_dir configurable_mdp/data/${exp[0]} \
        > configurable_mdp/experiment_${method[1]}_${exp[0]}.log 2>&1 && \
    echo 'Experiment (${method[1]} - ${exp[0]}) completed' && \

    CUDA_VISIBLE_DEVICES=0 python configurable_mdp/train_bilevel_lqr_${method[5]}.py \
        --experiment_dir configurable_mdp/data/${exp[0]} \
        > configurable_mdp/experiment_${method[5]}_${exp[0]}.log 2>&1 && \
    echo 'Experiment (${method[5]} - ${exp[0]}) completed'

    " >> "configurable_mdp/experiment_gpu0.log" 2>&1 &

# gpu:1 - hpgd_td, proposal
nohup bash -c "

    CUDA_VISIBLE_DEVICES=1 python configurable_mdp/train_bilevel_lqr_${method[2]}.py \
        --experiment_dir configurable_mdp/data/${exp[0]} \
        > configurable_mdp/experiment_${method[2]}_${exp[0]}.log 2>&1 && \
    echo 'Experiment (${method[2]} - ${exp[0]}) completed' && \

    CUDA_VISIBLE_DEVICES=1 python configurable_mdp/train_bilevel_lqr_${method[4]}.py \
        --experiment_dir configurable_mdp/data/${exp[0]} \
        > configurable_mdp/experiment_${method[4]}_${exp[0]}.log 2>&1 && \
    echo 'Experiment (${method[4]} - ${exp[0]}) completed'

    " >> "configurable_mdp/experiment_gpu1.log" 2>&1 &

# gpu:2 - hpgd_oracle
nohup bash -c "

    CUDA_VISIBLE_DEVICES=2 python configurable_mdp/train_bilevel_lqr_${method[3]}.py \
        --experiment_dir configurable_mdp/data/${exp[0]} \
        > configurable_mdp/experiment_${method[3]}_${exp[0]}.log 2>&1 && \
    echo 'Experiment (${method[3]} - ${exp[0]}) completed'

    " >> "configurable_mdp/experiment_gpu2.log" 2>&1 &

echo "Experiments started in background."
