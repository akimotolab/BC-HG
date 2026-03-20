#!/bin/bash
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

if [[ "$#" -gt 0 ]]; then
    echo "Usage: bash configurable_mdp/exp_4room_subopt.sh [--dry-run]"
    exit 1
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "dry-run モード: 実行計画のみ表示します。"
else
    echo "このスクリプトを実行しますか？"
    read -r -p "実行する場合は 'y' を入力してください (y/n): " confirm

    case "$confirm" in
        [Yy]*)
            echo "スクリプトを実行します..."
            ;;
        [Nn]*)
            echo "実行をキャンセルしました。"
            exit 0
            ;;
        *)
            echo "無効な入力です。実行をキャンセルしました。"
            exit 1
            ;;
    esac
fi

# Methods to run
methods=(
    baseline sobirl hpgd_sarsa hpgd proposal hpgd_oracle
)
# Experiments to run
exps=(
    experiment_reg_lambda_0_003_total_steps_100_g19_subopt_0_1
    experiment_reg_lambda_0_003_total_steps_100_g19_subopt_0_01
    experiment_reg_lambda_0_003_total_steps_100_g19_subopt_0_5
    experiment_reg_lambda_0_003_total_steps_100_g19_subopt_0_05
    experiment_reg_lambda_0_003_total_steps_200_g19_subopt_0_1
    experiment_reg_lambda_0_003_total_steps_200_g19_subopt_0_01
    experiment_reg_lambda_0_003_total_steps_200_g19_subopt_0_5
    experiment_reg_lambda_0_003_total_steps_200_g19_subopt_0_05
    experiment_reg_lambda_0_003_total_steps_400_g19_subopt_0_1
    experiment_reg_lambda_0_003_total_steps_400_g19_subopt_0_01
    experiment_reg_lambda_0_003_total_steps_400_g19_subopt_0_5
    experiment_reg_lambda_0_003_total_steps_400_g19_subopt_0_05
    experiment_reg_lambda_0_003_total_steps_1000_g19_subopt_0_1
    experiment_reg_lambda_0_003_total_steps_1000_g19_subopt_0_01
    experiment_reg_lambda_0_003_total_steps_1000_g19_subopt_0_5
    experiment_reg_lambda_0_003_total_steps_1000_g19_subopt_0_05
)
# GPU assignment for each method
declare -A method_gpu=(
    [baseline]=0
    [sobirl]=0
    [hpgd]=0
    [hpgd_sarsa]=1
    [proposal]=1
    [hpgd_oracle]=2
)

declare -A gpu_methods=()
for method in "${methods[@]}"; do
    gpu="${method_gpu[$method]:-}"
    if [[ -z "$gpu" ]]; then
        echo "[SKIP] GPU未設定のため method=${method} をスキップします"
        continue
    fi
    gpu_methods["$gpu"]+=" $method"
done

run_gpu_queue() {
    local gpu="$1"
    local method_list="$2"

    echo "[GPU ${gpu}] queue start:${method_list}"

    for method in $method_list; do
        for exp in "${exps[@]}"; do
            experiment_dir="configurable_mdp/data/${exp}"
            log_file="configurable_mdp/experiment_${method}_${exp}.log"
            train_script="configurable_mdp/train_stochastic_bilevel_${method}_subopt.py"

            if [[ "$DRY_RUN" == true ]]; then
                echo "[PLAN][GPU ${gpu}] CUDA_VISIBLE_DEVICES=${gpu} python ${train_script} --experiment_dir ${experiment_dir} > ${log_file} 2>&1"
            else
                echo "[RUN][GPU ${gpu}] method=${method}, exp=${exp}"
                CUDA_VISIBLE_DEVICES="$gpu" python "$train_script" \
                    --experiment_dir "$experiment_dir" \
                    > "$log_file" 2>&1
                echo "[DONE][GPU ${gpu}] ${method} - ${exp}"
            fi
        done
    done

    echo "[GPU ${gpu}] queue finished"
}

pids=()
while IFS= read -r gpu; do
    if [[ "$DRY_RUN" == true ]]; then
        run_gpu_queue "$gpu" "${gpu_methods[$gpu]}"
    else
        run_gpu_queue "$gpu" "${gpu_methods[$gpu]}" &
        pids+=("$!")
    fi
done < <(printf '%s\n' "${!gpu_methods[@]}" | sort -n)

if [[ "$DRY_RUN" == false ]]; then
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo "全GPUキューの実験が完了しました。"
else
    echo "dry-run 完了: 実験は実行していません。"
fi
