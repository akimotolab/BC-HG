#!/bin/bash
set -euo pipefail

DRY_RUN=false
BACKGROUND=false
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            ;;
        --background)
            BACKGROUND=true
            ;;
        *)
            echo "Usage: bash configurable_mdp/exp_4rooms_subopt_2.sh [--dry-run] [--background]"
            exit 1
            ;;
    esac
    shift
done

if [[ "$BACKGROUND" == true && "$DRY_RUN" == true ]]; then
    echo "[WARN] --dry-run ではバックグラウンド実行しないため、--background を無効化します。"
    BACKGROUND=false
fi

if [[ "$BACKGROUND" == true && "${EXP_4ROOMS_SUBOPT_DAEMONIZED:-0}" != "1" ]]; then
    timestamp="$(date +%Y%m%d_%H%M%S)"
    daemon_log_file="HPGD/exp_4rooms_subopt_2_${timestamp}.log"

    nohup env EXP_4ROOMS_SUBOPT_DAEMONIZED=1 bash "$SCRIPT_PATH" > "$daemon_log_file" 2>&1 &
    echo "バックグラウンド実行を開始しました。"
    echo "PID: $!"
    echo "LOG: $daemon_log_file"
    exit 0
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "dry-run モード: 実行計画のみ表示します。"
else
    if [[ "${EXP_4ROOMS_SUBOPT_DAEMONIZED:-0}" == "1" ]]; then
        echo "デーモン実行のため確認プロンプトをスキップします。"
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
fi

# Methods to run
methods=(
    baseline sobirl hpgd_sarsa hpgd proposal_5 hpgd_oracle
)
# Experiments to run
data_dir="/home/mikoto/workspace/ptia/HPGD/data/journal/subopt2"
exps=(
    experiment_reg_lambda_0_003_total_steps_100_subopt_1
    experiment_reg_lambda_0_003_total_steps_200_subopt_1
    experiment_reg_lambda_0_003_total_steps_400_subopt_1
    experiment_reg_lambda_0_003_total_steps_1000_subopt_1
    experiment_reg_lambda_0_003_total_steps_100_subopt_2
    experiment_reg_lambda_0_003_total_steps_200_subopt_2
    experiment_reg_lambda_0_003_total_steps_400_subopt_2
    experiment_reg_lambda_0_003_total_steps_1000_subopt_2
    experiment_reg_lambda_0_003_total_steps_100_subopt_5
    experiment_reg_lambda_0_003_total_steps_200_subopt_5
    experiment_reg_lambda_0_003_total_steps_400_subopt_5
    experiment_reg_lambda_0_003_total_steps_1000_subopt_5
    experiment_reg_lambda_0_003_total_steps_100_subopt_10
    experiment_reg_lambda_0_003_total_steps_200_subopt_10
    experiment_reg_lambda_0_003_total_steps_400_subopt_10
    experiment_reg_lambda_0_003_total_steps_1000_subopt_10
    experiment_reg_lambda_0_003_total_steps_100_subopt_20
    experiment_reg_lambda_0_003_total_steps_200_subopt_20
    experiment_reg_lambda_0_003_total_steps_400_subopt_20
    experiment_reg_lambda_0_003_total_steps_1000_subopt_20
    experiment_reg_lambda_0_003_total_steps_100_subopt_50
    experiment_reg_lambda_0_003_total_steps_200_subopt_50
    experiment_reg_lambda_0_003_total_steps_400_subopt_50
    experiment_reg_lambda_0_003_total_steps_1000_subopt_50
)
# GPU assignment for each method
declare -A method_gpu=(
    [baseline]=0
    [sobirl]=0
    [hpgd]=1
    [hpgd_sarsa]=0
    [proposal_5]=1
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
            experiment_dir="${data_dir}/${exp}"
            log_file="HPGD/experiment_${method}_${exp}.log"
            train_script="HPGD/train_stochastic_bilevel_${method}_subopt_2.py"

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
