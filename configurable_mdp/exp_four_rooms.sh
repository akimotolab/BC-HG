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
            echo "Usage: bash configurable_mdp/exp_four_rooms.sh [--dry-run] [--background]"
            exit 1
            ;;
    esac
    shift
done

if [[ "$BACKGROUND" == true && "$DRY_RUN" == true ]]; then
    echo "[WARN] --dry-run does not run in background mode."
    BACKGROUND=false
fi

if [[ "$BACKGROUND" == true && "${EXP_FOUR_ROOMS_DAEMONIZED:-0}" != "1" ]]; then
    timestamp="$(date +%Y%m%d_%H%M%S)"
    log_file="four_rooms_experiments_${timestamp}.log"

    nohup env EXP_FOUR_ROOMS_DAEMONIZED=1 bash "$SCRIPT_PATH" > "$log_file" 2>&1 &
    echo "Started background execution."
    echo "PID: $!"
    echo "LOG: $log_file"
    exit 0
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry-run mode: showing execution plan only."
else
    echo "Executing script..."
fi

# Methods to run
methods=(
    bchg baseline sobirl hpgd hpgd_sarsa hpgd_oracle
)
# Experiments to run
exps=(
    experiment_reg_lambda_0_001_total_steps_100
    experiment_reg_lambda_0_001_total_steps_200
    experiment_reg_lambda_0_001_total_steps_400
    experiment_reg_lambda_0_001_total_steps_1000
    experiment_reg_lambda_0_003_total_steps_100
    experiment_reg_lambda_0_003_total_steps_200
    experiment_reg_lambda_0_003_total_steps_400
    experiment_reg_lambda_0_003_total_steps_1000
    experiment_reg_lambda_0_005_total_steps_100
    experiment_reg_lambda_0_005_total_steps_200
    experiment_reg_lambda_0_005_total_steps_400
    experiment_reg_lambda_0_005_total_steps_1000
)
# GPU assignment for each method
declare -A method_gpu=(
    [baseline]=0
    [sobirl]=0
    [hpgd]=0
    [hpgd_sarsa]=1
    [bchg]=1
    [hpgd_oracle]=2
)

declare -A gpu_methods=()
for method in "${methods[@]}"; do
    gpu="${method_gpu[$method]:-}"
    if [[ -z "$gpu" ]]; then
        echo "[SKIP] Skipping method=${method} because GPU is not configured"
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
            train_script="configurable_mdp/train_four_rooms_${method}.py"

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
    echo "All GPU queue experiments have completed."
else
    echo "Dry-run complete: no experiments were executed."
fi
