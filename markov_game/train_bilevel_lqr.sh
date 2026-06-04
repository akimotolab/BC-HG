#!/bin/bash

# Execution check
echo "Do you want to run this script?"
read -p "Enter 'y' to run (y/n): " confirm

case $confirm in
    [Yy]* )
        echo "Executing script..."
        ;;
    [Nn]* )
        echo "Execution canceled."
        exit 0
        ;;
    * )
        echo "Invalid input. Execution canceled."
        exit 1
        ;;
esac

# markov_game/config/LQREnv-v4/config_*.yaml
configs=(
    "config_bchg.yaml"
    "config_baseline_onpolicy.yaml"
    "config_baseline_offpolicy.yaml"
)
cfg_idx=0  # Config file index to use

# Experiment settings
datetime=$(date +"%Y%m%d_%H%M%S")
log_label="$datetime"
config="${configs[cfg_idx]}"  # Config file to use
seed="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]"
gpu_id=0  # GPU ID to use (e.g., 0, 1, 2, ...)

# Leader grid search parameters
actor_update_steps_ns=(1 2 5 10)

for actor_update_steps_n in "${actor_update_steps_ns[@]}"; do

    safe_cfg_label=$(echo "$config" | sed 's/\.yaml$//')
    log_file="$log_label"_"$safe_cfg_label"_an_"$actor_update_steps_n".log

    # Run experiments in the background for all the combinations of parameters
    nohup python markov_game/train_bilevel_lqr.py \
        --config "markov_game/config/LQREnv-v4/$config" \
        datetime="'$datetime'" \
        seed="$seed" \
        gpu_id="$gpu_id" \
        leader.actor_update_steps_n="$actor_update_steps_n" \
        > "$log_file" 2>&1 &
done

# Note: if you use --no_aggregate option, execute "scripts/aggregate_results.py" after all the experiments are done