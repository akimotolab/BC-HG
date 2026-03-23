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

# markov_game/config/config_discrete_toy_*.yaml
configs=(
    "config_discrete_toy_bchg.yaml"
    "config_discrete_toy_biac.yaml"
    "config_discrete_toy_baseline.yaml"
    # "config_discrete_toy_biac_on.yaml"
    # "config_discrete_toy_baseline_off.yaml"
)

# Experiment settings
datetime=$(date +"%Y%m%d_%H%M%S")
log_label="$datetime"
config="${configs[0]}"  # Config file to use
gpu_id=0  # GPU ID to use (e.g., 0, 1, 2, ...)

# Leader grid search parameters
actor_update_steps_ns=(1 2 5 10 20)
critic_update_steps_ns=(1 2 5 10 20)

for actor_update_steps_n in "${actor_update_steps_ns[@]}"; do
    for critic_update_steps_n in "${critic_update_steps_ns[@]}"; do

        safe_cfg_label=$(echo "$config" | sed 's/\.yaml$//')
        log_file="$log_label"_"$safe_cfg_label"_an_"$actor_update_steps_n"_cn_"$critic_update_steps_n".log

        # Run experiments in the background for all the combinations of parameters
        nohup python markov_game/train_discrete_toy.py \
            --config "markov_game/config/$config" \
            --no_aggregate \
            datetime="'$datetime'" \
            gpu_id="$gpu_id" \
            leader.actor_update_steps_n="$actor_update_steps_n" \
            leader.critic_update_steps_n="$critic_update_steps_n" \
            > "$log_file" 2>&1 &
    done
done

# Note: if you use --no_aggregate option, execute "scripts/aggregate_results.py" after all the experiments are done