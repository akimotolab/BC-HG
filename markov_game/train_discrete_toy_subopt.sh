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

# Environment
env_id=DiscreteToy4_R1-v0  # DiscreteToy4_R1-v0 / DiscreteToy4_R2-v0

# markov_game/config/$env_id/config_*.yaml
configs=(
    "config_bchg.yaml"
    "config_biac_offpolicy.yaml"
    "config_baseline_onpolicy.yaml"
    "config_biac_onpolicy.yaml"
    "config_baseline_offpolicy.yaml"
)
cfg_idx=0  # Config file index to use

# Experiment settings
datetime=$(date +"%Y%m%d_%H%M%S")
log_label="$datetime"
config="${configs[cfg_idx]}"  # Config file to use
seed="[0,1,2,3,4,5,6,7,8,9]"
gpu_id=0  # GPU ID to use (e.g., 0, 1, 2, ...)

# Overwrite config parameters
algos=(
    "BCHGDiscrete_Subopt"
    "BiAC_Subopt"
    "BaselineDiscrete_Subopt"
    "BiAC_Subopt"
    "BaselineDiscrete_Subopt"
)
algo=${algos[cfg_idx]}
sweep_follower="['stop_q_iteration','reset_q']"
algo_follower=SoftQIteration_Subopt
stop_q_iterations=(1 2 5 10 50 100)
reset_qs=(True False)  # True: Reset-Q, False: Carry-Q

for stop_q_iteration in "${stop_q_iterations[@]}"; do
    for reset_q in "${reset_qs[@]}"; do

        if [ "$reset_q" = "True" ]; then
            name=${env_id}_${algos[cfg_idx]}_reset_q
        else
            name=${env_id}_${algos[cfg_idx]}_carry_q
        fi

        safe_cfg_label=$(echo "$config" | sed 's/\.yaml$//')
        log_file="$log_label"_"$env_id"_"$safe_cfg_label"_stop_"$stop_q_iteration"_reset_"$reset_q".log

        # Run experiments in the background for all the combinations of parameters
        nohup python markov_game/train_discrete_toy.py \
            --config "markov_game/config/$env_id/$config" \
            name="$name" \
            datetime="'$datetime'" \
            seed="$seed" \
            gpu_id="$gpu_id" \
            sweep.follower="$sweep_follower" \
            leader.algo="$algo" \
            follower.algo="$algo_follower" \
            follower.stop_q_iteration="$stop_q_iteration" \
            follower.reset_q="$reset_q" \
            > "$log_file" 2>&1 &
    done
done

# Note: if you use --no_aggregate option, execute "scripts/aggregate_results.py" after all the experiments are done