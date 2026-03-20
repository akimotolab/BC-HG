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

name="MG_Discrete"
datetime=$(date +"%Y%m%d_%H%M%S")
seed="[0,1,2,3,4,5,6,7,8,9]"
gpu_id=0

log_label="$datetime"

#sweep
sweep_env="[]"
# sweep_leader="['algo', 'gradient_steps_n', 'policy_lr', 'qf_lr', 'replay_buffer_size']"  # BCHGDiscrete_Opt_2
sweep_leader="['algo', 'actor_update_steps_n', 'critic_update_steps_n', 'replay_buffer_size']"  # BCHGDiscrete_Opt
sweep_follower="[]"

# ctxt
snapshot_mode="last"
snapshot_gap=1

# env
id=DiscreteToy4_1b-v1
#    DiscreteToy1_1a-v0
#    DiscreteToy1_2a-v0
#    DiscreteToy1_2b-v0
#    DiscreteToy1_2c-v0
#    DiscreteToy1_2d-v0
#    DiscreteToy1_2e-v0
#    DiscreteToy1_2f-v0
#    DiscreteToy1_2g-v0
#    DiscreteToy4_1a-v0
#    DiscreteToy4_1a-v1
#    DiscreteToy4_1b-v0
#    DiscreteToy4_1b-v1

# leader
algo=BCHGDiscrete_Opt  # BCHGDiscrete_Opt / BCHGDiscrete_Opt_2 / BaselineDiscrete_SQI / BaselineDiscrete_SQI_2 / BiAC_SQI
replay_buffer_size=450  # 450 / 1e6
n_epochs=250
max_total_env_steps=225000
sample_num_eps=3
eval_num_eps=10
# For BaselineDiscrete_SQI / BCHGDiscrete_Opt
buffer_batch_size=64
min_buffer_size=450  # Steps after which the leader starts updating
target_update_tau=1e-2
policy_lr=1e-4
# policy_lrs=(1e-1 1e-2 1e-3 1e-4)  # BCHGDiscrete_Opt_2
qf_lr=1e-3
# qf_lrs=(1e-1 1e-2 1e-3 1e-4)  # BCHGDiscrete_Opt_2
actor_update_steps_n=10
actor_update_steps_ns=(1 2 5 10 20)  # BCHGDiscrete_Opt
critic_update_steps_n=20
critic_update_steps_ns=(1 2 5 10 20)  # BCHGDiscrete_Opt
on_policy=True  # True / False
reset_leader_qf=False
reset_leader_qf_optimizer=False
use_advantage=True
use_advantage_in_influence=True
# For BaselineDiscrete_SQI_2 / BCHGDiscrete_Opt_2 / BiAC_SQI
gradient_steps_n=1

# follower
# For SoftQIteration
temperature=0.05
max_iterations=1000

# for policy_lr in "${policy_lrs[@]}"; do  # BCHGDiscrete_Opt_2
    # for qf_lr in "${qf_lrs[@]}"; do  # BCHGDiscrete_Opt_2
        # log_file="$log_label"_"$name"_"$algo"_plr_"$policy_lr"_qlr_"$qf_lr".log  # BCHGDiscrete_Opt_2
for actor_update_steps_n in "${actor_update_steps_ns[@]}"; do  # BCHGDiscrete_Opt
    for critic_update_steps_n in "${critic_update_steps_ns[@]}"; do  # BCHGDiscrete_Opt
        log_file="$log_label"_"$name"_"$algo"_"an_$actor_update_steps_n"_cn_"$critic_update_steps_n".log  # BCHGDiscrete_Opt

        # run the script with the current seed
        nohup python scripts/train_discrete_toy.py \
            --no_aggregate \
            name="$name"_"$id" \
            datetime="'$datetime'" \
            seed="$seed" \
            gpu_id="$gpu_id" \
            sweep.leader="$sweep_leader" \
            sweep.follower="$sweep_follower" \
            sweep.env="$sweep_env" \
            ctxt.snapshot_mode="$snapshot_mode" \
            ctxt.snapshot_gap="$snapshot_gap" \
            env.id="$id" \
            leader.algo="$algo" \
            leader.replay_buffer_size="$replay_buffer_size" \
            leader.n_epochs="$n_epochs" \
            leader.max_total_env_steps="$max_total_env_steps" \
            leader.eval_num_eps="$eval_num_eps" \
            leader.sample_num_eps="$sample_num_eps" \
            leader.buffer_batch_size="$buffer_batch_size" \
            leader.min_buffer_size="$min_buffer_size" \
            leader.target_update_tau="$target_update_tau" \
            leader.policy_lr="$policy_lr" \
            leader.qf_lr="$qf_lr" \
            leader.actor_update_steps_n="$actor_update_steps_n" \
            leader.critic_update_steps_n="$critic_update_steps_n" \
            leader.on_policy="$on_policy" \
            leader.reset_leader_qf="$reset_leader_qf" \
            leader.reset_leader_qf_optimizer="$reset_leader_qf_optimizer" \
            leader.use_advantage="$use_advantage" \
            leader.use_advantage_in_influence="$use_advantage_in_influence" \
            leader.gradient_steps_n="$gradient_steps_n" \
            follower.temperature="$temperature" \
            follower.max_iterations="$max_iterations" \
            > "$log_file" 2>&1 &
    done
done