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

name="LQR"
datetime=$(date +"%Y%m%d_%H%M%S")
seed="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]"
gpu_id=2

log_label="$datetime"

#sweep
sweep_env="['id']"
sweep_leader="['algo', 'init_mean_K_1', 'init_mean_K_2', 'actor_update_steps_n', 'critic_update_steps_n', 'replay_buffer_size']"
sweep_follower="[]"

# ctxt
snapshot_mode="gap"
snapshot_gap=100

# env
id="LQREnv-v4"
cost_lambda=1.0

# leader
algo=BCHG_Opt  # Baseline_LQR / BCHG_Opt
init_mean_K_1=-2.0
init_mean_K_1s=(0.0)
init_mean_K_2=-1.0
init_mean_K_2s=(0.0)
init_std_K_1=1.0
init_std_K_2=1.0
fixed_W=1e-3
replay_buffer_size=400  # 400 / 1e6
n_epochs=500
max_total_env_steps=200000  # 400 * 500
sample_num_eps=4
eval_num_eps=10
# For Baseline_LQR / BCHG_Opt
buffer_batch_size=64
min_buffer_size=400  # Steps after which the leader starts updating
init_steps=4000  # Steps until which the leader acts randomly
l_discount=0.99
max_policy_grad_norm=1.0
policy_lr=1e-1
qf_lr=1e-1
actor_update_steps_n=10
actor_update_steps_ns=(1 2 5 10)
critic_update_steps_n=20
critic_update_steps_ns=(5)
batch_size_for_la_exp=32
on_policy=true  # true / false
reset_leader_qf=false
reset_leader_qf_optimizer=false
use_advantage=true
use_advantage_in_influence=false
use_closed_form_gradient=false
use_K_L2_regularization=false
K_L2_reg_coef=0.0

# follower
# For MaxEntLQR
f_discount=0.95
beta=1e-1
max_iterations=1000
tol=1e-10

for actor_update_steps_n in "${actor_update_steps_ns[@]}"; do
    for critic_update_steps_n in "${critic_update_steps_ns[@]}"; do
        for init_mean_K_1 in "${init_mean_K_1s[@]}"; do
            for init_mean_K_2 in "${init_mean_K_2s[@]}"; do
                
                log_file="$log_label"_"$name"_"$algo"_K_"$init_mean_K_1"_"$init_mean_K_2"_an_"$actor_update_steps_n"_cn_"$critic_update_steps_n".log

                # run the script with the current seed
                nohup python scripts/train_bilevel_lqr.py \
                    --no_aggregate \
                    name="$name"_"$id" \
                    datetime="'$datetime'" \
                    seed="$seed" \
                    gpu_id="$gpu_id" \
                    sweep.env="$sweep_env" \
                    sweep.leader="$sweep_leader" \
                    sweep.follower="$sweep_follower" \
                    ctxt.snapshot_mode="$snapshot_mode" \
                    ctxt.snapshot_gap="$snapshot_gap" \
                    env.id="$id" \
                    env.cost_lambda="$cost_lambda" \
                    leader.algo="$algo" \
                    leader.init_mean_K_1="$init_mean_K_1" \
                    leader.init_mean_K_2="$init_mean_K_2" \
                    leader.init_std_K_1="$init_std_K_1" \
                    leader.init_std_K_2="$init_std_K_2" \
                    leader.fixed_W="$fixed_W" \
                    leader.replay_buffer_size="$replay_buffer_size" \
                    leader.n_epochs="$n_epochs" \
                    leader.max_total_env_steps="$max_total_env_steps" \
                    leader.sample_num_eps="$sample_num_eps" \
                    leader.eval_num_eps="$eval_num_eps" \
                    leader.buffer_batch_size="$buffer_batch_size" \
                    leader.min_buffer_size="$min_buffer_size" \
                    leader.init_steps="$init_steps" \
                    leader.discount="$l_discount" \
                    leader.max_policy_grad_norm="$max_policy_grad_norm" \
                    leader.policy_lr="$policy_lr" \
                    leader.qf_lr="$qf_lr" \
                    leader.actor_update_steps_n="$actor_update_steps_n" \
                    leader.critic_update_steps_n="$critic_update_steps_n" \
                    leader.batch_size_for_la_exp="$batch_size_for_la_exp" \
                    leader.on_policy="$on_policy" \
                    leader.reset_leader_qf="$reset_leader_qf" \
                    leader.reset_leader_qf_optimizer="$reset_leader_qf_optimizer" \
                    leader.use_advantage="$use_advantage" \
                    leader.use_advantage_in_influence="$use_advantage_in_influence" \
                    leader.use_closed_form_gradient="$use_closed_form_gradient" \
                    leader.use_K_L2_regularization="$use_K_L2_regularization" \
                    leader.K_L2_reg_coef="$K_L2_reg_coef" \
                    follower.discount="$f_discount" \
                    follower.beta="$beta" \
                    follower.max_iterations="$max_iterations" \
                    follower.tol="$tol" \
                    > "$log_file" 2>&1 &
            done
        done
    done
done
