import os
import sys
import time
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from garage.torch import prefer_gpu, set_gpu_mode
from garage.torch.policies import DeterministicMLPPolicy, TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.experiment.deterministic import set_seed, get_seed

from src.experiment import (
    Trainer, 
    wrap_experiment, 
    set_seed as custom_set_seed,
    kwargs_from_cfg, get_algo, 
    args_for_experiments, check_all_keys_exist, HierarchicalSweeper
)
from src.policies import JointPolicy, LinearGaussianPolicy
from src.envs import GymEnv, normalize
from src.replay_buffer import GammaReplayBuffer
from src.follower import FollowerWrapper
from src.sampler import LocalSampler, VecWorker

@wrap_experiment
def main(ctxt, cfg):

    # Save config file
    output_config_path = os.path.join(ctxt.snapshot_dir, 'config.yaml')
    OmegaConf.save(config=cfg, f=output_config_path)

    l_params = cfg['leader']
    f_params = cfg['follower']
    
    set_seed(cfg['seed'])
    custom_set_seed(get_seed())

    trainer = Trainer(ctxt)

    env_kwargs = {k:v for k,v in cfg['env'].items() if k != 'id'}
    env = GymEnv(cfg['env']['id'], **env_kwargs)
    env.seed(get_seed())
    eval_env = trainer.get_env_copy(env)
    eval_env.seed(get_seed())

    # Define the follower agent
    follower_algo = get_algo(f_params['algo'])
    follower = FollowerWrapper(
        algo=follower_algo,
        fixed_policy=f_params['fixed_policy'],
        env=env,
        **kwargs_from_cfg(f_params, follower_algo)
        )
    
    # Define the leader agent
    leader_policy = LinearGaussianPolicy(
        env_spec=env.spec.leader_policy_env_spec,
        init_mean_K=[[l_params['init_mean_K_1'], l_params['init_mean_K_2']]],
        init_std_K=[[l_params['init_std_K_1'], l_params['init_std_K_2']]],
        fixed_W=[[l_params['fixed_W']]],
        )
    leader_qf = ContinuousMLPQFunction(
        env_spec=env.spec.leader_qf_env_spec,
        hidden_sizes=l_params['qf_hidden_sizes'],
        hidden_nonlinearity=nn.ReLU
        )
    leader_replay_buffer = GammaReplayBuffer(
        env_spec=env.spec.leader_policy_env_spec,
        size=int(l_params['replay_buffer_size']),
        gamma=l_params['discount']
        )
    algo = get_algo(l_params['algo'])
    leader = algo(
        env_spec=env.spec,
        policy=leader_policy,
        qf=leader_qf,
        replay_buffer=leader_replay_buffer,
        **kwargs_from_cfg(l_params, algo)
        )

    joint_policy = JointPolicy(
        env_spec=env.spec,
        leader_policy=leader.policy, 
        follower_policy=follower.policy,
        )
    sampler = LocalSampler(
        agents=joint_policy,
        envs=env,
        max_episode_length=env.spec.max_episode_length,
        n_workers=1,
        worker_class=VecWorker,
        worker_args=dict(
            n_envs=l_params['sample_num_eps'],
            )
        )
    eval_sampler = LocalSampler(
        agents=joint_policy,
        envs=eval_env,
        max_episode_length=eval_env.spec.max_episode_length,
        n_workers=1,
        worker_class=VecWorker,
        worker_args=dict(
            n_envs=l_params['eval_num_eps'],
            )
        )
    
    # Move to GPU
    follower.to()
    leader.to()

    # Train
    trainer.setup(
        env=env,
        leader=leader,
        follower=follower,
        sampler=sampler,
        eval_env=eval_env,
        eval_sampler=eval_sampler,
        learner='leader'
        )
    trainer.train(
        n_epochs=l_params['n_epochs'],
        batch_size=int(l_params['sample_num_eps'] * env.spec.max_episode_length),
        max_total_env_steps=l_params['max_total_env_steps'],
        )

    return ctxt.snapshot_dir

    
if __name__ == '__main__':
    try:
        #multiprocessing.set_start_method('spawn', force=True)
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 既に設定されている場合は無視する

    # パラメータを読み込む
    parser = argparse.ArgumentParser(description='Train a model with given config and experiment name.')
    parser.add_argument(
        '--config', type=str, default=os.path.join('markov_game', 'config', 'config_bilevel_lqr_bchg.yaml'), 
        help='Path to the config file.'
        )
    parser.add_argument('--no_aggregate', action='store_true', help='Do not aggregate the results.')
    args, unknown_args = parser.parse_known_args()

    # config.yamlを読み込む
    config = OmegaConf.load(args.config)
    print(f"Loaded config: {args.config}")

    # config.yamlの内容を上書きする
    # 上書きしたい内容をコマンドライン引数で指定する
    # 例: name=experiment_001 leader.policy_lr=1e-4 env.id=GuidedPendulum-v0
    try:
        cli_conf = OmegaConf.from_cli(unknown_args)
        # None, none, NULL, null, ~ を None に変換
        for k,v in cli_conf.items():
            if isinstance(v, str) and v.lower() in ['none', 'null', '~']:
                cli_conf[k] = None
        # キーの存在チェック
        ok, missing_keys = check_all_keys_exist(cli_conf, config)
        if not ok:
            raise ValueError(f"Missing config keys in command line arguments: {missing_keys}")
        # config を cli_conf で上書き
        config = OmegaConf.merge(config, cli_conf)
        print("Overwritten by command line arguments (OmegaConf style):")
        print(OmegaConf.to_yaml(cli_conf))
    except ValueError as e:
        print(f"Error parsing OmegaConf style arguments: {e}")
        # 必要に応じてエラー処理を追加
        sys.exit(1)

    # GPUの設定
    if torch.cuda.is_available():
        if config['gpu_id'] is not None:
            set_gpu_mode(True, gpu_id=config['gpu_id'])
        else:
            set_gpu_mode(True)
    else:
        set_gpu_mode(False)

    # parameter の組み合わせでmainを実行
    log_dirs = {}
    exp_num = 0
    datetime = config.get('datetime', None)
    data_dir = os.path.join('markov_game', 'data', 'local')
    start = time.time()
    for cfg, timestamp in HierarchicalSweeper(config):
        exp_num += 1
        cfg['datetime'] = timestamp if datetime is None else datetime
        ctxt, params = args_for_experiments(cfg)
        ctxt['log_dir'] = os.path.join(data_dir, ctxt['prefix'], ctxt['name'])
        log_dir = main(ctxt, cfg=params)  # train on each config
        if params['exp_name'] not in log_dirs:
            log_dirs[params['exp_name']] = []
        log_dirs[params['exp_name']].append(log_dir)    
    end = time.time()
    exp_result_dir = os.path.join(data_dir, ctxt['prefix'])
    print(f'All the results are saved in {exp_result_dir}.')

    # 結果をaggregarteし，かつそれをtensorboardに書き込む
    if not args.no_aggregate:
        from scripts.aggregate_results import aggregate_progresses
        for exp_name, l_dirs in log_dirs.items():
            average_log_dir = os.path.join(
                exp_result_dir,
                "aggregated",
                f"{exp_name}_average" if len(exp_name) > 0 else "average"
                )
            aggregate_progresses(log_dirs=l_dirs, 
                                 aggregate_log_dir=average_log_dir, 
                                 mode='average', 
                                 use_tensorboard=True,
                                 is_separate_logs=True)
            std_log_dir = os.path.join(
                exp_result_dir,
                "aggregated",
                f"{exp_name}_std" if len(exp_name) > 0 else "std"
                )
            aggregate_progresses(log_dirs=l_dirs, 
                                 aggregate_log_dir=std_log_dir, 
                                 mode='std', 
                                 use_tensorboard=True,
                                 is_separate_logs=True)

    print(f"All trainings finished ({exp_num} experiments) in {end - start:.2f} seconds.")
    print(f'Results are saved in {exp_result_dir}.')