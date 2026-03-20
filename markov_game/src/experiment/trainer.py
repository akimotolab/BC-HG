"""Provides algorithms with access to most of garage's features."""
import copy
import os
import time

import cloudpickle
from dowel import logger, tabular


# This is avoiding a circular import
from garage.experiment.deterministic import get_seed, set_seed
from garage.experiment.snapshotter import Snapshotter

from ..utils import log_performance
from ..envs import GlobalEnvSpec
from ..policies import JointPolicy
from .experiment import dump_json


tf = None


class ExperimentStats:
    # pylint: disable=too-few-public-methods
    """Statistics of a experiment.

    Args:
        total_epoch (int): Total epoches.
        total_itr (int): Total Iterations.
        total_env_steps (int): Total environment steps collected by workers.
        last_episode (list[dict]): Last sampled episodes.

    """

    def __init__(self, total_epoch, total_itr, 
                 total_env_steps, total_sim_steps, last_episode):
        self.total_epoch = total_epoch
        self.total_itr = total_itr
        self.total_env_steps = total_env_steps
        self.total_sim_steps = total_sim_steps
        self.last_episode = last_episode


class TrainArgs:
    # pylint: disable=too-few-public-methods
    """Arguments to call train() or resume().

    Args:
        n_epochs (int): Number of epochs.
        batch_size (int): Number of environment steps in one batch.
        plot (bool): Visualize an episode of the policy after after each epoch.
        store_episodes (bool): Save episodes in snapshot.
        pause_for_plot (bool): Pause for plot.
        start_epoch (int): The starting epoch. Used for resume().

    """

    def __init__(self, n_epochs, batch_size, max_total_env_steps,
                 plot, store_episodes, pause_for_plot, start_epoch,
                 itr_per_epoch, steps_per_itr, learner):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_total_env_steps = max_total_env_steps
        self.plot = plot
        self.store_episodes = store_episodes
        self.pause_for_plot = pause_for_plot
        self.start_epoch = start_epoch
        self.itr_per_epoch = itr_per_epoch
        self.steps_per_itr = steps_per_itr
        self.learner = learner


class Trainer:
    """Base class of trainer.

    Use trainer.setup(algo, env) to setup algorithm and environment for trainer
    and trainer.train() to start training.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by Trainer to create the snapshotter.
            If None, it will create one with default settings.

    Note:
        For the use of any TensorFlow environments, policies and algorithms,
        please use TFTrainer().

    Examples:
        | # to train
        | trainer = Trainer()
        | env = Env(...)
        | policy = Policy(...)
        | algo = Algo(
        |         env=env,
        |         policy=policy,
        |         ...)
        | trainer.setup(algo, env)
        | trainer.train(n_epochs=100, batch_size=4000)

        | # to resume immediately.
        | trainer = Trainer()
        | trainer.restore(resume_from_dir)
        | trainer.resume()

        | # to resume with modified training arguments.
        | trainer = Trainer()
        | trainer.restore(resume_from_dir)
        | trainer.resume(n_epochs=20)

    """

    def __init__(self, snapshot_config):
        self._snapshotter = Snapshotter(snapshot_config.snapshot_dir,
                                        snapshot_config.snapshot_mode,
                                        snapshot_config.snapshot_gap)
        
        self._has_setup = False
        self._plot = False

        self._seed = None
        self._train_args = None
        self._stats = ExperimentStats(total_itr=0,
                                      total_env_steps=0,
                                      total_sim_steps=0,
                                      total_epoch=0,
                                      last_episode=None)

        self._leader = None
        self._follower = None
        self._env = None
        self._eval_env = None
        self._sampler = None
        self._eval_sampler = None
        self._plotter = None
        self._learner = None

        self._start_time = None
        self._itr_start_time = None

        # They have to be updated manually in each epoch 
        self.step_itr = None
        self.step_episode = None

        # only used for off-policy algorithms
        self.enable_leader_logging = False
        self.enable_follower_logging = False
        self.follower_logger, self.leader_logger, self.eval_logger = None, None, None
        self.follower_tabular, self.leader_tabular, self.eval_tabular = None, None, None
        self._log_x_axis = snapshot_config.x_axis

        self._n_workers = None
        self._worker_class = None
        self._worker_args = None

    def setup(self, 
              env, 
              leader, 
              follower,
              sampler,
              eval_env=None,
              eval_sampler=None,
              learner="leader"):
        """Set up trainer for algorithm and environment.

        This method saves algo and env within trainer and creates a sampler.

        Note:
            After setup() is called all variables in session should have been
            initialized. setup() respects existing values in session so
            policy weights can be loaded before setup().

        Args:
            leader (RLAlgorithm): A leader's algorithm instance. If this algo want to use
                samplers, it should have a `_sampler` field.
            env (Environment): An environment instance.
            follower (FollowerWrapper): A wrapped follower's algorithm instance.

        """
        self._env = env
        self._eval_env = eval_env if eval_env else self.get_env_copy(env)
        self._leader = leader
        self._follower = follower
        self._learner = learner
        self._sampler = sampler
        self._eval_sampler = eval_sampler if eval_sampler else copy.deepcopy(sampler)
        noise_sigma_l = leader.noise_sigma if hasattr(leader, 'noise_sigma') else None
        noise_sigma_f = follower.noise_sigma if hasattr(follower, 'noise_sigma') else None
        self._joint_policy = JointPolicy(
            env_spec=env.spec,
            leader_policy=leader.policy,
            follower_policy=follower.policy,
            noise_sigma_l=noise_sigma_l,
            noise_sigma_f=noise_sigma_f
        )
        self._seed = get_seed()
        self._env.seed(self._seed)
        self._eval_env.seed(self._seed)

        # Separate the csv_output and tensorboard_output for leader, follower, and evaluation
        logger.log("Logging separately")
        from dowel import Logger, TabularInput, CsvOutput, TensorBoardOutput, StdOutput, TextOutput
        self.follower_logger, self.leader_logger, self.eval_logger = logger, Logger(), Logger()
        self.follower_tabular, self.leader_tabular, self.eval_tabular = tabular, TabularInput(), TabularInput()
        text_log_file = os.path.join(self._snapshotter.snapshot_dir, 'debug.log')
        leader_logdir = os.path.join(self._snapshotter.snapshot_dir, 'leader')
        eval_logdir = os.path.join(self._snapshotter.snapshot_dir, 'eval')

        logger_to_use = [self.leader_logger, self.eval_logger] if self._learner == 'leader' else [self.eval_logger]
        logdirs = [leader_logdir, eval_logdir] if self._learner == 'leader' else [eval_logdir]
        for l, d in zip(logger_to_use, logdirs):
            l.remove_output_type(CsvOutput)
            l.remove_output_type(TensorBoardOutput)
            l.add_output(CsvOutput(os.path.join(d, 'progress.csv')))
            l.add_output(TensorBoardOutput(d, x_axis=self._log_x_axis))
            l.add_output(StdOutput())
            l.add_output(TextOutput(text_log_file))
        
        self._has_setup = True

    def _start_worker(self):
        """Start Plotter and Sampler workers."""
        if self._plot:
            # pylint: disable=import-outside-toplevel
            from garage.plotter import Plotter
            self._plotter = Plotter()
            self._plotter.init_plot(self.get_env_copy(self._env), self._joint_policy)

    def _shutdown_worker(self):
        """Shutdown Plotter and Sampler workers."""
        if self._sampler is not None:
            self._sampler.shutdown_worker()
        if self._plot:
            self._plotter.close()

    def obtain_episodes(self,
                        itr,
                        batch_size=None,
                        agent_update=None,
                        env_update=None,
                        init_state=None,
                        init_follower_action=None,
                        sim=False):
        """Obtain one batch of episodes.

        Args:
            itr (int): Index of iteration (epoch).
            batch_size (int): Number of steps in batch. This is a hint that the
                sampler may or may not respect.
            agent_update (list[object]): Value which will be passed into the
                `agent_update_fn` before doing sampling episodes. If a list is
                passed in, it must have length exactly `factory.n_workers`, and
                will be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            sim (bool): Whether to simulate the environment.

        Raises:
            ValueError: If the trainer was initialized without a sampler, or
                batch_size wasn't provided here or to train.

        Returns:
            EpisodeBatch: Batch of episodes.

        """
        if self._sampler is None:
            raise ValueError('trainer was not initialized with `sampler`. '
                             'the algo should have a `_sampler` field when'
                             '`setup()` is called')
        if batch_size is None and self._train_args.batch_size is None:
            raise ValueError(
                'trainer was not initialized with `batch_size`. '
                'Either provide `batch_size` to trainer.train, '
                ' or pass `batch_size` to trainer.obtain_samples.')
        episodes = None
        if agent_update is None:
            l_policy = getattr(self._leader, 'exploration_policy', None)
            f_policy = getattr(self._follower, 'exploration_policy', None)
            if l_policy is None:
                # This field should exist, since self.make_sampler would have
                # failed otherwise.
                l_policy = self._leader.policy
            if f_policy is None:
                # This field should exist, since self.make_sampler would have
                # failed otherwise.
                f_policy = self._follower.policy
            noise_sigma_l = self._leader.noise_sigma if hasattr(self._leader, 'noise_sigma') else None
            noise_sigma_f = self._follower.noise_sigma if hasattr(self._follower, 'noise_sigma') else None
            agent_update = JointPolicy(
                env_spec=self._env.spec,
                leader_policy=l_policy,
                follower_policy=f_policy,
                noise_sigma_l=noise_sigma_l,
                noise_sigma_f=noise_sigma_f
            )

        if init_state is not None or init_follower_action is not None:
            sim = True
        episodes = self._sampler.obtain_samples(
            itr, (batch_size or self._train_args.batch_size),
            agent_update=agent_update,
            env_update=env_update,
            init_state=init_state,
            init_follower_action=init_follower_action,
            get_actions_kwargs={
                'deterministic_l': (self._leader.fixed_policy 
                                    and self._leader.is_deterministic_policy),
                'deterministic_f': (self._follower.fixed_policy 
                                    and self._follower.is_deterministic_policy),
                'explorate_l': (not self.total_env_steps > self.leader._init_steps),
                'explorate_f': (not self.total_env_steps > self.follower._init_steps)
            })
        self.total_env_steps += sum(episodes.lengths)
        if sim:
            self._stats.total_sim_steps += sum(episodes.lengths)
        return episodes

    def obtain_samples(self,
                       itr,
                       batch_size=None,
                       agent_update=None,
                       env_update=None,
                       init_state=None,
                       init_follower_action=None,
                       sim=False):
        """Obtain one batch of samples.

        Args:
            itr (int): Index of iteration (epoch).
            batch_size (int): Number of steps in batch.
                This is a hint that the sampler may or may not respect.
            agent_update (list[object]): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Raises:
            ValueError: Raised if the trainer was initialized without a
                        sampler, or batch_size wasn't provided here
                        or to train.

        Returns:
            list[dict]: One batch of samples.

        """
        eps = self.obtain_episodes(itr, 
                                   batch_size, 
                                   agent_update,
                                   env_update,
                                   init_state,
                                   init_follower_action,
                                   sim)
        return eps.to_list()

    def save(self, epoch):
        """Save snapshot of current batch.

        Args:
            epoch (int): Epoch.

        Raises:
            NotSetupError: if save() is called before the trainer is set up.

        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup trainer before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        params['seed'] = self._seed
        params['train_args'] = self._train_args
        params['stats'] = self._stats

        # Save states
        params['env'] = self._env
        params['eval_env'] = self._eval_env
        params['leader'] = self._leader
        params['follower'] = self._follower
        params['sampler'] = self._sampler
        params['eval_sampler'] = self._eval_sampler
        params['n_workers'] = self._n_workers
        params['worker_class'] = self._worker_class
        params['worker_args'] = self._worker_args

        self._snapshotter.save_itr_params(epoch, params)

        logger.log('Saved')

    def restore(self, from_dir, from_epoch='last'):
        """Restore experiment from snapshot.

        Args:
            from_dir (str): Directory of the pickle file
                to resume experiment from.
            from_epoch (str or int): The epoch to restore from.
                Can be 'first', 'last' or a number.
                Not applicable when snapshot_mode='last'.

        Returns:
            TrainArgs: Arguments for train().

        """
        saved = self._snapshotter.load(from_dir, from_epoch)

        self._seed = saved['seed']
        self._train_args = saved['train_args']
        self._stats = saved['stats']

        set_seed(self._seed)

        self.setup(
            env=saved['env'], 
            algo=saved['leader'], 
            follower=saved['follower'],
            sampler=saved['sampler'],
            eval_env=saved['eval_env'],
            eval_sampler=saved['eval_sampler']
            )

        n_epochs = self._train_args.n_epochs
        last_epoch = self._stats.total_epoch
        last_itr = self._stats.total_itr
        total_env_steps = self._stats.total_env_steps
        total_sim_steps = self._stats.total_sim_steps
        batch_size = self._train_args.batch_size
        store_episodes = self._train_args.store_episodes
        pause_for_plot = self._train_args.pause_for_plot
        itr_per_epoch = self._train_args.itr_per_epoch
        steps_per_itr = self._train_args.steps_per_itr
        learner = self._train_args.learner

        fmt = '{:<20} {:<15}'
        logger.log('Restore from snapshot saved in %s' %
                   self._snapshotter.snapshot_dir)
        logger.log(fmt.format('-- Train Args --', '-- Value --'))
        logger.log(fmt.format('n_epochs', n_epochs))
        logger.log(fmt.format('last_epoch', last_epoch))
        logger.log(fmt.format('batch_size', batch_size))
        logger.log(fmt.format('store_episodes', store_episodes))
        logger.log(fmt.format('pause_for_plot', pause_for_plot))
        logger.log(fmt.format('itr_per_epoch', itr_per_epoch))
        logger.log(fmt.format('steps_per_itr', steps_per_itr))
        logger.log(fmt.format('learner', learner))
        logger.log(fmt.format('-- Stats --', '-- Value --'))
        logger.log(fmt.format('last_itr', last_itr))
        logger.log(fmt.format('total_env_steps', total_env_steps))
        logger.log(fmt.format('total_sim_steps', total_sim_steps))

        self._train_args.start_epoch = last_epoch + 1
        return copy.copy(self._train_args)

    def log_diagnostics(self, pause_for_plot=False):
        """Log diagnostics.

        Args:
            pause_for_plot (bool): Pause for plot.

        """
        logger.log('Time %.2f s' % (time.time() - self._start_time))
        logger.log('EpochTime %.2f s' % (time.time() - self._itr_start_time))

        self.eval_tabular.record('TotalEnvSteps', self.total_env_steps)
        self.eval_tabular.record('StepItr', self.step_itr)
        self.eval_logger.log(self.eval_tabular)
        self.eval_logger.dump_all(self.total_env_steps)
        self.eval_tabular.clear()

        if self.enable_follower_logging:
            self.follower_tabular.record('TotalEnvSteps', self.total_env_steps)
            self.follower_tabular.record('StepItr', self.step_itr)
            self.follower_logger.log(self.follower_tabular)
            self.follower_logger.dump_all(self.total_env_steps)
            self.follower_tabular.clear()
        if self.enable_leader_logging:
            self.leader_tabular.record('TotalEnvSteps', self.total_env_steps)
            self.leader_tabular.record('StepItr', self.step_itr)
            self.leader_tabular.record('Leader/TotalSimSteps', self._stats.total_sim_steps)
            self.leader_logger.log(self.leader_tabular)
            self.leader_logger.dump_all(self.total_env_steps)
            self.leader_tabular.clear()
        
        if self._plot:
            self._plotter.update_plot(self._joint_policy,
                                      self._env.spec.max_episode_length)
            if pause_for_plot:
                input('Plotting evaluation run: Press Enter to " "continue...')

    def evaluate_once(self, 
                      agent_update=None, 
                      env_update=None, 
                      reset_env=True, 
                      prefix='Evaluation'):
        """Evaluate the policy once.

        Returns:
            dict: The performance metrics.
        """
        eval_env = self._eval_env if env_update is None else env_update
        if reset_env:
            eval_env.seed(self._seed)
        eval_eps = self._eval_sampler.obtain_exact_episodes(
            n_eps_per_worker=1,
            agent_update=self._joint_policy if agent_update is None else agent_update,
            env_update=eval_env,
            get_actions_kwargs=dict(
                deterministic_l=self.leader.is_deterministic_policy,
                deterministic_f=self.follower.is_deterministic_policy,
            )
        )
        performance = log_performance(
            self.eval_tabular,
            self.total_env_steps,
            eval_eps,
            discount=[self.leader.discount, self.follower.discount],
            prefix=prefix
            )
        return performance

    def train(self,
              n_epochs,
              batch_size=None,
              max_total_env_steps=None,
              plot=False,
              store_episodes=False,
              pause_for_plot=False):
        """Start training.

        Args:
            n_epochs (int): Number of epochs.
            batch_size (int or None): Number of environment steps in one batch.
            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
            pause_for_plot (bool): Pause for plot.
            learner (str): Which agent to train, either 'leader' or 'follower'.

        Raises:
            NotSetupError: If train() is called before setup().

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._has_setup:
            raise NotSetupError(
                'Use setup() to setup trainer before training.')
                
        if self._learner == "leader":
            algo = self._leader
        elif self._learner == "follower":
            algo = self._follower
        else:
            raise ValueError("learner must be either 'leader' or 'follower'")

        # Note:
        # max_total_env_steps: maxmum total number of steps in the environment
        # itr_per_epoch: number of iterations per epoch
        # steps_per_itr: number of environment steps per iteration
        # _gradient_steps_n: number of gradient steps per iteration
        itr_per_epoch = max_total_env_steps // batch_size // n_epochs
        steps_per_itr = batch_size
        max_total_env_steps = n_epochs * itr_per_epoch * steps_per_itr

        assert itr_per_epoch > 0, (
            f'max_total_env_steps must be at least {batch_size * n_epochs} '
            f'(batch_size={batch_size}, n_epochs={n_epochs})'
        )

        logger.log('Training with the following parameters:')
        logger.log('n_epochs: %d' % n_epochs)
        logger.log('itr_per_epoch: %d' % itr_per_epoch)
        logger.log('steps_per_itr: %d' % steps_per_itr)
        logger.log('max_total_env_steps: %d' % max_total_env_steps)

        # Save arguments for restore
        self._train_args = TrainArgs(n_epochs=n_epochs,
                                     batch_size=batch_size,
                                     max_total_env_steps=max_total_env_steps,
                                     plot=plot,
                                     store_episodes=store_episodes,
                                     pause_for_plot=pause_for_plot,
                                     start_epoch=0,
                                     itr_per_epoch=itr_per_epoch,
                                     steps_per_itr=steps_per_itr,
                                     learner=self._learner)

        self._plot = plot
        self._start_worker()

        log_dir = self._snapshotter.snapshot_dir
        summary_file = os.path.join(log_dir, 'experiment.json')
        dump_json(summary_file, self)

        average_return = algo.train(self)
        self._shutdown_worker()

        self.leader_logger.remove_all()
        self.eval_logger.remove_all()
        # Note: logger (self.follower_logger) is not removed here, because 
        # it is removed in the ExperimentTemplate.__call__() method.

        return average_return

    def step_epochs(self):
        """Step through each epoch.

        This function returns a magic generator. When iterated through, this
        generator automatically performs services such as snapshotting and log
        management. It is used inside train() in each algorithm.

        The generator initializes two variables: `self.step_itr` and
        `self.step_episode`. To use the generator, these two have to be
        updated manually in each epoch, as the example shows below.

        Yields:
            int: The next training epoch.

        Examples:
            for epoch in trainer.step_epochs():
                trainer.step_episode = trainer.obtain_samples(...)
                self.train_once(...)
                trainer.step_itr += 1

        """
        self._start_time = time.time()
        self.step_itr = self._stats.total_itr
        self.step_episode = None

        # Used by integration tests to ensure examples can run one epoch.
        n_epochs = int(
            os.environ.get('GARAGE_EXAMPLE_TEST_N_EPOCHS',
                           self._train_args.n_epochs))

        logger.log('Obtaining samples...')

        for epoch in range(self._train_args.start_epoch, n_epochs):
            self._itr_start_time = time.time()
            with logger.prefix('epoch #%d | ' % epoch):
                yield epoch
                save_episode = (self.step_episode
                                if self._train_args.store_episodes else None)

                self._stats.last_episode = save_episode
                self._stats.total_epoch = epoch
                self._stats.total_itr = self.step_itr

                self.save(epoch)
                self.log_diagnostics(self._train_args.pause_for_plot)
        
        epoch += 1
        with logger.prefix('epoch #%d | ' % epoch):
            self.save(epoch)

    def resume(self,
               n_epochs=None,
               batch_size=None,
               max_total_env_steps=None,
               plot=None,
               store_episodes=None,
               pause_for_plot=None,
               itr_per_epoch=None,
               steps_per_itr=None,
               learner=None):
        """Resume from restored experiment.

        This method provides the same interface as train().

        If not specified, an argument will default to the
        saved arguments from the last call to train().

        Args:
            n_epochs (int): Number of epochs.
            batch_size (int): Number of environment steps in one batch.
            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
            pause_for_plot (bool): Pause for plot.

        Raises:
            NotSetupError: If resume() is called before restore().

        Returns:
            float: The average return in last epoch cycle.

        """
        if self._train_args is None:
            raise NotSetupError('You must call restore() before resume().')

        self._train_args.n_epochs = n_epochs or self._train_args.n_epochs
        self._train_args.batch_size = batch_size or self._train_args.batch_size
        self._train_args.max_total_env_steps = max_total_env_steps or self._train_args.max_total_env_steps

        if plot is not None:
            self._train_args.plot = plot
        if store_episodes is not None:
            self._train_args.store_episodes = store_episodes
        if pause_for_plot is not None:
            self._train_args.pause_for_plot = pause_for_plot

        self._train_args.itr_per_epoch = itr_per_epoch or self._train_args.itr_per_epoch
        self._train_args.steps_per_itr = steps_per_itr or self._train_args.steps_per_itr
        self._train_args.learner = learner or self._train_args.learner

        if learner == "leader":
            algo = self._leader
        elif learner == "follower":
            algo = self._follower
        else:
            raise ValueError("learner must be either 'leader' or 'follower'")

        average_return = algo.train(self)
        self._shutdown_worker()

        return average_return

    @classmethod
    def get_env_copy(cls, env):
        """Get a copy of the environment.

        Args:
            env (Environment): An environment instance.

        Returns:
            Environment: A copy of the environment instance.
        """
        if env:
            return cloudpickle.loads(cloudpickle.dumps(env))
        else:
            return None

    @property
    def total_env_steps(self):
        """Total environment steps collected by workers.

        Returns:
            int: Total environment steps collected.

        """
        return self._stats.total_env_steps

    @total_env_steps.setter
    def total_env_steps(self, value):
        """Total environment steps collected by workers.

        Args:
            value (int): Total environment steps collected.

        """
        self._stats.total_env_steps = value

    @property
    def joint_policy(self):
        """Joint Policy."""
        return self._joint_policy

    @property
    def env_spec(self) -> GlobalEnvSpec:
        """Environment Global Specification."""
        return self._env.spec

    @property
    def env(self):
        """Environment."""
        return self._env
    
    @property
    def eval_env(self):
        """Evaluation Environment."""
        return self._eval_env
    
    @property
    def leader(self):
        """Leader."""
        return self._leader

    @property
    def follower(self):
        """Follower."""
        return self._follower
    
    @property
    def sampler(self):
        """Sampler."""
        return self._sampler
    
    @property
    def eval_sampler(self):
        """Evaluation Sampler."""
        return self._eval_sampler

    @property
    def train_args(self):
        """Training arguments."""
        return self._train_args


class NotSetupError(Exception):
    """Raise when an experiment is about to run without setup."""
