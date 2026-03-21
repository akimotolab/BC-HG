import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors

from train_four_rooms_hpgd import upper_level_reward
from src.models.IncentiveModel import incentive_transform
from src.algorithms.value_iteration_and_prediction import (
    general_value_iteration,
    initial_value_prediction,
)


def visualize_four_rooms_map(env, env_params, config, rng, savepath=None):
    """
    Visualize the four rooms map with the initial states, goals and target state
    :param env: ConfigurableFourRooms environment
    :param env_params: Environment parameters
    :param config: Config dictionary
    :param rng: Random number generator
    :param savepath: Path to save the figure
    :return: None
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    _, init_state = env.reset(rng, env_params)
    _, ax = env.render(init_state, env_params, ax=ax, annotate_positions=False)

    init_probs = env.state_initialization_distribution(
        env_params.state_initialization_params
    ).probs
    init_states_counter = 1
    n_init_state = sum(init_probs > 1e-8)
    for pos, p in zip(env.available_init_pos, init_probs):
        if p <= 1e-8:
            continue
        ax.annotate(
            rf"$\textbf{{S^{init_states_counter}}}$"
            if n_init_state > 1
            else rf"$\textbf{{S}}$",
            weight="bold",
            xy=(pos[1], pos[0]),
            xycoords="data",
            xytext=(pos[1] - 0.3, pos[0] + 0.25),
        )
        init_states_counter += 1
    for i, pos in enumerate(env.available_goals):
        ax.annotate(
            rf"$\textbf{{G}}^{i+1}$",
            weight="bold",
            xy=(pos[1], pos[0]),
            xycoords="data",
            xytext=(pos[1] - 0.3, pos[0] + 0.25),
        )
    pos = config["upper_optimisation"]["reward_function"]["target_state"]
    ax.annotate(
        rf"$\textbf{{+1}}$"
        if config["upper_optimisation"]["reward_function"]["type"] == "positive"
        else rf"$\textbf{{-1}}$",
        weight="bold",
        xy=(pos[1], pos[0]),
        xycoords="data",
        xytext=(pos[1] - 0.3, pos[0] + 0.25),
    )

    # Add gridlines
    for i in range(env.occupied_map.shape[0]):
        ax.axhline(i + 0.5, color="black", linewidth=0.75)
    for i in range(env.occupied_map.shape[1]):
        ax.axvline(i + 0.5, color="black", linewidth=0.75)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close()


def rolling_average_NaN(arr, mask, window_size, axis=1):
    """
    Compute the rolling average of an array with NaN values
    :param arr: 1 or 2D array
    :param mask: Mask array with the same shape as arr
    :param window_size: Length of the rolling window
    :param axis: Axis along which to compute the rolling average
    :return: Array with rolling average values
    """
    def get_moving_average(x):
        """Moving average with expanding window at the beginning"""
        assert len(x.shape) == 1
        x = jnp.convolve(x, jnp.ones(window_size), mode="full")
        x = x[:-(window_size-1)]
        normalization = jnp.clip(jnp.arange(1, x.shape[0]+1), a_max=window_size)
        x = x/normalization
        return x
    if axis == 0:
        rolling_average = get_moving_average(arr)
    elif axis == 1:
        rolling_average = jax.vmap(get_moving_average)(arr)
    else:
        raise ValueError("Invalid axis")
    return rolling_average


def plot_episode_lengths(
    xi_values,
    episode_lengths,
    average_num_of_episodes,
    config,
    rolling_window=100,
    savefig_path=None,
):
    print("Rolling-window: ", rolling_window)
    data = {
        "Average episode length": episode_lengths,
        "Average number of episodes": average_num_of_episodes,
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    for i, (title, input_values) in enumerate(data.items()):
        for j in range(len(config["environment"]["available_goals"])):
            mask_goal = xi_values == j
            # arr = jnp.where(mask_goal, input_values, jnp.nan) # Shape: (num_seeds, n_steps)
            arr = rolling_average_NaN(input_values, mask_goal, rolling_window, axis=1)
            # Plot averages over the first axis with confidence bound
            mean = jnp.mean(arr, axis=0)
            std = jnp.std(arr, axis=0) / jnp.sqrt(arr.shape[0])
            axes[i].plot(mean, label=f"Goal {j+1}")
            axes[i].fill_between(
                jnp.arange(mean.shape[0]),
                mean - std,
                mean + std,
                alpha=0.2,
            )
        axes[i].legend()
        axes[i].set_title(title)
    plt.tight_layout()
    if savefig_path is not None:
        plt.savefig(savefig_path)
    plt.show()
    plt.close()
    

def plot_UL_rewards(
    UL_rewards: dict,  # {key: jnp.ndarray of shape (n_seeds, n_steps)}
    rolling_window=100,
    title=None,
    savefig_path=None,
    figsize=(10, 5),
    ax=None,  # Added: allow specifying an existing axis
    legend_position=None,
    add_oracle_xaxis_to_zero_order=False,
    xlim=None,
    ylim=None,
    legend_names={
        "bilevel": r"\textsc{Bilevel}",
        "benchmark": r"\textsc{AMD}",
        "zero_order": r"\textsc{Zero-order}",
        "zero_order_oracle": r"\textsc{Zero-order oracle}",
    },
    line_styles={
        "bilevel": "-",
        "benchmark": "--",
        "zero_order": "-.",
        "zero_order_oracle": ":",
    },
    algo_colors=None,
    zorder=None,
    y_ticks=jnp.arange(-1.0, 1.6, 0.2),
    xlabel=rf"Outer iterations: $N$",
    ylabel=rf"Upper-level objective",
    yscale="linear",  # "linear", "log"
    statistic="mean",  # "mean", "median"
    alpha=0.8,
    show_legend=True,  # Added: control whether to show the legend
    separated_in_seeds=False,  # Added: whether data is separated in seeds
):
    # Setup axes
    if ax is None:
        plt.clf()
        plt.figure(figsize=figsize)
        current_ax = plt.gca()
        is_standalone = True
    else:
        current_ax = ax
        is_standalone = False

    def add_line_to_plot(y, label, x=None):
        if x is None:
            x = jnp.arange(y.shape[1])
        if not separated_in_seeds:
            if statistic == "mean":
                mean = jnp.nanmean(y, axis=0)  # Shape: (n_steps,)
                std = jnp.sqrt(jnp.nanvar(y, axis=0)) / jnp.sqrt(y.shape[0])
                mean, std = mean.reshape(-1, mean.shape[0]), std.reshape(-1, std.shape[0])  # Shape: (1, n_steps)
                UCB, LCB = mean + std, mean - std
            elif statistic == "median":
                mean = jnp.nanmedian(y, axis=0)  # Shape: (n_steps,)
                u_quantile = jnp.nanpercentile(y, 75, axis=0)
                l_quantile = jnp.nanpercentile(y, 25, axis=0)
                mean = mean.reshape(-1, mean.shape[0])  # Shape: (1, n_steps)
                u_quantile, l_quantile = u_quantile.reshape(-1, u_quantile.shape[0]), l_quantile.reshape(-1, l_quantile.shape[0])  # Shape: (1, n_steps)
                UCB, LCB = u_quantile, l_quantile
            else:
                raise ValueError("Invalid statistic")
            
            labels = [legend_names.get(label, label)]  # Fix: use .get() to provide a default value
            linewidth = plt.rcParams["lines.linewidth"]
        else:
            mean = y  # Shape: (n_seeds, n_steps,)
            std = jnp.zeros_like(mean)
            UCB, LCB = mean + std, mean - std
            labels = [legend_names.get(label, label)] + [None] * (mean.shape[0] - 1)
            linewidth = plt.rcParams["lines.linewidth"] / 2.0
        if rolling_window > 1:
            mean_list = []
            UCB_list = []
            LCB_list = []
            for i in range(mean.shape[0]):
                mean_smoothed = rolling_average_NaN(
                    mean[i], jnp.ones_like(mean[i], dtype=bool), rolling_window, axis=0
                )
                UCB_smoothed = rolling_average_NaN(
                    UCB[i], jnp.ones_like(UCB[i], dtype=bool), rolling_window, axis=0
                )
                LCB_smoothed = rolling_average_NaN(
                    LCB[i], jnp.ones_like(LCB[i], dtype=bool), rolling_window, axis=0
                )
                mean_list.append(mean_smoothed)
                UCB_list.append(UCB_smoothed)
                LCB_list.append(LCB_smoothed)
            mean = jnp.stack(mean_list, axis=0)
            UCB = jnp.stack(UCB_list, axis=0)
            LCB = jnp.stack(LCB_list, axis=0)
        for i in range(mean.shape[0]):
            current_ax.plot(
                x,
                mean[i],
                label=labels[i],
                linestyle=line_styles.get(label, "-"),  # Fix: use .get() to provide a default value
                linewidth=linewidth,
                color=algo_colors[label] if algo_colors is not None else None,
                zorder=zorder[label] if zorder is not None else None,
                alpha=alpha if not separated_in_seeds else alpha / 2.0,
            )
            current_ax.fill_between(
                x,
                LCB[i],
                UCB[i],
                alpha=0.2,
                color=algo_colors[label] if algo_colors is not None else None,
                zorder=zorder[label] if zorder is not None else None,
            )

    for key, value in UL_rewards.items():
        add_line_to_plot(value, key)
        if key == "zero_order" and add_oracle_xaxis_to_zero_order:
            add_line_to_plot(
                value, "zero_order_oracle", x=jnp.arange(1, 2 * value.shape[1], 2)
            )

    # Legend setup
    if show_legend:
        if legend_position is None:
            legend = current_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        else:
            legend = current_ax.legend(**legend_position)
        legend.set_zorder(100)

    # Axis configuration
    if xlim is not None:
        current_ax.set_xlim(*xlim)
    if xlabel:
        current_ax.set_xlabel(xlabel)
    if ylabel:
        current_ax.set_ylabel(ylabel)
    if y_ticks is not None:
        current_ax.set_yticks(ticks=y_ticks)
    if ylim is not None:
        current_ax.set_ylim(*ylim)
    if title:
        current_ax.set_title(title)
    current_ax.set_yscale(yscale)

    # Save and show (only if standalone)
    if is_standalone:
        if savefig_path is not None:
            plt.savefig(savefig_path, bbox_inches="tight")
        plt.show()
        plt.close()


def plot_policy(
    env,
    env_params,
    config,
    seed_idx,
    incentive_params,
    plot_input=None,
    plot_only_most_likely=False,
    rng=jax.random.PRNGKey(0),
    savefig_path=None,
):
    env_params_viz = env_params.replace(
        incentive_params=jax.tree_map(lambda x: x[seed_idx], incentive_params)
    )

    config_lower_level = config["lower_optimisation"]
    q_final, _ = general_value_iteration(
        env,
        env_params_viz,
        config_lower_level["discount_factor"],
        n_policy_iter=config_lower_level["n_policy_iter"],
        n_value_iter=config_lower_level["n_value_iter"],
        regularization=config_lower_level["regularization"],
        reg_lambda=config_lower_level["reg_lambda"],
        return_q_value=True,
    )
    print("Q-value shape: ", q_final.shape)
    br_policy = jax.nn.softmax(
        q_final / config_lower_level["reg_lambda"], axis=-1
    )  # Shape: (n_goals, n_states, n_actions)

    config_upper_level = config["upper_optimisation"]
    V_LL_initial_pos, _ = initial_value_prediction(
        env,
        env_params_viz,
        gamma=config_lower_level["discount_factor"],
        n_policy_iter=config_lower_level["n_policy_iter"],
        n_value_iter=config_lower_level["n_value_iter"],
        policy=br_policy,
        regularization=config_lower_level["regularization"],
        reg_lambda=config_lower_level["reg_lambda"],
    )
    V_UL_initial_pos, _ = initial_value_prediction(
        env,
        env_params_viz,
        gamma=config_upper_level["discount_factor"],
        n_policy_iter=config_lower_level["n_policy_iter"],
        n_value_iter=config_lower_level["n_value_iter"],
        policy=br_policy,
        external_reward=lambda s, a, env_p: upper_level_reward(
            env_p.incentive_params, s, a, config
        ),
    )
    print("Lower-level value: ", V_LL_initial_pos)
    print("Upper-level value: ", V_UL_initial_pos)

    config_env = config["environment"]
    if plot_input is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    else:
        fig, axes = plot_input
    for j in range(len(config["environment"]["available_goals"])):  # For each goal
        goal_logits = jnp.log(jnp.array([1 - j, j], dtype=jnp.float32) + 1e-8)
        env_params_tmp = env_params_viz.replace(
            resample_goal_logits=goal_logits,
        )
        V_LL_fixed_goal, _ = initial_value_prediction(
            env,
            env_params_tmp,
            gamma=config_lower_level["discount_factor"],
            n_policy_iter=config_lower_level["n_policy_iter"],
            n_value_iter=config_lower_level["n_value_iter"],
            policy=br_policy,
            regularization=config_lower_level["regularization"],
            reg_lambda=config_lower_level["reg_lambda"],
        )
        V_UL_fixed_goal, _ = initial_value_prediction(
            env,
            env_params_tmp,
            gamma=config_upper_level["discount_factor"],
            n_policy_iter=config_lower_level["n_policy_iter"],
            n_value_iter=config_lower_level["n_value_iter"],
            policy=br_policy,
            external_reward=lambda s, a, env_p: upper_level_reward(
                env_p.incentive_params, s, a, config
            ),
        )
        print(f"Initial LL state value with goal {j+1} ", V_LL_fixed_goal)
        print(f"Initial UL state value with goal {j+1} ", V_UL_fixed_goal)

        # Visualize the greedy policy
        policy = br_policy[j]
        _, init_state = env.reset(rng, env_params_tmp)
        if plot_input is not None:
            ax = axes
        else:
            ax = axes[j]
        _, ax = env.render(init_state, env_params_tmp, ax=ax, annotate_positions=False)
        goal_pos = config_env["available_goals"][j]
        for i, pol in enumerate(policy):
            pos = env.coords[i]
            most_likely_direction = jnp.argmax(pol)
            # Add arrow to ax
            for k, action_direction in enumerate(env.directions):
                if plot_only_most_likely and k != most_likely_direction:
                    continue
                # Add arrow to ax
                color = "k" if k != most_likely_direction else "b"
                ax.arrow(
                    pos[1],
                    pos[0],
                    pol[k] * action_direction[1] / 2.0,
                    pol[k] * action_direction[0] / 2.0,
                    head_width=0.1,
                    head_length=0.1,
                    fc=color,
                    ec=color,
                    alpha=0.6,
                )
        ax.annotate(
            rf"$G^{j+1}$",
            xy=(goal_pos[1], goal_pos[0]),
            xycoords="data",
            xytext=(goal_pos[1] - 0.3, goal_pos[0] + 0.25),
        )
        for i, pos in enumerate(config_env["available_init_pos"]):
            pos = jnp.array(pos)
            # Add initial positions to ax
            ax.annotate(
                rf"$S^{i+1}$" if len(config_env["available_init_pos"]) > 1 else rf"$S$",
                # fontsize=16,
                xy=(pos[1], pos[0]),
                xycoords="data",
                xytext=(pos[1] - 0.3, pos[0] + 0.25),
            )
    plt.tight_layout()
    if savefig_path is not None:
        plt.savefig(savefig_path, bbox_inches="tight")
    plt.show()
    plt.close()
    return q_final


def plot_incentive_grid(
    env,
    env_params,
    incentives,
    incentive_coords,
    config,
    verbose=False,
    plot_input=None,
    cmap="PuBu_r",
):
    config_incentive = config["configuration"]["incentive"]
    rng = jax.random.PRNGKey(0)
    if plot_input is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot_input
    _, init_state = env.reset(rng, env_params)
    _, ax = env.render(init_state, env_params, ax=ax, annotate_positions=False)
    incentives_transformed = incentive_transform(
        incentives,
        activation_function=config_incentive["activation_function"],
        range=config_incentive["range"],
        temperature=config_incentive["temperature"],
    )
    if verbose:
        print("Incentive percentages: ", jax.nn.softmax(incentives))
        print("Leftover incentives: ", jax.nn.softmax(incentives)[-1])
        print(min(incentives_transformed), max(incentives_transformed))
        print("Incentive range: ", config_incentive["range"])
    heatmap = jnp.full_like(env.occupied_map, jnp.nan, dtype=jnp.float32)
    for pos, incentive_value in zip(incentive_coords, incentives_transformed):
        if verbose:
            print(pos, incentive_value)
        heatmap = heatmap.at[pos[0], pos[1]].set(incentive_value)

    pcm = ax.imshow(
        heatmap,
        cmap=cmap,
        norm=mpl_colors.SymLogNorm(
            vmin=min(config_incentive["range"]),
            vmax=max(config_incentive["range"]),
            linthresh=0.001,
            linscale=1.0,
        ),
    )
    return pcm

