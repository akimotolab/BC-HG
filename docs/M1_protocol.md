# M1 frozen evaluation protocol

Only fixed leader. No M2 implementation or experiments. The original AsyncMARL collector and Trainer are reused; SAC uses the existing SACDiscrete and FollowerWrapper. The new TabularSoftQ plugs into the same follower interface. The leader is a zero-logit categorical policy (uniform); Four-Rooms has one dummy leader action and a frozen incentive vector exported from the existing environment, seed 0. This dummy action does not change the Conf-MDP problem.

JAX and Torch environments remain separate. `export_fixed_four_rooms.py` exports physical transition probabilities, immediate rewards, goal masks, initial distribution and fixed incentives, never a trained Q or policy. The Gym adapter samples this kernel. Its state is `(goal_index, position_index)` encoded as a single integer, then one-hot; goal cannot be merged across contexts. Export uses the corrected kernel tested against the original JAX step. Reward is paid in the current goal state and that step terminates, matching the existing code; a horizon-200 truncation bootstraps using the physical next observation. MG uses `(s,a)` and samples the next leader action from the fixed policy. Both critic targets use true termination only.

## Pilot and evaluation separation

Pilot seed: 100. Initial Four-Rooms tabular budget 20,000, 256 random steps, no mixing failed (return 0.291 to 0.149, unvisited state-actions 17.5%). Second pilot used 100,000 steps, 10,000 initial random actions and constant 10% uniform behavior mixing. The reference policy and update targets always use the current unmixed actor/soft policy. Exploration choices are in the resolved config.

Independent evaluation seeds: 0 and 1. Acceptance fixed before those runs: final entropy-regularized return gain > 1e-5 AND either uniform KL or Q RMSE improves by > 1%. These are minimum learning checks, not convergence to the reference. All seeds are reported, and no best checkpoint is selected. Original pilot configurations remain in their output directories.

Order: Four-Rooms tabular, MG tabular, Four-Rooms SAC, MG SAC. Per-condition budgets/settings are in `configs/m1/*.yaml`. Actor/target intervals are 1 and validated. One follower update per four new transitions after warm-up. For tabular, each sampled transition is applied sequentially, so repeated entries observe previous updates in that batch. A batch of 16 means 16 table assignments, not one optimizer step. SAC makes two critic steps and one actor step per batch, then updates both targets. Fixed beta, reward scale 1.

## Evaluation and information isolation

Model-based diagnostics receive copies of the frozen task model and current arrays, and return only metrics. Double-precision reference soft Q iteration stops at sup residual < 1e-10 (max 10,000 sweeps, otherwise explicit failure). Fixed-policy return uses a linear solve, entropy included. The reference is called a high-precision numerical reference. No reference Q/policy is passed to the follower, replay, target or action selector. Oracle toggle integration tests compare final Q, policy, visits and update counters bit-for-bit; same-seed repetition is also checked. Evaluation saves and restores Python/NumPy/Torch RNG, so evaluation does not consume the learner's random stream.

KL direction is current policy || reference, uniform over all extended states; visit-weighted KL is auxiliary. Report un-clipped return gap, Q RMSE, soft optimality residual (mean/sup), actor-Boltzmann KL and beta-scaled value gap. SAC Q diagnostics use the minimum current twin critics. Held-out TD MSE uses independent evaluation trajectories and current policy-evaluation value (tabular uses soft value); it is not labelled a true Bellman residual. Full visit arrays are saved at each evaluation.

`metrics.json` includes training/evaluation transition counts, follower operation counts, zero leader counts, elapsed wall time, reference time, diagnostic time and evaluation rollout time. `final.npz` includes Q/policy/visits; full follower+optimizer+target+replay+RNG is saved separately. Existing Trainer snapshots also save both agents. Arbitrary mid-rollout sampler continuation is not guaranteed.
