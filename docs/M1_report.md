# M1: fixed-leader discrete follower learning

Date: 2026-09-15. Base revision: `82753d6c4f3bc813158c1819879edcdb5f81f5ac` plus the reviewed working-tree changes. Scope is M0–M1 only. Run-specific source fingerprints, dependency versions and resolved settings are in [provenance](results/M1_provenance.json). CPU: Python 3.8, Torch 1.7.1+cu110, garage 2021.3.0, NumPy 1.23.5; JAX 0.4.21/Python 3.11 is used separately to export the frozen Four-Rooms environment.

## Implementation

The existing `BCHGDiscrete` fixed-policy path, `AsyncMARL` collection loop, `Trainer`, replay buffer, `SACDiscrete` and `FollowerWrapper` are reused. The new tabular soft Q learner uses this same collector and follower interface. `bchg.py` remains the existing continuous-task implementation; no continuous experiment or M2 update was introduced.

Four-Rooms samples an exported physical model consistent with the original JAX step kernel. It preserves goal/position context, current-state goal reward, fixed incentives and the distinction between true termination and horizon truncation. MG retains leader action in `(s,a)` and samples the next leader action. Policy-evaluation SAC targets and soft-optimality diagnostic targets are distinct. Fixed temperature, value mode, target coefficient, exploration, replay warm-up and all update budgets are explicit in the settings.

Reference soft Q iteration runs in float64 until sup residual is below `1e-10`. Fixed-policy return is evaluated by a linear solve with entropy. References are diagnostic-only; they never supply replay data, targets, actions or optimizer inputs.

## Evaluation protocol and results

Pilot seed 100 was separate from evaluation seeds 0 and 1. Four-Rooms uses 100,000 transitions per run, MG 20,000. The predeclared final-checkpoint gate requires return gain above `1e-5` and at least 1% relative improvement in uniform policy KL or Q RMSE. No threshold was adjusted after evaluation. See [frozen protocol](M1_protocol.md), [complete metrics](results/M1_results.json), [learning curves](results/M1_curves.json) and [figure](results/M1_curves.pdf).

| Task | Follower | Seed | Return: initial → final | KL: initial → final | Q RMSE: initial → final | Seconds | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| four_rooms | tabular | 0 | 0.290557 → 0.785543 | 1.9123 → 0.7704 | 0.8185 → 0.5536 | 126.23 | PASS |
| four_rooms | tabular | 1 | 0.290557 → 0.784340 | 1.9123 → 0.9004 | 0.8185 → 0.4867 | 113.94 | PASS |
| four_rooms | sac | 0 | 0.262820 → 0.791967 | 1.9831 → 0.2995 | 0.9321 → 0.0326 | 188.99 | PASS |
| four_rooms | sac | 1 | 0.325352 → 0.793051 | 1.8438 → 0.5173 | 0.8830 → 0.0728 | 237.84 | PASS |
| mg | tabular | 0 | 26.408532 → 50.274863 | 4.8170 → 0.0084 | 49.5806 → 0.2368 | 26.15 | PASS |
| mg | tabular | 1 | 26.408532 → 49.450515 | 4.8170 → 0.2189 | 49.5806 → 0.2292 | 20.90 | PASS |
| mg | sac | 0 | 26.774568 → 50.255633 | 4.3723 → 0.0313 | 49.6645 → 30.7116 | 34.35 | PASS |
| mg | sac | 1 | 25.233845 → 50.258345 | 5.0060 → 0.0293 | 49.7477 → 30.7653 | 34.81 | PASS |

Reference returns are approximately 0.794110 (Four-Rooms) and 50.282738 (MG). These runs demonstrate learning, not uniform convergence: Four-Rooms tabular Q errors remain appreciable, MG tabular seed 1 finishes below seed 0 despite both passing the learning gate, and MG SAC Q RMSE remains about 30.7 although its return and policy KL are close to the reference. A near-reference return does not establish accurate absolute critic values. All intermediate evaluations and all seeds are retained. Wall times include setup, logging and evaluation overhead within the timed run; they are not a controlled speed comparison between algorithms.

## Validation

- M0 baseline regressions, environment contracts, real optimizer counters and checkpoint continuation passed; see [M0 report](M0_report.md).
- Four M1 unit tests passed: hand-computed tabular updates including duplicate entries and truncation; stable soft values; SAC action-axis expectations, signs and value-gap identity; analytic one-state reference; true termination at the horizon and Four-Rooms adapter context.
- All four oracle-toggle integration tests passed. Each compares oracle on/off and a same-seed repeat over 800 training transitions; final Q, policy and visit counts agree bit-for-bit, and update counters agree at every evaluation.
- Every evaluation checks unchanged leader parameters and zero leader actor/critic/target updates. Full state-action visit counts sum to the collected transitions. SAC critic 1, critic 2, actor and target-cycle counts agree, with zero temperature steps. Tabular assignments equal 16 times its batch count.
- Existing Trainer snapshots retain both agents and replay; follower snapshots retain actor/critics/targets/optimizer/replay/RNG. Exact arbitrary mid-episode sampler continuation is not claimed.

## Failures and limitations

The first Four-Rooms adapter smoke run exposed a reset keyword mismatch, which was corrected before evaluation. The first tabular exploration pilot (20,000 steps, 256 random steps, no mixing) failed: return fell from 0.290557 to 0.149218 and KL increased. Q RMSE alone improved, demonstrating why it was insufficient for acceptance. Unvisited state-actions were 17.5%. A separate pilot used 10,000 initial random actions, 10% behavior mixing and a 100,000-transition budget, then passed. Those settings were frozen before evaluation. No failed evaluation seed was discarded. Pilot configurations, metrics and gate outcomes are retained in [pilot results](results/M1_pilots.json).

This evaluation covers one frozen Four-Rooms incentive/goal configuration and one uniform MG leader, two evaluation seeds per condition, and CPU execution. It is not a full reproduction of the paper or evidence about leader tracking, continuous tasks or high-dimensional scaling. GPU resource/performance measurements and fresh dependency reconstruction remain unverified. The existing MG environment has an unrelated missing `pikepdf` dependency for img2pdf, documented with a reproduction command in the runbook.

Operation counts are exact at the corresponding update sites. Wall-clock fields distinguish reference calculation, rollout evaluation and model diagnostics, but `online_wall_seconds` is an approximate remainder including non-optimizer overhead and earlier held-out calculations. Do not use it as isolated learner compute time. See [commands, resources and schema](M0_M1_reproduction.md) for these boundaries and reproduction steps.

## Gate

All eight evaluation runs passed the predeclared M1 learning gate. M0 contracts/regressions and all four isolation tests passed. The fixed-leader learning and diagnostic prerequisites for planning M2 are satisfied, subject to the limitations above; this is not evidence that an online M2 algorithm will succeed.

The implementation stops at M1. Any future M2 comparison needs a separate task and experiment specification; none was run here.
