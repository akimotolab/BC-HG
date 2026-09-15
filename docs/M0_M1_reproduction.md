# M0–M1 reproduction and artifact schema

Run commands from the repository root. M0 and M1 reports describe the tested CPU environment; JAX and Torch/garage need separate environments. The CPU reconstruction recipes below derive from the original environment files. A fresh installation was **not** tested; `configs/environments/*-observed-freeze.txt` records installed package versions, not a portable, solver-validated lock. In particular, garage 2021.3.0 and Torch 1.7.1 require the legacy Python 3.8 stack. Do not resolve them into the JAX Python 3.11 environment. If reconstruction fails, retain the solver output alongside these exact recipes and inventories.

Dependency audit on 2026-09-15: `python -m pip check` passed in bchg-cmdp. In bchg-mg it reported `img2pdf 0.4.4 requires pikepdf, which is not installed.` This unresolved PDF-conversion dependency is outside the exercised training/test imports; all M0/M1 runtime checks passed without it. The original environment recipe still includes img2pdf. Fresh-environment and PDF-conversion validation remain outstanding; no dependency installation was silently performed.

```bash
conda env create -n bchg-cmdp -f configs/environments/cmdp-cpu.yaml
conda env create -n bchg-mg -f configs/environments/mg-cpu.yaml
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bchg-mpl
conda activate bchg-cmdp
export JAX_PLATFORMS=cpu
python configurable_mdp/verify_experiment_environment.py
python tests/m0_cmdp.py
mkdir -p artifacts/m0/four_rooms_entry
cp configs/m0/four_rooms.yaml artifacts/m0/four_rooms_entry/config.yaml
python configurable_mdp/train_four_rooms_bchg.py --experiment_dir artifacts/m0/four_rooms_entry
python configurable_mdp/export_fixed_four_rooms.py

conda activate bchg-mg
python markov_game/verify_experiment_environment.py
python tests/m0_mg.py
python tests/m1_unit.py
python markov_game/train_discrete_toy.py --config configs/m0/mg.yaml --no_aggregate name=M0_reset_fixed follower.reset_q=true
python markov_game/train_discrete_toy.py --config configs/m0/mg.yaml --no_aggregate name=M0_carry_fixed
python tests/m0_restore.py
```

Use fresh output directories. `tests/m0_restore.py` locates a `M0_carry_fixed` snapshot; retain a single matching run when reproducing this test. Unit checks take seconds to tens of seconds; the fixed-leader runs below are integration/experiment work and need not run on each CI invocation.

For each condition, run pilot seed 100 first and inspect `summary.json`; then run both evaluation seeds with unchanged settings. The checked-in settings and acceptance criteria were fixed before evaluation. Execute conditions in the displayed order. The original failed exploration pilot is documented in the M1 report and protocol.

```bash
for condition in four_rooms_tabular mg_tabular four_rooms_sac mg_sac; do
  python markov_game/train_fixed_follower.py --config configs/m1/$condition.yaml --seed 100 --output artifacts/m1/pilot_$condition
  for seed in 0 1; do
    python markov_game/train_fixed_follower.py --config configs/m1/$condition.yaml --seed $seed --output artifacts/m1/eval_${condition}_$seed
  done
  python tests/m1_isolation.py --config configs/m1/$condition.yaml --output artifacts/m1/isolation_$condition
done
python tests/report_m1.py
```

Inspect each pilot and condition gate before proceeding to the next condition. This shell block supplies commands, not an automatic approval of a failed gate. `--no-oracle` disables reference computation. Its summary has `passed: null`, because it cannot judge reference improvement.

GPU command (not tested here): activate the original CUDA-capable `bchg-mg` environment, set `CUDA_VISIBLE_DEVICES=0`, and add `--device cuda` to `train_fixed_follower.py`. SAC uses two 64-unit hidden layers, twin critics and their targets; replay capacity is 20,000, batch size 64. One compatible CUDA GPU per process is sufficient architecturally; no VRAM or GPU performance claim was measured. Tabular conditions and reference linear solves are small and suited to CPU. This does not authorize any M2 or continuous-task runs.

## Artifacts

- `config.yaml`: complete resolved run settings, including seed, fixed beta/gamma, exploration, replay, update intervals and acceptance thresholds.
- `metrics.json`: initial and per-epoch evaluations. `environment_steps` counts newly collected training transitions; `evaluation_steps` counts independent rollout transitions. `counts` separates train calls, batch/table assignments, critic 1/2, actor, target cycles and temperature updates. A target cycle updates both SAC target critics. `leader_counts` must remain zero.
- Return and reference return include discounted entropy; `optimality_gap` is un-clipped reference minus current return. KL direction is current || reference. Uniform and visit-weighted KL, Q RMSE, mean/sup Bellman optimality residual, actor–Boltzmann KL/value gap, entropy, unvisited fraction and held-out TD MSE are distinct fields.
- `oracle_sweeps` counts reference fixed-point iterations (each also evaluates a residual); `diagnostic_policy_solves` counts fixed-policy linear solves including the reference solve. These are outside online counters.
- `wall_seconds` is elapsed time at evaluation entry after the rollout; `reference_seconds` is the separate initial reference calculation; `diagnostic_evaluation_seconds` and `evaluation_wall_seconds` are cumulative model diagnostics and rollout times. The legacy `online_wall_seconds` estimate subtracts these from elapsed time and includes setup, logging, snapshots and earlier held-out calculations. It is an approximate runtime measure, **not** isolated optimizer time; differences of a diagnostic-call duration may occur at its boundary.
- `visits_<step>.npz`, `final.npz`: full state/leader-action/follower-action visit counts; final Q and current policy. Visit counts must sum to collected transitions.
- `follower_checkpoint.pkl`: follower, optimizer, targets, replay and global RNG state. `params.pkl` is the existing Trainer snapshot with both agents. Restore only trusted local pickle files. Exact continuation at an arbitrary sampler mid-episode is not supported.
- `summary.json`: final-checkpoint gate and code provenance (HEAD, diff hash, per-source hashes and runtime versions). No best checkpoint selection. `training.log` retains the existing Trainer output.

Four-Rooms' export is a physical task model, not an oracle policy: it includes fixed incentive weights, transitions, rewards, context, initial distribution and terminal masks. Training samples this model through the environment adapter. Model-based reference values are only evaluated in the diagnostic module. The exported metadata fixes beta/gamma and is checked by the runner.

Large runtime outputs remain under ignored `artifacts/` and the original MG data directory. Reviewable M0/M1 result summaries are retained under `docs/results/`.
