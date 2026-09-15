# M0 implementation and validation

Date: 2026-09-15. Base revision: 82753d6 (documentation directory rename only since the investigation). Scope: M0; M1 runs follow this gate. Python environments remain separate.

## Results

- Existing Four-Rooms BC-HG command completed, CPU, seed 0, two outer iterations, 20 new transitions per iteration. Final full-save verification: `artifacts/m0/cmdp_entry_verified.log` and `four_rooms_entry_final/metrics_bchg.pkl`, 5.87 seconds. Reset/Carry CLIs also completed with full saves, 7.85 seconds each (`four_rooms_{reset,carry}_entry`).
- Existing Four-Rooms Reset-Q and Carry-Q update functions completed twice with identical seeds and identical results. Per iteration: 20 environment steps, 20 leader SARSA assignments, one leader optimizer step, 10 adopted follower sweeps, 40 diagnostic continuation sweeps, 200 diagnostic policy-evaluation sweeps. Actual fixed-length scan sizes are counted; these are not optimizer steps. See `artifacts/m0/cmdp/results.json`.
- Existing MG command completed for Reset-Q and Carry-Q, seed 0, 900 training transitions. Each: leader actor 2 / critic 2 / target cycles 2; follower adopted sweeps 4 / diagnostic continuation sweeps 36. Repeated Carry-Q returns matched exactly. See `artifacts/m0/mg_results.json`.
- MG unit tests passed: environment/action context, timeout bootstrap, observed-leader expectation, diagnostic isolation for Reset/Carry, SAC warm-up, real optimizer counters and full follower pickle + RNG round-trip followed by an identical next update.
- Four-Rooms tests passed: position+goal context/dtype, reward signs, true termination/truncation, corrected model against 20,000 sampled transitions, diagnostic-budget isolation, full numerical carry serialization followed by an identical next update.

## Changes and compatibility

Published-code Bellman/model behavior remains the default (`legacy`, `consistent_model=False`). M1 explicitly uses expectation of soft values over next leader action and Four-Rooms' sampled transition law. Raw Four-Rooms observations are now float32, matching the declared space. Extra info fields retain final observation and separate termination/truncation. In consistent mode a time limit no longer freezes the last physical transition.

MG timeout checks inspect the boolean flag, not mere key presence. Seed code no longer overwrites a torch API. Trainer restore uses `leader=` and stores global Python/NumPy/Torch RNG state. MG snapshots retain algorithm buffers; sampler workers are recreated (exact arbitrary mid-rollout sampler continuation is not claimed). Full SAC next-update continuation and JAX numerical carry continuation are tested. The Four-Rooms CLI retains the original incentive checkpoint and additionally saves full numerical carry, resolved settings and source fingerprints; restore uses a matching initialized template.

Existing JAX reference Q includes current-state entropy, and its regularized value-prediction helper re-derives a Boltzmann policy. Those legacy diagnostics are not the M1 standard-soft-Q reference. M1 requires its own mathematically consistent diagnostic operator and fixed-policy evaluation. No existing cosine metric is labelled true hypergradient accuracy.

A first measurement-JSON run failed on a NumPy int64; serialization was corrected and both conditions rerun successfully. Failed logs were followed by successful logs; no seed was dropped.

The final JAX CLI save check exposed an existing shallow-copy mutation: traced hyperparameters replaced entries of the original nested configuration. Each run now copies the upper-optimisation mapping before substitution. All three CLIs then saved resolved configuration, full state, original incentive checkpoint and finite metrics successfully. The full Trainer restore test also passed for leader networks, follower Q, optimizer state, replay and RNG.

## Reproduce

From repository root, use the Python of the indicated environment (or activate it). CPU flags: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bchg-mpl`, plus `JAX_PLATFORMS=cpu` for JAX.

```
python tests/m0_cmdp.py                      # bchg-cmdp
python tests/m0_mg.py                        # bchg-mg
python configurable_mdp/train_four_rooms_bchg.py --experiment_dir artifacts/m0/four_rooms_entry
python markov_game/train_discrete_toy.py --config configs/m0/mg.yaml --no_aggregate name=M0_reset_fixed follower.reset_q=true
python markov_game/train_discrete_toy.py --config configs/m0/mg.yaml --no_aggregate name=M0_carry_fixed
```

Copy `configs/m0/four_rooms.yaml` into a fresh experiment directory as `config.yaml` before the Four-Rooms CLI command. Existing output directories should not be reused for comparisons. MG commands create timestamped directories. The summarizer `tests/summarize_m0.py` additionally compares the named initial Carry run `M0_carry`.

Dependency versions are captured in `artifacts/m0/{cmdp,mg}-freeze.txt`; original conda definitions are retained. Existing installed environments passed the import/runtime checks in the investigation. Fresh conda reconstruction and GPU execution are not certified by these CPU checks; CPU-only JAX can replace the CUDA jaxlib wheel with jaxlib==0.4.21, and the MG Torch wheel with torch==1.7.1+cpu using the existing PyTorch wheel index. Pin the remaining versions from the recorded environment. Environment verifiers now use the active Python's pip.

Portable package-version inventories, proposed CPU environment recipes, commands and log schema are retained in [M0–M1 reproduction](M0_M1_reproduction.md). Fresh solver resolution remains unverified and is explicitly separated from the successful installed-environment checks.

Final `python -m pip check`: JAX passed; MG reports the reproducible outstanding dependency `img2pdf 0.4.4 requires pikepdf, which is not installed`. PDF conversion is not used by these runs. This is reported under the handoff's unresolved-dependency allowance, not counted as a clean dependency build.

## Gate

M0 short regressions and contract tests pass. Proceed to fixed-leader M1 only. Full historical paper reproduction, exact mid-episode sampler resume, GPU profiling and M2 are outside this gate. M1 must verify actual sample learning and oracle isolation; baseline execution alone does not establish these.
