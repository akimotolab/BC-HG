# BC-HG: Boltzmann Covariance HyperGradient

BC-HG (Boltzmann Covariance HyperGradient) is a bilevel reinforcement learning algorithm for decentralized bilevel RL.
This repository is the official implementation for the ICAPS 2026 paper, "Sample-Efficient Hypergradient Estimation for Decentralized Bi-Level Reinforcement Learning" ([arXiv:2603.14867](https://arxiv.org/abs/2603.14867)).
It includes two bilevel RL problem settings and supports end-to-end reproducibility (training → figures):

- `configurable_mdp`: bilevel RL in configurable MDPs (JAX-based experiments)
- `markov_game`: bilevel RL in Markov games (PyTorch/garage-based experiments)

Third-party integration in this two-setting structure:

- `configurable_mdp` extends **HPGD** (https://github.com/lasgroup/HPGD), licensed under MIT.
- See `configurable_mdp/LICENSE` for details.

The two settings use different dependency stacks and must be run in separate conda environments.

## Method and Paper

Reference paper:

- **Sample-Efficient Hypergradient Estimation for Decentralized Bi-Level Reinforcement Learning**  
  Mikoto Kudo, Takumi Tanabe, Akifumi Wachi, Youhei Akimoto  
  ICAPS 2026, [arXiv:2603.14867](https://arxiv.org/abs/2603.14867)

To reproduce figures and tables:

1. Run experiments with `python ...` commands (or `.sh` sweeps).
2. Execute the corresponding notebooks.

## Quick Start (Minimal Reproduction)

Use this path to run one BC-HG experiment and reproduce one figure with the fewest steps.

```bash
# 1) Create and activate environment (Configurable-MDP setting)
conda env create -n bchg-cmdp -f configurable_mdp/environment.yaml
conda activate bchg-cmdp

# 2) Run one BC-HG experiment (FourRooms)
python configurable_mdp/train_four_rooms_bchg.py \
	--experiment_dir configurable_mdp/data/experiment_reg_lambda_0_001_total_steps_100

# 3) Open notebook for visualization
jupyter notebook configurable_mdp/notebooks/experiment_visualization_four_rooms.ipynb
```

Expected outcome:

- Training logs/results are updated under `configurable_mdp/data/experiment_reg_lambda_0_001_total_steps_100`.
- Running the notebook reproduces the corresponding FourRooms plots.

## Quick Start (Markov Game, Optional)

Use this path if you want the shortest reproducible run in the Markov-game setting.

```bash
# 1) Create and activate environment (Markov-game setting)
conda env create -n bchg-mg -f markov_game/environment.yaml
conda activate bchg-mg

# 2) Run one BC-HG experiment (DiscreteToy)
python markov_game/train_discrete_toy.py \
	--config markov_game/config/config_discrete_toy_bchg.yaml

# 3) Monitor with TensorBoard (optional)
tensorboard --logdir markov_game/data/local/

# 4) Open notebook for visualization
jupyter notebook markov_game/notebooks/plot_notebook_DiscreteToy.ipynb
```

Expected outcome:

- Training logs/results are created under `markov_game/data/local/<EXPERIMENT_NAME>/`.
- TensorBoard shows training metrics from the generated log directory.
- Running the notebook reproduces the corresponding DiscreteToy plots.

## Repository Structure

```text
BC-HG/
├── configurable_mdp/    # Configurable-MDP setting: FourRooms / BilevelLQR (JAX)
├── markov_game/         # Markov-game setting: DiscreteToy / BuildingThermalControl (Torch + garage)
└── README.md
```

## Environment Setup

**Important:** The two problem-setting implementations (`configurable_mdp` and `markov_game`) require separate conda environments with different dependency stacks. Do not mix these environments.

### 1) `configurable_mdp` Environment (JAX)

```bash
conda env create -n bchg-cmdp -f configurable_mdp/environment.yaml
conda activate bchg-cmdp
```

**Dependencies:** JAX 0.4.21 (with CUDA 12), Flax, Optax, Gymnax, etc.  
**Quick check:**
```bash
python -c "import jax; print(jax.devices())"
```

### 2) `markov_game` Environment (PyTorch + Garage)

This environment is more complex due to legacy dependencies. **Read this carefully.**

```bash
conda env create -n bchg-mg -f markov_game/environment.yaml
conda activate bchg-mg
```

**Why this is complex:**

- The project uses `garage==2021.3.0`, a research RL framework that requires **PyTorch 1.7.1** (released late 2020).
- This repository pins `torch==1.7.1+cu110`, i.e., the CUDA 11.0 build of PyTorch.
- In typical pip/conda wheel usage, you mainly need an NVIDIA driver compatible with the wheel runtime; a local CUDA toolkit is usually not required unless you build CUDA extensions.
- GPU Compute Capability and driver compatibility still matter; see the table below.

**Environment contents (key pins):**

```yaml
# markov_game/environment.yaml (relevant lines)
torch==1.7.1+cu110
torchvision==0.8.2+cu110
garage==2021.3.0
tensorflow==2.13.1
```

**GPU Compute Capability & CUDA Version Notes (Examples):**

These are practical examples, not strict universal rules. Always verify with your GPU, driver, and PyTorch wheel combination:

| GPU Model | Compute Capability | Compatible CUDA Versions |
|-----------|-------------------|--------------------------|
| RTX 3090, A6000 | 8.6 | 11.1–11.4 (11.0 may work) |
| A100 | 8.0 | 11.0, 11.1–11.4 |
| Older GPUs (e.g., V100, T4) | 7.x | 10.1, 10.2, 11.0+ |

For more details, consult [CUDA—Wikipedia](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

**Verify installation:**

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import garage; print('garage import ok')"
```

## Reproducing Experiments (Canonical: `python ...`)

The canonical execution path is direct `python ...` commands.
Paper-result settings are defined in:

- `configurable_mdp/data/*/config.yaml`
- `markov_game/config/*.yaml`

### Problem Setting 1: Configurable-MDP (`configurable_mdp`)

#### FourRooms

```bash
conda activate bchg-cmdp

python configurable_mdp/train_four_rooms_bchg.py \
	--experiment_dir configurable_mdp/data/experiment_reg_lambda_0_001_total_steps_100
```

You can switch methods by changing the script name (e.g., `train_four_rooms_baseline.py`, `train_four_rooms_hpgd.py`, etc.) and switch paper settings by changing `--experiment_dir`.

#### BilevelLQR

```bash
conda activate bchg-cmdp

python configurable_mdp/train_bilevel_lqr_bchg.py \
	--experiment_dir configurable_mdp/data/experiment_bilevel_lqr_reg_lambda_0_1_steps_10000
```

You can switch methods similarly (e.g., `train_bilevel_lqr_baseline.py`, `train_bilevel_lqr_hpgd.py`, etc.).

### Problem Setting 2: Markov Game (`markov_game`)

#### DiscreteToy

```bash
conda activate bchg-mg

python markov_game/train_discrete_toy.py \
	--config markov_game/config/config_discrete_toy_bchg.yaml
```

#### BuildingThermalControl

```bash
conda activate bchg-mg

python markov_game/train_bilevel_lqr.py \
	--config markov_game/config/config_bilevel_lqr_bchg.yaml
```

Both scripts accept OmegaConf-style overrides from CLI, e.g.:

```bash
python markov_game/train_discrete_toy.py \
	--config markov_game/config/config_discrete_toy_bchg.yaml \
	leader.actor_update_steps_n=5 leader.critic_update_steps_n=2
```

## Full Grid Search / Batch Execution (`.sh`)

Use the provided shell scripts for broad sweeps over multiple hyperparameter settings.

### Problem Setting 1: Configurable-MDP (`configurable_mdp`)

```bash
bash configurable_mdp/exp_four_rooms.sh
bash configurable_mdp/exp_bilevel_lqr.sh
```

Optional modes:

```bash
bash configurable_mdp/exp_four_rooms.sh --dry-run
bash configurable_mdp/exp_four_rooms.sh --background
```

### Problem Setting 2: Markov Game (`markov_game`)

```bash
bash markov_game/train_discrete_toy.sh
bash markov_game/train_bilevel_lqr.sh
```

These scripts target full-sweep runs rather than single reproducible runs.

## Reproducing Figures and Tables

Figures and tables in the paper are generated from experiment data via Jupyter notebooks.
Typical workflow:

1. **Collect experiment data** using `python ...` or `.sh` grid search (see "Reproducing Experiments" above).
2. **Run the corresponding notebook** to visualize and aggregate results.

### Data Directory Mapping

- **Problem Setting 1 (Configurable-MDP):** `configurable_mdp/notebooks` reads from `configurable_mdp/data/experiment_*/` directories.
- **Problem Setting 2 (Markov game):** `markov_game/notebooks` reads from `markov_game/data/local/` directories.

### FourRooms & Random Search Figures

```bash
# First, collect data (if not already present):
conda activate bchg-cmdp
python configurable_mdp/train_four_rooms_bchg.py --experiment_dir configurable_mdp/data/experiment_reg_lambda_0_001_total_steps_100
# ... or run bash configurable_mdp/exp_four_rooms.sh for full sweep

# Then, visualize:
jupyter notebook configurable_mdp/notebooks/experiment_visualization_four_rooms.ipynb
jupyter notebook configurable_mdp/notebooks/random_search_bilevel_lqr.ipynb
```

### BilevelLQR Figures (JAX)

```bash
conda activate bchg-cmdp
python configurable_mdp/train_bilevel_lqr_bchg.py --experiment_dir configurable_mdp/data/experiment_bilevel_lqr_reg_lambda_0_1_steps_10000
# or: bash configurable_mdp/exp_bilevel_lqr.sh

jupyter notebook configurable_mdp/notebooks/experiment_visualization_bilevel_lqr.ipynb
```

### DiscreteToy & BuildingThermalControl Figures (PyTorch)

```bash
conda activate bchg-mg

# Collect data:
python markov_game/train_discrete_toy.py --config markov_game/config/config_discrete_toy_bchg.yaml
python markov_game/train_bilevel_lqr.py --config markov_game/config/config_bilevel_lqr_bchg.yaml
# or: bash markov_game/train_discrete_toy.sh && bash markov_game/train_bilevel_lqr.sh

# Visualize (note: data is in markov_game/data/local/):
jupyter notebook markov_game/notebooks/plot_notebook_DiscreteToy.ipynb
jupyter notebook markov_game/notebooks/plot_notebook_LQR.ipynb
jupyter notebook markov_game/notebooks/MaxEntLQR.ipynb
```

**Note on aggregation:**  
The `markov_game` scripts auto-aggregate results into CSV and TensorBoard logs during training. If you skip aggregation (`--no_aggregate`), run:
```bash
python markov_game/aggregate_results.py markov_game/data/local/<LOG_DIR>
```

## Monitoring Training Progress with TensorBoard

### `markov_game`: Real-Time Training Metrics

During or after `markov_game` training, visualize metrics (loss, reward, etc.) with TensorBoard:

```bash
conda activate bchg-mg
tensorboard --logdir markov_game/data/local/<LOG_DIR>
```

Then open your browser to `http://localhost:6006/` (default TensorBoard port).

## Notes

- Some plotting code uses LaTeX rendering; install TeX packages (e.g., `texlive-latex-base`, `texlive-latex-extra`) if needed.

## License

This project is licensed under Apache License 2.0. See `LICENSE`.