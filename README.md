# Deep Reinforcement Learning with MuJuCo Humanoid Environment

> This is the final project for the course "Deep Reinforcement Learning" at the National Tsing Hua University.
> The goal of this project is to train an agent to solve [the MuJuCo Humanoid environment](https://gymnasium.farama.org/main/environments/mujoco/humanoid/).

## Installation

### Create a Conda Environment

```bash
conda create -n humanoid python=3.11 -y    # python >= 3.8 or <= 3.11 is required
conda activate humanoid
```

### Install Dependencies

```bash
conda install pytorch -c pytorch -y        # install pytorch as per your operating system
pip3 install "gymnasium[mujoco]" stable-baselines3 numpy==2.0.0rc2 tensorboard
```

### Fix MuJuCo Rendering Bug

In `/home/<username>/miniconda3/envs/humanoid/lib/python3.11/site-packages/gymnasium/envs/mujoco/mujoco_rendering.py`, there is an bug relating to the rendering of the MuJuCo environment.
To fix this bug, you need to replace the following code in line 592:

- Before:
    ```python
            self.add_overlay(
                bottomleft, "Solver iterations", str(self.data.solver_iter + 1)
            )
    ```

- After:
    ```python
            self.add_overlay(
                bottomleft, "Solver iterations", str(self.data.solver_niter + 1)
            )
    ```

After this fix, the rendering of the MuJuCo environment should work properly.

## Usage

```bash
conda activate humanoid
python3 train.py <options...>
```

### Options

- `-h`, `--help`

    Show help message and exit

- `-t`, `--timesteps`: `[int]`
                    
    The total number of timesteps to train for. (Default: `1_000_000`)

- `-a`, `--algo`: `["sac","ppo","td3"]`

    The algorithm to use for training. (Default: `sac`)

- `-her`, `--use-her`

    Use hindsight experience replay. (Default: `False`)

- `-l`, `--lr`, `--learning-rate`: `[float]`

    Learning rate. (Default: `0.00025`)
