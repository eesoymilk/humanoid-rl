# Deep Reinforcement Learning with MuJuCo Humanoid Environment

> This is the final project for the course "Deep Reinforcement Learning" at the National Tsing Hua University.
> The goal of this project is to train an agent to solve [the MuJuCo Humanoid environment](https://gymnasium.farama.org/main/environments/mujoco/humanoid/).

![The image of result in render mode](./Walk-Away.png)

## Environment Installation

### Create a Conda Environment

```bash
conda create -n humanoid python=3.11 -y    # python >= 3.8 or <= 3.11 is required
conda activate humanoid
```

### Install Dependencies

```bash
# install pytorch as per your operating system, the below command applies to Linux
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip3 install "gymnasium[mujoco]" "stable-baselines3[extra]" tensorboard numpy
```

### Fix MuJuCo Rendering Bug

In `$CONDA_PATH/envs/humanoidrun -n libpython3.11/site-packages/gymnasium/envs/mujoco/mujoco_rendering.py`, there is an bug relating to the rendering of the MuJuCo environment.
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

### Training

```bash
conda activate humanoid && python3 train.py <options...>
```

#### Options

- `-h`, `--help`

    Show help message and exit

- `-t`, `--timesteps`: `[int]`

    The total number of timesteps to train for. (Default: `5_000_000`)

- `--no-wrapper`

    Disable the wrapper for the environment. (Default: `False`)

- `-a`, `--algo`: `["sac","ppo","td3", "a2c", "ddpg"]`

    The algorithm to use for training. (Default: `sac`)

### Evaluating

```bash
conda activate humanoid && python3 eval.py <options...>
```

#### Options

- `-h`, `--help`

    Show help message and exit

- (**required**) `-m`, `--model`, `--model-name`: `[str]`

    The name of the model to evaluate.
    This repo offer basic pre-trained model
    |Model Name|Parameter|
    |---|---|
    |SAC|`sac_wrapped`|
    |SAC without wrapped|`sac_nowrapped`|
    |PPO|`PPO_wrapped`|
    |PPO without wrapped|`PPO_nowrapped`|
    |A2C|`A2C_wrapped`|
    |A2C without wrapped|`A2C_nowrapped`|
    |TD3|`TD3_wrapped`|
    |TD3 without wrapped|`TD3_nowrapped`|
    |DDPG|`DDPG_wrapped`|
    |DDPG without wrapped|`DDPG_nowrapped`|

- `-r`, `--render`

     Render the environment. (Default: `False`)
