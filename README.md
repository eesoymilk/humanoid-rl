# Deep Reinforcement Learning with MuJuCo Humanoid Environment

> This is the final project for the course "Deep Reinforcement Learning" at the National Tsing Hua University.
> The goal of this project is to train an agent to solve [the MuJuCo Humanoid environment](https://gymnasium.farama.org/main/environments/mujoco/humanoid/).

## Installation

### Create a Conda Environment

```bash
conda create -n humanoid python=3.11    # python >=3.8 or <=3.11 is required
conda activate humanoid
```

### Install Dependencies

#### PyTorch

```bash
conda install pytorch -c pytorch        # install pytorch as per your operating system
```

#### Gymnasium with MuJoCo

We will use the `v5` environment of the MuJoCo Humanoid environment and the pre-release gymnasium (`v1.0.0a2`) is required to use the latest version of the MuJoCo (`v3.1.5`).
Therefore, we need to use the `--pre` when installing gymnasium.
You can see more about the incompatibility issue in [this PR](https://github.com/Farama-Foundation/Gymnasium/pull/746).

```bash
pip install --pre "gymnasium[mujoco]"   # install pre-release gymnasium with mujoco
```
