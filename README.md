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

```bash
conda install pytorch -c pytorch        # install pytorch as per your operating system
pip install "gymnasium[mujoco]" stable-baselines3
```
