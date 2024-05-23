# Deep Reinforcement Learning with MuJuCo Humanoid Environment

This is the final project for the course "Deep Reinforcement Learning" at the National Tsing Hua University. The goal of this project is to train an agent to solve the MuJuCo Humanoid environment.

## Python Setup

> Note: the pre-release gymnasium is required to use the latest version of the MuJoCo environment. Therefore, we need to use the `--pre` flag to install `gymnasium v1.0.0a2`.

```bash
conda create -n humanoid python=3.11    # python >=3.8 or <=3.11 is required
conda activate humanoid
conda install pytorch -c pytorch        # install pytorch as per your operating system
pip install --pre "gymnasium[mujoco]"   # install pre-release gymnasium with mujoco
```
