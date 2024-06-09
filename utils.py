import sys
import numpy as np
import gymnasium as gym
from typing import Optional, Literal
from pathlib import Path
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append(str(Path(__file__).resolve().parent))

from humanoid.wrapper import HumanoidCustomObservation


def get_humanoid_env(
    no_wrapper: bool = False,
    version: int = 4,
    render_mode: Optional[str] = None,
) -> gym.Env:
    env = gym.make(f"Humanoid-v{version}", render_mode=render_mode)
    if not no_wrapper:
        env = HumanoidCustomObservation(env)
    return env


def load_model(
    env: gym.Env,
    algo: Literal["sac", "td3", "ppo", "a2c", "ddpg"],
    tensorboard_log: Optional[str] = None,
    chkpt: Optional[str] = None,
) -> SAC | PPO | TD3 | A2C | DDPG:
    args = ("MlpPolicy", env)
    kwargs = {"verbose": 1, "tensorboard_log": tensorboard_log}

    print("Algorithm: ", end="")
    if algo == "sac":
        print("SAC")
        model = SAC(*args, **kwargs)
    elif algo == "td3":
        print("TD3")
        n_actions = env.action_space.shape[-1]
        mu, std = np.zeros(n_actions), 0.1 * np.ones(n_actions)
        kwargs["action_noise"] = NormalActionNoise(mu, std)
        model = TD3(*args, **kwargs)
    elif algo == "ppo":
        print("PPO")
        model = PPO(*args, **kwargs)
    elif algo == "a2c":
        print("A2C")
        model = A2C(*args, **kwargs)
    elif algo == "ddpg":
        print("DDPG")
        model = DDPG(*args, **kwargs)
    else:
        print("Error")
        raise ValueError(f"Invalid algorithm: {algo}")

    if chkpt is not None:
        try:
            model.set_parameters(chkpt)
            print(f"Loaded model from {chkpt}.")
        except ValueError:
            print(f"Failed to load model from {chkpt}.\nTraining from scratch.")

    return model
