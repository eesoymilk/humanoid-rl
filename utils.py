import sys
import numpy as np
import gymnasium as gym

from pathlib import Path
from typing import Optional, Literal
from datetime import datetime
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.logger import configure, Logger

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


def get_logger(logger_dir: Path | str) -> Logger:
    logger = configure(
        folder=str(logger_dir),
        format_strings=["stdout", "log", "csv", "tensorboard"],
    )
    return logger


def load_model(
    env: gym.Env,
    algo: Literal["sac", "td3", "ppo"],
    lr: float = 0.00025,
    target_update_interval: int = 4,
    logger: Optional[Logger] = None,
    chkpt: Optional[str] = None,
) -> SAC | PPO | TD3:
    args = ("MlpPolicy", env)
    kwargs = {
        "learning_rate": lr,
        "target_update_interval": target_update_interval,
        "verbose": 1,
    }

    print("Algorithm: ", end="")
    if algo == "sac":
        print("SAC")
        model = SAC(*args, **kwargs)
    elif algo == "td3":
        print("TD3")
        model = TD3(*args, **kwargs)
    elif algo == "ppo":
        print("PPO")
        model = PPO(*args, **kwargs)
    else:
        print("Error")
        raise ValueError(f"Invalid algorithm: {algo}")

    if logger:
        model.set_logger(logger)

    if chkpt is not None:
        try:
            model.set_parameters(chkpt)
            print(f"Loaded model from {chkpt.name}.")
        except ValueError:
            print(f"Failed to load model from {chkpt.name}.")
            print("Training from scratch.")

    return model


def train(
    model: SAC | PPO | TD3,
    total_timesteps: int,
    save_dir: Path,
    no_wrapper: bool,
    log_interval: int = 10,
) -> None:
    try:
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    except KeyboardInterrupt:
        now = datetime.now()
        print(f"Training interrupted at {now.strftime('%m/%d %H:%M:%S')}")

    algo_name = model.__class__.__name__.lower()
    fname = f"{algo_name}_{'' if no_wrapper else 'wrapped_'}humanoid"
    model.save(save_dir / fname)


def eval(
    env: gym.Env,
    model: SAC,
    n_episodes: int = 10,
    verbose: bool = True,
) -> None:
    ep = 0
    rewards = []

    try:
        while ep < n_episodes:
            ep_reward = 0
            obs, _ = env.reset()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)

                ep_reward += reward

                if terminated or truncated:
                    break

            if verbose:
                print(f"Episode {ep + 1} reward: {ep_reward}")

            rewards.append(ep_reward)

    except KeyboardInterrupt:
        print(f"Evaluation interrupted at episode {ep + 1}")

    print(f"Mean episode reward: {np.mean(rewards)}")
