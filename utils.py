import sys
import gymnasium as gym

from pathlib import Path
from typing import Optional, Literal
from datetime import datetime
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.logger import configure, Logger

sys.path.append(str(Path(__file__).resolve().parent))

from humanoid.wrapper import HumanoidCustomObservation


def get_humanoid_env(no_wrapper: bool = False, version: int = 4) -> gym.Env:
    env = gym.make(f"Humanoid-v{version}")
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
    logger: Logger,
    algo: Literal["sac", "td3", "ppo"],
    lr: float,
    target_update_interval: int = 4,
    chkpt: Optional[Path] = None,
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
        print("Training interrupted at", now.strftime("%m/%d %H:%M:%S"))

    algo_name = model.__class__.__name__.lower()
    fname = f"{algo_name}_{'' if no_wrapper else 'wrapped_'}humanoid"
    model.save(save_dir / fname)
