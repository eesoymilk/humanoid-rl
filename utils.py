import sys
import gymnasium as gym

from pathlib import Path
from typing import Optional, Literal
from datetime import datetime
from stable_baselines3 import SAC, PPO, TD3, HerReplayBuffer
from stable_baselines3.common.logger import configure, Logger

sys.path.append(str(Path(__file__).resolve().parent))

from humanoid.wrapper import HumanoidCustomObservation


def get_humanoid_env(version: int = 4) -> gym.Env:
    env = gym.make(f"Humanoid-v{version}")
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
    algo: Literal["sac", "ppo", "td3"],
    use_her: bool,
    lr: float,
    target_update_interval: int = 4,
    chkpt: Optional[Path] = None,
) -> SAC | PPO | TD3:
    args = ("MlpPolicy", env)
    kwargs = {
        "learning_rate": lr,
        "target_update_interval": target_update_interval,
        "replay_buffer_class": (
            HerReplayBuffer if use_her and algo != "ppo" else None
        ),
        "verbose": 1,
    }

    if algo == "sac":
        model = SAC(*args, **kwargs)
    elif algo == "ppo":
        model = PPO(*args, **kwargs)
    elif algo == "td3":
        model = TD3(*args, **kwargs)
    else:
        raise ValueError(f"Invalid algorithm: {algo}")

    model.set_logger(logger)

    if chkpt is not None:
        try:
            model.set_parameters(chkpt)
            print(f"Loaded model from {chkpt.name}.")
        except ValueError:
            print(f"Failed to load model from {chkpt.name}.")

    return model


def train(model: SAC | PPO | TD3, total_timesteps: int, save_dir: Path) -> None:
    interrupted = False
    try:
        model.learn(total_timesteps=total_timesteps, log_interval=10)
    except KeyboardInterrupt:
        interrupted = True

    if interrupted:
        now = datetime.now()
        print("Training interrupted at", now.strftime("%m/%d %H:%M:%S"))

    fname = f"{model.__class__.__name__.lower()}_humanoid"
    model.save(save_dir / fname)
