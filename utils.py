import gymnasium as gym

from pathlib import Path
from typing import Optional
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure, Logger

from humanoid.wrapper import HumanoidCustomObservation


def get_humanoid_env() -> gym.Env:
    env = gym.make("Humanoid-v4")
    env = HumanoidCustomObservation(env)
    return env


def get_logger(logger_dir: Path | str) -> Logger:
    logger = configure(
        folder=str(logger_dir),
        format_strings=["stdout", "log", "csv", "tensorboard"],
    )
    return logger


def load_sac_model(
    env: gym.Env,
    logger: Logger,
    lr: float = 0.00025,
    target_update_interval: int = 4,
    chkpt: Optional[Path] = None,
) -> SAC:
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=lr,
        target_update_interval=target_update_interval,
        verbose=1,
    )

    model.set_logger(logger)

    if chkpt is not None:
        try:
            model.set_parameters(chkpt)
            print(f"Loaded model from {chkpt.name}.")
        except ValueError:
            print(f"Failed to load model from {chkpt.name}.")

    return model
