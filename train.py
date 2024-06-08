import sys
import argparse
from pathlib import Path
from datetime import datetime
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from utils import get_humanoid_env, get_logger, load_model


def parse_args() -> tuple[int, bool, str, float]:
    """
    Parse the command line arguments.

    return:
        total_timesteps: int
        no_wrapper: bool
        algo: str
        lr: float
    """
    parser = argparse.ArgumentParser(
        "train", description="Train the Humanoid environment."
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=10_000_000,
        help="The total number of timesteps to train for. [Default: 1_000_000]",
    )
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo", "td3", "a2c", "ddpg"],
        help="The algorithm to use for training. [Default: sac]",
    )
    parser.add_argument(
        "--no-wrapper",
        action="store_true",
        dest="no_wrapper",
        default=False,
        help="Disable the custom observation wrapper.",
    )
    args = parser.parse_args()

    return (
        args.timesteps,
        args.no_wrapper,
        args.algo,
    )


def train(
    model: SAC | PPO | TD3 | A2C | DDPG,
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


def main() -> None:
    total_timesteps, no_wrapper, algo = parse_args()

    start_time = datetime.now().strftime("%m%d%H%M")
    
    folder_name = f"{start_time}_{algo}{'_nowrapped' if no_wrapper else '_wrapped'}"

    checkpoints_dir = SCRIPT_DIR / "models" / "checkpoints" / folder_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger_dir = checkpoints_dir / "logs"
    logger_dir.mkdir(parents=True, exist_ok=True)

    env = get_humanoid_env(no_wrapper)
    logger = get_logger(logger_dir)

    model = load_model(env, algo, logger=logger)
    train(model, total_timesteps, checkpoints_dir, no_wrapper)


if __name__ == "__main__":
    main()
