import sys
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from utils import get_humanoid_env, get_logger, load_model, train


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
        default=1_000_000,
        help="The total number of timesteps to train for. [Default: 1_000_000]",
    )
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo", "td3"],
        help="The algorithm to use for training. [Default: sac]",
    )
    parser.add_argument(
        "--no-wrapper",
        action="store_true",
        dest="no_wrapper",
        default=False,
        help="Disable the custom observation wrapper.",
    )
    parser.add_argument(
        "-l",
        "--lr",
        "--learning-rate",
        type=float,
        default=0.00025,
        help="The learning rate. [Default: 0.00025]",
    )
    args = parser.parse_args()

    return (
        args.timesteps,
        args.no_wrapper,
        args.algo,
        args.lr,
    )


def main() -> None:
    total_timesteps, no_wrapper, algo, lr = parse_args()

    start_time = datetime.now().strftime("%m%d%H%M")

    checkpoints_dir = SCRIPT_DIR / "models" / "checkpoints" / start_time
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger_dir = SCRIPT_DIR / "logs" / start_time
    logger_dir.mkdir(parents=True, exist_ok=True)

    env = get_humanoid_env(no_wrapper)
    logger = get_logger(logger_dir)

    model = load_model(env, algo, lr, logger=logger)
    train(model, total_timesteps, checkpoints_dir, no_wrapper)


if __name__ == "__main__":
    main()
