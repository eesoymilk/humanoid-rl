import sys
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from utils import get_humanoid_env, get_logger, load_model, train


def parse_args():
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
        "-her",
        "--use-her",
        action="store_true",
        default=False,
        help="Use hindsight experience replay. [Default: False]",
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
    return args


def main() -> None:
    args = parse_args()
    total_timesteps: int = args.timesteps
    algo: str = args.algo
    use_her: bool = args.use_her
    lr: float = args.lr

    start_time = datetime.now().strftime("%m%d%H%M")

    checkpoints_dir = SCRIPT_DIR / "models" / "checkpoints" / start_time
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger_dir = SCRIPT_DIR / "logs" / start_time
    logger_dir.mkdir(parents=True, exist_ok=True)

    env = get_humanoid_env()
    logger = get_logger(logger_dir)

    model = load_model(env, logger, algo, use_her, lr)
    train(model, total_timesteps, checkpoints_dir)


if __name__ == "__main__":
    main()
