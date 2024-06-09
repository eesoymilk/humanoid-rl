import sys
import argparse
from pathlib import Path
from datetime import datetime
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from utils import get_humanoid_env, get_logger, load_model


def parse_args() -> tuple[int, int, bool, str]:
    """
    Parse the command line arguments.

    return:
        total_timesteps: int
        n_envs: int
        no_wrapper: bool
        algo: str
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
    chkpt_dir: Path,
    no_wrapper: bool,
    log_interval: int = 10,
    progress_bar: bool = True,
) -> None:
    try:
        model_name = model.__class__.__name__
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            tb_log_name=f"{model_name}_{'nowrapped' if no_wrapper else 'wrapped'}",
            progress_bar=progress_bar,
        )
    except KeyboardInterrupt:
        now = datetime.now()
        print(f"Training interrupted at {now.strftime('%m/%d %H:%M:%S')}")

    algo_name = model.__class__.__name__.lower()
    fname = f"{algo_name}_{'' if no_wrapper else 'wrapped_'}humanoid"
    model.save(chkpt_dir / fname)


def main() -> None:
    total_timesteps, no_wrapper, algo = parse_args()

    start_time = datetime.now().strftime("%m%d%H%M")

    folder_name = (
        f"{start_time}_{algo}{'_nowrapped' if no_wrapper else '_wrapped'}"
    )

    chkpt_dir = SCRIPT_DIR / "models" / "checkpoints" / folder_name
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    tb_log = chkpt_dir / "logs"
    tb_log.mkdir(parents=True, exist_ok=True)

    env = get_humanoid_env(no_wrapper=no_wrapper)

    model = load_model(env, algo, tensorboard_log=tb_log)
    train(model, total_timesteps, chkpt_dir, no_wrapper)


if __name__ == "__main__":
    main()
