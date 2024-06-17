import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from utils import get_humanoid_env, load_model


def parse_args() -> tuple[int, str, bool, int, int]:
    """
    Parse the command line arguments.

    return:
        total_timesteps: int
        algo: str
        no_wrapper: bool
        replay_buffer: int
        save_interval: int
    """
    parser = argparse.ArgumentParser(
        "train", description="Train the Humanoid environment."
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=5_000_000,
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
    parser.add_argument(
        "-rb",
        "--replay-buffer",
        type=int,
        dest="replay_buffer",
        default=1_000_000,
        help="The size of the replay buffer. [Default: 1_000_000]",
    )
    parser.add_argument(
        "-si",
        "--save-interval",
        type=int,
        default=1_000_000,
        help="The number of timesteps between saving the model. [Default: 1_000_000]",
    )
    args = parser.parse_args()

    return (
        args.timesteps,
        args.no_wrapper,
        args.algo,
        args.replay_buffer,
        args.save_interval,
    )


class TensorboardAndIntervalSaveCallback(BaseCallback):
    """
    Callback for saving a model every 'save_interval' steps
    """

    def __init__(self, save_interval: int, save_path: str, wrapped: str, verbose=1):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path
        self.best_mean_reward = -float('inf')
        self.wrapped = wrapped

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            self.logger.dump(self.n_calls)
            reward = evaluate_policy(self.model, self.training_env, 5)
            self.logger.record("eval/reward", reward[0])
            self.logger.record("eval/std", reward[1])
        if self.n_calls != 0 and self.n_calls % self.save_interval == 0:
            path = os.path.join(
                self.save_path, f'{self.model.__class__.__name__}_{self.wrapped}_{self.num_timesteps}.zip'
            )
            self.model.save(path)
            if self.verbose > 0:
                print(
                    f"Saving model checkpoint to {path} on step {self.num_timesteps}"
                )
        return True


def train(
    model: SAC | PPO | TD3 | A2C | DDPG,
    start_time_str: str,
    total_timesteps: int,
    chkpt_dir: Path,
    no_wrapper: bool,
    log_interval: int = 10,
    progress_bar: bool = True,
    save_interval: int = 1_000_000,
) -> None:
    run_name = f"{model.__class__.__name__}_{'no' if no_wrapper else ''}wrapped_{start_time_str}"
    save_path = str(chkpt_dir / run_name)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            tb_log_name=run_name,
            progress_bar=progress_bar,
            callback=TensorboardAndIntervalSaveCallback(save_interval, save_path, "nowrapped" if no_wrapper else "wrapped"),
        )
    except KeyboardInterrupt:
        now = datetime.now()
        print(f"Training interrupted at {now.strftime('%m/%d %H:%M:%S')}")

    model.save(chkpt_dir / run_name)


def main() -> None:
    total_timesteps, no_wrapper, algo, replay_buffer, save_interval = (
        parse_args()
    )

    start_time_str = datetime.now().strftime("%m%d%H%M")

    folder_name = (
        f"{start_time_str}_{algo}{'_nowrapped' if no_wrapper else '_wrapped'}"
    )

    chkpt_dir = SCRIPT_DIR / "models" / "checkpoints" / folder_name
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    tb_log = chkpt_dir / "logs"
    tb_log.mkdir(parents=True, exist_ok=True)

    env = get_humanoid_env(no_wrapper=no_wrapper)

    model = load_model(env, algo, replay_buffer, tensorboard_log=tb_log)
    train(
        model,
        start_time_str,
        total_timesteps,
        chkpt_dir,
        no_wrapper,
        save_interval,
    )


if __name__ == "__main__":
    main()
