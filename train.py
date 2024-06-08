import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from datetime import datetime
from stable_baselines3 import SAC

from utils import get_humanoid_env, get_logger, load_sac_model

TOTAL_TIMESTEPS = 1_000_000


def train(model: SAC, total_timesteps: int, save_dir: Path) -> None:
    interrupted = False
    try:
        model.learn(total_timesteps=total_timesteps, log_interval=10)
    except KeyboardInterrupt:
        interrupted = True

    now = datetime.now()
    if interrupted:
        print("Training interrupted at", now.strftime("%m/%d %H:%M:%S"))

    fname = f"sac_humanoid_{now.strftime('%m%d%H%M')}"
    model.save(save_dir / fname)


def main() -> None:
    start_time = datetime.now().strftime("%m%d%H%M")

    checkpoints_dir = SCRIPT_DIR / "models" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger_dir = SCRIPT_DIR / "logs" / start_time
    logger_dir.mkdir(parents=True, exist_ok=True)

    env = get_humanoid_env()
    logger = get_logger(logger_dir)

    model = load_sac_model(env, logger)
    train(model, TOTAL_TIMESTEPS, checkpoints_dir)


if __name__ == "__main__":
    main()
