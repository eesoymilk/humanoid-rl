import gymnasium as gym
from gymnasium import ObservationWrapper
from typing import Optional
from pathlib import Path
from datetime import datetime
from stable_baselines3 import SAC

SCRIPT_DIR = Path(__file__).resolve().parent


class HumanoidCustomObservation(ObservationWrapper):
    """Custom observation wrapper for the Humanoid environment."""

    def observation(self, observation):
        # Custom observation logic here
        return super().observation(observation)


def get_humanoid_env() -> gym.Env:
    env = gym.make("Humanoid-v4")
    env = HumanoidCustomObservation(env)
    return env


def load_sac_model(
    env: gym.Env,
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

    if chkpt is not None:
        try:
            model.set_parameters(chkpt)
            print(f"Loaded model from {chkpt.name}.")
        except ValueError:
            print(f"Failed to load model from {chkpt.name}.")

    return model


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
    env = get_humanoid_env()
    models_dir = SCRIPT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model = load_sac_model(env)
    train(model, 50_000_000, models_dir)


if __name__ == "__main__":
    main()
