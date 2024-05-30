import gymnasium as gym
from pathlib import Path
from stable_baselines3 import SAC

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    env = gym.make("Humanoid-v4", render_mode="human")
    model = SAC.load(SCRIPT_DIR / "models" / "sac_humanoid_05302228")

    total_reward = 0.0
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        if terminated or truncated:
            break

    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    main()
