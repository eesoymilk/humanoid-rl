import numpy as np
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import SAC

SCRIPT_DIR = Path(__file__).resolve().parent


def eval(
    env: gym.Env,
    model: SAC,
    n_episodes: int = 10,
    verbose: bool = True,
) -> None:
    episode_rewards = []
    for ep in range(n_episodes):
        episode_reward = 0
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward

            if terminated or truncated:
                break

        if verbose:
            print(f"Episode {ep + 1} reward: {episode_reward}")

        episode_rewards.append(episode_reward)

    print(f"Mean episode reward: {np.mean(episode_rewards)}")


def main() -> None:
    # env = gym.make("Humanoid-v4", render_mode="human")
    env = gym.make("Humanoid-v4")
    model: SAC = SAC.load(SCRIPT_DIR / "models" / "sac_humanoid_1M")
    eval(env, model)


if __name__ == "__main__":
    main()
