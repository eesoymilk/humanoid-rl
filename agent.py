import gymnasium as gym
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


class Agent:
    def __init__(self, env: gym.Env) -> None:
        self._env = env

    @property
    def env(self) -> gym.Env:
        return self._env

    def act(self, observation: npt.NDArray) -> npt.NDArray:
        return self.env.action_space.sample()

    def evaluate(self, episodes: int = 10) -> float:
        rewards: list[float] = []

        for _ in tqdm(range(episodes), desc="Evaluating"):
            obs, info = self.env.reset()
            episode_reward = 0.0

            while True:
                action = self.act(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            rewards.append(episode_reward)

        return np.mean(rewards)

    def close(self) -> None:
        self.env.close()
