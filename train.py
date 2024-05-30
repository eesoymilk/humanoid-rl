import gymnasium as gym
from stable_baselines3 import SAC


def main() -> None:
    env = gym.make("Humanoid-v4")
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("sac_humanoid")


if __name__ == "__main__":
    main()
