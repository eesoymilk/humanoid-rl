import gymnasium as gym
from agent import Agent


def main() -> None:
    env = gym.make("Humanoid-v5", render_mode="human")
    agent = Agent(env)
    mean_reward = agent.evaluate()
    print(f"Mean reward: {mean_reward}")
    agent.close()


if __name__ == "__main__":
    main()
