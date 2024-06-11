import sys
import argparse
import numpy as np
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import SAC, PPO, TD3, A2C, DDPG

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from utils import get_humanoid_env


def parse_args() -> tuple[str, int, bool]:
    """
    Parse the command line arguments.

    return:
        model_name: str
        n_episodes: int
        render: bool
    """
    parser = argparse.ArgumentParser(
        "eval", description="Evaluate the Humanoid environment."
    )
    parser.add_argument(
        "-m",
        "--model",
        "--model-name",
        dest="model_name",
        type=str,
        required=True,
        help="The name of the model to evaluate. E.g. SAC_nowrapped.",
    )
    parser.add_argument(
        "--e",
        "--episodes",
        "--n-episodes",
        dest="n_episodes",
        type=int,
        default=10,
        help="The total number of episode to evaluate. [Default: 10]",
    )
    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        required=False,
        help="Render the environment. [Default: False]",
    )
    args = parser.parse_args()

    return args.model_name, args.n_episodes, args.render


def eval(
    env: gym.Env,
    model: SAC | PPO | TD3 | A2C | DDPG,
    n_episodes: int = 10,
    verbose: bool = True,
) -> None:
    ep = 0
    rewards = []

    try:
        while ep < n_episodes:
            ep_reward = 0
            obs, _ = env.reset()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)

                ep_reward += reward

                if terminated or truncated:
                    break

            if verbose:
                print(f"Episode {ep + 1} reward: {ep_reward}")

            ep += 1
            rewards.append(ep_reward)

    except KeyboardInterrupt:
        print(f"Evaluation interrupted at episode {ep + 1}")

    print(f"Mean episode reward: {np.mean(rewards)}")


def main() -> None:
    model_name, n_episodes, render = parse_args()
    parts = model_name.split("_")
    algo_name = parts[0]
    use_wrapper = "wrapped" in model_name.split("_")

    print(
        f"Evaluation: {algo_name.upper()} {'(use wrapper)' if use_wrapper else ''}"
    )

    env = get_humanoid_env(
        no_wrapper=not use_wrapper, render_mode="human" if render else None
    )

    chkpt = str(SCRIPT_DIR / "final_models" / model_name)

    if algo_name == "SAC":
        model = SAC.load(chkpt)
    elif algo_name == "TD3":
        model = TD3.load(chkpt)
    elif algo_name == "PPO":
        model = PPO.load(chkpt)
    elif algo_name == "A2C":
        model = A2C.load(chkpt)
    elif algo_name == "DDPG":
        model = DDPG.load(chkpt)
    else:
        raise ValueError(f"Invalid algorithm: {algo_name}")

    eval(env, model, n_episodes)


if __name__ == "__main__":
    main()
