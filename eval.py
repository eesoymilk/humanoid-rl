import sys
import argparse
from pathlib import Path
from stable_baselines3 import SAC, PPO, TD3

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from utils import get_humanoid_env, load_model, eval


def parse_args() -> tuple[str, bool]:
    """
    Parse the command line arguments.

    return:
        model_name: str
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
        help="The name of the model to evaluate.",
    )
    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        required=False,
        help="Render the environment. [Default: False]",
    )
    args = parser.parse_args()

    return args.model_name, args.render


def main() -> None:
    model_name, render = parse_args()
    parts = model_name.split("_")
    algo_name = parts[0]
    use_wrapper = "wrapped" in model_name.split("_")

    print(
        f"Evaluation: {algo_name.upper()} {'(use wrapper)' if use_wrapper else ''}"
    )

    env = get_humanoid_env(
        no_wrapper=not use_wrapper, render_mode="human" if render else None
    )

    chkpt = str(SCRIPT_DIR / "models" / model_name)

    if algo_name == "sac":
        model = SAC.load(chkpt)
    elif algo_name == "td3":
        model = TD3.load(chkpt)
    elif algo_name == "ppo":
        model = PPO.load(chkpt)
    else:
        raise ValueError(f"Invalid algorithm: {algo_name}")

    eval(env, model)


if __name__ == "__main__":
    main()
