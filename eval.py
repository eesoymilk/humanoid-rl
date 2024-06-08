import sys
import argparse
from pathlib import Path

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
        type=bool,
        required=False,
        help="Render the environment. [Default: False]",
    )
    args = parser.parse_args()

    return (args.model_name, args.render)


def main() -> None:
    model_name, render = parse_args()
    parts = model_name.split("_")
    algo_name = parts[0]
    use_wrapper = "wrapped" in model_name.split("_")

    env = get_humanoid_env(no_wrapper=not use_wrapper, render_mode=render)
    model = load_model(
        env,
        algo_name,
        learning=False,
        chkpt=str(SCRIPT_DIR / "models" / model_name),
    )
    eval(env, model)


if __name__ == "__main__":
    main()
