import gymnasium as gym
import argparse
import os
import time
from agent2 import Agent, handle_state, ACTION_MAP
import json
import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import shutil
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MiniGrid environment with an agent"
    )

    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-Empty-8x8-v0",
        help="MiniGrid environment name",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--timeout", type=int, default=360, help="Maximum time per episode (in seconds)"
    )
    parser.add_argument(
        "--render", action="store_true", help="Whether to render the environment"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between steps (for visualization)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use - either OpenAI or local model (e.g. 'Qwen/Qwen2.5-7B-Instruct'), depending on the API URL",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information during execution",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save trajectory data and images",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="default",
        help="Identifier for grouping multiple env runs",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="URL for the local model API, e.g. http://localhost:8081/v1. Leave empty to use OpenAI API",
    )
    parser.add_argument(
        "--use-absolute-location",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="If True, calculate robot absolute position from the moment he starts at 0,0",
    )
    parser.add_argument(
        "--use-object-memory",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="If True, store objects with their absolute position as the robot explores",
    )
    parser.add_argument(
        "--use-cot",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="If True, use Chain of Thought reasoning every 6 steps",
    )
    parser.add_argument(
        "--use-math-cot",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="If True, use Chain of Thought reasoning with calculations about distance and orientation from the current robot position to the object of interest every 6 steps",
    )

    return parser.parse_args()


def save_step_data(
    base_path: Path,
    mission: str,
    step: int,
    actions: List[int],
    won: bool,
    elapsed: float,
    rgb_array: np.ndarray,
    metadata: dict,
):
    """Save step data and image"""
    # Save image
    Image.fromarray(rgb_array).save(base_path / f"{step}.png")

    # Save step info
    episode_data = {
        "mission": mission,
        "steps": step,
        "actions": [ACTION_MAP[action] for action in actions],
        "won": won,
        "elapsed": elapsed,
    }
    with open(base_path / f"state.json", "w") as f:
        json.dump(episode_data, f, indent=2)

    metadata_path = base_path / f"metadata/{step}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    # Parse command line arguments
    args = parse_args()

    print(f"--- Running {args.episodes} episodes in {args.env}...")

    if args.render and args.save:
        raise ValueError("Cannot save frames and render at the same time!")

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        if args.api_url is None:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")
        else:
            api_key = "placeholder"

    # Initialize environment
    render_mode = "human" if args.render else ("rgb_array" if args.save else None)
    env = gym.make(args.env, render_mode=render_mode)

    # Track metrics
    wins = 0

    # Setup results directory if saving trajectories
    if args.save:
        results_dir = Path("results") / args.run_id
        results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize results tracking
        results_file = results_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results_data = json.load(f)
        else:
            results_data = {"envs": {}}

    # Run episodes
    for episode in range(args.episodes):
        # Initialize agent
        agent = Agent(api_key=api_key, model=args.model, api_url=args.api_url,
                      use_absolute_location=args.use_absolute_location, 
                      use_object_memory=args.use_object_memory,
                      use_cot=args.use_cot, use_math_cot=args.use_math_cot)

        if args.verbose:
            print(f"\nStarting Episode {episode + 1}")

        obs, _ = env.reset(seed=episode + args.seed)
        mission = getattr(env.unwrapped, "mission")
        steps = 0
        start_time = time.time()

        if args.save:
            env_dir = results_dir / args.env
            if steps == 0 and episode == 0 and env_dir.exists():
                shutil.rmtree(env_dir)
            env_dir.mkdir(parents=True, exist_ok=True)
            episode_dir = env_dir / str(episode)
            episode_dir.mkdir(exist_ok=True)

        # Run episode
        won = False
        actions = []
        for step_i in range(args.max_steps):
            # Get action from agent
            action, metadata = handle_state(obs, mission, agent, args.verbose)
            actions.append(action)

            # Take action in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if reward > 0 and terminated:
                won = True
            steps += 1
            elapsed = time.time() - start_time
            if args.verbose:
                print(f"step {step_i + 1}, elapsed: {elapsed:.2f}s")

            if args.save:
                # Get RGB render for saving
                rgb_array = env.render()
                save_step_data(
                    episode_dir,
                    mission,
                    steps,
                    actions,
                    won,
                    elapsed,
                    rgb_array,
                    metadata,
                )

            obs = next_obs

            # Add delay if rendering
            if args.render and args.delay > 0:
                time.sleep(args.delay)

            if terminated or truncated:
                break
            if elapsed > args.timeout:
                print("timeout!")
                break

        if won:
            wins += 1

        print(f"Episode {episode + 1} finished:")
        print(f"- Steps: {steps}")
        print(f"- Wins: {wins}/{episode + 1}")

    # Save final results
    if args.save:
        results_data["envs"][args.env] = {
            "wins": wins,
            "episodes": args.episodes,
            "accuracy": wins / args.episodes,
        }
        results_data["last_updated"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        total_wins = sum(env_data["wins"] for env_data in results_data["envs"].values())
        total_eposides = sum(
            env_data["episodes"] for env_data in results_data["envs"].values()
        )
        results_data["total_wins"] = total_wins
        results_data["total_episodes"] = total_eposides
        results_data["total_accuracy"] = total_wins / total_eposides

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
