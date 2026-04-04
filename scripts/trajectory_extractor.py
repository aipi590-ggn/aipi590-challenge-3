"""Extract trajectory data from policy rollouts for visualization."""

import json
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from pathlib import Path
from typing import Optional


def extract_trajectory(
    model,
    env_id: str = 'FetchPickAndPlace-v4',
    n_episodes: int = 1,
    deterministic: bool = True,
) -> list[dict]:
    """
    Run policy and extract trajectory data (joint angles, gripper state, etc).

    Returns list of episodes, each with timestep data.
    """
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id)

    episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        trajectory = {
            'episode': ep,
            'task': env_id,
            'timesteps': [],
        }

        done = False
        step = 0

        while not done:
            # For MultiInputPolicy, predict expects the dict-based observation directly
            action, _ = model.predict(obs, deterministic=deterministic)

            # Extract state before step
            state = env.unwrapped.data.qpos.copy()  # Joint positions

            # Get goal positions from observation dict
            achieved = obs.get('achieved_goal', np.array([0, 0, 0]))
            desired = obs.get('desired_goal', np.array([0, 0, 0]))

            # Record timestep
            timestep = {
                'step': step,
                'joint_positions': state[:7].tolist(),  # 7-DOF arm
                'gripper_position': state[7:9].tolist() if len(state) > 7 else [0, 0],
                'object_position': achieved[:3].tolist() if hasattr(achieved, 'tolist') else achieved[:3],
                'goal_position': desired[:3].tolist() if hasattr(desired, 'tolist') else desired[:3],
            }
            trajectory['timesteps'].append(timestep)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        trajectory['success'] = info.get('is_success', False)
        trajectory['length'] = step
        episodes.append(trajectory)

    env.close()
    return episodes


def save_trajectories(
    episodes: list[dict],
    output_path: Path,
) -> None:
    """Save trajectory data as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(episodes, f, indent=2)
    print(f'Saved {len(episodes)} trajectories to {output_path}')


if __name__ == '__main__':
    # Example usage
    from stable_baselines3 import SAC

    model_path = Path('results/models/best_model')
    if model_path.exists():
        model = SAC.load(str(model_path))
        episodes = extract_trajectory(model, n_episodes=5)
        save_trajectories(episodes, Path('docs/data/trajectories.json'))
