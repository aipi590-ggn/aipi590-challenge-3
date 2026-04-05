"""Extract trajectory data and record videos from the same rollouts."""

import json
import numpy as np
import mujoco
import gymnasium as gym
import gymnasium_robotics
from pathlib import Path


FETCH_MESH_GEOMS = [
    'robot0:base_link',
    'robot0:torso_lift_link',
    'robot0:shoulder_pan_link',
    'robot0:shoulder_lift_link',
    'robot0:upperarm_roll_link',
    'robot0:elbow_flex_link',
    'robot0:forearm_roll_link',
    'robot0:wrist_flex_link',
    'robot0:wrist_roll_link',
    'robot0:gripper_link',
]

FETCH_FINGER_GEOMS = [
    'robot0:r_gripper_finger_link',
    'robot0:l_gripper_finger_link',
]

# Minimum initial object-to-goal distance (meters) for an episode to
# be considered visually interesting.
MIN_INTERESTING_DIST = 0.10

# Maximum number of rollout attempts when filtering for interesting
# episodes, to avoid infinite loops on poorly trained policies.
MAX_FILTER_ATTEMPTS = 50


def _get_geom_transforms(model, data):
    """Get world-frame position and quaternion for each visualization geom."""
    transforms = {}
    for geom_name in FETCH_MESH_GEOMS + FETCH_FINGER_GEOMS:
        try:
            gid = model.geom(geom_name).id
            pos = data.geom_xpos[gid].tolist()
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, data.geom_xmat[gid])
            transforms[geom_name] = {
                'pos': pos,
                'quat': quat.tolist(),
            }
        except Exception:
            pass
    return transforms


def _get_table_info(model, data):
    """Get table position from the MuJoCo model."""
    for name in ['table0', 'table']:
        try:
            bid = model.body(name).id
            return data.xpos[bid].tolist()
        except Exception:
            continue
    return None


def _is_interesting(trajectory):
    """Check whether an episode shows visually meaningful movement."""
    ts0 = trajectory['timesteps'][0]
    ts_end = trajectory['timesteps'][-1]

    obj_start = np.array(ts0['object_position'])
    goal = np.array(ts0['goal_position'])
    obj_end = np.array(ts_end['object_position'])

    start_dist = np.linalg.norm(obj_start - goal)
    end_dist = np.linalg.norm(obj_end - goal)

    if start_dist < MIN_INTERESTING_DIST:
        return False

    # Object should move at least 30% closer to goal
    if end_dist > start_dist * 0.7:
        return False

    return True


def _run_episode(env, model, mj_model, mj_data, ep_index, deterministic):
    """Run a single episode and return the trajectory dict."""
    obs, _ = env.reset()

    trajectory = {
        'episode': ep_index,
        'task': env.spec.id if hasattr(env, 'spec') and env.spec else '',
        'table_position': _get_table_info(mj_model, mj_data),
        'timesteps': [],
    }

    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)

        achieved = obs.get('achieved_goal', np.array([0, 0, 0]))
        desired = obs.get('desired_goal', np.array([0, 0, 0]))

        timestep = {
            'step': step,
            'geoms': _get_geom_transforms(mj_model, mj_data),
            'object_position': achieved[:3].tolist() if hasattr(achieved, 'tolist') else list(achieved[:3]),
            'goal_position': desired[:3].tolist() if hasattr(desired, 'tolist') else list(desired[:3]),
        }
        trajectory['timesteps'].append(timestep)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    trajectory['success'] = bool(info.get('is_success', False))
    trajectory['length'] = step
    return trajectory


def extract_trajectory(
    model,
    env_id: str = 'FetchPickAndPlace-v4',
    n_episodes: int = 1,
    deterministic: bool = True,
    video_dir: str | Path | None = None,
    video_prefix: str | None = None,
    filter_interesting: bool = True,
) -> list[dict]:
    """Run policy rollouts, extract trajectory data and optionally record videos.

    When video_dir is provided, videos are recorded from the same environment
    instance and episodes as the trajectory data, so they match exactly.

    When filter_interesting is True, trivial episodes (object already near goal)
    are discarded and re-rolled until n_episodes interesting ones are collected.
    Videos for discarded episodes are cleaned up automatically.
    """
    gym.register_envs(gymnasium_robotics)

    render_mode = 'rgb_array' if video_dir else None
    env = gym.make(env_id, render_mode=render_mode)

    if video_dir:
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        if video_prefix is None:
            video_prefix = env_id.split('-')[0].lower()
        env = gym.wrappers.RecordVideo(
            env,
            str(video_dir),
            episode_trigger=lambda ep: True,
            name_prefix=video_prefix,
        )

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    episodes = []
    successes = 0
    raw_ep = 0

    while len(episodes) < n_episodes and raw_ep < MAX_FILTER_ATTEMPTS:
        traj = _run_episode(env, model, mj_model, mj_data, len(episodes), deterministic)
        raw_ep += 1

        if filter_interesting and not _is_interesting(traj):
            # Remove the video file for this discarded episode
            if video_dir:
                for f in sorted(video_dir.glob(f'{video_prefix}-episode-{raw_ep - 1}*')):
                    f.unlink(missing_ok=True)
            continue

        traj['episode'] = len(episodes)
        episodes.append(traj)
        if traj['success']:
            successes += 1

    env.close()

    # Rename video files to match final episode indices
    if video_dir:
        _renumber_videos(video_dir, video_prefix, n_episodes)

    attempted = raw_ep
    kept = len(episodes)
    print(f'Kept {kept}/{attempted} episodes (filtered {attempted - kept} trivial)')
    print(f'Success rate: {successes}/{kept}')
    return episodes


def _renumber_videos(video_dir, prefix, n_expected):
    """Rename video files to sequential episode-0..N after filtering."""
    video_dir = Path(video_dir)
    existing = sorted(video_dir.glob(f'{prefix}-episode-*'))
    for new_idx, path in enumerate(existing):
        if new_idx >= n_expected:
            path.unlink(missing_ok=True)
            continue
        ext = path.suffix
        new_name = f'{prefix}-episode-{new_idx}{ext}'
        new_path = path.parent / new_name
        if path != new_path:
            path.rename(new_path)


def save_trajectories(episodes, output_path):
    """Save trajectory data as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)): return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(episodes), f)
    print(f'Saved {len(episodes)} trajectories to {output_path}')


def generate_versioned_filename(env_id, n_episodes):
    task_name = env_id.split('-')[0].lower()
    return f'trajectories-{task_name}-{n_episodes}ep.json'
