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
# count as non-trivial for visualization selection.
MIN_NONTRIVIAL_DIST = 0.10


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


def _initial_distance(trajectory):
    """Object-to-goal distance at episode start."""
    ts0 = trajectory['timesteps'][0]
    obj = np.array(ts0['object_position'])
    goal = np.array(ts0['goal_position'])
    return float(np.linalg.norm(obj - goal))


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
    n_episodes: int = 5,
    n_viz: int = 5,
    deterministic: bool = True,
    video_dir: str | Path | None = None,
    video_prefix: str | None = None,
) -> list[dict]:
    """Run n_episodes rollouts, return the n_viz most representative ones.

    All episodes are evaluated and the full success rate is reported.
    For visualization, episodes are ranked by initial object-to-goal
    distance (descending) to surface the most informative behavior.
    Trivial episodes where the object starts within MIN_NONTRIVIAL_DIST
    of the goal are ranked last.

    When video_dir is provided, videos are recorded from the same
    environment instance as the trajectory data, so they match exactly.
    Only videos for selected episodes are kept; the rest are removed.
    """
    n_run = max(n_episodes, n_viz * 3)
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

    all_episodes = []
    successes = 0

    for ep in range(n_run):
        traj = _run_episode(env, model, mj_model, mj_data, ep, deterministic)
        all_episodes.append(traj)
        if traj['success']:
            successes += 1

    env.close()
    print(f'Eval: {successes}/{n_run} success ({100 * successes / n_run:.0f}%)')

    # Rank by initial distance (largest first) for visualization selection.
    ranked = sorted(all_episodes, key=_initial_distance, reverse=True)
    selected = ranked[:n_viz]

    # Reassign episode indices
    selected_raw_indices = set()
    for i, traj in enumerate(selected):
        selected_raw_indices.add(traj['episode'])
        traj['episode'] = i

    # Clean up video files: keep selected, remove the rest, renumber, make GIFs
    if video_dir:
        _keep_videos(video_dir, video_prefix, selected_raw_indices, n_run)
        _convert_gifs(video_dir, video_prefix)

    kept_success = sum(1 for t in selected if t['success'])
    print(f'Selected {n_viz}/{n_run} episodes for visualization ({kept_success} successful)')
    return selected


def _keep_videos(video_dir, prefix, keep_indices, total):
    """Keep only videos for selected episodes, renumber sequentially."""
    video_dir = Path(video_dir)

    # Collect paths for kept episodes in selection order
    kept = []
    for raw_idx in range(total):
        matches = sorted(video_dir.glob(f'{prefix}-episode-{raw_idx}.*'))
        if raw_idx in keep_indices:
            kept.extend(matches)
        else:
            for f in matches:
                f.unlink(missing_ok=True)

    # Renumber kept files: episode-0, episode-1, ...
    for new_idx, path in enumerate(kept):
        ext = path.suffix
        new_name = f'{prefix}-episode-{new_idx}{ext}'
        new_path = path.parent / new_name
        if path != new_path:
            # Rename via temp to avoid collisions
            tmp = path.parent / f'_tmp_{new_idx}{ext}'
            path.rename(tmp)
            tmp.rename(new_path)


def _convert_gifs(video_dir, prefix, scale=320, fps=5):
    """Convert kept MP4s to GIFs and remove stale GIFs."""
    import subprocess as sp

    video_dir = Path(video_dir)

    # Remove any pre-existing GIFs (may be stale from previous runs)
    for old_gif in video_dir.glob(f'{prefix}-episode-*.gif'):
        old_gif.unlink(missing_ok=True)

    for mp4 in sorted(video_dir.glob(f'{prefix}-episode-*.mp4')):
        gif = mp4.with_suffix('.gif')
        try:
            sp.run([
                'ffmpeg', '-i', str(mp4),
                '-vf', f'fps={fps},scale={scale}:-1:flags=lanczos',
                '-y', str(gif),
            ], check=True, capture_output=True)
            size_kb = gif.stat().st_size // 1024
            print(f'  {gif.name} ({size_kb}KB)')
        except FileNotFoundError:
            print('Warning: ffmpeg not found, skipping GIF conversion')
            return
        except sp.CalledProcessError as e:
            print(f'Warning: failed to convert {mp4.name}: {e.stderr.decode()[:200]}')


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
