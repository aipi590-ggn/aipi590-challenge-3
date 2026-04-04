# AIPI 590: Challenge 3 — RL in the Physical World

Training a robotic grasping/reaching policy in MuJoCo simulation using SAC + Hindsight Experience Replay, with analysis of sim-to-real transfer gaps.

## Notebooks

- **[challenge3.ipynb](notebooks/challenge3.ipynb)** — Main: 1M timesteps, FetchPickAndPlace-v4 (manipulation task)
- **[challenge3-v1.ipynb](notebooks/challenge3-v1.ipynb)** — Alternative: 200k timesteps, FetchReach-v4 (reaching task)

## Trained Policies

Downloadable from [GitHub Releases](../../releases):

- `v1-challenge3-1m` — SAC+HER policy (1M steps, FetchPickAndPlace-v4)
- `v1-challenge3-200k` — SAC+HER policy (200k steps, FetchReach-v4)

## Rollout Videos

### Challenge 3 (FetchReach-v4, 200k steps)

- [Episode 0](https://github.com/jonasneves/aipi590-challenge-3/blob/main/results/videos/fetchreach-episode-0.mp4)
- [Episode 1](https://github.com/jonasneves/aipi590-challenge-3/blob/main/results/videos/fetchreach-episode-1.mp4)
- [Episode 2](https://github.com/jonasneves/aipi590-challenge-3/blob/main/results/videos/fetchreach-episode-2.mp4)
- [Episode 3](https://github.com/jonasneves/aipi590-challenge-3/blob/main/results/videos/fetchreach-episode-3.mp4)
- [Episode 4](https://github.com/jonasneves/aipi590-challenge-3/blob/main/results/videos/fetchreach-episode-4.mp4)

## Key Decisions

- **Algorithm**: SAC + HER (Soft Actor-Critic + Hindsight Experience Replay)
  - Off-policy, sample-efficient, handles sparse rewards
  - HER relabels failures as successes toward achieved goal

- **Simulation Budget**: 1M timesteps (main), 200k (v1)
  - ~25 min on A100, ~60 min on T4

- **Live Visualization**: 4-panel training dashboard (ECharts)
  - Episode reward, success rate, actor/critic loss, entropy coefficient
  - Updates every 2k steps

## Known Gaps (Sim-to-Real)

1. **Contact & Gripper Modeling** — finger compliance, micro-slip
2. **Actuator Fidelity** — backlash, control loop latency (~10ms ROS 2)
3. **Observation Noise** — encoder resolution, camera pipeline latency
4. **Zero Calibration Drift** — per-joint errors compound through kinematic chain
5. **Domain Randomization** — table friction, object properties, action delay

## Setup

```bash
pip install mujoco gymnasium-robotics stable-baselines3 moviepy
```

Then run a notebook in [Google Colab](https://colab.research.google.com) or Kaggle.


## Rollout Videos (GIF)

<table>
  <tr>
    <td><strong>fetchreach-episode-0</strong><br><img src="results/videos/fetchreach-episode-0.gif" width="100%"></td>
    <td><strong>fetchreach-episode-1</strong><br><img src="results/videos/fetchreach-episode-1.gif" width="100%"></td>
  </tr>
  <tr>
    <td><strong>fetchreach-episode-2</strong><br><img src="results/videos/fetchreach-episode-2.gif" width="100%"></td>
    <td><strong>fetchreach-episode-3</strong><br><img src="results/videos/fetchreach-episode-3.gif" width="100%"></td>
  </tr>
  <tr>
    <td><strong>fetchreach-episode-4</strong><br><img src="results/videos/fetchreach-episode-4.gif" width="100%"></td>
    <td></td>
  </tr>
</table>

![fetchreach-episode-3](results/videos/fetchreach-episode-3.gif)

### fetchreach-episode-4

![fetchreach-episode-4](results/videos/fetchreach-episode-4.gif)

