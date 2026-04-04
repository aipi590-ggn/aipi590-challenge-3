# AIPI 590: Challenge 3 — RL in the Physical World

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3%2B-green.svg)](https://mujoco.org/)
[![Stable Baselines3](https://img.shields.io/badge/SB3-1.8%2B-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Training robotic grasping and reaching policies in MuJoCo simulation using SAC + Hindsight Experience Replay, with analysis of sim-to-real transfer gaps against physical robot hardware.

---

## 🎮 Interactive Visualization

**[View Interactive Policy Rollouts →](https://jonasneves.github.io/aipi590-challenge-3/)**

Explore trained policy episodes with playback controls, speed adjustment, and real-time statistics.

## Notebooks

| Task | Notebook | Open in Colab |
|------|----------|---------------|
| Manipulation (Main) | [challenge3-pickandplace.ipynb](notebooks/challenge3-pickandplace.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonasneves/aipi590-challenge-3/blob/main/notebooks/challenge3-pickandplace.ipynb) |
| Reaching (Experimentation) | [challenge3-reach-experimentation.ipynb](notebooks/challenge3-reach-experimentation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonasneves/aipi590-challenge-3/blob/main/notebooks/challenge3-reach-experimentation.ipynb) |

- **pickandplace**: 1M timesteps, FetchPickAndPlace-v4 (grasping & manipulation)
- **reach**: 200k timesteps, FetchReach-v4 (reaching task)

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

## Structure

```
aipi590-challenge-3/
├── notebooks/
│   ├── challenge3-pickandplace.ipynb     # Main: 1M steps, FetchPickAndPlace-v4
│   └── challenge3-reach-experimentation.ipynb  # Experimentation: 200k steps, FetchReach-v4
├── scripts/
│   ├── colab_utils.py                    # Colab automation (publish, live charts)
│   └── trajectory_extractor.py           # Extract trajectories for visualization
├── docs/
│   ├── index.html                        # Interactive Three.js viewer
│   └── data/trajectories.json            # Trajectory data
├── results/
│   ├── models/                           # Trained policy checkpoints
│   ├── videos/                           # Rollout episodes (mp4)
│   └── plots/                            # Training curves (matplotlib)
├── requirements.txt
└── README.md
```

---

## Setup

Each notebook is self-contained and installs dependencies automatically. To run locally:

```bash
pip install -r requirements.txt
```

Then run a notebook in [Google Colab](https://colab.research.google.com) or Kaggle.

---



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

---

## Team

Lindsay Gross · Yifei Guo · Jonas Neves

Duke University · AIPI 590 · Spring 2026
