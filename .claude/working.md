# Challenge 3: RL in the Physical World — Working Context

## Current Status

**Status**: Complete. Presentation prep in progress.

**Project**: SAC+HER policies for FetchPickAndPlace-v4 (1M steps) and FetchReach-v4 (200k steps) in MuJoCo, with sim-to-real transfer analysis.

**Repo**: https://github.com/aipi590-ggn/aipi590-challenge-3
**Live viewer**: https://aipi590-ggn.github.io/aipi590-challenge-3/
**Team**: Lindsay Gross, Yifei Guo, Jonas Neves

## What the professor actually asked for

Source: Professor Bent, Week 9 + Week 12 transcripts.

1. Use a physical or simulated embodiment platform (PyBullet, MuJoCo, etc.)
2. Train an RL policy on an embodied task (grasping, navigation, balance)
3. Discuss challenges with sim-to-real transfer in your chosen domain
4. Present to class (Week 12, March 31)

She explicitly mentioned the four sim-to-real gaps from a 2025 paper covered in Week 9: action gap, reward gap, next-state gap, observation gap. Mitigation strategies discussed: domain randomization, exploratory policies, VLAs for task descriptions.

Physical robot is bonus ("would be really awesome") but not required. Any technique from class is fair game.

See `REQUIREMENTS_CHECKLIST.md` for tracked status against these requirements.

## Key Decisions

1. **Task**: FetchPickAndPlace-v4 (grasping with gripper control) + FetchReach-v4 (simpler reaching variant)
2. **Algorithm**: SAC + HER. HER is essential because FetchPickAndPlace has sparse reward (almost never positive early in training). HER relabels failures as successes toward the achieved goal.
3. **Simulation Budget**: 1M steps (main), 200k (reach). ~25 min on A100.

## Architecture

### Notebooks
- `notebooks/challenge3-pickandplace.ipynb` — main: 1M steps, full sim2real analysis
- `notebooks/challenge3-reach-experimentation.ipynb` — 200k steps, rapid experimentation

### Scripts
- `scripts/colab_utils.py` — Colab automation (clone, publish, LiveChartCallback, video conversion)
- `scripts/trajectory_extractor.py` — extract body positions from policy rollouts for web viewer
- `scripts/auto_check.py` — CI script that auto-marks REQUIREMENTS_CHECKLIST.md items

### CI
- `.github/workflows/auto-check.yml` — on push to main + manual dispatch: runs auto_check.py, writes job summary with checklist, commits if changed

### Web Viewer
- `docs/index.html` — Three.js interactive trajectory playback
- `docs/data/trajectories*.json` — per-task trajectory data
- GitHub Pages deployed from branch (docs/ directory)

## Sim-to-Real Analysis

Five gaps documented in README:
1. Contact & Gripper Modeling (finger compliance, micro-slip)
2. Actuator Fidelity (backlash, ~10ms ROS 2 latency)
3. Observation Noise (encoder resolution, camera pipeline latency)
4. Zero Calibration Drift (22mm error at 650mm reach from 2 degrees)
5. Domain Randomization Strategy (7 parameters proposed)

Maps to the four gaps from Week 9 lecture: action (actuator fidelity), reward (implicit in task success definition), next-state (contact modeling, calibration drift), observation (noise, camera latency).

## Known Issues & Workarounds

- **Browser fan noise during training**: Reduced LiveChartCallback from 500 to 2000 step intervals, max 300 data points
- **Colab visualization broken (eval_js)**: Switched to clear_output + display(HTML)
- **MuJoCo headless rendering**: Xvfb virtual display prepended to eval cells
- **3D viewer wrong joint data**: Rewrote trajectory extractor to save data.xpos for 12 Fetch bodies

## Open Items

- [ ] Presentation slides (PRES1-5 in checklist)
- [ ] PickAndPlace rollout videos/GIFs not in README (only FetchReach shown)

## Metadata

- **Created**: 2026-04-04
- **Last Updated**: 2026-04-05 (session 5: checklist + CI workflows)
- **Challenge Due**: 2026-03-31
