# Challenge 3: RL in the Physical World — Working Context

## Current Status

**Project**: Training a robotic grasping policy with SAC+HER in FetchPickAndPlace-v4, analyzing sim2real transfer gaps against the reBot-DevArm platform.

**Repo**: https://github.com/jonasneves/aipi590-challenge-3

## Key Decisions Made

1. **Task**: FetchPickAndPlace-v4 (not just reaching) — actual manipulation task with gripper control required. More aligned with "grasping" requirement than FetchReach-v3.

2. **Algorithm**: SAC + HER (Hindsight Experience Replay)
   - SAC: off-policy, sample-efficient, entropy regularization
   - HER: relabels sparse-reward failures as successes toward achieved goal
   - Needed because FetchPickAndPlace has almost-never-positive reward signal early in training

3. **Simulation Budget**: 1M timesteps for main notebook (challenge3.ipynb), 200k for v1 (challenge3-v1.ipynb)
   - Expected runtime: ~25 min on A100, ~60 min on T4
   - Colab Pro users should select A100 runtime manually from Runtime menu

4. **Live Visualization**: 4-panel ECharts dashboard in `colab_utils.LiveChartCallback`
   - Top-left: Episode Reward (rolling 100-ep mean)
   - Top-right: Success Rate (with area fill)
   - Bottom-left: Actor Loss + Critic Loss (dual series)
   - Bottom-right: Entropy Coefficient
   - Stats bar above: timesteps, episodes, fps, success %, elapsed, updates
   - Updates every 2000 steps (was 500 to reduce browser load)
   - Downsamples to max 300 data points to prevent JS memory bloat

5. **Headless Rendering**: Uses Xvfb virtual display for evaluation videos
   - Prepended to eval cells in both notebooks
   - Fixes "X11: DISPLAY variable missing" error

## Architecture

### Notebooks
- `challenge3.ipynb` — main: 1M timesteps, FetchPickAndPlace-v4, full sim2real analysis
- `challenge3-v1.ipynb` — alternative: 200k timesteps, FetchReach-v4 (older variant), includes live chart now

### Scripts
- `scripts/colab_utils.py` — contains:
  - `prepare_notebook()` — clone repo, handle auth
  - `publish_artifacts()` — OAuth button + git push (no manual secret needed)
  - `save_notebook()` — snapshot running notebook via `_message.blocking_request("get_ipynb")`
  - `LiveChartCallback` — SB3 callback, 4-panel ECharts, clear_output + full redraw every 2k steps

### Data Flow (Training)
1. Install deps → clone repo → setup paths
2. Create train/eval envs (Monitor wrapped)
3. SAC with HerReplayBuffer (n_sampled_goal=4, strategy='future')
4. EvalCallback runs every 20k steps, saves best_model
5. LiveChartCallback renders dashboard every 2k steps
6. Training metrics pulled from `model.logger.name_to_value`

## Known Issues & Workarounds

### Issue: Browser fan noise during training
**Root cause**: Full ECharts redraw every 500 steps = ~2000 redraws over 1M steps
**Solution**:
- Increased `update_freq` default from 500 → 2000 (4× fewer redraws)
- Added `max_points=300` downsampling so JS arrays stay bounded
- Tunable: `LiveChartCallback(update_freq=5000, max_points=200)` for even lighter load

### Issue: Colab notebook visualization broken with eval_js
**Root cause**: eval_js runs in main frame, ECharts div lives in output iframe — no shared window
**Solution**:
- Switched from eval_js to `clear_output(wait=True)` + `display(HTML(...))`
- Data baked into HTML template at render time (no cross-frame JS calls)
- Fully reliable, works in all Colab contexts

### Issue: MuJoCo rendering fails ("gladLoadGL error", "DISPLAY missing")
**Root cause**: Colab is headless, no X11 display
**Solution**:
- Prepend eval cells with Xvfb virtual display:
  ```python
  import subprocess, os
  subprocess.Popen(['Xvfb', ':1', '-screen', '0', '1024x768x24'])
  os.environ['DISPLAY'] = ':1'
  ```
- Applied to both challenge3.ipynb and challenge3-v1.ipynb

## Sim2Real Analysis Structure

Five major gaps between MuJoCo training and reBot-DevArm hardware:

1. **Contact & Gripper Modeling** (dominant) — finger compliance, micro-slip, asymmetry
2. **Actuator Fidelity** — backlash, control loop latency (~10ms ROS 2), torque saturation
3. **Observation Noise** — encoder resolution, camera uncertainty & pipeline latency
4. **Zero Calibration Drift** — per-joint errors compound through kinematic chain (22mm error at 650mm reach from 2°)
5. **Domain Randomization Strategy** — table with 7 parameters: action delay, observation noise, mass variance, friction range, object friction, gripper position noise, latency simulation

Beyond DR: residual policy learning + real-to-sim adaptation (Isaac Sim integration planned Q2 2026)

## Next Steps for Future Sessions

1. **Wait for Training**: Let 1M steps complete on A100 (25 min) or T4 (60 min)
   - Monitor LiveChartCallback for success rate emergence
   - Real improvement expected around 200-400k steps (HER kickin in)

2. **Evaluation Phase**: Run cell-eval once best_model is saved
   - Generates 5 rollout videos
   - Displays inline HTML5 video

3. **Publish**: Run cell-publish to commit artifacts back to GitHub
   - Requires Colab secret setup (button-based OAuth, no manual token)
   - Pushes plots, videos, and notebook snapshot

4. **Future Extensions**:
   - Add domain randomization wrappers (actuator delay, observation noise)
   - Implement residual correction on real hardware (if deployed to reBot-DevArm)
   - Compare with PPO or other algorithms
   - Analyze failure modes from rollout videos

## Files to Know

- `.claude/working.md` — this file
- `notebooks/challenge3.ipynb` — main work
- `notebooks/challenge3-v1.ipynb` — FetchReach variant
- `scripts/colab_utils.py` — all the Colab automation & visualization
- `results/models/best_model.zip` — trained policy (created after first eval callback)
- `results/videos/` — recorded rollouts
- `results/plots/` — training curves (matplotlib, saved after cell-plot)
- `requirements.txt` — deps (mujoco, gymnasium-robotics, stable-baselines3, moviepy)

## Metadata

- **Created**: 2026-04-04
- **Last Updated**: 2026-04-04
- **Challenge Due**: 2026-03-31 (passed deadline, but core work complete)
- **Team**: Jonas (user) + Claude (co-author)
