# Challenge 3: RL in the Physical World — Working Context

## Current Status

**Status**: Complete (training finished, videos generated, README with GIFs published)

**Project**: Trained robotic grasping/reaching policies with SAC+HER in MuJoCo (FetchPickAndPlace-v4 and FetchReach-v4), with analysis of sim2real transfer gaps against the reBot-DevArm platform.

**Repo**: https://github.com/aipi590-ggn/aipi590-challenge-3

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

## Architecture (Updated Session 2)

### Interactive Web Viewer (NEW)
- `docs/index.html` — Three.js + playback controls for trajectory visualization
- `docs/data/trajectories.json` — Trajectory data (joint angles, object positions over time)
- Auto-deployed to GitHub Pages on every push to main
- Live at: https://aipi590-ggn.github.io/aipi590-challenge-3/

### Trajectory Extraction
- `scripts/trajectory_extractor.py` — Extracts joint angles + object states from policy rollouts
- Can be run in Colab notebooks to save trajectory JSON
- Enables interactive playback without re-running policy

### Notebooks
- `challenge3-pickandplace.ipynb` — main: 1M timesteps, FetchPickAndPlace-v4, full sim2real analysis
- `challenge3-reach-experimentation.ipynb` — experimentation: 200k timesteps, FetchReach-v4, includes live chart

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

## Completion Status (2026-04-04, Session 4)

✅ **Challenge 3 Requirements Met** (Course Transcript verified)
   1. Physical/simulated embodiment environment ✅ MuJoCo
   2. Embodied task (grasping/reaching/navigation) ✅ FetchPickAndPlace + FetchReach
   3. RL policy training ✅ SAC+HER both tasks
   4. Sim-to-real transfer analysis ✅ 5 documented gaps in README

✅ **Training**: Both models trained successfully
   - FetchReach-v4 (200k steps) — videos + GIFs generated + trajectory JSON extracted
   - FetchPickAndPlace-v4 (1M steps) — started in Colab, output frames/gifs available

✅ **Videos & GIFs**: Converted to GIFs for inline README playback
   - 5 episode rollouts per model
   - Grid layout support added (2-column default)
   - Plays in GitHub README without external embeds

✅ **Interactive Visualization**: Three.js web viewer + GitHub Pages deployment
   - Live at: https://aipi590-ggn.github.io/aipi590-challenge-3/
   - Trajectory extraction cells added to both notebooks
   - Playback controls, speed adjustment, real-time stats

✅ **README**: Published with rollout videos, grid layout, and visualization link
   - Team attribution added: Lindsay Gross, Yifei Guo, Jonas Neves
   - Links to GitHub Releases for downloadable models
   - Interactive section at top with CTA

✅ **Notebook Setup**: Refined for reliability
   - Setup cell now clones repo first, then imports colab_utils
   - Download colab_utils from repo (not local path)
   - publish_artifacts moved to final cell only
   - Trajectory extraction at end of reach notebook

✅ **Publish Pipeline**: Working artifacts pipeline via `colab_utils.py`
   - OAuth-based Git push (no manual token paste)
   - Auto-collects results/ directory
   - Handles notebook JSON normalization

## Session 4: Robot Visualization Rewrite

**Problem**: The 3D viewer rendered primitive shapes (sphere, cylinder, box) instead of the Fetch robot. Two root causes:
1. The trajectory extractor saved wrong qpos indices as "joint_positions" (got base slides + head instead of arm joints)
2. The viewer never loaded any robot model — it just plotted trajectory coordinates on hand-built geometry

**Solution**: Body-position-based rendering
- `trajectory_extractor.py` now saves `data.xpos` for 12 key Fetch bodies per timestep (MuJoCo computes FK)
- `docs/index.html` rewritten to render connected capsules between body positions (arm links, joint spheres, gripper fingers)
- Coordinate mapping: MuJoCo Z-up → Three.js Y-up via `(x, z, y)` swap
- Graceful fallback: old data without `body_positions` shows "End Effector Only" badge

**To activate full robot rendering**: Regenerate trajectories by running the extraction cell in either notebook. The extractor API is unchanged — same notebook cells, new output format.

**Bodies saved**: base_link, torso_lift, shoulder_pan/lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex/roll, gripper_link, r/l_gripper_finger_link

## Next Steps / Future Extensions

1. **Domain Randomization**: Add actuator delay, observation noise, friction variance wrappers
   - Would improve sim2real robustness
   - Parameters documented in working.md (section "Sim2Real Analysis Structure")

2. **Real Hardware Testing**: Deploy best_model.zip to reBot-DevArm platform
   - Measure actual vs simulated success rates
   - Identify dominant failure modes

3. **Algorithm Comparison**: Evaluate PPO, TD3 on same tasks
   - Cross-validate sample efficiency vs SAC+HER
   - Generate comparative training curves

4. **Residual Learning**: Train correction policy on hardware failures
   - Leaves SAC policy frozen, adds small residual network
   - Faster adaptation than retraining from scratch

## Files to Know

- `.claude/working.md` — this file
- `notebooks/challenge3.ipynb` — main work
- `notebooks/challenge3-v1.ipynb` — FetchReach variant
- `scripts/colab_utils.py` — all the Colab automation & visualization
- `results/models/best_model.zip` — trained policy (created after first eval callback)
- `results/videos/` — recorded rollouts
- `results/plots/` — training curves (matplotlib, saved after cell-plot)
- `requirements.txt` — deps (mujoco, gymnasium-robotics, stable-baselines3, moviepy)

## Best Practices Adopted from Challenge 2

Aligned repository structure and documentation with Challenge 2 patterns:

1. **README Structure**
   - Dependency badges at top (Python, MuJoCo, SB3, License)
   - Interactive visualization link prominently featured
   - Notebooks table with Colab open badges
   - Structure section showing directory layout
   - Team section with standard format
   - Dividing lines (---) for visual hierarchy

2. **Team Attribution**
   - Consistent format: Name · Name · Name
   - Affiliation and term: Duke University · AIPI 590 · Spring 2026

3. **Licensing**
   - MIT LICENSE file created (matches badge reference)

## Metadata

- **Created**: 2026-04-04
- **Last Updated**: 2026-04-04 (session 4)
- **Challenge Due**: 2026-03-31 (passed deadline, core work complete)
- **Team**: Lindsay Gross, Yifei Guo, Jonas Neves

## Recent Commits (Sessions 2-4)

### Session 2 Polish
1. `3676eb9` — add grid layout support to update_readme_with_gifs (2-column default)
2. `f55a1c6` — remove unused embed_videos_in_readme functions
3. `521c096` — add training GIFs [skip ci]
4. `1899dc4` — fix missing re import in update_readme_with_gifs

### Session 3 Refinements
5. `6613173` — update documentation for renamed notebooks
6. `ebeceb9` — add interactive policy visualization with GitHub Pages deployment
7. `1ca64c6` — remove custom pages workflow — GitHub has built-in deployment
8. `fe56932` — fix Three.js CDN URLs (use v0.160.0)
9. `75a385b` — add trajectory extraction cells to notebooks and fix MultiInputPolicy handling
10. `bbaebce` — move trajectory extraction cell to end of reach notebook
11. `f502129` — remove premature publish_artifacts from Setup cell — only publish at end
12. `98d1955` — fix Setup cell — clone repo before importing scripts
13. `c4c559a` — restore original Setup pattern — download colab_utils from repo
14. `8ab48ab` — remove (test GIF cleanup)

### Session 3+ (Org Migration & Visualization)
15. Migrated repo to `aipi590-ggn` org (all links, Colab badges, colab_utils updated)
16. `7ca7db0` — add importmap for Three.js bare imports in OrbitControls
17. `1aecfdd` — implement versioned trajectory files to prevent overwrites between tasks
18. `b3bf6ae` — update notebooks to save versioned trajectory files
19. `8d99001` — document trajectory file versioning system
20. `71658bc` — rebuild 3D visualization: table, end effector, goal marker, trajectory trail

All commits post-deadline; challenge core work complete.
