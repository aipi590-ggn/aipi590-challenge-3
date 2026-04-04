"""Colab notebook helpers: clone, run, publish artifacts back to GitHub."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_URL = "https://github.com/jonasneves/aipi590-challenge-3.git"
DEFAULT_REPO_DIR = Path("/content/aipi590-challenge-3")
TOKEN_SECRET_NAME = "GITHUB_TOKEN_AIPI590_CHALLENGE_3"

_GITHUB_SVG = """
<svg width="18" height="18" viewBox="0 0 98 96" fill="white"
     xmlns="http://www.w3.org/2000/svg">
  <path fill-rule="evenodd" clip-rule="evenodd"
    d="M48.854 0C21.839 0 0 22 0 49.217c0 21.756 13.993 40.172 33.405
    46.69 2.427.49 3.316-1.059 3.316-2.362 0-1.141-.08-5.052-.08-9.127
    -13.59 2.934-16.42-5.867-16.42-5.867-2.184-5.704-5.42-7.17-5.42-7.17
    -4.448-3.015.324-3.015.324-3.015 4.934.326 7.523 5.052 7.523 5.052
    4.367 7.496 11.404 5.378 14.235 4.074.404-3.178 1.699-5.378 3.074-6.6
    -10.839-1.141-22.243-5.378-22.243-24.283 0-5.378 1.94-9.778 5.014-13.2
    -.485-1.222-2.184-6.275.486-13.038 0 0 4.125-1.304 13.426 5.052
    a46.97 46.97 0 0 1 12.214-1.63c4.125 0 8.33.571 12.213 1.63
    9.302-6.356 13.427-5.052 13.427-5.052 2.67 6.763.97 11.816.485 13.038
    3.155 3.422 5.015 7.822 5.015 13.2 0 18.905-11.404 23.06-22.324 24.283
    1.78 1.548 3.316 4.481 3.316 9.126 0 6.6-.08 11.897-.08 13.526
    0 1.304.89 2.853 3.316 2.364 19.412-6.52 33.405-24.935 33.405-46.691
    C97.707 22 75.788 0 48.854 0z"/>
</svg>
"""

_BTN_STYLE = """
<style>
  .ch3-btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 10px 20px; border: none; border-radius: 6px;
    background: #24292f; color: #fff; font-size: 14px;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    cursor: pointer; transition: background 0.15s;
  }
  .ch3-btn:hover:not(:disabled) { background: #32383f; }
  .ch3-btn:disabled { opacity: 0.7; cursor: default; }
  .ch3-status { margin-top: 8px; font-size: 13px;
    font-family: monospace; color: #555; }
</style>
"""


def _do_publish(
    token: str,
    rel_paths: list[str],
    message: str,
    repo_path: Path,
    dry_run: bool,
) -> bool:
    import json as _json

    missing = [p for p in rel_paths if not (repo_path / p).exists()]
    if missing:
        raise FileNotFoundError("Cannot publish — files not found: " + ", ".join(missing))

    for rel in rel_paths:
        if not rel.endswith(".ipynb"):
            continue
        nb_path = repo_path / rel
        with open(nb_path) as f:
            nb = _json.load(f)
        if "ipynb" in nb and "cells" not in nb:
            nb = nb["ipynb"]
            nb.setdefault("nbformat", 4)
            nb.setdefault("nbformat_minor", 5)
            with open(nb_path, "w") as f:
                _json.dump(nb, f, indent=1)
                f.write("\n")

    repo_url = f"https://x-access-token:{token}@github.com/jonasneves/aipi590-challenge-3.git"

    subprocess.run(["git", "config", "user.email", "colab-bot@scaffold"], check=True, cwd=repo_path)
    subprocess.run(["git", "config", "user.name", "Colab Bot"], check=True, cwd=repo_path)
    subprocess.run(["git", "remote", "set-url", "origin", repo_url], check=True, cwd=repo_path)

    if (repo_path / ".git" / "shallow").exists():
        subprocess.run(["git", "fetch", "--unshallow", "origin", "main"], check=True, cwd=repo_path)

    subprocess.run(["git", "add", "--force", "--", *rel_paths], check=True, cwd=repo_path)

    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *rel_paths],
        cwd=repo_path, capture_output=True, text=True, check=True,
    )
    staged = [l for l in status.stdout.splitlines() if l and l[0] not in (" ", "?")]
    if not staged:
        print("No artifact changes to commit.")
        return False

    if dry_run:
        print(f"[dry_run] Would commit: {', '.join(rel_paths)}")
        return False

    subprocess.run(["git", "commit", "-m", f"{message} [skip ci]"], check=True, cwd=repo_path)

    subprocess.run(["git", "rebase", "--abort"], cwd=repo_path, capture_output=True)
    fetch = subprocess.run(["git", "fetch", "origin", "main"], cwd=repo_path, capture_output=True, text=True)
    if fetch.returncode != 0:
        raise RuntimeError(f"git fetch failed:\n{fetch.stderr or fetch.stdout}")

    rebase = subprocess.run(["git", "rebase", "origin/main"], cwd=repo_path, capture_output=True, text=True)
    if rebase.returncode != 0:
        subprocess.run(["git", "rebase", "--abort"], cwd=repo_path, capture_output=True)
        raise RuntimeError(f"git rebase failed:\n{rebase.stderr or rebase.stdout}")

    push = subprocess.run(["git", "push", "origin", "main"], cwd=repo_path, capture_output=True, text=True)
    if push.returncode != 0:
        raise RuntimeError(f"git push failed:\n{push.stderr or push.stdout}")

    print(f"Pushed: {', '.join(rel_paths)}")
    return True


def prepare_notebook(
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    *,
    pull_latest: bool = False,
) -> Path:
    """Clone the repo into Colab workspace if needed. Returns repo root."""
    repo_path = Path(repo_dir)
    if not repo_path.exists():
        subprocess.run(
            ["git", "clone", "--depth=1", REPO_URL, str(repo_path)],
            check=True,
        )

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    os.chdir(repo_path)

    if pull_latest:
        try:
            subprocess.run(["git", "pull", "origin", "main"], check=True, cwd=repo_path)
        except subprocess.CalledProcessError:
            print("Warning: git pull failed — continuing with local state.")

    return repo_path


def publish_release(
    version: str,
    paths: Iterable[str | Path] | None = None,
    description: str = "",
    repo_dir: str | Path = DEFAULT_REPO_DIR,
) -> bool:
    """Create a GitHub release and upload model artifacts.

    Args:
        version: Release tag (e.g., 'v1', 'v1.0.0').
        paths: Files to upload. If None, includes all .zip files from results/models/.
        description: Release description (markdown).
        repo_dir: Repository directory.

    Returns:
        True if successful, False otherwise.
    """
    repo_path = Path(repo_dir)

    if paths is None:
        # Default: include all .zip files in results/models/
        models_dir = repo_path / "results" / "models"
        if models_dir.exists():
            paths = sorted(models_dir.glob("*.zip"))
        else:
            paths = []

    paths = [Path(p) for p in paths]
    if not paths:
        print("No files to upload.")
        return False

    # Check if all files exist
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Files not found: {', '.join(str(p) for p in missing)}")

    # Try to use gh CLI first (simplest, most reliable)
    try:
        import shutil
        if shutil.which("gh"):
            return _publish_release_gh(version, paths, description)
    except Exception:
        pass

    # Fall back to GitHub API
    return _publish_release_api(version, paths, description, repo_path)


def _publish_release_gh(
    version: str,
    paths: list[Path],
    description: str,
) -> bool:
    """Create release using GitHub CLI."""
    import subprocess

    cmd = ["gh", "release", "create", version, "--title", version]

    if description:
        cmd.extend(["--notes", description])

    cmd.extend(str(p) for p in paths)

    try:
        subprocess.run(cmd, check=True)
        print(f"Release {version} created with {len(paths)} asset(s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"gh release failed: {e}")
        return False


def _publish_release_api(
    version: str,
    paths: list[Path],
    description: str,
    repo_path: Path,
) -> bool:
    """Create release using GitHub REST API."""
    try:
        from google.colab import userdata
        token = userdata.get(TOKEN_SECRET_NAME)
    except Exception:
        token = None

    if not token:
        print("Error: No GitHub token available. Set GITHUB_TOKEN_AIPI590_CHALLENGE_3 in Colab secrets.")
        return False

    import urllib.request
    import json as _json

    owner, repo = "jonasneves", "aipi590-challenge-3"
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"

    # Create release
    release_data = {
        "tag_name": version,
        "name": version,
        "body": description,
        "draft": False,
        "prerelease": False,
    }

    req = urllib.request.Request(
        api_url,
        data=_json.dumps(release_data).encode(),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            release = _json.loads(response.read())
        upload_url = release["upload_url"].replace("{?name,label}", "")
    except Exception as e:
        print(f"Failed to create release: {e}")
        return False

    # Upload assets
    success_count = 0
    for path in paths:
        try:
            with open(path, "rb") as f:
                asset_data = f.read()

            asset_url = f"{upload_url}?name={path.name}"
            asset_req = urllib.request.Request(
                asset_url,
                data=asset_data,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/octet-stream",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                method="POST",
            )
            with urllib.request.urlopen(asset_req) as response:
                response.read()
            print(f"Uploaded {path.name}")
            success_count += 1
        except Exception as e:
            print(f"Failed to upload {path.name}: {e}")

    if success_count == len(paths):
        print(f"Release {version} created with {len(paths)} asset(s)")
        return True
    else:
        print(f"Partial upload: {success_count}/{len(paths)} assets uploaded")
        return success_count > 0


def save_notebook(
    notebook_name: str,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
) -> str | None:
    """Snapshot the running Colab notebook and write it to the repo."""
    try:
        from google.colab import _message
        import json

        nb = _message.blocking_request("get_ipynb", request="", timeout_sec=30)
        if not nb:
            print("Warning: get_ipynb returned empty.")
            return None

        if "ipynb" in nb and "cells" not in nb:
            nb = nb["ipynb"]

        nb.setdefault("nbformat", 4)
        nb.setdefault("nbformat_minor", 5)

        out = Path(repo_dir) / "notebooks" / notebook_name
        with open(out, "w") as f:
            json.dump(nb, f, indent=1)

        rel = f"notebooks/{notebook_name}"
        print(f"Notebook snapshot saved to {rel}")
        return rel
    except Exception as e:
        print(f"Warning: could not save notebook — {e}")
        return None


def publish_artifacts(
    message: str,
    paths: Iterable[str | Path] | None = None,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    dry_run: bool = False,
) -> bool | None:
    """Commit and push artifacts from Colab back to GitHub.

    Uses a stored secret if available; otherwise shows a Sign in & Publish
    button — no manual setup required.

    If paths is None, includes all files under results/ directory.
    """
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("publish_artifacts only works from Google Colab.") from exc

    repo_path = Path(repo_dir)

    if paths is None:
        # Default: include all files in results/
        results_dir = repo_path / "results"
        if results_dir.exists():
            paths = sorted(results_dir.rglob("*"))
            paths = [p for p in paths if p.is_file()]
        else:
            paths = []

    rel_paths = [str(Path(p).relative_to(repo_path)) for p in paths]

    try:
        from google.colab import userdata
        token = userdata.get(TOKEN_SECRET_NAME)
    except Exception:
        token = None

    if token:
        return _do_publish(token, rel_paths, message, repo_path, dry_run)

    from IPython.display import display, HTML
    from google.colab import output

    def _on_token(token: str) -> None:
        print("Authenticated. Publishing…")
        try:
            _do_publish(token, rel_paths, message, repo_path, dry_run)
        except Exception as e:
            print(f"Publish failed: {e}")

    output.register_callback("_ch3_publish_cb", _on_token)

    display(HTML(f"""
    {_BTN_STYLE}
    <button class="ch3-btn" id="ch3-pub-btn">
      {_GITHUB_SVG}
      Sign in &amp; Publish
    </button>
    <div class="ch3-status" id="ch3-pub-status"></div>
    <script type="module">
      import {{ connectGitHub }} from 'https://neevs.io/auth/lib.js';
      const btn = document.getElementById('ch3-pub-btn');
      const status = document.getElementById('ch3-pub-status');
      btn.addEventListener('click', async () => {{
        btn.disabled = true;
        status.textContent = 'Waiting for GitHub authorization\u2026';
        try {{
          const {{ token }} = await connectGitHub('repo', 'jonasneves');
          btn.style.background = '#2da44e';
          btn.innerHTML = '\u2713 Authorized \u2014 publishing\u2026';
          status.textContent = '';
          google.colab.kernel.invokeFunction('_ch3_publish_cb', [token], {{}});
        }} catch (e) {{
          btn.disabled = false;
          btn.style.background = '';
          status.textContent = 'Authorization failed \u2014 try again.';
          console.error(e);
        }}
      }});
    </script>
    """))
    return None


# ---------------------------------------------------------------------------
# Live training chart
# ---------------------------------------------------------------------------

_TRAINING_CHART_HTML = """\
<style>
.tb-stats{display:flex;gap:24px;padding:12px 16px;background:#12122a;
  border-radius:8px 8px 0 0;font-family:monospace;flex-wrap:wrap}
.tb-stat{display:flex;flex-direction:column}
.tb-val{color:#e8e8e8;font-size:15px;font-weight:600}
.tb-lbl{color:#555;font-size:10px;margin-top:2px;text-transform:uppercase;letter-spacing:.5px}
</style>
<div class="tb-stats">
  <div class="tb-stat"><span class="tb-val">__TIMESTEPS__</span><span class="tb-lbl">timesteps</span></div>
  <div class="tb-stat"><span class="tb-val">__EPISODES__</span><span class="tb-lbl">episodes</span></div>
  <div class="tb-stat"><span class="tb-val">__FPS__ fps</span><span class="tb-lbl">throughput</span></div>
  <div class="tb-stat"><span class="tb-val">__SUCCESS_PCT__%</span><span class="tb-lbl">success rate</span></div>
  <div class="tb-stat"><span class="tb-val">__ELAPSED__s</span><span class="tb-lbl">elapsed</span></div>
  <div class="tb-stat"><span class="tb-val">__N_UPDATES__</span><span class="tb-lbl">updates</span></div>
</div>
<div id="tb-chart" style="height:460px;border-radius:0 0 8px 8px;overflow:hidden;background:#1a1a2e"></div>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script>
(function() {
  var dom = document.getElementById('tb-chart');
  var w = Math.max(window.innerWidth - 32, 400);
  dom.style.width = w + 'px';
  var chart = echarts.init(dom, 'dark', {width: w, height: 460});
  var rewardData  = __REWARD_DATA__;
  var successData = __SUCCESS_DATA__;
  var actorData   = __ACTOR_DATA__;
  var criticData  = __CRITIC_DATA__;
  var entData     = __ENT_DATA__;
  var fmtK   = function(v){ return (v/1000).toFixed(0)+'k'; };
  var fmtPct = function(v){ return (v*100).toFixed(0)+'%'; };
  var sl     = {lineStyle:{opacity:.1}};
  chart.setOption({
    backgroundColor: '#1a1a2e',
    animation: false,
    grid: [
      {top:'14%', left:'7%',  width:'38%', height:'30%'},
      {top:'14%', left:'57%', right:'3%',  height:'30%'},
      {top:'62%', left:'7%',  width:'38%', bottom:'8%'},
      {top:'62%', left:'57%', right:'3%',  bottom:'8%'},
    ],
    xAxis: [
      {gridIndex:0, type:'value', axisLabel:{show:false}, splitLine:sl},
      {gridIndex:1, type:'value', axisLabel:{show:false}, splitLine:sl},
      {gridIndex:2, type:'value', axisLabel:{formatter:fmtK}, name:'Timesteps', nameLocation:'middle', nameGap:25, splitLine:sl},
      {gridIndex:3, type:'value', axisLabel:{formatter:fmtK}, name:'Timesteps', nameLocation:'middle', nameGap:25, splitLine:sl},
    ],
    yAxis: [
      {gridIndex:0, type:'value', min:-50, max:0, splitLine:sl},
      {gridIndex:1, type:'value', min:0,   max:1, axisLabel:{formatter:fmtPct}, splitLine:sl},
      {gridIndex:2, type:'value', splitLine:sl},
      {gridIndex:3, type:'value', splitLine:sl},
    ],
    title: [
      {text:'Episode Reward',     textStyle:{fontSize:11,color:'#888'}, left:'7%',  top:'5%'},
      {text:'Success Rate',       textStyle:{fontSize:11,color:'#888'}, left:'57%', top:'5%'},
      {text:'Actor / Critic Loss',textStyle:{fontSize:11,color:'#888'}, left:'7%',  top:'53%'},
      {text:'Entropy Coefficient',textStyle:{fontSize:11,color:'#888'}, left:'57%', top:'53%'},
    ],
    legend: [
      {data:['Actor Loss','Critic Loss'], top:'55%', left:'7%', itemGap:16, textStyle:{color:'#888',fontSize:10}},
    ],
    series: [
      {type:'line', xAxisIndex:0, yAxisIndex:0, data:rewardData,  smooth:.3, symbol:'none', lineStyle:{color:'#5b8ff9',width:2}},
      {type:'line', xAxisIndex:1, yAxisIndex:1, data:successData, smooth:.3, symbol:'none', lineStyle:{color:'#5ad8a6',width:2}, areaStyle:{color:'rgba(90,216,166,.07)'}},
      {name:'Actor Loss',  type:'line', xAxisIndex:2, yAxisIndex:2, data:actorData,  smooth:.3, symbol:'none', lineStyle:{color:'#ff7875',width:2}},
      {name:'Critic Loss', type:'line', xAxisIndex:2, yAxisIndex:2, data:criticData, smooth:.3, symbol:'none', lineStyle:{color:'#ffd666',width:2}},
      {type:'line', xAxisIndex:3, yAxisIndex:3, data:entData,     smooth:.3, symbol:'none', lineStyle:{color:'#d3adf7',width:2}},
    ]
  });
})();
</script>"""


class LiveChartCallback:
    """SB3 callback that renders a live 4-panel ECharts training dashboard.

    Shows episode reward, success rate, actor/critic loss, and entropy
    coefficient. Uses clear_output + full redraw on each update — avoids
    eval_js iframe scoping issues in Colab.

    Usage::

        model.learn(total_timesteps=1_000_000, callback=LiveChartCallback())
    """

    # Lazy __new__ so this file can be imported without stable-baselines3.
    def __new__(cls, *args, **kwargs):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Impl(BaseCallback):
            def __init__(self, update_freq=2000, window=100, max_points=300, verbose=0):
                super().__init__(verbose)
                self.update_freq = update_freq
                self.window = window
                self.max_points = max_points  # downsample history sent to chart
                self._ep_rewards: list[float] = []
                self._ep_successes: list[float] = []
                self._history: dict[str, list] = {
                    "timesteps": [], "reward": [], "success": [],
                    "actor_loss": [], "critic_loss": [], "ent_coef": [],
                }

            def _on_training_start(self) -> None:
                from IPython.display import clear_output, display, HTML
                clear_output(wait=True)
                display(HTML(self._render()))

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    if "episode" in info:
                        self._ep_rewards.append(info["episode"]["r"])
                    if "is_success" in info:
                        self._ep_successes.append(float(info["is_success"]))

                if self.n_calls % self.update_freq == 0 and self._ep_rewards:
                    w = min(self.window, len(self._ep_rewards))
                    mean_r = sum(self._ep_rewards[-w:]) / w
                    mean_s = sum(self._ep_successes[-w:]) / w if self._ep_successes else 0.0
                    lv = self.model.logger.name_to_value

                    h = self._history
                    h["timesteps"].append(self.num_timesteps)
                    h["reward"].append(round(mean_r, 3))
                    h["success"].append(round(mean_s, 3))
                    h["actor_loss"].append(round(float(lv.get("train/actor_loss") or 0), 4))
                    h["critic_loss"].append(round(float(lv.get("train/critic_loss") or 0), 4))
                    h["ent_coef"].append(round(float(lv.get("train/ent_coef") or 0), 6))

                    from IPython.display import clear_output, display, HTML
                    clear_output(wait=True)
                    display(HTML(self._render()))

                return True

            def _render(self) -> str:
                h = self._history
                ts = h["timesteps"]
                n = len(ts)

                # Downsample to max_points evenly spaced indices to cap browser work.
                if n > self.max_points:
                    step = n / self.max_points
                    indices = [int(i * step) for i in range(self.max_points)]
                else:
                    indices = list(range(n))

                def pairs(key: str) -> str:
                    return json.dumps([[ts[i], h[key][i]] for i in indices])

                lv = self.model.logger.name_to_value if self.model else {}
                success_pct = round(h["success"][-1] * 100, 1) if h["success"] else 0.0

                return (
                    _TRAINING_CHART_HTML
                    .replace("__TIMESTEPS__",   f"{self.num_timesteps:,}")
                    .replace("__EPISODES__",    str(len(self._ep_rewards)))
                    .replace("__FPS__",         str(int(lv.get("time/fps") or 0)))
                    .replace("__SUCCESS_PCT__", str(success_pct))
                    .replace("__ELAPSED__",     str(int(lv.get("time/time_elapsed") or 0)))
                    .replace("__N_UPDATES__",   f"{int(lv.get('train/n_updates') or 0):,}")
                    .replace("__REWARD_DATA__",  pairs("reward"))
                    .replace("__SUCCESS_DATA__", pairs("success"))
                    .replace("__ACTOR_DATA__",   pairs("actor_loss"))
                    .replace("__CRITIC_DATA__",  pairs("critic_loss"))
                    .replace("__ENT_DATA__",     pairs("ent_coef"))
                )

        return _Impl(*args, **kwargs)
