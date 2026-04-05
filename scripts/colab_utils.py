"""Colab notebook helpers: clone, run, publish artifacts back to GitHub."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Iterable

REPO_URL = "https://github.com/aipi590-ggn/aipi590-challenge-3.git"
DEFAULT_REPO_DIR = Path("/content/aipi590-challenge-3")
TOKEN_SECRET_NAME = "GITHUB_TOKEN"

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

    missing = [p for p in rel_paths if not (repo_path / p).exists()]
    if missing:
        raise FileNotFoundError("Cannot publish — files not found: " + ", ".join(missing))

    for rel in rel_paths:
        if not rel.endswith(".ipynb"):
            continue
        nb_path = repo_path / rel
        with open(nb_path) as f:
            nb = json.load(f)
        if "ipynb" in nb and "cells" not in nb:
            nb = nb["ipynb"]
            nb.setdefault("nbformat", 4)
            nb.setdefault("nbformat_minor", 5)
            with open(nb_path, "w") as f:
                json.dump(nb, f, indent=1)
                f.write("\n")

    repo_url = f"https://x-access-token:{token}@github.com/aipi590-ggn/aipi590-challenge-3.git"

    subprocess.run(["git", "config", "user.email", "colab-bot@scaffold"], check=True, cwd=repo_path)
    subprocess.run(["git", "config", "user.name", "Colab Bot"], check=True, cwd=repo_path)
    subprocess.run(["git", "remote", "set-url", "origin", repo_url], check=True, cwd=repo_path)

    if (repo_path / ".git" / "shallow").exists():
        subprocess.run(["git", "fetch", "--unshallow", "origin", "main"], check=True, cwd=repo_path)

    # Collect the parent directories of artifacts so -A captures deletions too
    artifact_dirs = sorted({str(Path(p).parts[0]) for p in rel_paths})
    subprocess.run(["git", "add", "-A", "--", *artifact_dirs], check=True, cwd=repo_path)

    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *artifact_dirs],
        cwd=repo_path, capture_output=True, text=True, check=True,
    )
    staged = [l for l in status.stdout.splitlines() if l and l[0] not in (" ", "?")]
    if not staged:
        print("No artifact changes to commit.")
        return False

    if dry_run:
        print(f"[dry_run] Would commit: {', '.join(artifact_dirs)}")
        return False

    # Sync with remote: reset to origin/main, then recommit only our artifacts.
    # This avoids rebase conflicts when the Colab clone is many commits behind.
    fetch = subprocess.run(["git", "fetch", "origin", "main"], cwd=repo_path, capture_output=True, text=True)
    if fetch.returncode != 0:
        raise RuntimeError(f"git fetch failed:\n{fetch.stderr or fetch.stdout}")

    subprocess.run(["git", "reset", "--soft", "origin/main"], check=True, cwd=repo_path)

    # Re-stage only the artifact directories (soft reset moved everything to staged)
    subprocess.run(["git", "reset", "HEAD", "--", "."], cwd=repo_path, capture_output=True)
    subprocess.run(["git", "add", "-A", "--", *artifact_dirs], check=True, cwd=repo_path)

    # Check if there's still something to commit after re-staging
    status2 = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        cwd=repo_path, capture_output=True, text=True,
    )
    if not status2.stdout.strip():
        print("Artifacts already match remote. Nothing to push.")
        return False

    subprocess.run(["git", "commit", "-m", f"{message} [skip ci]"], check=True, cwd=repo_path)

    push = subprocess.run(["git", "push", "origin", "main"], cwd=repo_path, capture_output=True, text=True)
    if push.returncode != 0:
        err = push.stderr or push.stdout
        if "403" in err or "denied" in err.lower():
            raise RuntimeError(
                f"Push denied. Your GitHub account needs write access to aipi590-ggn/aipi590-challenge-3.\n"
                f"Ask an org admin to add you as a collaborator, then re-run this cell.\n\n{err}"
            )
        raise RuntimeError(f"git push failed:\n{err}")

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


def videos_to_gifs(
    video_paths: Iterable[str | Path],
    scale: int = 320,
    fps: int = 5,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
) -> list[Path]:
    """Convert MP4 videos to animated GIFs for README inline playback.

    Uses ffmpeg to downsample and convert. GIFs play inline in GitHub markdown
    without needing drag-drop API or authentication.

    Args:
        video_paths: List of MP4 files to convert.
        scale: Output width in pixels (height scaled proportionally).
        fps: Frames per second in output GIF (lower = smaller file).
        repo_dir: Repository directory.

    Returns:
        List of created GIF paths (relative to repo_dir).
    """
    import subprocess

    repo_path = Path(repo_dir)
    video_paths = [repo_path / Path(p) if not Path(p).is_absolute() else Path(p) for p in video_paths]

    missing = [p for p in video_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Videos not found: {', '.join(str(p) for p in missing)}")

    gif_paths = []

    for video_path in video_paths:
        gif_path = video_path.with_suffix(".gif")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(video_path),
                    "-vf", f"fps={fps},scale={scale}:-1:flags=lanczos",
                    "-y",  # Overwrite without asking
                    str(gif_path),
                ],
                check=True,
                capture_output=True,
            )
            gif_paths.append(gif_path.relative_to(repo_path))
            file_size_mb = gif_path.stat().st_size / (1024 * 1024)
            print(f"Created {gif_path.name} ({file_size_mb:.1f} MB)")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {video_path.name}: {e.stderr.decode()}")
        except FileNotFoundError:
            print("Error: ffmpeg not found. Install with: apt-get install ffmpeg")
            return []

    return gif_paths


def update_readme_with_gifs(
    gif_paths: list[str | Path],
    readme_path: str | Path = "README.md",
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    columns: int = 2,
) -> bool:
    """Update README with inline GIF links in a grid layout.

    Replaces video section with GIF markdown arranged in columns.

    Args:
        gif_paths: List of GIF files (relative to repo_dir).
        readme_path: Path to README.
        repo_dir: Repository directory.
        columns: Number of columns (1=vertical, 2=horizontal pairs).

    Returns:
        True if successful.
    """
    import re

    repo_path = Path(repo_dir)
    readme_full = repo_path / readme_path

    if not readme_full.exists():
        print(f"README not found: {readme_full}")
        return False

    try:
        with open(readme_full) as f:
            readme_content = f.read()

        # Build GIF section with grid layout
        gif_section = "\n## Rollout Videos (GIF)\n\n"

        if columns == 1:
            # Vertical layout
            for gif_path in gif_paths:
                gif_rel = Path(gif_path).relative_to(repo_path) if Path(gif_path).is_absolute() else gif_path
                name = Path(gif_path).stem
                gif_section += f"### {name}\n\n![{name}]({gif_rel})\n\n"
        else:
            # Grid layout (HTML table for better control)
            gif_section += '<table>\n'
            for i in range(0, len(gif_paths), columns):
                gif_section += '  <tr>\n'
                for j in range(columns):
                    if i + j < len(gif_paths):
                        gif_path = gif_paths[i + j]
                        gif_rel = Path(gif_path).relative_to(repo_path) if Path(gif_path).is_absolute() else gif_path
                        name = Path(gif_path).stem
                        gif_section += f'    <td><strong>{name}</strong><br><img src="{gif_rel}" width="100%"></td>\n'
                    else:
                        gif_section += '    <td></td>\n'
                gif_section += '  </tr>\n'
            gif_section += '</table>\n'

        # Replace or append GIF section
        if "## Rollout Videos (GIF)" in readme_content:
            readme_content = re.sub(
                r"## Rollout Videos \(GIF\).*?(?=\n##|\Z)",
                gif_section.rstrip() + "\n",
                readme_content,
                flags=re.DOTALL,
            )
        else:
            readme_content += gif_section

        with open(readme_full, "w") as f:
            f.write(readme_content)

        print(f"Updated {readme_path} with {len(gif_paths)} GIF(s) ({columns} column(s))")
        return True
    except Exception as e:
        print(f"Failed to update README: {e}")
        return False


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

    # Use GitHub API (gh CLI requires separate auth in notebooks)
    return _publish_release_api(version, paths, description, repo_path)


def _publish_release_api(
    version: str,
    paths: list[Path],
    description: str,
    repo_path: Path,
) -> bool:
    """Create release using GitHub REST API."""
    import os

    # Try env var first (works in Kaggle, GitHub Actions, etc.)
    token = os.environ.get("GITHUB_TOKEN")

    # Try Colab secret as fallback
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN")
        except Exception:
            pass

    if not token:
        print("No GitHub token found. Use publish_artifacts() to sign in via OAuth,")
        print("or add a GITHUB_TOKEN secret in Colab (Settings > Secrets).")
        return False

    import urllib.request
    import json as json

    owner, repo = "aipi590-ggn", "aipi590-challenge-3"
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
        data=json.dumps(release_data).encode(),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            release = json.loads(response.read())
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
    _is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    _is_colab = False
    if not _is_kaggle:
        try:
            from google.colab import output as _colab_output
            _is_colab = hasattr(_colab_output, "register_callback")
        except ImportError:
            pass

    repo_path = Path(repo_dir)

    if paths is None:
        # Default: include results/ and docs/data/ (trajectory files)
        paths = []
        for subdir in ["results", "docs/data"]:
            d = repo_path / subdir
            if d.exists():
                paths.extend(p for p in sorted(d.rglob("*")) if p.is_file())

    rel_paths = [str(Path(p).relative_to(repo_path)) for p in paths]

    token = os.environ.get("GITHUB_TOKEN")

    # Kaggle secrets (Add-ons → Secrets)
    if not token and _is_kaggle:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        except Exception as e:
            print(f"Warning: Kaggle secret lookup failed: {e}")
            token = None

    # Colab secrets
    if not token and _is_colab:
        try:
            from google.colab import userdata
            token = userdata.get(TOKEN_SECRET_NAME)
        except Exception:
            token = None

    if token:
        return _do_publish(token, rel_paths, message, repo_path, dry_run)

    if not _is_colab:
        print("No GITHUB_TOKEN found.")
        print()
        print("On Kaggle:  Add-ons → Secrets → add GITHUB_TOKEN (a personal access token with repo scope),")
        print("            then toggle it ON for this notebook and re-run the cell")
        print("Elsewhere:  export GITHUB_TOKEN=ghp_...")
        print()
        print("Then re-run this cell.")
        return None

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
          const {{ token }} = await connectGitHub('repo', 'aipi590-ggn');
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
