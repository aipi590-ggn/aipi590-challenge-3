"""Colab notebook helpers: clone, run, publish artifacts back to GitHub."""

from __future__ import annotations

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
    paths: Iterable[str | Path],
    message: str,
    repo_dir: str | Path = DEFAULT_REPO_DIR,
    dry_run: bool = False,
) -> bool | None:
    """Commit and push artifacts from Colab back to GitHub.

    Uses a stored secret if available; otherwise shows a Sign in & Publish
    button — no manual setup required.
    """
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("publish_artifacts only works from Google Colab.") from exc

    repo_path = Path(repo_dir)
    rel_paths = [str(Path(p)) for p in paths]

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
