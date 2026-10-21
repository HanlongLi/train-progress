"""Live train-progress footer for an unmodified Codex CLI, hosted by tmux."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid

import tp_view

ROWS = 3
RECENT_S = 10 * 60


def clean(value):
    # Job output is untrusted terminal text, including tmux styling directives.
    return "".join(c if not unicodedata.category(c).startswith("C") else " " for c in str(value))


def clip(text, width):
    result, used = [], 0
    for c in text:
        cells = 0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
        if used + cells > width:
            break
        result.append(c)
        used += cells
    return "".join(result)


def render(jobs, row, width, now=None):
    now = time.time() if now is None else now
    jobs = [j for j in jobs if j["status"] in ("running", "stalled") or now - j["mtime"] <= RECENT_S]
    priority = {"running": 0, "stalled": 1, "failed": 2, "done": 3}
    jobs.sort(key=lambda j: (priority[j["status"]], -j["mtime"]))
    if row >= len(jobs):
        return clip("train-progress · No active jobs", width) if row == 0 else ""
    job = jobs[row]
    step, total = max(0, int(job.get("step") or 0)), max(0, int(job.get("total") or 0))
    status = job["status"]
    fraction = 1 if status == "done" else min(1, step / total) if total else 0
    cells = min(16, max(4, width // 8))
    filled = round(fraction * cells)
    bar = "█" * filled + "░" * (cells - filled)
    progress = f"{int(fraction * 100)}% · {step}/{total}" if total else "…"
    if status == "done" and not total:
        progress = "100%"
    text = f"{tp_view.ICON[status]} {clip(clean(job.get('name', '?')), max(4, min(28, width // 3)))}  {bar} {progress}"
    if status != "running":
        text += f" · {status}"
    started, updated = job.get("started_at"), job.get("updated_at")
    if status == "running" and started and updated and step and total > step and updated - started >= tp_view.ETA_MIN_S:
        text += " · ETA " + tp_view.duration((total - step) * (updated - started) / step)
    metric = job.get("metric", job.get("loss"))
    if isinstance(metric, (int, float)):
        text += f" · {clean(job.get('metric_name') or 'loss')} {metric:.4g}"
    if row == ROWS - 1 and len(jobs) > ROWS:
        suffix = f" · +{len(jobs) - ROWS} more"
        return clip(clip(text, max(0, width - len(suffix))) + suffix, width)
    return clip(text, width)


def main(argv):
    ap = argparse.ArgumentParser(prog="tp footer", description="Print one compact footer row without starting a watcher.")
    ap.add_argument("--row", type=int, choices=range(ROWS), default=0)
    ap.add_argument("--width", type=int, default=80)
    ap.add_argument("--session")
    args = ap.parse_args(argv)
    session = args.session
    if not session and os.environ.get("TRAIN_PROGRESS_SCOPE") == "session":
        session = os.environ.get("TRAIN_PROGRESS_SESSION") or os.environ.get("CODEX_THREAD_ID")
    # Double # so job names cannot inject tmux styles such as #[hidden].
    print(render(tp_view.load(session), args.row, max(1, args.width)).replace("#", "##"))
    return 0


def config():
    lines = ["set -g status on", f"set -g status {ROWS}", "set -g status-interval 1",
             "set -g status-style 'fg=cyan,bg=default'", "set -g escape-time 10",
             "set -g default-terminal tmux-256color"]
    for row in range(ROWS):
        command = shlex.join([sys.executable, os.path.abspath(__file__), "--row", str(row), "--width"])
        lines.append(f"set -g status-format[{row}] " + shlex.quote(f"#({command} #{{client_width}})"))
    return "\n".join(lines) + "\n"


def launch(argv):
    if argv[:1] in (["-h"], ["--help"]):
        print("Usage: tp codex [CODEX_ARGUMENTS...]\n\nLaunch Codex with live progress bars below its prompt (requires tmux).\n"
              "All arguments are forwarded to codex. Exit Codex normally to close the footer.\n"
              "Use TRAIN_PROGRESS_SCOPE=session to show only this launch's jobs.")
        return 0
    for command in ("tmux", "codex"):
        if not shutil.which(command):
            print(f"tp codex: {command} is not on PATH." +
                  (" Install it with `brew install tmux` (macOS) or your system package manager." if command == "tmux" else ""), file=sys.stderr)
            return 127
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("tp codex: an interactive terminal is required; use `tp --once` for a snapshot.", file=sys.stderr)
        return 2
    token = uuid.uuid4().hex[:12]
    env = dict(os.environ)
    env["TRAIN_PROGRESS_SESSION"] = env.get("TRAIN_PROGRESS_SESSION") or f"codex-{token}"
    # Use a dedicated server so the footer never changes another tmux session.
    # Clearing TMUX also permits launching from an existing tmux pane.
    env.pop("TMUX", None)
    base = ["tmux", "-L", f"train-progress-{token}"]
    with tempfile.TemporaryDirectory(prefix="train-progress-") as folder:
        path = os.path.join(folder, "tmux.conf")
        with open(path, "w") as f:
            f.write(config())
        result = subprocess.call(base + ["-f", path, "new-session", "-s", "codex", shlex.join(["codex"] + argv)], env=env)
        # A normal exit shuts down this server automatically. Detaching keeps
        # Codex alive, as expected in tmux; tell the person how to reattach.
        alive = subprocess.run(base + ["has-session", "-t", "codex"], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if alive.returncode == 0:
            print("Codex is still running. Reattach with: " + shlex.join(base + ["attach-session", "-t", "codex"]))
        return result


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
