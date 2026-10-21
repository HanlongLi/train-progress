"""Hook commands for agents without a side panel (Codex), reading the hook's JSON on stdin.

    python3 agent_hook.py session-start   # watch <cwd>/logs, start the watcher, list running jobs
    python3 agent_hook.py turn            # say which jobs finished, failed or stalled since last time

Each prints one JSON object whose `systemMessage` the agent shows the person, or
nothing. TRAIN_PROGRESS_SCOPE=session limits both to jobs started from the session.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_watch  # noqa: E402
import tp_view  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _jobs(session):
    scoped = os.environ.get("TRAIN_PROGRESS_SCOPE") == "session"
    session = os.environ.get("TRAIN_PROGRESS_SESSION") or session
    return tp_view.load(session if scoped and session else None)


def _seen_path(session):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session or "default")
    return os.path.join(tb_watch.out_dir(), ".seen", f"{safe}.json")


def _remember(session, jobs):
    path = _seen_path(session)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({j["id"]: j["status"] for j in jobs}, f)


def session_start(hook):
    cwd = hook.get("cwd") or os.getcwd()
    logs = os.path.join(cwd, "logs")
    if os.path.isdir(logs):
        roots_file = os.path.join(tb_watch.out_dir(), "roots.txt")
        try:
            with open(roots_file) as f:
                known = f.read().split("\n")
        except OSError:
            known = []
        if logs not in known:
            os.makedirs(tb_watch.out_dir(), exist_ok=True)
            with open(roots_file, "a") as f:
                f.write(logs + "\n")
    # Exits at once when a watcher already runs on this machine.
    subprocess.Popen([sys.executable, os.path.join(HERE, "tb_watch.py")], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    session = hook.get("session_id")
    jobs = _jobs(session)
    _remember(session, jobs)
    running = [j for j in jobs if j["status"] == "running"]
    if running:
        listed = ", ".join(tp_view.summary(j) for j in running[:4])
        more = f" and {len(running) - 4} more" if len(running) > 4 else ""
        return f"{len(running)} running: {listed}{more}. Run `tp` in another terminal for live bars."
    return None


def turn(hook):
    session = hook.get("session_id")
    path = _seen_path(session)
    try:
        with open(path) as f:
            seen = json.load(f)
    except (OSError, ValueError):
        seen = None  # nothing to compare with yet
    jobs = _jobs(session)
    _remember(session, jobs)
    if seen is None:
        return None
    # A job that is new since last time counts as having been running.
    changed = [j for j in jobs if j["status"] != "running" and seen.get(j["id"], "running") != j["status"]]
    if changed:
        return " · ".join(tp_view.summary(j) for j in changed)
    return None


def main():
    try:
        hook = json.load(sys.stdin)
    except ValueError:
        hook = {}
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    message = {"session-start": session_start, "turn": turn}[action](hook)
    if message:
        print(json.dumps({"systemMessage": f"train-progress: {message}"}))


if __name__ == "__main__":
    main()
