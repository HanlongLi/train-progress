"""Training progress bars in any terminal: beside Codex, OpenCode, or on their own.

    python3 tp_view.py                 # every job on this machine
    python3 tp_view.py --session ID    # only jobs started with TRAIN_PROGRESS_SESSION=ID
    python3 tp_view.py --once          # print once and exit (for scripts / status lines)
    python3 tp_view.py --watch DIR ... # also watch these TensorBoard log folders

Reads the same ~/.train_progress/*.json files as the Claude Code panel. While it
runs it also keeps the TensorBoard watcher going (unless one already runs), so
rsl_rl / TensorBoard runs under ./logs show up with no setup. Ctrl+C quits.
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_watch  # noqa: E402

STALL_S = 15 * 60
KEEP_S = 24 * 3600
ICON = {"running": "●", "done": "✓", "failed": "✗", "stalled": "◌"}
COLOR = {"running": "36", "done": "32", "failed": "31", "stalled": "33"}


def duration(sec):
    s = max(0, round(sec))
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m{s % 60:02d}s"
    h = m // 60
    return f"{h}h{m % 60:02d}m" if h < 48 else f"{h // 24}d{h % 24}h"


def load(session):
    now, jobs = time.time(), []
    folder = tb_watch.out_dir()
    try:
        names = [n for n in os.listdir(folder) if n.endswith(".json")]
    except OSError:
        return []
    for name in names:
        path = os.path.join(folder, name)
        try:
            mtime = os.path.getmtime(path)
            with open(path) as f:
                raw = json.load(f)
        except (OSError, ValueError):
            continue
        # When the job last made progress; the watcher rewrites files every scan.
        last = raw.get("updated_at") or mtime
        status = raw.get("status") if raw.get("status") in ("done", "failed") else "running"
        if status == "running" and now - last > STALL_S:
            status = "stalled"
        if status != "running" and now - last > KEEP_S:
            continue
        if session and raw.get("session") != session:
            continue
        raw.update(status=status, mtime=last)
        jobs.append(raw)
    return sorted(jobs, key=lambda j: -j["mtime"])


def render(jobs, width, color):
    def c(code, text):
        return f"\033[{code}m{text}\033[0m" if color else text

    lines = []
    for j in jobs:
        step, total, st = int(j.get("step") or 0), int(j.get("total") or 0), j["status"]
        frac = 1.0 if st == "done" else (min(1.0, step / total) if total else 0.0)
        right = ""
        started, updated = j.get("started_at"), j.get("updated_at")
        if st == "running" and started and updated and step and total > step and updated > started:
            right = "ETA " + duration((total - step) * (updated - started) / step)
        elif st != "running":
            right = f"{st} {duration(time.time() - j['mtime'])} ago"
        name = f"{ICON[st]} {j.get('name', '?')}"
        pad = max(1, width - len(name) - len(right))
        lines.append(c(COLOR[st], name) + " " * pad + c("2", right))
        cells = max(4, width - 6)
        filled = round(frac * cells)
        lines.append(c(COLOR[st], "█" * filled + "░" * (cells - filled)) + f"{int(frac * 100):>5}%")
        metric = j.get("metric", j.get("loss"))
        details = [f"{step}/{total}" if total else f"step {step}"]
        if metric is not None:
            details.append(f"{j.get('metric_name', 'loss')} {metric:.4g}")
        if j.get("host"):
            details.append(j["host"])
        lines.append(c("2", " · ".join(details)[:width]))
        lines.append("")
    return "\n".join(lines) if lines else c("2", f"No training jobs in {tb_watch.out_dir()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="only jobs tagged with this session id")
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--watch", nargs="*", default=[], help="extra TensorBoard log folders to watch")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    roots = list(args.watch)
    if os.path.isdir("logs"):
        roots.append("logs")
    if args.once:
        tb_watch.scan_forever(roots, once=True)
    else:
        lock = tb_watch.take_lock()
        if lock is not None:  # no watcher running yet: be the one
            threading.Thread(target=tb_watch.scan_forever, args=(roots,), daemon=True).start()

    color = sys.stdout.isatty()
    try:
        while True:
            width = min(100, shutil.get_terminal_size((80, 24)).columns)
            out = render(load(args.session), width, color)
            if args.once:
                print(out)
                return
            sys.stdout.write("\033[H\033[2J" + out + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
