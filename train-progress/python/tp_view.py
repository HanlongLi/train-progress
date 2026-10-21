"""Progress bars for long tasks in any terminal: beside Codex, OpenCode, or on their own.

    tp                         # live view of every job on this machine; Ctrl+C quits
    tp --session ID            # only jobs started from this agent session
    tp --once                  # print once and exit (for scripts / status lines)
    tp --watch DIR ...         # also watch these TensorBoard log folders

    tp run [-n NAME] -- CMD    # run CMD as a job: progress read from its output,
                               # done or failed by its exit code
    tp set NAME [STEP] [--total N] [--note TEXT] [--metric NAME=VALUE]
    tp done NAME [--note TEXT] # mark a job reported with `tp set` finished
    tp fail NAME [--note TEXT]

Reads the same ~/.train_progress/*.json files as the Claude Code panel. While the
live view runs it also keeps the TensorBoard watcher going (unless one already
runs), so rsl_rl / TensorBoard runs under ./logs show up with no setup.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb_watch  # noqa: E402
from train_progress import Progress  # noqa: E402

STALL_S = 15 * 60
KEEP_S = 24 * 3600
ETA_MIN_S = 5  # no ETA before this: `tp set` alone would give "ETA 0s"
ICON = {"running": "●", "done": "✓", "failed": "✗", "stalled": "◌"}
COLOR = {"running": "36", "done": "32", "failed": "31", "stalled": "33"}
COMMANDS = ("run", "set", "done", "fail")

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
# "45/300" (tqdm, ninja's [45/300], "epoch 3/10"), but not dates, times or versions.
FRACTION = re.compile(r"(?<![\d./:-])(\d+)\s*/\s*(\d+)(?![\d./:-])")
PERCENT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%")


def duration(sec):
    s = max(0, round(sec))
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m{s % 60:02d}s"
    h = m // 60
    return f"{h}h{m % 60:02d}m" if h < 48 else f"{h // 24}d{h % 24}h"


def load(session=None):
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
        raw.update(id=name[:-5], status=status, mtime=last)
        jobs.append(raw)
    return sorted(jobs, key=lambda j: -j["mtime"])


def summary(job):
    """One short line per job, for agent hooks and toasts."""
    step, total, st = int(job.get("step") or 0), int(job.get("total") or 0), job["status"]
    parts = [f"{ICON[st]} {job.get('name', '?')}"]
    if st == "running" and total:
        parts.append(f"{min(100, int(step * 100 / total))}%")
    elif st != "running":
        parts.append(st)
    if job.get("note"):
        parts.append(f"({job['note'][:60]})")
    return " ".join(parts)


def render(jobs, width, color):
    def c(code, text):
        return f"\033[{code}m{text}\033[0m" if color else text

    lines = []
    for j in jobs:
        step, total, st = int(j.get("step") or 0), int(j.get("total") or 0), j["status"]
        frac = 1.0 if st == "done" else (min(1.0, step / total) if total else 0.0)
        right = ""
        started, updated = j.get("started_at"), j.get("updated_at")
        if st == "running" and started and updated and step and total > step and updated - started >= ETA_MIN_S:
            right = "ETA " + duration((total - step) * (updated - started) / step)
        elif st == "running" and not total and started:
            right = "running " + duration(time.time() - started)
        elif st != "running":
            right = f"{st} {duration(time.time() - j['mtime'])} ago"
        name = f"{ICON[st]} {j.get('name', '?')}"
        pad = max(1, width - len(name) - len(right))
        lines.append(c(COLOR[st], name) + " " * pad + c("2", right))
        cells = max(4, width - 6)
        if total or st == "done":
            filled = round(frac * cells)
            lines.append(c(COLOR[st], "█" * filled + "░" * (cells - filled)) + f"{int(frac * 100):>5}%")
        else:  # no total: nothing to measure the bar against
            lines.append(c("2", "░" * cells) + "    …")
        metric = j.get("metric", j.get("loss"))
        count = f"{step}/{total}" if total else f"step {step}" if step else ""
        details = [count]
        if metric is not None:
            details.append(f"{j.get('metric_name') or 'loss'} {metric:.4g}")
        if j.get("note") and j["note"] != count:  # `tp run` notes are often just "16/30"
            details.append(j["note"])
        if j.get("host"):
            details.append(j["host"])
        lines.append(c("2", " · ".join(d for d in details if d)[:width]))
        lines.append("")
    return "\n".join(lines) if lines else c("2", f"No jobs in {tb_watch.out_dir()}")


def view(argv):
    ap = argparse.ArgumentParser(prog="tp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="only jobs tagged with this session id")
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--watch", nargs="*", default=[], help="extra TensorBoard log folders to watch")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args(argv)

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


def parse_progress(line):
    """(step, total) found in one line of output, or None."""
    for m in reversed(list(FRACTION.finditer(line))):
        step, total = int(m.group(1)), int(m.group(2))
        if 0 < total and step <= total:
            return step, total
    m = None
    for m in PERCENT.finditer(line):
        pass
    if m and float(m.group(1)) <= 100:
        return int(float(m.group(1))), 100
    return None


def run(argv):
    ap = argparse.ArgumentParser(prog="tp run", description="Run a command as a tracked job.")
    ap.add_argument("-n", "--name", help="job name (default: the command)")
    ap.add_argument("--total", type=int, default=0, help="total steps, when the output has none")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command and its arguments")
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command given: tp run -- CMD ...")
    name = args.name or " ".join([os.path.basename(cmd[0])] + cmd[1:])[:48]

    p = Progress(name, total=args.total)
    lock = threading.Lock()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    try:
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    except OSError as e:
        p.finish("failed", note=str(e))
        print(f"tp run: {e}", file=sys.stderr)
        return 127

    def pump(src, dst):
        pending = ""
        while True:
            chunk = os.read(src.fileno(), 65536)
            if not chunk:
                break
            dst.buffer.write(chunk)
            dst.buffer.flush()
            # \r-separated pieces too: tqdm redraws one line in place.
            pending += chunk.decode("utf-8", "replace")
            *done, pending = re.split(r"[\r\n]", pending)
            pending = pending[-4096:]
            with lock:
                for raw in done:
                    line = ANSI.sub("", raw).strip()
                    if not line:
                        continue
                    found = parse_progress(line)
                    step, total = found if found else (p.step, None)
                    p.update(step, total=None if args.total else total, note=line[:120])

    threads = [threading.Thread(target=pump, args=pair, daemon=True)
               for pair in ((child.stdout, sys.stdout), (child.stderr, sys.stderr))]
    for t in threads:
        t.start()
    try:
        code = child.wait()
    except KeyboardInterrupt:  # the child got the Ctrl+C too
        code = child.wait()
    for t in threads:
        t.join(timeout=2)
    if code == 0:
        p.finish("done")
    else:
        p.finish("failed", note=f"exit {code}" + (f": {p.note}" if p.note else ""))
    return code if code >= 0 else 128 - code


def report(command, argv):
    ap = argparse.ArgumentParser(prog=f"tp {command}", description="Report a job's progress from a shell script.")
    ap.add_argument("name")
    if command == "set":
        ap.add_argument("step", nargs="?", type=int, help="current step (default: one more)")
        ap.add_argument("--total", type=int)
        ap.add_argument("--metric", metavar="NAME=VALUE", help="a number shown beside the bar")
    ap.add_argument("--note", help="a short line of text under the bar")
    args = ap.parse_args(argv)

    p = Progress(args.name, resume=True)
    if command == "set":
        metrics = {}
        if args.metric:
            key, _, value = args.metric.partition("=")
            metrics[key] = float(value)
        p.min_interval = 0
        p.update(args.step, total=args.total, note=args.note, **metrics)
    else:
        p.finish("done" if command == "done" else "failed", note=args.note)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        sys.exit(run(argv[1:]))
    if argv[:1] and argv[0] in COMMANDS:
        sys.exit(report(argv[0], argv[1:]))
    view(argv)


if __name__ == "__main__":
    main()
