"""Feed TensorBoard runs into the train-progress panel, with no change to training code.

    python3 tb_watch.py [ROOT ...]

Every few seconds it finds run folders under the roots (any folder holding
events.out.tfevents.*), reads the new records of each event file, and writes
~/.train_progress/<experiment>__<run>.json. The roots are the ones given plus
the lines of ~/.train_progress/roots.txt, which the Claude Code, Codex and OpenCode
integrations fill with each session's `<project>/logs`. No TensorBoard install is needed: the event
file format is parsed here directly. Only one watcher runs per machine; a second
one exits at once.

- total: --total, else `max_iterations` from the run's params/agent.yaml
  (Isaac Lab / rsl_rl), offset by the first step logged so resumed runs count right.
- metric: the scalar shown beside the bar, --metric (default Train/mean_reward).
- done: the step reached the total, or the final model_<total-1>.pt checkpoint exists.
- failed: the event file names the trainer's PID; when that process is gone from
  this host before the run is done, the run is marked failed.
- session: read from the trainer's environment (TRAIN_PROGRESS_SESSION, else
  CLAUDE_CODE_SESSION_ID, else CODEX_THREAD_ID), so the panel can show one
  session's runs.
"""

import argparse
import fcntl
import json
import os
import re
import socket
import struct
import time

HOST = socket.gethostname()
SESSION_VARS = (b"TRAIN_PROGRESS_SESSION=", b"CLAUDE_CODE_SESSION_ID=", b"CODEX_THREAD_ID=")


def out_dir():
    return os.environ.get("TRAIN_PROGRESS_DIR") or os.path.expanduser("~/.train_progress")


def _varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) of one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _varint(buf, i)
        elif wire == 1:
            val, i = buf[i:i + 8], i + 8
        elif wire == 2:
            size, i = _varint(buf, i)
            val, i = buf[i:i + size], i + size
        elif wire == 5:
            val, i = buf[i:i + 4], i + 4
        else:
            return
        yield field, wire, val


def _scalars(event):
    """(step, wall_time, {tag: value}) of one tensorflow.Event record."""
    step, wall, values = 0, 0.0, {}
    for field, wire, val in _fields(event):
        if field == 1 and wire == 1:
            wall = struct.unpack("<d", val)[0]
        elif field == 2 and wire == 0:
            step = val
        elif field == 5 and wire == 2:  # Summary
            for f, w, v in _fields(val):
                if f != 1 or w != 2:  # Summary.Value
                    continue
                tag, simple = None, None
                for vf, vw, vv in _fields(v):
                    if vf == 1 and vw == 2:
                        tag = vv.decode("utf-8", "replace")
                    elif vf == 2 and vw == 5:
                        simple = struct.unpack("<f", vv)[0]
                if tag is not None and simple is not None:
                    values[tag] = simple
    return step, wall, values


def _session_of(pid):
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            env = f.read().split(b"\0")
    except OSError:
        return None
    for var in SESSION_VARS:
        for item in env:
            if item.startswith(var):
                return item[len(var):].decode("utf-8", "replace") or None
    return None


class Run:
    def __init__(self, folder, name):
        self.folder, self.name = folder, name
        self.offsets = {}  # event file -> bytes read so far
        self.first_step = None
        self.step = 0
        self.metric = None
        self.started = None
        self.updated = None
        self.session = None

    def read(self, metric_tag):
        for fname in sorted(os.listdir(self.folder)):
            if not fname.startswith("events.out.tfevents"):
                continue
            path = os.path.join(self.folder, fname)
            with open(path, "rb") as f:
                f.seek(self.offsets.get(path, 0))
                while True:
                    start = f.tell()
                    head = f.read(12)
                    if len(head) < 12:
                        f.seek(start)
                        break
                    (length,) = struct.unpack("<Q", head[:8])
                    body = f.read(length + 4)
                    if len(body) < length + 4:  # record still being written
                        f.seek(start)
                        break
                    step, wall, values = _scalars(body[:length])
                    if wall:
                        self.started = self.started or wall
                        self.updated = wall
                    # rsl_rl also logs `<tag>/time` scalars stepped by seconds
                    if any(not t.endswith("/time") for t in values):
                        if self.first_step is None:
                            self.first_step = step
                        self.step = max(self.step, step)
                        if metric_tag in values:
                            self.metric = values[metric_tag]
                self.offsets[path] = f.tell()

    def pids(self):
        """PIDs of trainers on this host, from event file names (...<host>.<pid>.<n>)."""
        out = []
        for path in self.offsets:
            parts = os.path.basename(path).split(".")
            if len(parts) >= 6 and parts[-1].isdigit() and parts[-2].isdigit():
                if ".".join(parts[4:-2]) == HOST:
                    out.append(int(parts[-2]))
        return out


def _max_iterations(folder):
    try:
        with open(os.path.join(folder, "params", "agent.yaml")) as f:
            m = re.search(r"^max_iterations:\s*(\d+)", f.read(), re.M)
        return int(m.group(1)) if m else None
    except OSError:
        return None


def _write(folder, key, data):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{key}.json")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _roots(given, roots_file):
    roots = list(given)
    try:
        with open(roots_file) as f:
            roots += [line.strip() for line in f if line.strip()]
    except OSError:
        pass
    seen, out = set(), []
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if root not in seen and os.path.isdir(root):
            seen.add(root)
            out.append(root)
    return out


def take_lock():
    """The one-watcher-per-machine lock, or None when another watcher holds it."""
    os.makedirs(out_dir(), exist_ok=True)
    lock = open(os.path.join(out_dir(), ".watcher.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        return None
    return lock


def scan_forever(given=(), metric="Train/mean_reward", total=None, interval=5.0,
                 max_age_hours=24.0, once=False):
    folder = out_dir()
    label = metric.split("/")[-1].replace("mean_", "")
    runs = {}
    while True:
        now = time.time()
        for root in _roots(given, os.path.join(folder, "roots.txt")):
            for run_dir, _dirs, files in os.walk(root):
                events = [f for f in files if f.startswith("events.out.tfevents")]
                if not events:
                    continue
                newest = max(os.path.getmtime(os.path.join(run_dir, f)) for f in events)
                if now - newest > max_age_hours * 3600:
                    continue
                project = os.path.basename(os.path.dirname(root)) if os.path.basename(root) == "logs" \
                    else os.path.basename(root)
                rel = os.path.relpath(run_dir, root)
                key = re.sub(r"[^A-Za-z0-9._-]", "_", f"{project}__{rel.replace(os.sep, '__')}")
                run = runs.get(key) or runs.setdefault(key, Run(run_dir, rel))
                try:
                    run.read(metric)
                except OSError:
                    continue

                iters = total or _max_iterations(run_dir)
                run_total = (run.first_step or 0) + iters if iters else 0
                done = bool(run_total) and (
                    run.step >= run_total - 1
                    or os.path.exists(os.path.join(run_dir, f"model_{run_total - 1}.pt"))
                    or os.path.exists(os.path.join(run_dir, f"model_{run_total}.pt"))
                )
                pids = run.pids()
                alive = [p for p in pids if os.path.exists(f"/proc/{p}")]
                if run.session is None and alive:
                    run.session = _session_of(alive[0])
                status = "done" if done else ("failed" if pids and not alive else "running")

                _write(folder, key, {
                    "name": rel,
                    "step": run.step,
                    "total": run_total,
                    "metric": run.metric,
                    "metric_name": label,
                    "status": status,
                    "started_at": run.started,
                    "updated_at": run.updated,
                    "host": HOST,
                    "session": run.session,
                })
        if once:
            return
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", help="log folders to watch, e.g. logs/rsl_rl")
    ap.add_argument("--metric", default="Train/mean_reward", help="scalar tag to show")
    ap.add_argument("--total", type=int, help="total iterations, when params/agent.yaml lacks it")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--max-age-hours", type=float, default=24.0, help="ignore runs idle longer than this")
    ap.add_argument("--once", action="store_true", help="scan once and exit")
    args = ap.parse_args()

    lock = None if args.once else take_lock()
    if not args.once and lock is None:
        return  # another watcher is already running on this machine
    scan_forever(args.roots, args.metric, args.total, args.interval, args.max_age_hours, args.once)


if __name__ == "__main__":
    main()
