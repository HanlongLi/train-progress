"""Run with: python3 -m unittest discover -s train-progress/python -v"""

import fcntl
import json
import os
import pty
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import termios
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import agent_hook
import codex_footer as footer


def job(name="training", status="running", step=50, total=100, **extra):
    return dict(name=name, status=status, step=step, total=total, mtime=1000, **extra)


class FooterTests(unittest.TestCase):
    def test_progress_metric_and_eta(self):
        line = footer.render([job(metric=0.12, metric_name="loss", started_at=900, updated_at=1000)], 0, 120, now=1000)
        for value in ("training", "50%", "50/100", "loss 0.12", "ETA 1m40s", "█", "░"):
            self.assertIn(value, line)

    def test_active_first_and_expiry(self):
        jobs = [job("old", "done"), job("fresh", "done"), job("active")]
        jobs[0]["mtime"] = 0
        self.assertIn("active", footer.render(jobs, 0, 80, now=1001))
        self.assertIn("fresh", footer.render(jobs, 1, 80, now=1001))
        self.assertEqual("", footer.render(jobs, 2, 80, now=1001))

    def test_empty_unknown_and_failed(self):
        self.assertIn("No active jobs", footer.render([], 0, 80))
        self.assertIn("…", footer.render([job(total=0)], 0, 80, now=1000))
        self.assertIn("failed", footer.render([job(status="failed")], 0, 80, now=1000))

    def test_narrow_unicode_and_overflow(self):
        jobs = [job("学習\x1b\n" + str(i)) for i in range(5)]
        for width in (1, 8, 20, 40, 80):
            line = footer.render(jobs, 2, width, now=1000)
            cells = sum(0 if footer.unicodedata.combining(c) else 2 if footer.unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in line)
            self.assertLessEqual(cells, width)
            self.assertNotIn("\x1b", line)
            self.assertNotIn("\n", line)
        self.assertIn("+2 more", footer.render(jobs, 2, 80, now=1000))

    def test_scope_and_style_escaping(self):
        with patch.dict(os.environ, {"TRAIN_PROGRESS_SCOPE": "session", "TRAIN_PROGRESS_SESSION": "launch"}), \
             patch("tp_view.load", return_value=[job("#[hidden]")]) as load, \
             patch("builtins.print") as output:
            footer.main([])
            load.assert_called_once_with("launch")
            self.assertIn("##[hidden]", output.call_args[0][0])
        with patch.dict(os.environ, {"TRAIN_PROGRESS_SCOPE": "session", "TRAIN_PROGRESS_SESSION": "launch"}), \
             patch("tp_view.load", return_value=[]) as load:
            agent_hook._jobs("codex-thread")
            load.assert_called_once_with("launch")

    def test_missing_tmux_and_noninteractive(self):
        with patch("shutil.which", return_value=None), patch("sys.stderr"):
            self.assertEqual(127, footer.launch([]))
        with patch("shutil.which", return_value="/bin/command"), patch("sys.stdin.isatty", return_value=False), patch("sys.stderr"):
            self.assertEqual(2, footer.launch([]))

    def test_detaching_preserves_session_and_prints_reattach(self):
        with patch("shutil.which", return_value="/bin/command"), \
             patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True), \
             patch("subprocess.call", return_value=0), \
             patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run, \
             patch("builtins.print") as output:
            self.assertEqual(0, footer.launch(["resume", "--last"]))
            self.assertIn("attach-session", output.call_args[0][0])
            self.assertEqual("has-session", run.call_args[0][0][-3])


@unittest.skipUnless(shutil.which("tmux"), "tmux is required for the terminal integration test")
class TerminalTests(unittest.TestCase):
    def test_live_footer_arguments_resize_and_shutdown(self):
        # macOS Unix socket paths are limited to 104 bytes; its default
        # per-user temporary directory is too long for a tmux socket name.
        with tempfile.TemporaryDirectory(prefix="tp-footer-test-", dir="/tmp") as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "jobs").mkdir()
            fake = root / "bin" / "codex"
            fake.write_text("#!" + sys.executable + "\nimport json, os, sys\n"
                            "from pathlib import Path\n"
                            "Path(os.environ['TP_TEST_ARGS']).write_text(json.dumps([sys.argv[1:], os.environ.get('TRAIN_PROGRESS_SESSION')]))\n"
                            "print('CODEX_PROMPT>', flush=True)\ninput()\n")
            fake.chmod(0o755)
            jobfile = root / "jobs" / "training.json"
            raw = dict(name="footer-live-test", step=50, total=100, status="running", updated_at=time.time())
            jobfile.write_text(json.dumps(raw))
            env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"],
                       TERM="xterm-256color", TRAIN_PROGRESS_DIR=str(root / "jobs"),
                       TRAIN_PROGRESS_SCOPE="all", TP_TEST_ARGS=str(root / "args.json"), TMUX_TMPDIR=tmp)
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
            args = ["a space", "literal;echo SHOULD_NOT_RUN", "$(echo bad)"]
            proc = subprocess.Popen([sys.executable, str(Path(footer.__file__).with_name("tp_view.py")), "codex"] + args,
                                    stdin=slave, stdout=slave, stderr=slave, env=env)
            os.close(slave)
            received = bytearray()

            def until(text):
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if text in received:
                        return
                    if select.select([master], [], [], 0.1)[0]:
                        try:
                            chunk = os.read(master, 65536)
                        except OSError:
                            break
                        if not chunk:
                            break
                        received.extend(chunk)
                self.fail(f"Missing {text!r} in terminal output: {bytes(received)!r}")

            try:
                until(b"50%")
                until(b"CODEX_PROMPT>")
                saved_args, session = json.loads((root / "args.json").read_text())
                self.assertEqual(args, saved_args)
                self.assertTrue(session.startswith("codex-"))
                raw.update(step=75, updated_at=time.time())
                jobfile.write_text(json.dumps(raw))
                until(b"75%")
                fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 70, 0, 0))
                raw.update(step=100, status="done", updated_at=time.time())
                jobfile.write_text(json.dumps(raw))
                until(b"done")
                os.write(master, b"\n")
                self.assertEqual(0, proc.wait(timeout=5))
                sockets = list(root.glob("tmux-*/*"))
                for sock in sockets:
                    check = subprocess.run(["tmux", "-S", str(sock), "has-session"], capture_output=True)
                    self.assertNotEqual(0, check.returncode, "tmux server was left running after Codex exited")
            finally:
                for sock in root.glob("tmux-*/*"):
                    subprocess.run(["tmux", "-S", str(sock), "kill-server"], capture_output=True)
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
                os.close(master)


if __name__ == "__main__":
    unittest.main()
