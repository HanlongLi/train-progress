"""Report the progress of any long task (training, a data job, a batch of
downloads) to the train-progress panel in Claude Code, Codex, OpenCode or `tp`.

Each job writes ~/.train_progress/<name>.json (or $TRAIN_PROGRESS_DIR).

    from train_progress import Progress

    with Progress("bert-finetune", total=num_steps) as p:
        for step, batch in enumerate(loader, 1):
            loss = train_step(batch)
            p.update(step, loss=loss)
    # leaving the block marks the job done, or failed on an exception

    with Progress("resize-images") as p:          # no total: shows elapsed time
        for path in paths:
            p.update(note=path.name)              # a short line of text under the bar

`update` takes any named numbers (`loss=`, `reward=`, `files=`); the first one
given is shown beside the bar. Also: `p.track(iterable)` wraps a loop like tqdm,
and `HFCallback` / `LightningCallback` plug into Hugging Face Trainer / PyTorch
Lightning.
"""

import json
import os
import socket
import time

__all__ = ["Progress", "HFCallback", "LightningCallback"]


# The agent session a job belongs to: set explicitly, or by the host agent in
# every command it runs (Claude Code, Codex; the OpenCode plugin sets the first).
SESSION_VARS = ("TRAIN_PROGRESS_SESSION", "CLAUDE_CODE_SESSION_ID", "CODEX_THREAD_ID")


def _dir():
    return os.environ.get("TRAIN_PROGRESS_DIR") or os.path.expanduser("~/.train_progress")


def session_id():
    for var in SESSION_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return None


class Progress:
    def __init__(self, name, total=0, min_interval=2.0, rank_zero_only=True, resume=False):
        self.name = name
        self.total = int(total)
        self.step = 0
        self.metric_name = None
        self.metric = None
        self.note = None
        self.min_interval = min_interval
        self.started_at = time.time()
        # The session that started the job, for the panel's "this session" view.
        self.session = session_id()
        self._last_write = 0.0
        # Under DDP only rank 0 reports, so one job shows as one row.
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.enabled = not (rank_zero_only and rank != 0)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        self.path = os.path.join(_dir(), f"{safe}.json")
        if resume:  # carry on a job another process started (`tp set`)
            self._load()
        self._write("running", force=True)

    def _load(self):
        try:
            with open(self.path) as f:
                old = json.load(f)
        except (OSError, ValueError):
            return
        if old.get("status") != "running":
            return  # a finished job of the same name: start over
        self.step = int(old.get("step") or 0)
        self.total = self.total or int(old.get("total") or 0)
        self.metric_name, self.metric = old.get("metric_name"), old.get("metric")
        self.note = old.get("note")
        self.started_at = old.get("started_at") or self.started_at
        self.session = self.session or old.get("session")

    def update(self, step=None, total=None, note=None, **metrics):
        self.step = self.step + 1 if step is None else int(step)
        if total is not None:
            self.total = int(total)
        if note is not None:
            self.note = str(note)[:200]
        for name, value in metrics.items():
            if value is not None:
                self.metric_name, self.metric = name, float(value)
                break
        self._write("running")

    def track(self, iterable, total=None):
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        if total:
            self.total = int(total)
        for item in iterable:
            yield item
            self.update()

    def finish(self, status="done", note=None):
        if note is not None:
            self.note = str(note)[:200]
        self._write(status, force=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish("failed" if exc_type else "done")
        return False

    def _write(self, status, force=False):
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self._last_write < self.min_interval:
            return
        self._last_write = now
        data = {
            "name": self.name,
            "step": self.step,
            "total": self.total,
            "metric": self.metric,
            "metric_name": self.metric_name,
            "note": self.note,
            "status": status,
            "started_at": self.started_at,
            "updated_at": now,
            "host": socket.gethostname(),
            "session": self.session,
        }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)  # atomic, so the panel never reads half a file


def HFCallback(name):
    """A transformers.TrainerCallback: Trainer(..., callbacks=[HFCallback("run")])."""
    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.p = Progress(name, total=state.max_steps)

        def on_log(self, args, state, control, logs=None, **kw):
            self.p.update(state.global_step, loss=(logs or {}).get("loss"))

        def on_step_end(self, args, state, control, **kw):
            self.p.update(state.global_step)

        def on_train_end(self, args, state, control, **kw):
            self.p.finish("done")

    return _CB()


def LightningCallback(name):
    """A lightning Callback: Trainer(callbacks=[LightningCallback("run")])."""
    try:
        from lightning.pytorch.callbacks import Callback
    except ImportError:
        from pytorch_lightning.callbacks import Callback

    class _CB(Callback):
        def on_train_start(self, trainer, module):
            self.p = Progress(name, total=trainer.estimated_stepping_batches)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = trainer.callback_metrics.get("train_loss", trainer.callback_metrics.get("loss"))
            self.p.update(trainer.global_step, loss=None if loss is None else float(loss))

        def on_train_end(self, trainer, module):
            self.p.finish("done")

        def on_exception(self, trainer, module, exception):
            self.p.finish("failed")

    return _CB()
