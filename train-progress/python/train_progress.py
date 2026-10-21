"""Report training progress to the Claude Code `train-progress` panel.

Each job writes ~/.train_progress/<name>.json (or $TRAIN_PROGRESS_DIR).

    from train_progress import Progress

    with Progress("bert-finetune", total=num_steps) as p:
        for step, batch in enumerate(loader, 1):
            loss = train_step(batch)
            p.update(step, loss=loss)
    # leaving the block marks the job done, or failed on an exception

Also: `p.track(iterable)` wraps a loop like tqdm, and `HFCallback` /
`LightningCallback` plug into Hugging Face Trainer / PyTorch Lightning.
"""

import json
import os
import socket
import time

__all__ = ["Progress", "HFCallback", "LightningCallback"]


def _dir():
    return os.environ.get("TRAIN_PROGRESS_DIR") or os.path.expanduser("~/.train_progress")


class Progress:
    def __init__(self, name, total=0, min_interval=2.0, rank_zero_only=True):
        self.name = name
        self.total = int(total)
        self.step = 0
        self.loss = None
        self.min_interval = min_interval
        self.started_at = time.time()
        # The session that started the job, for the panel's "this session" view.
        self.session = os.environ.get("TRAIN_PROGRESS_SESSION") or os.environ.get("CLAUDE_CODE_SESSION_ID")
        self._last_write = 0.0
        # Under DDP only rank 0 reports, so one job shows as one row.
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.enabled = not (rank_zero_only and rank != 0)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        self.path = os.path.join(_dir(), f"{safe}.json")
        self._write("running", force=True)

    def update(self, step=None, loss=None, total=None):
        self.step = self.step + 1 if step is None else int(step)
        if loss is not None:
            self.loss = float(loss)
        if total is not None:
            self.total = int(total)
        self._write("running")

    def track(self, iterable, total=None):
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        if total:
            self.total = int(total)
        for item in iterable:
            yield item
            self.update()

    def finish(self, status="done"):
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
            "loss": self.loss,
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
