---
name: track-jobs
description: Track long-running commands (model training, builds, data processing, downloads, big test suites) as progress bars the person can watch, and check on running jobs. Use when starting a command that will take more than a minute or two, or when asked how a training run or background job is doing.
---

# Tracking long tasks with train-progress

Jobs are JSON files in `~/.train_progress/` (or `$TRAIN_PROGRESS_DIR`), shown in
the Claude Code panel, as Codex/OpenCode notices, and by the `tp` command.

`tp` is the command below. If it is not on PATH, use
`python3 <this skill's folder>/../../python/tp_view.py` instead.

## Start a long command as a job

```sh
tp run -n "build-index" -- python build_index.py --shards 64
```

The progress comes from the command's own output (`45/300`, `[12/340]`, `37%`,
tqdm bars); the job is marked done or failed by the exit code, and the last
output line shows under the bar. Add `--total N` when the output only counts up.
Run it in the background as you would the bare command; its output passes through.

TensorBoard runs under `<project>/logs` show up by themselves; don't wrap those.

## Report from a script or a loop

```sh
tp set nightly-backup 3 --total 12 --note "copying photos"   # step 3 of 12
tp set nightly-backup                                          # one more step
tp done nightly-backup        # or: tp fail nightly-backup --note "disk full"
```

From Python, `from train_progress import Progress` (next to `tp_view.py`):
`with Progress("name", total=n) as p: ... p.update(step, loss=x, note="...")`.

## Check on jobs

`tp --once` prints every job with its bar, step, ETA and status;
`tp --once --session <id>` limits it to one agent session. Read it before
answering "how is the training going?" rather than guessing from logs.
