# train-progress

Live progress bars for ML training jobs: a side panel in Claude Code, and a
`tp` command for any other terminal (Codex, OpenCode, tmux, plain shell).

```
● rsl_rl/my_task/2026-10-21_11-30-11                       ETA 2h12m
█████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   22%
674/3000 · reward 40.06 · my-gpu-box
```

## Install

**Claude Code** (the panel), at the prompt:

```
/plugin install train-progress --marketplace hanlongli/train-progress
```

Then `/train` opens the panel. It opens by itself when a job is running.

**Codex, OpenCode, or any terminal** (the `tp` command):

```sh
git clone https://github.com/hanlongli/train-progress && train-progress/train-progress/install.sh
tp                  # live view; Ctrl+C quits
tp --once           # print once
```

Both need `python3` (standard library only).

## Where jobs come from

- **TensorBoard runs** (rsl_rl, Isaac Lab, anything with a SummaryWriter) show
  up with no code change. A watcher reads the event files under each Claude Code
  session's `<project>/logs`, under the folders in the plugin's `logRoots`
  option, and under `./logs` wherever you run `tp`. The total comes from
  `params/agent.yaml`'s `max_iterations` (or `--total`), and the metric shown is
  `Train/mean_reward`. A run whose trainer process exits before the end shows as failed.
- **Your own code** can report directly:

  ```python
  from train_progress import Progress
  with Progress("bert-ft", total=num_steps) as p:
      for step, batch in enumerate(loader, 1):
          p.update(step, loss=train_step(batch))
  ```

  It also comes with `HFCallback("name")` for Hugging Face Trainer and `LightningCallback("name")` for Lightning.

Every job is one JSON file in `~/.train_progress/` (or `$TRAIN_PROGRESS_DIR`).
Only one watcher runs per machine.

## All jobs, or this session's

Each job records the session that started it, read from
`CLAUDE_CODE_SESSION_ID`, which Claude Code sets in every command it runs.

- **Claude Code:** choose the default in `/config` → train-progress → *Jobs to
  show* (`all` or `session`). The panel's `all sessions ⇄` button or
  `/train all` / `/train session` switches it for the current session.
- **Other tools:** start training with `TRAIN_PROGRESS_SESSION=<name>` set, and
  view it with `tp --session <name>`.

## Options (Claude Code `/config`)

| Option | Default | |
| --- | --- | --- |
| Jobs to show | `all` | `all` or `session` |
| Extra TensorBoard log folders | (empty) | comma-separated |
| Watch TensorBoard logs | on | turn off to show only helper-reported jobs |
