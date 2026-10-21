# train-progress

Live progress bars for training runs and any other long task (builds, data
jobs, downloads, test suites). In Claude Code they sit in a bar above the
prompt; Codex and OpenCode tell you when a job finishes, fails or stalls; and
the `tp` command shows them in any terminal.

```
● rsl_rl/my_task/2026-10-21_11-30-11  ████░░░░░░░░░░░░  22%  674/3000 · reward 40.06 · ETA 2h12m
● build-index                         ░░░░░░░░░░░░░░░░    …  shard 17 done · 3m12s
✓ bert-ft                             ████████████████ 100%  1000/1000 · loss 0.4213 · done 2m ago
```

## Install

**Claude Code**, at the prompt:

```
/plugin install train-progress --marketplace hanlongli/train-progress
```

The bar shows above the prompt while a job is running or just ended.
`/progress hide` / `/progress` hides and shows it, `/progress panel` opens the
full list in a side panel, `/progress all|session` picks whose jobs to show.

**Codex**:

```sh
codex plugin marketplace add hanlongli/train-progress
codex plugin add train-progress@local-mods
```

Then review and trust its hooks with `/hooks` in Codex. A new session lists the
jobs that are running, and after each turn Codex shows which jobs finished,
failed or stalled. The plugin's `progress` skill teaches Codex to start long
commands with `tp run` and to check on them with `tp --once`.

**OpenCode, or any terminal** (the `tp` command):

```sh
git clone https://github.com/hanlongli/train-progress && train-progress/train-progress/install.sh
tp                  # live view; Ctrl+C quits
tp --once           # print once
```

When OpenCode is set up on the machine (or with `install.sh --opencode`) this
also installs the OpenCode plugin: a toast when a job finishes, fails or stalls,
a `progress` tool the model can read jobs with, and a `/progress` command.

All of them need `python3` (standard library only).

## Where jobs come from

- **Any command**: `tp run` runs it as a job. The progress comes from its own
  output (`45/300`, `[12/340]`, `37%`, tqdm bars), the last line it printed
  shows under the bar, and the exit code marks it done or failed.

  ```sh
  tp run -n build-index -- python build_index.py --shards 64
  tp run -- make -j8
  ```

- **Shell scripts** report steps themselves:

  ```sh
  tp set nightly-backup 3 --total 12 --note "copying photos"
  tp set nightly-backup            # one more step
  tp done nightly-backup           # or: tp fail nightly-backup --note "disk full"
  ```

- **Python code**:

  ```python
  from train_progress import Progress
  with Progress("bert-ft", total=num_steps) as p:
      for step, batch in enumerate(loader, 1):
          p.update(step, loss=train_step(batch))

  with Progress("resize-images") as p:      # no total: shows elapsed time
      for path in paths:
          p.update(note=path.name)
  ```

  `update` takes any named number (`loss=`, `reward=`, `files=`). It also comes
  with `HFCallback("name")` for Hugging Face Trainer and `LightningCallback("name")`
  for Lightning.

- **TensorBoard runs** (rsl_rl, Isaac Lab, anything with a SummaryWriter) show
  up with no code change. A watcher reads the event files under each session's
  `<project>/logs`, under the folders in the plugin's `logRoots` option, and
  under `./logs` wherever you run `tp`. The total comes from
  `params/agent.yaml`'s `max_iterations` (or `--total`), and the metric shown is
  `Train/mean_reward`. A run whose trainer process exits before the end shows as failed.

Every job is one JSON file in `~/.train_progress/` (or `$TRAIN_PROGRESS_DIR`).
Only one watcher runs per machine. A running job that has not reported for 15
minutes shows as stalled.

## All jobs, or this session's

Each job records the agent session that started it: `CLAUDE_CODE_SESSION_ID`
and `CODEX_THREAD_ID` are set by Claude Code and Codex in every command they
run, and the OpenCode plugin sets `TRAIN_PROGRESS_SESSION`.

- **Claude Code:** choose the default in `/config` → train-progress → *Jobs to
  show* (`all` or `session`), or switch it with `/progress all|session`.
- **Codex, OpenCode:** set `TRAIN_PROGRESS_SCOPE=session` to hear only about
  the session's own jobs.
- **Anywhere else:** start the job with `TRAIN_PROGRESS_SESSION=<name>` set,
  and view it with `tp --session <name>`.

## Options (Claude Code `/config`)

| Option | Default | |
| --- | --- | --- |
| Jobs to show | `all` | `all` or `session` |
| Extra TensorBoard log folders | (empty) | comma-separated |
| Watch TensorBoard logs | on | turn off to show only jobs reported by `tp` or `train_progress.py` |
