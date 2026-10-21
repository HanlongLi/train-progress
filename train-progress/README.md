# train-progress

Live progress bars for training runs and any other long task (builds, data
jobs, downloads, test suites). In Claude Code they sit in a bar under the
prompt; Codex CLI gets a live footer with `tp codex`; Codex and OpenCode also
tell you when a job finishes, fails or stalls; and
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

The bar shows under the prompt while a job is running or just ended.
`/progress hide` / `/progress` hides and shows it, `/progress panel` opens the
full list in a side panel and `/progress close` closes it, and
`/progress all|session` picks whose jobs to show.

**Codex**:

```sh
codex plugin marketplace add hanlongli/train-progress
codex plugin add train-progress@local-mods
```

Then review and trust its hooks with `/hooks` in Codex. A new session lists the
jobs that are running, and after each turn Codex shows which jobs finished,
failed or stalled. The plugin's `progress` skill teaches Codex to start long
commands with `tp run` and to check on them with `tp --once`.

For **live bars below the Codex CLI prompt**, install the terminal helper and
tmux, then launch Codex through `tp`:

```sh
./train-progress/install.sh   # from this repository's root
brew install tmux            # macOS; on Linux use your package manager
tp codex
tp codex resume --last       # Codex arguments work here too
```

The bottom three terminal rows show job bars, percentages, step counts, metrics
and ETA. They refresh every second, including while Codex is busy or waiting
for input. Running and stalled jobs appear first; finished jobs stay for ten
minutes. Use `tp --once` to see the full list.

This uses a separate tmux server to reserve the footer; it does not patch Codex
or edit your Codex or tmux configuration. Ordinary `codex` sessions keep their
existing interface. Exit Codex normally to close the footer, or press tmux's
`Ctrl+B`, then `D` to detach and keep it running; the launcher prints the command
to reattach. The launcher also works inside an existing tmux pane.

To show only jobs started by this launch:

```sh
TRAIN_PROGRESS_SCOPE=session tp codex
```

The launcher gives its commands a shared `TRAIN_PROGRESS_SESSION` label, so
`tp run`, Python helpers and Codex hook notices agree on which jobs belong to
the launch. TensorBoard jobs without a session label appear in the default
all-jobs view. This footer is for the CLI; the Codex desktop app still uses
hook notices.

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

## Testing the Codex footer

```sh
python3 -m unittest discover -s train-progress/python -v
```

The tests include a real tmux/PTY check for live updates, argument forwarding,
terminal resizing and shutdown. That check needs tmux and local socket access.
