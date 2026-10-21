#!/bin/sh
# Installs the `tp` command (terminal progress bars, `tp run`, `tp set`) for use
# beside Codex, OpenCode or any shell, and the OpenCode plugin when OpenCode is
# set up here (or with --opencode). The Claude Code mod is installed with
# /plugin and the Codex plugin with `codex plugin add` instead.
set -e
here=$(cd "$(dirname "$0")" && pwd)
dest="${XDG_DATA_HOME:-$HOME/.local/share}/train-progress"
bin="$HOME/.local/bin"
mkdir -p "$dest" "$bin"
cp "$here"/python/*.py "$dest"/
cat > "$bin/tp" <<SH
#!/bin/sh
exec python3 "$dest/tp_view.py" "\$@"
SH
chmod +x "$bin/tp"
echo "Installed: $bin/tp  (python helper: $dest/train_progress.py)"

opencode="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
if [ "$1" = "--opencode" ] || [ -d "$opencode" ] || command -v opencode >/dev/null 2>&1; then
  mkdir -p "$opencode/plugins" "$opencode/commands"
  cp "$here/opencode/train-progress.js" "$opencode/plugins/"
  cat > "$opencode/commands/progress.md" <<MD
---
description: Progress of training runs and other long jobs
---
These are the jobs train-progress tracks on this machine:

!\`$bin/tp --once\`

Summarize them in a few lines: what is running and how far along, what finished
or failed and when. Say nothing else unless something failed or stalled.
MD
  echo "Installed: OpenCode plugin ($opencode/plugins/train-progress.js) and /progress command"
fi

case ":$PATH:" in *":$bin:"*) ;; *) echo "Add $bin to your PATH to run 'tp'." ;; esac
