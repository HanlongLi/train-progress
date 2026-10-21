#!/bin/sh
# Installs the `tp` command (terminal progress bars) for use beside Codex,
# OpenCode or any shell. The Claude Code panel is installed with /plugin instead.
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
case ":$PATH:" in *":$bin:"*) ;; *) echo "Add $bin to your PATH to run 'tp'." ;; esac
