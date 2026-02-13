#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_default="/home/mstf/aider/.env.cli2api"
env_local="$repo_root/.env.cli2api"

if command -v aider >/dev/null 2>&1; then
  aider_bin="$(command -v aider)"
elif [[ -x "/home/mstf/.local/bin/aider" ]]; then
  aider_bin="/home/mstf/.local/bin/aider"
else
  echo "aider binary not found. Install with: pipx install aider-chat" >&2
  exit 1
fi

if [[ -f "$env_default" ]]; then
  env_file="$env_default"
elif [[ -f "$env_local" ]]; then
  env_file="$env_local"
else
  cat >&2 <<'EOF'
No cli2api env file found.
Expected one of:
  /home/mstf/aider/.env.cli2api
  ./.env.cli2api

Create it from .env.cli2api.example, then run again.
EOF
  exit 1
fi

exec "$aider_bin" --env-file "$env_file" "$@"
