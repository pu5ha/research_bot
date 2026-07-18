#!/usr/bin/env bash
# One-time VPS bootstrap for research-paper-bot. Idempotent — safe to re-run.
# Run from the repo root:  bash deploy/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
echo "==> Setting up research-paper-bot in $(pwd)"

# 1. System deps (skip apt if we can't sudo, e.g. non-Debian host)
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y python3-venv python3-pip
fi

# 1b. On small boxes (<3GB RAM, no swap) add a 2GB swap file so embedding can't OOM.
if command -v free >/dev/null 2>&1; then
  mem_mb=$(free -m | awk '/^Mem:/{print $2}')
  swap_mb=$(free -m | awk '/^Swap:/{print $2}')
  if [ "${mem_mb:-9999}" -lt 3000 ] && [ "${swap_mb:-0}" -lt 100 ] && [ ! -f /swapfile ]; then
    echo "==> Low RAM (${mem_mb}MB) — creating a 2GB swap file"
    sudo fallocate -l 2G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  fi
fi

# 2. Virtualenv + CPU-only torch (avoids pulling ~2GB of CUDA wheels), then the rest
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip -q
# On x86 use the CPU wheel index to skip the ~2GB CUDA download. On ARM64 the
# default PyPI wheel is already CPU-only, so install it plainly.
case "$(uname -m)" in
  x86_64|amd64)
    echo "==> Installing CPU-only torch (x86)"
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    ;;
  *)
    echo "==> Installing torch ($(uname -m))"
    pip install -q torch
    ;;
esac
echo "==> Installing remaining requirements"
pip install -q -r requirements.txt

# 3. Host config: create from example and disable local-LLM summaries (no Ollama here)
if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
  sed -i 's/summarizer: {enabled: true/summarizer: {enabled: false/' config.yaml
  echo "==> config.yaml created (summaries disabled — messages use a truncated abstract)"
fi

# 4. Secrets stub + logs dir
[ -f .env ] || cp .env.example .env
mkdir -p logs

cat <<'NEXT'

==> Bootstrap done. Next steps:

  1. Fill in your secrets:
       nano .env          # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, CONTACT_EMAIL

  2. Build the taste profile and smoke-test (downloads the embedding model ~130MB):
       . .venv/bin/activate
       python -m src.main refresh-taste
       python -m src.main run-once

  3. Schedule it (see README §Deployment): cron for run-once + the systemd vote loop.
NEXT
