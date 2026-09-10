#!/usr/bin/env bash
# setup.sh - prepare the local security agent on Ubuntu.
# Run from inside the secagent directory:  bash setup.sh
set -euo pipefail

echo "== 1. checking ollama =="
command -v ollama >/dev/null || { echo "ollama not found"; exit 1; }
ollama --version
curl -fsS http://localhost:11434 || { echo "ollama not responding"; exit 1; }
echo

echo "== 2. host resources =="
nproc | xargs echo "vCPU:"
free -h | awk 'NR<=3'
df -h / | awk 'NR<=2'
echo

echo "== 3. swap safety net =="
if [ "$(swapon --show | wc -l)" -eq 0 ]; then
  echo "no swap found, creating 4G"
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || \
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
  echo "swap already present, skipping"
fi
echo

echo "== 4. python environment =="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q ollama
echo "ollama client installed"
echo

echo "== 5. sandbox =="
mkdir -p sandbox logs
[ -f sandbox/notes.txt ] || echo "hello from the sandbox" > sandbox/notes.txt
[ -f sandbox/users.csv ] || printf 'user,role\njayden,admin\nalice,analyst\n' > sandbox/users.csv
echo "sandbox ready: $(pwd)/sandbox"
echo

echo "== 6. models =="
echo "pulling llama3.2:1b (fast, no tool support in practice)"
ollama pull llama3.2:1b
echo "pulling qwen2.5:3b (slower, supports tool calling)"
ollama pull qwen2.5:3b
echo

echo "== 7. guardrail self-test =="
python3 tools.py
echo

echo "done. start the agent with:"
echo "  source .venv/bin/activate"
echo "  python3 secagent.py -m qwen2.5:3b --tools"
